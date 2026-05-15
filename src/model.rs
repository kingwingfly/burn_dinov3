use burn::{
    module::{Module, Param},
    nn::{
        Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{Bool, FloatDType, Tensor, backend::Backend, module, ops::AttentionModuleOptions},
};

#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    pub proj: Conv2d<B>,
    pub patch_size: usize,
    pub embed_dim: usize,
}

impl<B: Backend> PatchEmbed<B> {
    pub fn new(in_chans: usize, embed_dim: usize, patch_size: usize, device: &B::Device) -> Self {
        let proj = Conv2dConfig::new([in_chans, embed_dim], [patch_size, patch_size])
            .with_stride([patch_size, patch_size])
            .init(device);
        Self {
            proj,
            patch_size,
            embed_dim,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 3>, usize, usize) {
        let [batch_size, _, height, width] = x.dims();
        let (height, width) = (height / self.patch_size, width / self.patch_size);
        let x = self.proj.forward(x);
        let x = x.reshape([batch_size, self.embed_dim, height * width]);
        (x.swap_dims(1, 2), height, width)
    }
}

#[derive(Module, Debug)]
pub struct RopePositionEmbedding<B: Backend> {
    // Facebook engineers save periods in its pth model.
    pub periods: Param<Tensor<B, 1>>,
    pub d_head: usize,
}

impl<B: Backend> RopePositionEmbedding<B> {
    pub fn new(embed_dim: usize, num_heads: usize, base: f32, device: &B::Device) -> Self {
        let d_head = embed_dim / num_heads;

        let periods = Param::from_tensor(
            Tensor::from_floats([base], device)
                .powf(Tensor::arange_step(0..d_head as i64, 4, device).float() / d_head as f32),
        )
        .no_grad(); // do not update

        Self { periods, d_head }
    }

    pub fn forward(&self, height: usize, width: usize) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let device = self.periods.device();
        let hw = height * width;

        let coords_h = (Tensor::arange(0..height as i64, &device).float() + 0.5) / (height as f32);
        let coords_w = (Tensor::arange(0..width as i64, &device).float() + 0.5) / (width as f32);

        let ch = coords_h
            .reshape([height, 1])
            .repeat_dim(1, width)
            .reshape([hw, 1]);
        let cw = coords_w
            .reshape([1, width])
            .repeat_dim(0, height)
            .reshape([hw, 1]);
        let mut coords = Tensor::cat(vec![ch, cw], 1);
        coords = coords * 2.0 - 1.0;

        let angles = coords.reshape([hw, 2, 1]) * std::f64::consts::PI * 2.0
            / self
                .periods
                .val()
                .cast(FloatDType::F32) // After loaded from facebook pth, it's BF16
                .reshape([1, 1, self.periods.dims()[0]]);
        let angles = angles.reshape([hw, 2 * self.periods.dims()[0]]);
        let angles_tiled = Tensor::cat(vec![angles.clone(), angles], 1);

        let sin = angles_tiled.clone().sin();
        let cos = angles_tiled.cos();

        (sin, cos)
    }
}

#[derive(Module, Debug)]
pub struct LinearKMaskedBias<B: Backend> {
    pub linear: Linear<B>,
    pub bias_mask: Param<Tensor<B, 1>>,
}

impl<B: Backend> LinearKMaskedBias<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let masked_bias = self
            .linear
            .bias
            .as_ref()
            .map(|b| b.val() * self.bias_mask.val());

        module::linear(input, self.linear.weight.val(), masked_bias)
    }
}

#[derive(Module, Debug)]
pub struct LayerScale<B: Backend> {
    pub gamma: Param<Tensor<B, 1>>,
}

impl<B: Backend> LayerScale<B> {
    pub fn new(dim: usize, init_values: f32, device: &B::Device) -> Self {
        let gamma = Param::from_tensor(Tensor::ones([dim], device) * init_values);
        Self { gamma }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let dim = x.shape()[2];
        x * self.gamma.val().reshape([1, 1, dim])
    }
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    pub qkv: LinearKMaskedBias<B>,
    pub proj: Linear<B>,
    pub drop_out: Dropout,
    pub num_heads: usize,
}

impl<B: Backend> Attention<B> {
    pub fn new(dim: usize, num_heads: usize, device: &B::Device) -> Self {
        Self {
            qkv: LinearKMaskedBias {
                linear: LinearConfig::new(dim, dim * 3).with_bias(true).init(device),
                bias_mask: Param::from_tensor(Tensor::zeros([dim * 3], device)),
            },
            proj: LinearConfig::new(dim, dim).with_bias(true).init(device),
            drop_out: DropoutConfig::new(0.0).init(), // did not see any config other than 0 in facebook repo
            num_heads,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        repo: Option<&(Tensor<B, 2>, Tensor<B, 2>)>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, dim] = x.dims();
        let qkv = self.qkv.forward(x);

        let qkv = qkv.reshape([batch_size, seq_len, 3, self.num_heads, dim / self.num_heads]);

        let [mut q, mut k, v]: [Tensor<B, 4>; 3] = qkv
            .chunk(3, 2)
            .into_iter()
            .map(|tensor| tensor.squeeze_dim::<4>(2).swap_dims(1, 2))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        if let Some((sin, cos)) = repo {
            q = Self::apply_rope(q, sin, cos);
            k = Self::apply_rope(k, sin, cos);
        }

        let out = module::attention(q, k, v, None, None, AttentionModuleOptions::default());
        let out = out.swap_dims(1, 2).reshape([batch_size, seq_len, dim]);

        self.drop_out.forward(self.proj.forward(out))
    }

    fn apply_rope(x: Tensor<B, 4>, sin: &Tensor<B, 2>, cos: &Tensor<B, 2>) -> Tensor<B, 4> {
        let [b, nh, seq, head_dim] = x.dims();
        let [rope_seq, _h_dim] = sin.dims();
        let num_cls_and_storage_tokens = seq - rope_seq;
        let prefix = x
            .clone()
            .slice([..b, ..nh, ..num_cls_and_storage_tokens, ..head_dim]);
        let mut rope = x.slice([0..b, 0..nh, num_cls_and_storage_tokens..seq, 0..head_dim]);

        let half_dim = head_dim / 2;
        let x1 = rope.clone().slice([..b, ..nh, ..seq, ..half_dim]);
        let x2 = rope
            .clone()
            .slice([0..b, 0..nh, 0..seq, half_dim..head_dim]);

        let x_half = Tensor::cat(vec![x2.mul_scalar(-1.0), x1], 3);

        rope = (rope * cos.clone().reshape([1, 1, rope_seq, head_dim]))
            + (x_half * sin.clone().reshape([1, 1, rope_seq, head_dim]));

        Tensor::cat(vec![prefix, rope], 2)
    }
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    pub fc1: Linear<B>,
    pub act: Gelu,
    pub fc2: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    pub fn new(in_features: usize, hidden_features: usize, device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(in_features, hidden_features).init(device),
            act: Gelu::new(),
            fc2: LinearConfig::new(hidden_features, in_features).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);
        let x = self.act.forward(x);
        self.fc2.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    pub norm1: LayerNorm<B>,
    pub attn: Attention<B>,
    pub ls1: LayerScale<B>,
    pub norm2: LayerNorm<B>,
    pub mlp: Mlp<B>,
    pub ls2: LayerScale<B>,
}

impl<B: Backend> Block<B> {
    pub fn new(dim: usize, num_heads: usize, ffn_ratio: f64, device: &B::Device) -> Self {
        let hidden_dim = (dim as f64 * ffn_ratio) as usize;
        Self {
            norm1: LayerNormConfig::new(dim).with_bias(true).init(device),
            attn: Attention::new(dim, num_heads, device),
            ls1: LayerScale::new(dim, 1e-5, device),
            norm2: LayerNormConfig::new(dim).with_bias(true).init(device),
            mlp: Mlp::new(dim, hidden_dim, device),
            ls2: LayerScale::new(dim, 1e-5, device),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        repo: Option<&(Tensor<B, 2>, Tensor<B, 2>)>,
    ) -> Tensor<B, 3> {
        let x = self
            .ls1
            .forward(self.attn.forward(self.norm1.forward(x.clone()), repo))
            + x; // ResNet

        self.ls2
            .forward(self.mlp.forward(self.norm2.forward(x.clone())))
            + x
    }
}

#[derive(Module, Debug)]
pub struct DinoVisionTransformer<B: Backend> {
    pub patch_embed: PatchEmbed<B>,
    pub cls_token: Param<Tensor<B, 3>>,
    pub storage_tokens: Param<Tensor<B, 3>>,
    pub rope_embed: RopePositionEmbedding<B>,
    pub blocks: Vec<Block<B>>,
    pub norm: LayerNorm<B>,
    pub mask_token: Param<Tensor<B, 2>>,
}

impl<B: Backend> DinoVisionTransformer<B> {
    pub fn new(
        patch_size: usize,
        embed_dim: usize,
        num_storage_tokens: usize,
        depth: usize,
        num_heads: usize,
        ffn_ratio: f64,
        device: &B::Device,
    ) -> Self {
        let patch_embed = PatchEmbed::new(3, embed_dim, patch_size, device);
        let cls_token = Param::from_tensor(Tensor::zeros([1, 1, embed_dim], device));
        let storage_tokens =
            Param::from_tensor(Tensor::zeros([1, num_storage_tokens, embed_dim], device));
        let rope_embed = RopePositionEmbedding::new(embed_dim, num_heads, 100.0, device).no_grad();
        let blocks = vec![Block::new(embed_dim, num_heads, ffn_ratio, device); depth];
        let norm = LayerNormConfig::new(embed_dim).init(device);
        let mask_token = Param::from_tensor(Tensor::zeros([1, embed_dim], device));
        Self {
            patch_embed,
            cls_token,
            storage_tokens,
            rope_embed,
            blocks,
            norm,
            mask_token,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>, masks: Option<&Tensor<B, 2, Bool>>) -> Tensor<B, 3> {
        let (mut x, height, width) = self.patch_embed.forward(x);
        let [batch_size, seq, dim] = x.dims();

        if let Some(masks) = masks {
            x = x.mask_where(
                masks
                    .clone()
                    .reshape([1, seq, dim])
                    .repeat_dim(0, batch_size),
                self.mask_token
                    .val()
                    .reshape([1, 1, dim])
                    .repeat_dim(1, seq)
                    .repeat_dim(0, batch_size),
            );
        }

        // rotary position encoding 2d
        let repo = self.rope_embed.forward(height, width);

        let cls_token_batch = self.cls_token.val().repeat_dim(0, batch_size);
        let storage_tokens_batch = self.storage_tokens.val().repeat_dim(0, batch_size);
        x = Tensor::cat(vec![cls_token_batch, storage_tokens_batch, x], 1);

        for block in &self.blocks {
            x = block.forward(x, Some(&repo));
        }

        self.norm.forward(x)
    }
}

pub fn vit_small<B: Backend>(patch_size: usize, device: &B::Device) -> DinoVisionTransformer<B> {
    DinoVisionTransformer::new(patch_size, 384, 4, 12, 6, 4.0, device)
}

pub fn vit_base<B: Backend>(patch_size: usize, device: &B::Device) -> DinoVisionTransformer<B> {
    DinoVisionTransformer::new(patch_size, 768, 4, 12, 12, 4.0, device)
}

pub fn vit_large<B: Backend>(patch_size: usize, device: &B::Device) -> DinoVisionTransformer<B> {
    DinoVisionTransformer::new(patch_size, 1024, 4, 24, 16, 4.0, device)
}

pub fn vit_so400m<B: Backend>(patch_size: usize, device: &B::Device) -> DinoVisionTransformer<B> {
    DinoVisionTransformer::new(patch_size, 1152, 4, 27, 18, 3.777777778, device)
}

pub fn vit_huge2<B: Backend>(patch_size: usize, device: &B::Device) -> DinoVisionTransformer<B> {
    DinoVisionTransformer::new(patch_size, 1280, 4, 32, 20, 4.0, device)
}

pub fn vit_giant2<B: Backend>(patch_size: usize, device: &B::Device) -> DinoVisionTransformer<B> {
    DinoVisionTransformer::new(patch_size, 1536, 4, 40, 24, 4.0, device)
}

pub fn vit_7b<B: Backend>(patch_size: usize, device: &B::Device) -> DinoVisionTransformer<B> {
    DinoVisionTransformer::new(patch_size, 4096, 4, 40, 32, 3.0, device)
}
