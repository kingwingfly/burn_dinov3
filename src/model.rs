use std::iter::repeat_with;

use burn::{
    config::Config,
    module::{Module, Param},
    nn::{
        Dropout, DropoutConfig, Gelu, Initializer, LayerNorm, LayerNormConfig, Linear,
        LinearConfig,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{Bool, FloatDType, Tensor, backend::Backend, module, ops::AttentionModuleOptions},
};

#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    pub proj: Conv2d<B>,
}

impl<B: Backend> PatchEmbed<B> {
    pub fn new(in_chans: usize, embed_dim: usize, patch_size: usize, device: &B::Device) -> Self {
        let proj = Conv2dConfig::new([in_chans, embed_dim], [patch_size, patch_size])
            .with_stride([patch_size, patch_size])
            .init(device);
        Self { proj }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 3>, usize, usize) {
        let x = self.proj.forward(x);
        let [_, _, height, width] = x.dims();
        let x = x.flatten(2, -1);
        (x.swap_dims(1, 2), height, width)
    }
}

#[derive(Module, Debug)]
pub struct RopePositionEmbedding<B: Backend> {
    // Facebook engineers save periods in its pth model.
    pub periods: Param<Tensor<B, 1>>,
}

impl<B: Backend> RopePositionEmbedding<B> {
    pub fn new(embed_dim: usize, num_heads: usize, base: f32, device: &B::Device) -> Self {
        let d_head = embed_dim / num_heads;

        let periods = Param::from_tensor(
            Tensor::from_floats([base], device)
                .powf(Tensor::arange_step(0..d_head as i64, 4, device).float() / d_head as f32),
        )
        .no_grad(); // do not update

        Self { periods }
    }

    pub fn forward(&self, height: usize, width: usize) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let device = self.periods.device();

        let coords_h = (Tensor::arange(0..height as i64, &device).float() + 0.5) / (height as f32);
        let coords_w = (Tensor::arange(0..width as i64, &device).float() + 0.5) / (width as f32);

        let ch = coords_h
            .unsqueeze_dim::<2>(1) // [h, 1]
            .repeat_dim(1, width) // [h, w]
            .reshape([-1, 1]); // [hw, 1]
        let cw = coords_w
            .unsqueeze::<2>() // [1, w]
            .repeat_dim(0, height) // [h, w]
            .reshape([-1, 1]); // [hw, 1]
        let mut coords = Tensor::cat(vec![ch, cw], 1); // [hw, 2]
        coords = coords * 2.0 - 1.0;

        // [hw, 2, 1] / [1, 1, d_head/4] -> [hw, 2, d_head/4]
        let angles = coords.unsqueeze_dim::<3>(2) * std::f64::consts::PI * 2.0
            / self
                .periods
                .val()
                .cast(FloatDType::F32) // After loaded from facebook pth, it's BF16
                .unsqueeze::<3>();
        let angles = angles.flatten(1, 2); // [hw, d_head/2]
        let angles_tiled = Tensor::cat(vec![angles.clone(), angles], 1); // [hw, d_head]

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

pub trait LoRALayer<B: Backend>: Module<B> {
    type Config: LoRALayerConfig<B, LoRA = Self>;

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3>;
    fn alpha_div_dim(&self) -> f64;
}

pub trait LoRALayerConfig<B: Backend> {
    type LoRA: LoRALayer<B>;

    fn init(&self, dim: usize, device: &B::Device) -> Self::LoRA;
}

#[derive(Config, Debug)]
pub struct LoRAConfig {
    pub rank: usize,
    pub alpha: f64,
    #[config(default = "Initializer::KaimingUniform{gain:1.0/3.0f64.sqrt(), fan_out_only:false}")]
    pub a_initializer: Initializer,
    #[config(default = "Initializer::Zeros")]
    pub b_initializer: Initializer,
}

#[derive(Module, Debug)]
pub struct LoRA<B: Backend> {
    pub a: Param<Tensor<B, 2>>,
    pub b_q: Param<Tensor<B, 2>>,
    pub b_v: Param<Tensor<B, 2>>,
    pub alpha_div_dim: f64,
}

impl<B: Backend> LoRALayerConfig<B> for LoRAConfig {
    type LoRA = LoRA<B>;

    /// dim: one of q/k/v's dim
    fn init(&self, dim: usize, device: &B::Device) -> Self::LoRA {
        LoRA {
            a: self.a_initializer.init_with(
                [dim, self.rank],
                Some(dim * 3),
                Some(self.rank),
                device,
            ),
            b_q: self.b_initializer.init([self.rank, dim], device),
            b_v: self.b_initializer.init([self.rank, dim], device),
            alpha_div_dim: self.alpha / dim as f64,
        }
    }
}

impl<B: Backend> LoRALayer<B> for LoRA<B> {
    type Config = LoRAConfig;

    /// x: [batch_size, seq, dim]
    /// out: [batch_size, seq, dim * 3]
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // [b, seq, dim] @ [1, dim, r] -> [b, seq, r]
        let hidden = x.matmul(self.a.val().unsqueeze());

        // [b, seq, r] @ [1, r, dim] -> [b, seq, dim]
        let lora_q = hidden.clone().matmul(self.b_q.val().unsqueeze());
        // [b, seq, r] @ [1, r, dim] -> [b, seq, dim]
        let lora_v = hidden.matmul(self.b_v.val().unsqueeze());

        let lora_k = lora_q.zeros_like();

        Tensor::cat(vec![lora_q, lora_k, lora_v], 2) // [b, seq, dim * 3]
    }

    fn alpha_div_dim(&self) -> f64 {
        self.alpha_div_dim
    }
}

#[derive(Config, Debug)]
pub struct AttentionConfig {
    pub dim: usize,
    pub num_heads: usize,
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend, L: Module<B> = LoRA<B>> {
    pub qkv: LinearKMaskedBias<B>,
    pub proj: Linear<B>,
    pub drop_out: Dropout,
    pub lora: Option<L>,
    pub num_heads: usize,
}

impl AttentionConfig {
    pub fn init<B: Backend, L: LoRALayer<B>>(
        &self,
        lora: Option<L>,
        device: &B::Device,
    ) -> Attention<B, L> {
        Attention {
            qkv: LinearKMaskedBias {
                linear: LinearConfig::new(self.dim, self.dim * 3)
                    .with_bias(true)
                    .init(device),
                bias_mask: Param::from_tensor(Tensor::zeros([self.dim * 3], device)).no_grad(),
            },
            proj: LinearConfig::new(self.dim, self.dim)
                .with_bias(true)
                .init(device),
            drop_out: DropoutConfig::new(0.0).init(), // did not see any config other than 0 in facebook repo
            lora,
            num_heads: self.num_heads,
        }
    }
}

impl<B: Backend, L: LoRALayer<B>> Attention<B, L> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        repo: Option<&(Tensor<B, 2>, Tensor<B, 2>)>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, dim] = x.dims();
        // [b, seq, dim] @ [1, dim, dim * 3] -> [b, seq, dim * 3]
        let mut qkv = self.qkv.forward(x.clone());

        if let Some(lora) = self.lora.as_ref() {
            qkv = qkv + lora.alpha_div_dim() * lora.forward(x);
        }

        let qkv = qkv.reshape([batch_size, seq_len, 3, self.num_heads, dim / self.num_heads]);

        let [mut q, mut k, v]: [Tensor<B, 4>; 3] = qkv
            .chunk(3, 2)
            .into_iter()
            .map(|tensor| tensor.squeeze_dim::<4>(2).swap_dims(1, 2))
            .collect::<Vec<_>>() // [b, nh, s, dh]
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
        let [_, _, seq, head_dim] = x.dims();
        let [rope_seq, _h_dim] = sin.dims();
        let num_cls_and_storage_tokens = seq - rope_seq;

        let [prefix, mut rope] = x
            .split_with_sizes(vec![num_cls_and_storage_tokens, rope_seq], 2)
            .try_into()
            .unwrap();

        let half_head_dim = head_dim / 2;
        let [x1, x2] = rope.clone().split(half_head_dim, 3).try_into().unwrap();

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

#[derive(Config, Debug)]
pub struct BlockConfig {
    pub dim: usize,
    pub num_heads: usize,
    pub ffn_ratio: f64,
    pub lora: Option<LoRAConfig>,
}

#[derive(Module, Debug)]
pub struct Block<B: Backend, L: Module<B> = LoRA<B>> {
    pub norm1: LayerNorm<B>,
    pub attn: Attention<B, L>,
    pub ls1: LayerScale<B>,
    pub norm2: LayerNorm<B>,
    pub mlp: Mlp<B>,
    pub ls2: LayerScale<B>,
}

impl BlockConfig {
    pub fn init<B: Backend, L: LoRALayer<B>>(
        &self,
        lora: Option<L>,
        device: &B::Device,
    ) -> Block<B, L> {
        let hidden_dim = (self.dim as f64 * self.ffn_ratio) as usize;
        Block {
            norm1: LayerNormConfig::new(self.dim).with_bias(true).init(device),
            attn: AttentionConfig::new(self.dim, self.num_heads).init(lora, device),
            ls1: LayerScale::new(self.dim, 1e-5, device),
            norm2: LayerNormConfig::new(self.dim).with_bias(true).init(device),
            mlp: Mlp::new(self.dim, hidden_dim, device),
            ls2: LayerScale::new(self.dim, 1e-5, device),
        }
    }
}

impl<B: Backend, L: LoRALayer<B>> Block<B, L> {
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

#[derive(Config, Debug)]
pub struct DinoVisionTransformerConfig {
    pub patch_size: usize,
    pub embed_dim: usize,
    pub num_storage_tokens: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub ffn_ratio: f64,
    pub lora: Option<LoRAConfig>,
    #[config(default = "Initializer::Normal { mean: 0.0, std: 0.02 }")]
    pub initializer: Initializer,
}

#[derive(Module, Debug)]
pub struct DinoVisionTransformer<B: Backend, L: Module<B> = LoRA<B>> {
    pub patch_embed: PatchEmbed<B>,
    pub cls_token: Param<Tensor<B, 3>>,
    pub storage_tokens: Param<Tensor<B, 3>>,
    pub rope_embed: RopePositionEmbedding<B>,
    pub blocks: Vec<Block<B, L>>,
    pub norm: LayerNorm<B>,
    pub mask_token: Param<Tensor<B, 2>>,
}

impl DinoVisionTransformerConfig {
    pub fn init<B: Backend, L: LoRALayer<B>>(
        &self,
        lora: Option<L>,
        device: &B::Device,
    ) -> DinoVisionTransformer<B, L> {
        let patch_embed = PatchEmbed::new(3, self.embed_dim, self.patch_size, device);

        let cls_token = self.initializer.init([1, 1, self.embed_dim], device);
        let storage_tokens = self
            .initializer
            .init([1, self.num_storage_tokens, self.embed_dim], device);

        let rope_embed =
            RopePositionEmbedding::new(self.embed_dim, self.num_heads, 100.0, device).no_grad();

        let blocks = repeat_with(|| {
            BlockConfig::new(self.embed_dim, self.num_heads, self.ffn_ratio)
                .init(lora.clone(), device)
        })
        .take(self.depth)
        .collect();

        let norm = LayerNormConfig::new(self.embed_dim).init(device);

        let mask_token = Param::from_tensor(Tensor::zeros([1, self.embed_dim], device)).no_grad();

        DinoVisionTransformer {
            patch_embed,
            cls_token,
            storage_tokens,
            rope_embed,
            blocks,
            norm,
            mask_token,
        }
    }
}

impl<B: Backend, L: LoRALayer<B>> DinoVisionTransformer<B, L> {
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

pub fn vit_small<B: Backend, L: LoRALayer<B>>(
    patch_size: usize,
    lora_config: Option<L::Config>,
    device: &B::Device,
) -> DinoVisionTransformer<B, L> {
    DinoVisionTransformerConfig::new(patch_size, 384, 4, 12, 6, 4.0)
        .init(lora_config.map(|lc| lc.init(384, device)), device)
}

pub fn vit_base<B: Backend, L: LoRALayer<B>>(
    patch_size: usize,
    lora_config: Option<L::Config>,
    device: &B::Device,
) -> DinoVisionTransformer<B, L> {
    DinoVisionTransformerConfig::new(patch_size, 768, 4, 12, 12, 4.0)
        .init(lora_config.map(|lc| lc.init(768, device)), device)
}

pub fn vit_large<B: Backend, L: LoRALayer<B>>(
    patch_size: usize,
    lora_config: Option<L::Config>,
    device: &B::Device,
) -> DinoVisionTransformer<B, L> {
    DinoVisionTransformerConfig::new(patch_size, 1024, 4, 24, 16, 4.0)
        .init(lora_config.map(|lc| lc.init(1024, device)), device)
}

pub fn vit_so400m<B: Backend, L: LoRALayer<B>>(
    patch_size: usize,
    lora_config: Option<L::Config>,
    device: &B::Device,
) -> DinoVisionTransformer<B, L> {
    DinoVisionTransformerConfig::new(patch_size, 1152, 4, 27, 18, 3.777777778)
        .init(lora_config.map(|lc| lc.init(1152, device)), device)
}

pub fn vit_huge2<B: Backend, L: LoRALayer<B>>(
    patch_size: usize,
    lora_config: Option<L::Config>,
    device: &B::Device,
) -> DinoVisionTransformer<B, L> {
    DinoVisionTransformerConfig::new(patch_size, 1280, 4, 32, 20, 4.0)
        .init(lora_config.map(|lc| lc.init(1280, device)), device)
}

pub fn vit_giant2<B: Backend, L: LoRALayer<B>>(
    patch_size: usize,
    lora_config: Option<L::Config>,
    device: &B::Device,
) -> DinoVisionTransformer<B, L> {
    DinoVisionTransformerConfig::new(patch_size, 1536, 4, 40, 24, 4.0)
        .init(lora_config.map(|lc| lc.init(1536, device)), device)
}

pub fn vit_7b<B: Backend, L: LoRALayer<B>>(
    patch_size: usize,
    lora_config: Option<L::Config>,
    device: &B::Device,
) -> DinoVisionTransformer<B, L> {
    DinoVisionTransformerConfig::new(patch_size, 4096, 4, 40, 32, 3.0)
        .init(lora_config.map(|lc| lc.init(4096, device)), device)
}
