use std::collections::HashMap;

use burn::{
    config::Config,
    module::{Flag, Module, ModuleMapper, Param, ParamId},
    nn::{
        Dropout, DropoutConfig, Gelu, Initializer, LayerNorm, LayerNormConfig, Linear,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{Bool, DType, Device, Distribution, FloatDType, Tensor, activation, module},
};

#[derive(Module, Debug)]
pub struct PatchEmbed {
    pub proj: Conv2d,
}

impl PatchEmbed {
    pub fn new(in_chans: usize, embed_dim: usize, patch_size: usize, device: &Device) -> Self {
        let proj = Conv2dConfig::new([in_chans, embed_dim], [patch_size, patch_size])
            .with_stride([patch_size, patch_size])
            .init(device);
        Self { proj }
    }

    pub fn forward(&self, x: Tensor<4>) -> (Tensor<3>, usize, usize) {
        let x = self.proj.forward(x);
        let [_, _, height, width] = x.dims();
        (x.flatten(2, -1).swap_dims(1, 2), height, width)
    }
}

#[derive(Module, Debug)]
pub struct RopePositionEmbedding {
    pub periods: Param<Tensor<1>>,
    pub rescale_coords: Option<f32>,
    pub training: Param<Flag>,
}

impl RopePositionEmbedding {
    pub fn new(embed_dim: usize, num_heads: usize, base: f32, device: &Device) -> Self {
        assert!(
            embed_dim > 0
                && num_heads > 0
                && embed_dim.is_multiple_of(num_heads)
                && (embed_dim / num_heads).is_multiple_of(4),
            "2D RoPE requires an embedding dimension divisible by four times the number of heads"
        );
        let d_head = embed_dim / num_heads;
        let exponents =
            Tensor::arange_step(0..d_head as i64, 4, device).cast(FloatDType::F32) / d_head as f32;
        let periods = Tensor::full([1], base, (device, DType::F32)).powf(exponents);
        Self {
            periods: Param::from_tensor(periods).set_require_grad(false),
            rescale_coords: None,
            training: Param::from_bool(true),
        }
    }

    fn training_rescale(&self) -> Option<f32> {
        self.rescale_coords
            .filter(|_| self.training.is_enabled() && self.periods.lazy_device().is_autodiff())
    }

    pub fn forward(&self, height: usize, width: usize) -> (Tensor<2>, Tensor<2>) {
        let device = self.periods.lazy_device();
        let centers = |length: usize| {
            (Tensor::arange(0..length as i64, &device).cast(FloatDType::F32) + 0.5) / length as f32
        };
        let rows = centers(height)
            .unsqueeze_dim::<2>(1)
            .expand([height, width]);
        let cols = centers(width).unsqueeze_dim::<2>(0).expand([height, width]);
        let mut coords =
            Tensor::stack::<3>(vec![rows, cols], 2).reshape([height * width, 2]) * 2.0 - 1.0;
        if let Some(rescale) = self.training_rescale() {
            let bound = f64::from(rescale).ln();
            let scale = Tensor::<1>::random(
                [1],
                Distribution::Uniform(-bound, bound),
                (&device, DType::F32),
            )
            .exp();
            coords = coords * scale.unsqueeze::<2>();
        }
        // Each half of the head contains the row frequencies followed by the column frequencies.
        let angles = coords.unsqueeze_dim::<3>(2) * std::f32::consts::TAU
            / self
                .periods
                .val()
                .detach()
                .cast(FloatDType::F32)
                .unsqueeze::<3>();
        let angles = angles.flatten(1, 2);
        let angles = Tensor::cat(vec![angles.clone(), angles], 1);
        (angles.clone().sin(), angles.cos())
    }
}

// PyTorch initializes ViT linear weights with std=0.02 and truncation bounds of ±2 (100 std).
fn vit_linear(input: usize, output: usize, bias: bool, device: &Device) -> Linear {
    Linear {
        weight: Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        }
        .init([input, output], device),
        bias: bias.then(|| Initializer::Zeros.init([output], device)),
    }
}

#[derive(Module, Debug)]
pub struct LinearKMaskedBias {
    pub linear: Linear,
    pub bias_mask: Option<Param<Tensor<1>>>,
}

impl LinearKMaskedBias {
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let bias = self.linear.bias.as_ref().map(|b| {
            let bias = b.val();
            match &self.bias_mask {
                Some(mask) => mask.val().detach().cast(FloatDType::from(bias.dtype())) * bias,
                None => bias,
            }
        });
        module::linear(input, self.linear.weight.val(), bias)
    }
}

#[derive(Module, Debug)]
pub struct LayerScale {
    pub gamma: Param<Tensor<1>>,
}

impl LayerScale {
    pub fn new(dim: usize, init_values: f32, device: &Device) -> Self {
        let gamma = Initializer::Constant {
            value: init_values.into(),
        }
        .init([dim], device);
        Self { gamma }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        x * self.gamma.val().unsqueeze::<3>()
    }
}

pub trait LoRALayer: Module {
    type Config: LoRALayerConfig<LoRA = Self>;

    fn forward(&self, x: Tensor<3>) -> Tensor<3>;
    /// Multiplier for the adapter update. The built-in adapter uses alpha / rank.
    fn alpha_div_dim(&self) -> f32;
}

pub trait LoRALayerConfig {
    type LoRA: LoRALayer;

    fn init(&self, dim: usize, device: &Device) -> Self::LoRA;
}

#[derive(Config, Debug)]
pub struct LoRAConfig {
    pub rank: usize,
    pub alpha: f32,
    #[config(default = "Initializer::KaimingUniform{gain:1.0/3.0f64.sqrt(), fan_out_only:false}")]
    pub a_initializer: Initializer,
    #[config(default = "Initializer::Zeros")]
    pub b_initializer: Initializer,
}

#[derive(Module, Debug)]
pub struct LoRA {
    pub a: Param<Tensor<2>>,
    pub b_q: Param<Tensor<2>>,
    pub b_v: Param<Tensor<2>>,
    pub alpha_div_dim: f32,
}

impl LoRALayerConfig for LoRAConfig {
    type LoRA = LoRA;

    fn init(&self, dim: usize, device: &Device) -> LoRA {
        assert!(self.rank > 0, "LoRA rank must be positive");
        LoRA {
            a: self
                .a_initializer
                .init_with([dim, self.rank], Some(dim), Some(self.rank), device),
            b_q: self
                .b_initializer
                .init_with([self.rank, dim], Some(self.rank), Some(dim), device),
            b_v: self
                .b_initializer
                .init_with([self.rank, dim], Some(self.rank), Some(dim), device),
            alpha_div_dim: self.alpha / self.rank as f32,
        }
    }
}

impl LoRALayer for LoRA {
    type Config = LoRAConfig;

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let hidden = x.matmul(self.a.val().unsqueeze());
        let q = hidden.clone().matmul(self.b_q.val().unsqueeze());
        let v = hidden.matmul(self.b_v.val().unsqueeze());
        let k = q.zeros_like();
        Tensor::cat(vec![q, k, v], 2)
    }

    fn alpha_div_dim(&self) -> f32 {
        self.alpha_div_dim
    }
}

#[derive(Config, Debug)]
pub struct AttentionConfig {
    pub dim: usize,
    pub num_heads: usize,
    /// Defaults to a trained Q/V bias with the key bias masked.
    pub qkv_bias: Option<bool>,
}

#[derive(Module, Debug)]
pub struct Attention<L: LoRALayer = LoRA> {
    pub qkv: LinearKMaskedBias,
    pub proj: Linear,
    pub drop_out: Dropout,
    pub lora: Option<L>,
    pub num_heads: usize,
}

impl AttentionConfig {
    pub fn init<L: LoRALayer>(&self, lora: Option<L>, device: &Device) -> Attention<L> {
        assert!(
            self.num_heads > 0 && self.dim > 0 && self.dim.is_multiple_of(self.num_heads),
            "attention requires an embedding dimension divisible by the number of heads"
        );
        let dim = self.dim;
        let qkv_bias = self.qkv_bias.unwrap_or(true);
        Attention {
            qkv: LinearKMaskedBias {
                linear: vit_linear(dim, dim * 3, qkv_bias, device),
                bias_mask: qkv_bias.then(|| {
                    let ones = Tensor::ones([dim], device);
                    let mask = Tensor::cat(vec![ones.clone(), ones.zeros_like(), ones], 0);
                    Param::from_tensor(mask).set_require_grad(false)
                }),
            },
            proj: vit_linear(dim, dim, true, device),
            drop_out: DropoutConfig::new(0.0).init(),
            lora,
            num_heads: self.num_heads,
        }
    }
}

impl<L: LoRALayer> Attention<L> {
    pub fn forward(&self, x: Tensor<3>, rope: Option<&(Tensor<2>, Tensor<2>)>) -> Tensor<3> {
        let [batch_size, seq_len, dim] = x.dims();
        let mut qkv = self.qkv.forward(x.clone());
        if let Some(lora) = &self.lora {
            qkv = qkv + lora.forward(x) * lora.alpha_div_dim();
        }
        let qkv = qkv.reshape([batch_size, seq_len, 3, self.num_heads, dim / self.num_heads]);
        let [mut q, mut k, v] =
            [0, 1, 2].map(|i| qkv.clone().select_dim::<4>(2, i).swap_dims(1, 2));
        if let Some((sin, cos)) = rope {
            q = Self::apply_rope(q, sin, cos);
            k = Self::apply_rope(k, sin, cos);
        }
        let out = module::attention(q, k, v, None, None, Default::default());
        let out = out.swap_dims(1, 2).reshape([batch_size, seq_len, dim]);
        self.drop_out.forward(self.proj.forward(out))
    }

    fn apply_rope(x: Tensor<4>, sin: &Tensor<2>, cos: &Tensor<2>) -> Tensor<4> {
        let [_, _, seq, head_dim] = x.dims();
        let [patches, _] = sin.dims();
        let prefix = seq
            .checked_sub(patches)
            .expect("RoPE grid exceeds sequence length");
        let dtype = FloatDType::from(x.dtype());
        let tokens = x
            .clone()
            .narrow(2, prefix, patches)
            .cast(FloatDType::from(sin.dtype()));
        let half = head_dim / 2;
        let rotated = Tensor::cat(
            vec![
                -tokens.clone().narrow(3, half, half),
                tokens.clone().narrow(3, 0, half),
            ],
            3,
        );
        let tokens = (tokens * cos.clone().unsqueeze::<4>()
            + rotated * sin.clone().unsqueeze::<4>())
        .cast(dtype);
        if prefix == 0 {
            tokens
        } else {
            Tensor::cat(vec![x.narrow(2, 0, prefix), tokens], 2)
        }
    }
}

#[derive(Module, Debug)]
pub struct Mlp {
    pub fc1: Linear,
    pub act: Gelu,
    pub fc2: Linear,
}

impl Mlp {
    pub fn new(in_features: usize, hidden_features: usize, device: &Device) -> Self {
        Self {
            fc1: vit_linear(in_features, hidden_features, true, device),
            act: Gelu::new(),
            fc2: vit_linear(hidden_features, in_features, true, device),
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.fc2.forward(self.act.forward(self.fc1.forward(x)))
    }
}

#[derive(Module, Debug)]
pub struct SwiGlu {
    pub w1: Linear,
    pub w2: Linear,
    pub w3: Linear,
}

impl SwiGlu {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let hidden = activation::silu(self.w1.forward(x.clone())) * self.w2.forward(x);
        self.w3.forward(hidden)
    }
}

#[derive(Config, Debug, Copy)]
pub enum FfnConfig {
    Mlp,
    SwiGlu { align_to: usize },
}

#[derive(Module, Debug)]
pub enum FeedForward {
    Mlp(Mlp),
    SwiGlu(SwiGlu),
}

impl FfnConfig {
    fn init(self, dim: usize, hidden: usize, device: &Device) -> FeedForward {
        let hidden = if hidden == 0 { dim } else { hidden };
        match self {
            Self::Mlp => FeedForward::Mlp(Mlp::new(dim, hidden, device)),
            Self::SwiGlu { align_to } => {
                let width = (hidden * 2 / 3).next_multiple_of(align_to);
                FeedForward::SwiGlu(SwiGlu {
                    w1: vit_linear(dim, width, true, device),
                    w2: vit_linear(dim, width, true, device),
                    w3: vit_linear(width, dim, true, device),
                })
            }
        }
    }
}

impl FeedForward {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        match self {
            Self::Mlp(mlp) => mlp.forward(x),
            Self::SwiGlu(swiglu) => swiglu.forward(x),
        }
    }
}

#[derive(Config, Debug)]
pub struct BlockConfig {
    pub dim: usize,
    pub num_heads: usize,
    pub ffn_ratio: f64,
    /// New configs use the upstream GELU MLP. An absent field preserves the 0.1.7 rounded width.
    #[config(default = "Some(FfnConfig::Mlp)")]
    pub ffn_layer: Option<FfnConfig>,
    /// Defaults to a trained Q/V bias with the key bias masked.
    pub qkv_bias: Option<bool>,
}

#[derive(Module, Debug)]
pub struct Block<L: LoRALayer = LoRA> {
    pub norm1: LayerNorm,
    pub attn: Attention<L>,
    pub ls1: LayerScale,
    pub norm2: LayerNorm,
    pub mlp: FeedForward,
    pub ls2: LayerScale,
}

impl BlockConfig {
    pub fn init<L: LoRALayer>(&self, lora: Option<L>, device: &Device) -> Block<L> {
        let hidden = self.dim as f64 * self.ffn_ratio;
        let mlp = match self.ffn_layer {
            Some(layer) => layer.init(self.dim, hidden as usize, device),
            None => FeedForward::Mlp(Mlp::new(self.dim, hidden.round() as usize, device)),
        };
        Block {
            norm1: LayerNormConfig::new(self.dim).init(device),
            attn: AttentionConfig::new(self.dim, self.num_heads)
                .with_qkv_bias(self.qkv_bias)
                .init(lora, device),
            ls1: LayerScale::new(self.dim, 1e-5, device),
            norm2: LayerNormConfig::new(self.dim).init(device),
            mlp,
            ls2: LayerScale::new(self.dim, 1e-5, device),
        }
    }
}

impl<L: LoRALayer> Block<L> {
    pub fn forward(&self, x: Tensor<3>, rope: Option<&(Tensor<2>, Tensor<2>)>) -> Tensor<3> {
        let attn = self.attn.forward(self.norm1.forward(x.clone()), rope);
        let x = x + self.ls1.forward(attn);
        let ffn = self.mlp.forward(self.norm2.forward(x.clone()));
        x + self.ls2.forward(ffn)
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
    /// New configs use the upstream GELU MLP. An absent field preserves the 0.1.7 rounded width.
    #[config(default = "Some(FfnConfig::Mlp)")]
    pub ffn_layer: Option<FfnConfig>,
    /// Defaults to a trained Q/V bias with the key bias masked.
    pub qkv_bias: Option<bool>,
    /// Enable the extra prefix norm used by the 7B and satellite ViT-L checkpoints.
    pub untie_global_and_local_cls_norm: Option<bool>,
    #[config(default = "Some(2.0)")]
    pub rope_rescale_coords: Option<f32>,
    #[config(default = "Initializer::Normal { mean: 0.0, std: 0.02 }")]
    pub initializer: Initializer,
}

#[derive(Module, Debug)]
pub struct DinoVisionTransformer<L: LoRALayer = LoRA> {
    pub patch_embed: PatchEmbed,
    pub cls_token: Param<Tensor<3>>,
    pub storage_tokens: Option<Param<Tensor<3>>>,
    pub rope_embed: RopePositionEmbedding,
    pub blocks: Vec<Block<L>>,
    pub norm: LayerNorm,
    pub local_cls_norm: Option<LayerNorm>,
    pub training: Param<Flag>,
    pub mask_token: Param<Tensor<2>>,
}

impl DinoVisionTransformerConfig {
    pub fn init<L: LoRALayer>(
        &self,
        lora: Option<L::Config>,
        device: &Device,
    ) -> DinoVisionTransformer<L> {
        let dim = self.embed_dim;
        let mut rope_embed = RopePositionEmbedding::new(dim, self.num_heads, 100.0, device);
        if let Some(rescale) = self.rope_rescale_coords {
            assert!(
                rescale.is_finite() && rescale >= 1.0,
                "RoPE coordinate rescaling must be at least one"
            );
            rope_embed.rescale_coords = (rescale > 1.0).then_some(rescale);
        }
        let block = BlockConfig::new(dim, self.num_heads, self.ffn_ratio)
            .with_ffn_layer(self.ffn_layer)
            .with_qkv_bias(self.qkv_bias);
        let blocks = (0..self.depth)
            .map(|_| block.init(lora.as_ref().map(|config| config.init(dim, device)), device))
            .collect();
        DinoVisionTransformer {
            patch_embed: PatchEmbed::new(3, dim, self.patch_size, device),
            cls_token: self.initializer.init([1, 1, dim], device),
            // A 0.1.7 checkpoint has this parameter even when its token axis is empty.
            storage_tokens: (self.num_storage_tokens > 0 || self.ffn_layer.is_none()).then(|| {
                self.initializer
                    .init([1, self.num_storage_tokens, dim], device)
            }),
            rope_embed,
            blocks,
            norm: LayerNormConfig::new(dim).init(device),
            local_cls_norm: self
                .untie_global_and_local_cls_norm
                .unwrap_or(false)
                .then(|| LayerNormConfig::new(dim).init(device)),
            training: Param::from_bool(true),
            mask_token: Initializer::Zeros.init([1, dim], device),
        }
    }
}

impl<L: LoRALayer> DinoVisionTransformer<L> {
    fn prepare_tokens(
        &self,
        image: Tensor<4>,
        masks: Option<&Tensor<2, Bool>>,
    ) -> (Tensor<3>, usize, usize) {
        let (mut patches, height, width) = self.patch_embed.forward(image);
        let [batch, seq, dim] = patches.dims();
        let mut cls = self.cls_token.val();
        if let Some(masks) = masks {
            let masks = masks.clone().expand([batch, seq]).unsqueeze_dim(2);
            let replacement = self
                .mask_token
                .val()
                .cast(FloatDType::from(patches.dtype()));
            patches = patches.mask_where(masks, replacement.unsqueeze());
        } else {
            // Keep the mask parameter in the training graph for unmasked batches, as in DINOv3.
            cls = cls + self.mask_token.val().unsqueeze::<3>() * 0.0;
        }
        let mut tokens = vec![cls.expand([batch, 1, dim])];
        if let Some(storage) = &self.storage_tokens
            && storage.lazy_shape()[1] > 0
        {
            tokens.push(storage.val().expand([batch as i64, -1, -1]));
        }
        tokens.push(patches);
        (Tensor::cat(tokens, 1), height, width)
    }

    fn encode(&self, image: Tensor<4>, masks: Option<&Tensor<2, Bool>>) -> Tensor<3> {
        let (mut x, height, width) = self.prepare_tokens(image, masks);
        let shared_rope = self
            .rope_embed
            .training_rescale()
            .is_none()
            .then(|| self.rope_embed.forward(height, width));
        for block in &self.blocks {
            let rope = shared_rope
                .clone()
                .unwrap_or_else(|| self.rope_embed.forward(height, width));
            x = block.forward(x, Some(&rope));
        }
        x
    }

    /// Normalized tokens in CLS, storage, then row-major patch order.
    pub fn forward(&self, image: Tensor<4>, masks: Option<&Tensor<2, Bool>>) -> Tensor<3> {
        self.norm.forward(self.encode(image, masks))
    }

    /// The image embedding returned by the upstream inference forward pass.
    pub fn forward_cls(&self, image: Tensor<4>, masks: Option<&Tensor<2, Bool>>) -> Tensor<2> {
        self.norm
            .forward(self.encode(image, masks).select_dim(1, 0))
    }

    /// Normalized tokens for a local training crop, including the separate prefix norm when configured.
    pub fn forward_local(&self, image: Tensor<4>, masks: Option<&Tensor<2, Bool>>) -> Tensor<3> {
        let x = self.encode(image, masks);
        if let Some(local_norm) = &self.local_cls_norm
            && self.training.is_enabled()
            && x.device().is_autodiff()
        {
            let prefix = 1 + self.storage_tokens.as_ref().map_or(0, |s| s.dims()[1]);
            let patches = x.dims()[1] - prefix;
            Tensor::cat(
                vec![
                    local_norm.forward(x.clone().narrow(1, 0, prefix)),
                    self.norm.forward(x.narrow(1, prefix, patches)),
                ],
                1,
            )
        } else {
            self.norm.forward(x)
        }
    }

    /// Give each block independent optimizer IDs after importing a 0.1.7 Burnpack.
    /// Preserves tensor values, existing unique IDs, and ties within a custom adapter.
    pub fn deduplicate_lora_ids(&mut self) {
        struct AdapterIds {
            block: usize,
            seen: HashMap<ParamId, (usize, ParamId)>,
        }

        impl ModuleMapper for AdapterIds {
            fn map_float<const D: usize>(
                &mut self,
                mut param: Param<Tensor<D>>,
            ) -> Param<Tensor<D>> {
                let (last_block, id) = self.seen.entry(param.id).or_insert((self.block, param.id));
                if *last_block != self.block {
                    *last_block = self.block;
                    *id = ParamId::new();
                }
                param.id = *id;
                param
            }
        }

        let mut ids = AdapterIds {
            block: 0,
            seen: HashMap::new(),
        };
        for (index, block) in self.blocks.iter_mut().enumerate() {
            ids.block = index;
            block.attn.lora = block.attn.lora.take().map(|lora| lora.map(&mut ids));
        }
    }

    pub fn no_grad_except_lora(self) -> Self
    where
        Self: Module,
    {
        let mut model = self.no_grad();
        for block in &mut model.blocks {
            block.attn.lora = block
                .attn
                .lora
                .take()
                .map(|lora| lora.set_require_grad(true));
        }
        model
    }
}

pub fn vit_small<L: LoRALayer>(
    patch_size: usize,
    lora: Option<L::Config>,
    device: &Device,
) -> DinoVisionTransformer<L> {
    DinoVisionTransformerConfig::new(patch_size, 384, 4, 12, 6, 4.0).init(lora, device)
}

pub fn vit_small_plus<L: LoRALayer>(
    patch_size: usize,
    lora: Option<L::Config>,
    device: &Device,
) -> DinoVisionTransformer<L> {
    DinoVisionTransformerConfig::new(patch_size, 384, 4, 12, 6, 6.0)
        .with_ffn_layer(Some(FfnConfig::SwiGlu { align_to: 8 }))
        .init(lora, device)
}

pub fn vit_base<L: LoRALayer>(
    patch_size: usize,
    lora: Option<L::Config>,
    device: &Device,
) -> DinoVisionTransformer<L> {
    DinoVisionTransformerConfig::new(patch_size, 768, 4, 12, 12, 4.0).init(lora, device)
}

pub fn vit_large<L: LoRALayer>(
    patch_size: usize,
    lora: Option<L::Config>,
    device: &Device,
) -> DinoVisionTransformer<L> {
    DinoVisionTransformerConfig::new(patch_size, 1024, 4, 24, 16, 4.0).init(lora, device)
}

pub fn vit_large_plus<L: LoRALayer>(
    patch_size: usize,
    lora: Option<L::Config>,
    device: &Device,
) -> DinoVisionTransformer<L> {
    DinoVisionTransformerConfig::new(patch_size, 1024, 4, 24, 16, 6.0)
        .with_ffn_layer(Some(FfnConfig::SwiGlu { align_to: 8 }))
        .init(lora, device)
}

pub fn vit_so400m<L: LoRALayer>(
    patch_size: usize,
    lora: Option<L::Config>,
    device: &Device,
) -> DinoVisionTransformer<L> {
    DinoVisionTransformerConfig::new(patch_size, 1152, 4, 27, 18, 3.777777778).init(lora, device)
}

pub fn vit_huge2<L: LoRALayer>(
    patch_size: usize,
    lora: Option<L::Config>,
    device: &Device,
) -> DinoVisionTransformer<L> {
    DinoVisionTransformerConfig::new(patch_size, 1280, 4, 32, 20, 4.0).init(lora, device)
}

pub fn vit_huge_plus<L: LoRALayer>(
    patch_size: usize,
    lora: Option<L::Config>,
    device: &Device,
) -> DinoVisionTransformer<L> {
    DinoVisionTransformerConfig::new(patch_size, 1280, 4, 32, 20, 6.0)
        .with_ffn_layer(Some(FfnConfig::SwiGlu { align_to: 8 }))
        .init(lora, device)
}

pub fn vit_giant2<L: LoRALayer>(
    patch_size: usize,
    lora: Option<L::Config>,
    device: &Device,
) -> DinoVisionTransformer<L> {
    DinoVisionTransformerConfig::new(patch_size, 1536, 4, 40, 24, 4.0).init(lora, device)
}

pub fn vit_7b<L: LoRALayer>(
    patch_size: usize,
    lora: Option<L::Config>,
    device: &Device,
) -> DinoVisionTransformer<L> {
    DinoVisionTransformerConfig::new(patch_size, 4096, 4, 40, 32, 3.0)
        .with_ffn_layer(Some(FfnConfig::SwiGlu { align_to: 64 }))
        .with_qkv_bias(Some(false))
        .with_untie_global_and_local_cls_norm(Some(true))
        .init(lora, device)
}
