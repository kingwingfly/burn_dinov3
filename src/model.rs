use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    proj: Conv2d<B>,
    patch_size: usize,
    embed_dim: usize,
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

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch_size, _, height, width] = x.dims();
        let x = self.proj.forward(x);
        // B, C, H, W -> B, C, H*W
        let x = x.reshape([
            batch_size,
            self.embed_dim,
            (height / self.patch_size) * (width / self.patch_size),
        ]);
        // B, C, H*W -> B, H*W, C
        x.swap_dims(1, 2)
    }
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    fc1: Linear<B>,
    act: Gelu,
    fc2: Linear<B>,
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
pub struct Attention<B: Backend> {
    qkv: Linear<B>,
    proj: Linear<B>,
    num_heads: usize,
}

impl<B: Backend> Attention<B> {
    pub fn new(
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        proj_bias: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            qkv: LinearConfig::new(dim, dim * 3)
                .with_bias(qkv_bias)
                .init(device),
            proj: LinearConfig::new(dim, dim)
                .with_bias(proj_bias)
                .init(device),
            num_heads,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, dim] = x.dims();
        let qkv = self.qkv.forward(x);

        // Simplified attention for demonstration
        let qkv = qkv.reshape([batch_size, seq_len, 3, self.num_heads, dim / self.num_heads]);

        // TODO: rope, dot product attention etc.
        let out = burn::tensor::Tensor::zeros([batch_size, seq_len, dim], &qkv.device());

        self.proj.forward(out)
    }
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    norm1: LayerNorm<B>,
    attn: Attention<B>,
    norm2: LayerNorm<B>,
    mlp: Mlp<B>,
}

impl<B: Backend> Block<B> {
    pub fn new(dim: usize, num_heads: usize, ffn_ratio: f64, device: &B::Device) -> Self {
        let hidden_dim = (dim as f64 * ffn_ratio) as usize;
        Self {
            norm1: LayerNormConfig::new(dim).init(device),
            attn: Attention::new(dim, num_heads, true, true, device),
            norm2: LayerNormConfig::new(dim).init(device),
            mlp: Mlp::new(dim, hidden_dim, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x_norm = self.norm1.forward(x.clone());
        let x = x + self.attn.forward(x_norm);

        let x_norm = self.norm2.forward(x.clone());
        x + self.mlp.forward(x_norm)
    }
}

#[derive(Module, Debug)]
pub struct DinoVisionTransformer<B: Backend> {
    patch_embed: PatchEmbed<B>,
    cls_token: burn::module::Param<Tensor<B, 3>>,
    blocks: Vec<Block<B>>,
    norm: LayerNorm<B>,
}

impl<B: Backend> DinoVisionTransformer<B> {
    pub fn new(
        patch_size: usize,
        embed_dim: usize,
        depth: usize,
        num_heads: usize,
        ffn_ratio: f64,
        device: &B::Device,
    ) -> Self {
        let patch_embed = PatchEmbed::new(3, embed_dim, patch_size, device);
        let cls_token = burn::module::Param::from_tensor(burn::tensor::Tensor::zeros(
            [1, 1, embed_dim],
            device,
        ));

        let mut blocks = Vec::with_capacity(depth);
        for _ in 0..depth {
            blocks.push(Block::new(embed_dim, num_heads, ffn_ratio, device));
        }

        let norm = LayerNormConfig::new(embed_dim).init(device);

        Self {
            patch_embed,
            cls_token,
            blocks,
            norm,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let batch_size = x.dims()[0];
        let mut x = self.patch_embed.forward(x);

        let cls_tokens = self.cls_token.val().repeat_dim(0, batch_size);
        x = burn::tensor::Tensor::cat(vec![cls_tokens, x], 1);

        for block in &self.blocks {
            x = block.forward(x);
        }

        self.norm.forward(x)
    }
}

pub fn vit_small<B: Backend>(device: &B::Device) -> DinoVisionTransformer<B> {
    DinoVisionTransformer::new(16, 384, 12, 6, 4.0, device)
}
