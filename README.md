With [burn](https://github.com/tracel-ai/burn), This DINOv3 crate is a RIIR of [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)'s python project.

# Features

- Base on burn, so that you can got a cool TUI when training
- **LoRA is supported**: you can define `LoRA` module youself and inject it into ViT

# Usage

Download pretrained model from [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3).

Load the ViT-S/16 checkpoint with a rank-8 query/value LoRA adapter in each block. Enable a Burn backend such as `flex`, `cuda`, or `metal` in your application. Burn 0.22 selects the backend at runtime through `Device`. Use `None` for the adapter config to load the pretrained backbone on its own.

```rust no_run
use burn::tensor::{Device, Tensor};
use burn_dinov3::{DinoVisionTransformer, LoRAConfig, vit_small};
use burn_store::{ModuleSnapshot, PytorchStore};

fn main() {
    let device = Device::default();

    let mut dino: DinoVisionTransformer = vit_small(16, Some(LoRAConfig::new(8, 16.0)), &device);
    let mut store = PytorchStore::from_file("models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
        .with_key_remapping(r"\.attn\.qkv\.(weight|bias)$", ".attn.qkv.linear.$1")
        .allow_partial(true);
    let result = dino.load_from(&mut store).unwrap();
    assert!(
        result.unused.is_empty()
            && result
                .missing
                .iter()
                .all(|(key, _)| key.contains(".attn.lora.")),
        "{result}"
    );

    dino = dino.no_grad_except_lora();
    let tokens = dino.forward(Tensor::zeros([1, 3, 256, 256], &device), None);
    assert_eq!(tokens.dims(), [1, 261, 384]);
}
```

`forward` returns all normalized tokens in CLS, storage, then row-major patch order. `forward_cls` returns the upstream image embedding with shape `[batch, embed_dim]`. The boolean patch mask broadcasts to `[batch, patch_rows * patch_cols]`. Each `true` entry replaces that patch with the learned mask token. Apply the official image transforms before inference. [Image transforms](https://github.com/facebookresearch/dinov3#image-transforms)

Use `vit_small_plus`, `vit_large_plus`, or `vit_huge_plus` for S+, L+, or H+ weights. These checkpoints and `vit_7b` use SwiGLU. The generic `vit_huge2` constructor uses a GELU MLP. For the ViT-L/16 SAT493M checkpoint, set `with_untie_global_and_local_cls_norm(Some(true))` on the large model configuration. `forward_local` applies that checkpoint's local prefix norm on local training crops. [Upstream backbone configurations](https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/backbones.py)

The built-in LoRA multiplier is `alpha / rank`. To reproduce a trained 0.1.7 adapter's previous multiplier, pass `old_alpha * rank / embed_dim` as its new alpha.

For a GELU model saved by 0.1.7 in Burnpack format, add `.with_remap_pattern(r"\bmlp\.(fc[12])\.", "mlp.Mlp.$1.")` to the store when loading. This maps the saved feed-forward weights into the GELU variant. For a 0.1.7 SafeTensors file, use `.skip_enum_variants(true)`.

# Tests

Run `cargo test --locked` for the model tests. To compare every output token with the official PyTorch implementation, place the LVD1689M S/16, S+/16 and B/16 checkpoints in `models/`, make the upstream Python package available (a checkout in `dinov3/` works), and run `cargo test --locked --test checkpoints -- --ignored --nocapture`. The test uses the Flex CPU backend and `python3` by default. Set `BURN_DEVICE` to select a different enabled backend and `PYTHON` to select the Python executable with PyTorch and NumPy installed.

# Others

The API will be changed as I like, no sem ver guarentee (although there is likely no big change) in `v0.1`.

If I found any version is not correct, I'll simply yank it.

After my finishing fine tuning with this crate, and ensuring the implementation is correct, I'll bump the version up to `v0.2`.

# Contribution

Let me know if you want to add some functions by giving an issue, so that we'll not confict with each other.
