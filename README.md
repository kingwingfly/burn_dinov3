With [burn](https://github.com/tracel-ai/burn), This DINOv3 crate is a RIIR of [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)'s python project.

# Features

- Base on burn, so that you can got a cool TUI when training
- **LoRA is supported**: you can define `LoRA` module youself and inject it into ViT

# Usage

Download pretrained model from [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3).

Loaded pretrained model with LoRA enabled with rank `8`:
```rust no_run
use burn::{Tensor, backend};
use burn_dinov3::{DinoVisionTransformer, LoRA, LoRAConfig, vit_small};
use burn_store::{ModuleSnapshot, PytorchStore};

#[cfg(target_os = "macos")]
type Backend = backend::Metal;
#[cfg(not(target_os = "macos"))]
type Backend = backend::Cuda;

fn main() {
    let device = Default::default();
    let mut dino: DinoVisionTransformer<Backend, LoRA<Backend>> =
        vit_small(16, Some(LoRAConfig::new(8, 16.0)), &device); // modify to `None` to cancel LoRA

    let res = dino
        .load_from(
            &mut PytorchStore::from_file("models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
                .with_key_remapping(r"norm(\d*)\.weight$", r"norm$1.gamma")
                .with_key_remapping(r"norm(\d*)\.bias$", "norm$1.beta")
                .with_key_remapping(r"attn.qkv.weight$", "attn.qkv.linear.weight")
                .with_key_remapping(r"attn.qkv.bias$", "attn.qkv.linear.bias")
                .allow_partial(true),
        )
        .inspect_err(|e| println!("{e}"))
        .unwrap();

    println!("{}", res);

    dino = dino.no_grad_except_lora();

    dbg!(dino.forward(Tensor::zeros([1, 3, 256, 256], &device), None));
}
```

And got:
```text
┌─ Tensor Loading Summary ─────────────────────────
│
│ ✓ Successfully applied: 188 tensors
│ ⊘ Skipped (filtered):  0 tensors
│ ✗ Missing in source:    24 tensors
│ ? Unused in target:     0 tensors
│ ! Errors:               0 errors
│
├─ Missing Tensors (requested by model but not found in source)
│
│  First 10 missing tensors:
│    • blocks.0.attn.lora.a
│    • blocks.0.attn.lora.b
│    • blocks.1.attn.lora.a
│    • blocks.1.attn.lora.b
│    • blocks.10.attn.lora.a
│    • blocks.10.attn.lora.b
│    • blocks.11.attn.lora.a
│    • blocks.11.attn.lora.b
│    • blocks.2.attn.lora.a
│    • blocks.2.attn.lora.b
│    ... and 14 more
│
└───────────────────────────────────────────────────
[examples/load.rs:28:5] dino.forward(Tensor::zeros([1, 3, 256, 256], &device), None) = Tensor {
    primitive: Float(
        { id: TensorId { value: 1479 }, shape: Shape { dims: [1, 261, 384] }, device: DefaultDevice },
    ),
}
```

Only LoRA weights are missed.

Do not forget [image-transforms](https://github.com/facebookresearch/dinov3#image-transforms) in practice.

# Others

There's no test for `v0.1`, only all tensor loaded and just seems working.

The API will be changed as I like, no sem ver guarentee (although there is likely no big change) in `v0.1`.

If I found any version is not correct, I'll simply yank it.

After my finishing fine tuning with this crate, and ensuring the implementation is correct, I'll bump the version up to `v0.2`.

# Contribution

Let me know if you want to add some functions by giving an issue, so that we'll not confict with each other.
