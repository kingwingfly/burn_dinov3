With [burn](https://github.com/tracel-ai/burn), This DINOv3 crate is a RIIR of [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)'s python project.

# Usage

Download pretrained model from [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3).

Loaded pretrained model:
```rust no_run
use burn::{Tensor, backend::Cuda};
use burn_dinov3::{DinoVisionTransformer, vit_small};
use burn_store::{ModuleSnapshot, PytorchStore};

fn main() {
    let device = Default::default();
    let mut dino: DinoVisionTransformer<Cuda> = vit_small(16, &device);

    let res = dino
        .load_from(
            &mut PytorchStore::from_file("models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
                .with_key_remapping(r"norm(\d*)\.weight$", r"norm$1.gamma")
                .with_key_remapping(r"norm(\d*)\.bias$", "norm$1.beta")
                .with_key_remapping(r"attn.qkv.weight$", "attn.qkv.linear.weight")
                .with_key_remapping(r"attn.qkv.bias$", "attn.qkv.linear.bias"),
        )
        .inspect_err(|e| println!("{e}"))
        .unwrap();

    println!("{}", res);

    dbg!(dino.forward(Tensor::zeros([1, 3, 256, 256], &device), None));
}
```

And got:
```text
┌─ Tensor Loading Summary ─────────────────────────
│
│ ✓ Successfully applied: 188 tensors
│ ⊘ Skipped (filtered):  0 tensors
│ ✗ Missing in source:    0 tensors
│ ? Unused in target:     0 tensors
│ ! Errors:               0 errors
│
└───────────────────────────────────────────────────
[src/main.rs:24:5] dino.forward(Tensor::zeros([1, 3, 256, 256], &device), None) = Tensor {
    primitive: Float(
        { id: TensorId { value: 1289 }, shape: Shape { dims: [1, 261, 384] }, device: Cuda(0) },
    ),
}
```

Do not forget [image-transforms](https://github.com/facebookresearch/dinov3#image-transforms) in practice.

# Others

There's no test for now, only all tensor loaded and just seems working.

The API will be changed as I like, no sem ver guarentee (although there is likely no big change).

If I found any version is not correct, I'll simply yank it.
