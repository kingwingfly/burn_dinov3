Download pretrained model from [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)

Loaded pretrained model:
```rust
let mut dino: DinoVisionTransformer<Backend> = vit_small(16, &device);
dino.load_from(
    &mut PytorchStore::from_file("models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
        .with_key_remapping(r"norm(\d*)\.weight$", r"norm$1.gamma")
        .with_key_remapping(r"norm(\d*)\.bias$", "norm$1.beta")
        .with_key_remapping(r"attn.qkv.weight$", "attn.qkv.linear.weight")
        .with_key_remapping(r"attn.qkv.bias$", "attn.qkv.linear.bias"),
)
.inspect_err(|e| println!("{e}"))
.unwrap();
```
