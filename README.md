Download pretrained model:
```sh
mkdir models
wget -O models/model.onnx https://huggingface.co/onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX/resolve/main/onnx/model.onnx
wget -O models/model.onnx_data https://huggingface.co/onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX/resolve/main/onnx/model.onnx_data
```

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
