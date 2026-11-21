use burn::tensor::{Device, Tensor};
use burn_dinov3::{DinoVisionTransformer, vit_small};
use burn_store::{ModuleSnapshot, PytorchStore};

fn main() {
    let device = Device::default();

    let mut dino: DinoVisionTransformer = vit_small(16, None, &device);
    let mut store = PytorchStore::from_file("models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
        .with_key_remapping(r"\.attn\.qkv\.(weight|bias)$", ".attn.qkv.linear.$1");
    let result = dino.load_from(&mut store).unwrap();
    assert!(result.unused.is_empty(), "{result}");
    println!("{result}");

    let tokens = dino.forward(Tensor::zeros([1, 3, 256, 256], &device), None);
    println!("Token shape {:?}", tokens.dims());
}
