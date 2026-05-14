use burn::{Tensor, backend::Wgpu};
use burn_store::{ModuleSnapshot, PytorchStore};
use dino::model::{DinoVisionTransformer, vit_small};

type Backend = Wgpu;

fn main() {
    let device = Default::default();
    let mut dino: DinoVisionTransformer<Backend> = vit_small(&device);

    let res = dino
        .load_from(&mut PytorchStore::from_file(
            "models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        ))
        .unwrap();

    dbg!(res);

    dbg!(dino.forward(Tensor::zeros([1, 3, 256, 256], &device)));
}
