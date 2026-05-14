use burn::{Tensor, backend::Cuda};
use burn_store::{ModuleSnapshot, PytorchStore};
use dino::model::{DinoVisionTransformer, vit_small};

type Backend = Cuda;

fn main() {
    let device = Default::default();
    let mut dino: DinoVisionTransformer<Backend> = vit_small(16, 0., &device);

    let res = dino
        .load_from(
            &mut PytorchStore::from_file("models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
                .with_key_remapping(r"norm\.weight$", "norm.gamma")
                .with_key_remapping(r"norm\.bias$", "norm.beta")
                .with_key_remapping(r"norm1\.weight$", "norm1.gamma")
                .with_key_remapping(r"norm1\.bias$", "norm1.beta")
                .with_key_remapping(r"norm2\.weight$", "norm2.gamma")
                .with_key_remapping(r"norm2\.bias$", "norm2.beta")
                .with_key_remapping(r"attn.qkv.weight$", "attn.qkv.linear.weight")
                .with_key_remapping(r"attn.qkv.bias$", "attn.qkv.linear.bias"),
        )
        .inspect_err(|e| println!("{e}"))
        .unwrap();

    dbg!(res);

    dbg!(dino.forward(Tensor::zeros([1, 3, 256, 256], &device)));
}
