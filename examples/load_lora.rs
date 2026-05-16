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
        vit_small(16, Some(LoRAConfig::new(8)), &device);

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

    dbg!(dino.forward(Tensor::zeros([1, 3, 256, 256], &device), None));
}
