use std::{env, path::Path, process::Command};

use burn::tensor::{Bool, Device, Tensor};
use burn_dinov3::{DinoVisionTransformer, vit_base, vit_small, vit_small_plus};
use burn_store::{ModuleSnapshot, PytorchStore};

#[test]
#[ignore = "requires the three official checkpoints in models/, PyTorch, and the dinov3 Python package"]
fn pretrained_tokens_match_upstream() {
    let device = env::var_os("BURN_DEVICE").map_or_else(Device::flex, |_| Device::default());
    for (name, hash, builder) in [
        (
            "vits16",
            "08c60483",
            vit_small as fn(_, _, _) -> DinoVisionTransformer,
        ),
        ("vits16plus", "4057cbaa", vit_small_plus),
        ("vitb16", "73cec8be", vit_base),
    ] {
        let path = format!("models/dinov3_{name}_pretrain_lvd1689m-{hash}.pth");
        let reference = Command::new(env::var_os("PYTHON").unwrap_or_else(|| "python3".into()))
            .args(["-B", "tests/reference.py", name, &path])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .unwrap();
        assert!(
            reference.status.success(),
            "{}",
            String::from_utf8_lossy(&reference.stderr)
        );
        let (mut remaining, trailing_bytes) = reference.stdout.as_chunks::<4>();
        assert!(
            trailing_bytes.is_empty(),
            "incomplete float in {name} reference"
        );

        let mut model = builder(16, None, &device);
        let mut store = PytorchStore::from_file(Path::new(env!("CARGO_MANIFEST_DIR")).join(path))
            .with_key_remapping(r"\.attn\.qkv\.(weight|bias)$", ".attn.qkv.linear.$1");
        let loaded = model.load_from(&mut store).unwrap();
        assert!(loaded.unused.is_empty(), "{loaded}");

        let image = (Tensor::arange(0..2 * 3 * 35 * 49, &device)
            .remainder_scalar(251)
            .float()
            / 125.0
            - 1.0)
            .reshape([2, 3, 35, 49]);
        let mask = Tensor::<2, Bool>::from_data(
            [
                [false, true, false, false, true, false],
                [true, false, false, true, false, true],
            ],
            &device,
        );
        for masks in [None, Some(&mask)] {
            let actual = model.forward(image.clone(), masks);
            let [2, 11, _] = actual.dims() else {
                panic!("unexpected output shape {:?}", actual.dims())
            };
            let actual = actual.into_data().try_into_vec::<f32>().unwrap();
            let (expected, rest) = remaining.split_at(actual.len());
            remaining = rest;
            let mut max_error = 0.0_f32;
            for (i, (a, bytes)) in actual.iter().zip(expected).enumerate() {
                let b = f32::from_le_bytes(*bytes);
                let error = (a - b).abs();
                assert!(
                    error <= 2e-4 + 1e-5 * b.abs(),
                    "{name} token element {i}: {a} vs {b}"
                );
                max_error = max_error.max(error);
            }
            eprintln!(
                "{name} (masked={}) max absolute error {max_error:.7}",
                masks.is_some()
            );
        }
        assert!(remaining.is_empty());
    }
}
