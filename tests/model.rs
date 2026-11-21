use burn::{
    module::{Module, Param},
    nn::Initializer,
    tensor::{Bool, Device, Tensor},
};
use burn_dinov3::{
    DinoVisionTransformer, DinoVisionTransformerConfig, LoRA, LoRAConfig, LoRALayer,
    LoRALayerConfig,
    model::{AttentionConfig, FeedForward},
    vit_7b, vit_huge_plus, vit_large_plus, vit_small_plus,
};

#[test]
fn masks_replace_only_the_selected_patches_in_each_image() {
    let device = Device::flex().autodiff();
    let mut model: DinoVisionTransformer =
        DinoVisionTransformerConfig::new(2, 8, 0, 0, 2, 2.0).init(None, &device);
    model.mask_token = Param::from_tensor(Tensor::arange(0..8, &device).float().reshape([1, 8]));
    let images = Tensor::arange(0..96, &device).float().reshape([2, 3, 4, 4]);
    let mask = [[false, true, true, false], [true, false, false, true]];
    let shared = Tensor::<2, Bool>::from_data([mask[0]], &device);
    let repeated = Tensor::<2, Bool>::from_data([mask[0], mask[0]], &device);
    assert_eq!(
        model.forward(images.clone(), Some(&shared)).into_data(),
        model.forward(images.clone(), Some(&repeated)).into_data(),
    );
    let plain = model.forward(images.clone(), None);
    let masked = model.forward(images, Some(&Tensor::<2, Bool>::from_data(mask, &device)));
    assert_eq!(masked.dims(), [2, 5, 8]);
    let expected_mask = model
        .norm
        .forward(model.mask_token.val())
        .into_data()
        .try_into_vec::<f32>()
        .unwrap();
    let plain = plain.into_data().try_into_vec::<f32>().unwrap();
    let data = masked.to_data().try_into_vec::<f32>().unwrap();
    for (b, row) in mask.iter().enumerate() {
        for token in 0..5 {
            let offset = (b * 5 + token) * 8;
            let expected = if token > 0 && row[token - 1] {
                &expected_mask[..]
            } else {
                &plain[offset..offset + 8]
            };
            assert_eq!(&data[offset..offset + 8], expected);
        }
    }
    let grads = masked.select_dim::<2>(2, 0).sum().backward();
    let grad = model
        .mask_token
        .grad(&grads)
        .unwrap()
        .into_data()
        .try_into_vec::<f32>()
        .unwrap();
    assert!(grad.iter().any(|g| g.abs() > 0.01));
}

#[test]
fn qkv_masks_only_the_key_bias() {
    let device = Device::flex().autodiff();
    let mut attn = AttentionConfig::new(4, 1).init::<LoRA>(None, &device);
    attn.qkv.linear.bias = Some(Initializer::Ones.init([12], &device));
    let qkv = attn.qkv.forward(Tensor::zeros([1, 1, 4], &device));
    assert_eq!(
        qkv.to_data().try_into_vec::<f32>().unwrap(),
        [1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1., 1.]
    );
    let grads = qkv.sum().backward();
    assert!(attn.qkv.bias_mask.as_ref().unwrap().grad(&grads).is_none());
}

#[test]
fn lora_scales_by_rank_and_trains_independent_adapters() {
    let device = Device::flex().autodiff();
    let config = LoRAConfig::new(2, 4.0).with_a_initializer(Initializer::Ones);
    let adapter = config
        .clone()
        .with_b_initializer(Initializer::Ones)
        .init(4, &device);
    let update = adapter.forward(Tensor::ones([1, 1, 4], &device)) * adapter.alpha_div_dim();
    assert_eq!(
        update.into_data().try_into_vec::<f32>().unwrap(),
        [16., 16., 16., 16., 0., 0., 0., 0., 16., 16., 16., 16.]
    );
    let model: DinoVisionTransformer =
        DinoVisionTransformerConfig::new(2, 8, 1, 2, 2, 2.0).init(Some(config), &device);
    let model = model.no_grad_except_lora();
    let first = model.blocks[0].attn.lora.as_ref().unwrap();
    let second = model.blocks[1].attn.lora.as_ref().unwrap();
    assert_ne!(first.a.id, second.a.id);
    assert_ne!(first.b_q.id, second.b_q.id);
    assert_ne!(first.b_v.id, second.b_v.id);
    assert!(!model.mask_token.is_require_grad());
    let x = Tensor::arange(0..72, &device).float().reshape([1, 3, 4, 6]);
    let grads = model
        .forward_cls(x, None)
        .select_dim::<1>(1, 0)
        .sum()
        .backward();
    for block in &model.blocks {
        assert!(block.attn.proj.weight.grad(&grads).is_none());
        let lora = block.attn.lora.as_ref().unwrap();
        assert!(lora.a.grad(&grads).is_some());
        assert!(lora.b_q.grad(&grads).is_some());
        assert!(lora.b_v.grad(&grads).is_some());
    }
}

#[test]
fn swiglu_checkpoint_presets_have_the_released_shapes() {
    let device = Device::flex();
    for (builder, dim, depth, width, qkv_bias) in [
        (
            vit_small_plus::<LoRA> as fn(_, _, _) -> _,
            384,
            12,
            1536,
            true,
        ),
        (vit_large_plus::<LoRA>, 1024, 24, 4096, true),
        (vit_huge_plus::<LoRA>, 1280, 32, 5120, true),
        (vit_7b::<LoRA>, 4096, 40, 8192, false),
    ] {
        let model: DinoVisionTransformer = builder(16, None, &device);
        assert_eq!(model.blocks.len(), depth);
        let block = &model.blocks[depth - 1];
        assert_eq!(block.attn.qkv.linear.bias.is_some(), qkv_bias);
        assert_eq!(block.attn.qkv.bias_mask.is_some(), qkv_bias);
        assert_eq!(model.local_cls_norm.is_some(), dim == 4096);
        let FeedForward::SwiGlu(mlp) = &block.mlp else {
            panic!("expected SwiGLU")
        };
        assert_eq!(mlp.w1.weight.lazy_shape().as_slice(), &[dim, width]);
        assert_eq!(mlp.w2.weight.lazy_shape().as_slice(), &[dim, width]);
        assert_eq!(mlp.w3.weight.lazy_shape().as_slice(), &[width, dim]);
    }
}

#[test]
fn local_norm_only_applies_to_prefix_tokens_during_training() {
    let device = Device::flex().autodiff();
    let mut model: DinoVisionTransformer = DinoVisionTransformerConfig::new(2, 8, 2, 0, 1, 2.0)
        .with_untie_global_and_local_cls_norm(Some(true))
        .init(None, &device);
    let local = model.local_cls_norm.as_mut().unwrap();
    local.gamma = Initializer::Zeros.init([8], &device);
    local.beta = Some(Initializer::Ones.init([8], &device));
    let image = Tensor::arange(0..48, &device).float().reshape([1, 3, 4, 4]);
    let global = model.forward(image.clone(), None);
    let local = model.forward_local(image.clone(), None);
    assert_eq!(
        local
            .clone()
            .narrow(1, 0, 3)
            .into_data()
            .try_into_vec::<f32>()
            .unwrap(),
        [1.; 24]
    );
    assert_eq!(
        global.clone().narrow(1, 3, 4).into_data(),
        local.narrow(1, 3, 4).into_data()
    );
    let eval = model.valid();
    let image = image.without_autodiff();
    assert_eq!(
        eval.forward_local(image.clone(), None).into_data(),
        eval.forward(image, None).into_data()
    );
}

#[test]
fn rotary_grid_rescales_during_training_and_rotates_a_prefix_free_sequence() {
    let device = Device::flex().autodiff();
    let model: DinoVisionTransformer =
        DinoVisionTransformerConfig::new(2, 8, 0, 1, 1, 2.0).init(None, &device);
    let (sin1, _) = model.rope_embed.forward(2, 3);
    let (sin2, _) = model.rope_embed.forward(2, 3);
    assert_ne!(sin1.into_data(), sin2.into_data());
    let eval = model.valid();
    let (sin, cos) = eval.rope_embed.forward(2, 3);
    let (sin2, _) = eval.rope_embed.forward(2, 3);
    assert_eq!(sin.to_data(), sin2.into_data());
    // First patch center (-1/2, -2/3), with periods 1 and 10 in each axis.
    let expected = [-0.5_f32, -0.05, -2.0 / 3.0, -2.0 / 30.0];
    let row = sin
        .clone()
        .select_dim::<1>(0, 0)
        .into_data()
        .try_into_vec::<f32>()
        .unwrap();
    for (i, actual) in row.iter().enumerate() {
        assert!((actual - (expected[i % 4] * std::f32::consts::TAU).sin()).abs() < 1e-6);
    }
    let tokens = Tensor::ones([1, 6, 8], &Device::flex());
    let output = eval.blocks[0].attn.forward(tokens, Some(&(sin, cos)));
    assert_eq!(output.dims(), [1, 6, 8]);
}

#[test]
fn current_mlp_configs_preserve_upstream_widths_after_serialization() {
    use burn::config::{Config, config_to_json};

    for (ratio, width) in [(2.1, 16), (0.1, 8)] {
        let config = DinoVisionTransformerConfig::new(2, 8, 0, 1, 2, ratio);
        let json = config_to_json(&config);
        let config = DinoVisionTransformerConfig::load_binary(json.as_bytes()).unwrap();
        let model = config.init::<LoRA>(None, &Device::flex());
        assert!(model.storage_tokens.is_none());
        let FeedForward::Mlp(mlp) = &model.blocks[0].mlp else {
            panic!("expected GELU MLP")
        };
        assert_eq!(mlp.fc1.weight.lazy_shape().as_slice(), &[8, width]);
        assert_eq!(mlp.fc2.weight.lazy_shape().as_slice(), &[width, 8]);
    }
}

#[test]
fn legacy_burnpack_restores_fractional_mlp_and_trains_each_adapter() {
    use burn::{config::Config, optim::GradientsParams};
    use burn_store::{BurnpackStore, ModuleSnapshot};

    // Written by burn_dinov3 0.1.7 and burn-store 0.21 with a rank-2, alpha-4 all-ones adapter.
    let config =
        DinoVisionTransformerConfig::load_binary(include_bytes!("fixtures/model-0.1.7.json"))
            .unwrap();
    let device = Device::default().autodiff();
    let mut model: DinoVisionTransformer = config.init(Some(LoRAConfig::new(2, 1.0)), &device);
    let mut source = BurnpackStore::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/model-0.1.7.bpk"
    ))
    .with_remap_pattern(r"\bmlp\.(fc[12])\.", "mlp.Mlp.$1.");
    let result = model.load_from(&mut source).unwrap();
    assert!(result.unused.is_empty(), "{result}");
    assert_eq!(model.storage_tokens.as_ref().unwrap().dims(), [1, 0, 8]);
    let FeedForward::Mlp(mlp) = &model.blocks[0].mlp else {
        panic!("expected GELU MLP")
    };
    assert_eq!(mlp.fc1.weight.dims(), [8, 17]);
    assert_eq!(mlp.fc2.weight.dims(), [17, 8]);
    let mlp_id = mlp.fc1.weight.id;
    let adapter_ids = |model: &DinoVisionTransformer| {
        model
            .blocks
            .iter()
            .map(|block| {
                let lora = block.attn.lora.as_ref().unwrap();
                [lora.a.id, lora.b_q.id, lora.b_v.id]
            })
            .collect::<Vec<_>>()
    };
    let original = adapter_ids(&model);
    assert_eq!(original[0], original[1]);
    model.deduplicate_lora_ids();
    let migrated = adapter_ids(&model);
    assert_eq!(migrated[0], original[0]);
    model.deduplicate_lora_ids();
    assert_eq!(adapter_ids(&model), migrated);
    let FeedForward::Mlp(mlp) = &model.blocks[0].mlp else {
        unreachable!()
    };
    assert_eq!(mlp.fc1.weight.id, mlp_id);
    for block in &model.blocks {
        let adapter = block.attn.lora.as_ref().unwrap();
        for weight in [&adapter.a, &adapter.b_q, &adapter.b_v] {
            assert_eq!(weight.to_data().try_into_vec::<f32>().unwrap(), [1.; 16]);
        }
    }
    let model = model.no_grad_except_lora();
    let x = Tensor::arange(0..72, &device).float().reshape([1, 3, 4, 6]);
    let grads = model
        .forward_cls(x, None)
        .select_dim::<1>(1, 0)
        .sum()
        .backward();
    let grads = GradientsParams::from_grads(grads, &model);
    assert_eq!(grads.len(), 6);
    for id in migrated.iter().flatten() {
        assert!(grads.get::<2>(*id).is_some());
    }
}
