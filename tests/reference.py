"""Emit the official backbone's full token sequence for the Rust parity test."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dinov3"))
from dinov3.hub import backbones

name, path = sys.argv[1:]
torch.set_num_threads(2)
with torch.inference_mode():
    model = getattr(backbones, f"dinov3_{name}")(pretrained=False).eval()
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    image = ((torch.arange(2 * 3 * 35 * 49) % 251).float() / 125 - 1).reshape(2, 3, 35, 49)
    mask = torch.tensor([[False, True, False, False, True, False],
                         [True, False, False, True, False, True]])
    for masks in (None, mask):
        features = model.forward_features(image, masks)
        tokens = torch.cat([features["x_norm_clstoken"][:, None],
                            features["x_storage_tokens"], features["x_norm_patchtokens"]], dim=1)
        sys.stdout.buffer.write(tokens.contiguous().numpy().astype("<f4").tobytes())
