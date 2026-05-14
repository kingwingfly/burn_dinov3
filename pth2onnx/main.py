from functools import lru_cache
from pathlib import Path
import torch
import os

MODELS_DIR = "../models"


@lru_cache(maxsize=1)
def models() -> dict[str, Path]:
    return {
        key: Path(f"{root}/{file}")
        for root, _, files in os.walk(MODELS_DIR)
        for file in files
        if file.endswith(".pth") and (key := "_".join(file.split("_")[:2]))
    }


def main():
    for model, path in models().items():
        convert(model, path)


def convert(model: str, path: Path):
    torch_model = torch.hub.load(
        repo_or_dir="facebookresearch/dinov3",
        model=model,
        weights=str(path),
    )
    torch_model.eval()
    example_inputs = (
        torch.rand(1, 3, 256, 256, dtype=torch.float32, requires_grad=True),
    )
    torch_model(example_inputs[0])
    onnx_program = torch.onnx.export(
        torch_model,
        example_inputs,
        path.with_suffix(".onnx"),
        input_names=["input"],
        output_names=["output"],
        # dynamic_shapes={"args": ({0: "batch_size"},)},
        dynamo=False,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    # assert onnx_program is not None
    # onnx_program.save(path.with_suffix(".onnx"))


if __name__ == "__main__":
    main()
