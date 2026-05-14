import onnx
from onnx import version_converter, shape_inference

MODELS_DIR = "../models"

def main():
    # Load your ONNX model
    model = onnx.load('../models/model.onnx')

    # Convert the model to opset version 26
    upgraded_model = version_converter.convert_version(model, 26)

    # Apply shape inference to the upgraded model
    inferred_model = shape_inference.infer_shapes(upgraded_model)

    # Save the converted model
    onnx.save(inferred_model, '../models/upgraded_model.onnx')

if __name__ == "__main__":
    main()
