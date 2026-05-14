You can download `.pth` following facebook [DINOv3 repo](https://github.com/facebookresearch/dinov3).

And put them in `./models`.

Then convert the `.pth` model to `.onnx`:
```sh
cd pth2onnx
uv run main.py
```
