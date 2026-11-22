# Security Policy

## Supported versions

| Version                        | Supported |
| ------------------------------ | --------- |
| 2.x (PyTorch)                  | ✅        |
| 1.x (TensorFlow 1.13, retired) | ❌        |

## Reporting a vulnerability

Please **do not open a public issue**. Report privately via either:

- GitHub private vulnerability reporting: **Security** tab → **Report a vulnerability**, or
- email: kumaar324@gmail.com

Include a description, impact, steps to reproduce, affected version, and your environment (OS, Python, package versions). You should get an acknowledgement within a few days; we will coordinate a fix and disclosure with you and credit you unless you prefer otherwise.

## Things to be aware of when using this project

- **Checkpoints are pickles.** `.pth` files are loaded with `torch.load(..., weights_only=False)`, which can execute arbitrary code. Only load `.pth` files you trust; prefer the `.safetensors` releases on the [Hugging Face Hub](https://huggingface.co/dronefreak), which contain tensors only.
- **The Gradio demo has no authentication.** `hac-demo` listens on `0.0.0.0`; keep it on a trusted network, and only pass `--share` if you intend to create a public link.
- **Inputs are decoded by OpenCV/MediaPipe.** Treat untrusted images and videos as untrusted input and keep those dependencies up to date (`pip install -U opencv-python`, within the pinned MediaPipe range).

Model accuracy, adversarial examples and performance issues are not security vulnerabilities; please report them as regular issues.
