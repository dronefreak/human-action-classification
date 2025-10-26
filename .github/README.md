# Human Action Classification v2.0 🚀

[![🐍 Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white&style=for-the-badge)](https://www.python.org/downloads/)
[![🔥 PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white&style=for-the-badge)](https://pytorch.org/)
[![🤗 HuggingFace Models](https://img.shields.io/badge/HuggingFace-Models-FFD21E?logo=huggingface&logoColor=yellow&style=for-the-badge)](https://huggingface.co/dronefreak/human-action-classification-stanford40)
[![⚖️ License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green?logo=open-source-initiative&logoColor=white&style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)

Modern human action classification using **MediaPipe** pose estimation and **PyTorch** deep learning. Complete rewrite of the original TensorFlow 1.13 version with 10x faster inference and zero C++ compilation headaches.

> **⚠️ This is v2.0** - A complete modernization of the [original repo](https://github.com/dronefreak/human-action-classification). The legacy TF 1.13 code is preserved in the `legacy` branch.

## What's New in v2.0?

| Feature                   | v1.0 (TF 1.13)                  | v2.0 (PyTorch)                |
| ------------------------- | ------------------------------- | ----------------------------- |
| **Inference Speed**       | 1.4s per image                  | ~0.05s per image (30x faster) |
| **Installation**          | SWIG + C++ compilation required | Pure Python, pip install      |
| **Pose Estimation**       | tf-pose-estimation (OpenPose)   | MediaPipe (production-ready)  |
| **Framework**             | TensorFlow 1.13                 | PyTorch 2.0+                  |
| **Models**                | Fixed MobileNet/Inception       | Any timm model (100+ options) |
| **Config Management**     | Hardcoded paths                 | Hydra configs                 |
| **Dependency Management** | requirements.txt chaos          | pyproject.toml + Poetry       |
| **API**                   | Script-based                    | Clean Python API + CLI        |
| **Training**              | Custom loops                    | Modern trainer with metrics   |

## Features

- 🏃 **Fast inference**: MediaPipe pose detection + lightweight CNNs
- 🎯 **Dual-task classification**: Pose (sitting/standing/lying) + Action (40 classes)
- 🔧 **Flexible architecture**: Swap models easily with timm integration
- 📦 **Easy installation**: No C++ compilation, pure Python
- 🎥 **Multiple inputs**: Images, videos, webcam
- 🌐 **Web demo**: Gradio interface included
- 📊 **Training pipeline**: Ready-to-use training scripts
- 🚗 **AV-ready**: Can be adapted for pedestrian action prediction

## Examples

<table>
  <tr>
    <td><img src="../assets/gardening.jpg" width="300" height="200"/></td>
    <td><img src="../assets/looking_through_a_telescope.jpg" width="300" height="200"/></td>
  </tr>
</table>

## Quick Start

### Installation

```bash
# Creat the conda env
conda env create -f environment.yml

# Activate the conda env
conda activate human-action-classification

# Basic installation
pip install -e .

# With development tools
pip install -e ".[dev]"

# With Gradio demo
pip install -e ".[demo]"

# With training tools
pip install -e ".[train]"

# Everything
pip install -e ".[dev,demo,train]"
```

### Inference (No Training Required)

```python
from hac import ActionPredictor

# Initialize predictor
predictor = ActionPredictor(
    model_path=None,  # Uses pretrained backbone for demo
    device='cuda',
    use_pose_estimation=True
)

# Predict from image
result = predictor.predict_image('person.jpg')

print(f"Pose: {result['pose']['class']}")
print(f"Action: {result['action']['top_class']}")
print(f"Confidence: {result['action']['top_confidence']:.2f}")
```

### Command Line: Quickstart

```bash
# Single image
hac-infer --image photo.jpg --model weights/best.pth

# Video processing
hac-infer --video video.mp4 --model weights/best.pth

# Real-time webcam
hac-infer --webcam --model weights/best.pth

# Real-time webcam (uses pretrained ImageNet backbone)
python scripts/simple_demo.py --webcam

# Launch web demo
hac-demo --model weights/best.pth --share
```

## Training on Stanford 40 Dataset

### 1. Prepare Dataset

Download [Stanford 40 Actions Dataset](http://vision.stanford.edu/Datasets/40actions.html) and organize:

```
data/
├── train/
│   ├── applauding/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   ├── drinking/
│   └── ...
└── val/
    ├── applauding/
    └── ...
```

### 2. Train

```bash
python -m hac.training.train \
    --data_dir data/ \
    --model_name mobilenetv3_small_100 \
    --num_classes 40 \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-3 \
    --output_dir outputs/
```

### 3. Training Script Options

```python
from hac.training.train import Trainer
from hac.models.classifier import create_model
from hac.data.dataset import ActionDataset

# Create model
model = create_model(
    model_type='action',
    model_name='efficientnet_b0',  # Or any timm model
    num_classes=40,
    pretrained=True
)

# Create datasets
train_dataset = ActionDataset('data/', split='train')
val_dataset = ActionDataset('data/', split='val')

# Train
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    device='cuda',
    output_dir='outputs/',
    max_epochs=50
)

trainer.train()
```

### 4. Train on Your Data

```bash
# Organize data as: data/train/class1/, data/val/class1/, ...
python -m hac.training.train \
    --data_dir data/ \
    --model_name mobilenetv3_small_100 \
    --num_classes 40 \
    --epochs 50
```

## Architecture

```
Input Image
    ↓
[MediaPipe Pose Detection] → Keypoints (33 joints)
    ↓                              ↓
[Rule-based Classifier]      [CNN Backbone]
    ↓                              ↓
Pose Class                   Action Class
(sitting/standing/lying)     (40 activities)
```

### Available Models (via timm)

```python
# Lightweight (mobile/edge deployment)
- mobilenetv3_small_100
- mobilenetv3_large_100
- efficientnet_lite0

# Balanced (recommended)
- efficientnet_b0
- resnet18
- convnext_tiny

# Heavy (maximum accuracy)
- efficientnet_b4
- convnext_base
- vit_base_patch16_224
```

## Model Comparison for Stanfor40

| Model                 | Accuracy (%) | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :-------------------- | -----------: | --------------: | -----------: | -------: | ----------: |
| resnet50              |         88.5 |          0.8874 |        0.885 |   0.8842 |      0.8842 |
| resnet34              |         86.4 |          0.8686 |        0.864 |   0.8618 |      0.8618 |
| resnet18              |         82.3 |          0.8211 |        0.823 |   0.8178 |      0.8178 |
| mobilenetv3_large_100 |         82.1 |          0.8216 |        0.821 |   0.8169 |      0.8169 |
| vit_base_patch16_224  |         76.8 |          0.7774 |        0.768 |    0.765 |       0.765 |
| mobilenetv3_small_100 |        74.35 |          0.7382 |       0.7435 |    0.735 |       0.735 |

For more information on the available models in timm, please refer to [TIMM_MODELS](timm_model_families.md)

## 🤗 Pre-trained Models

All models trained on Stanford 40 Actions are available on HuggingFace Hub:

<details>
<summary><b>📊 View All Available Pre-trained Models</b></summary>

| Model             | Accuracy | F1 Score | Size | Speed | Link                                                                                                             |
| ----------------- | -------- | -------- | ---- | ----- | ---------------------------------------------------------------------------------------------------------------- |
| ResNet50          | 88.5%    | 0.8842   | 94MB | 30ms  | [Download](https://huggingface.co/dronefreak/human-action-classification-stanford40/tree/main/resnet50)          |
| ResNet34          | 86.4%    | 0.8618   | 85MB | 25ms  | [Download](https://huggingface.co/dronefreak/human-action-classification-stanford40/tree/main/resnet34)          |
| ResNet18          | 82.3%    | 0.8178   | 45MB | 18ms  | [Download](https://huggingface.co/dronefreak/human-action-classification-stanford40/tree/main/resnet18)          |
| MobileNetV3 Large | 82.1%    | 0.8169   | 20MB | 15ms  | [Download](https://huggingface.co/dronefreak/human-action-classification-stanford40/tree/main/mobilenetv3_large) |
| MobileNetV3 Small | 74.35%   | 0.7350   | 10MB | 10ms  | [Download](https://huggingface.co/dronefreak/human-action-classification-stanford40/tree/main/mobilenetv3_small) |

</details>

## Project Structure

```
human-action-classification-v2/
├── src/hac/
│   ├── models/          # PyTorch model definitions
│   │   └── classifier.py
│   ├── inference/       # Inference pipeline
│   │   ├── pose_extractor.py
│   │   └── predictor.py
│   ├── data/           # Dataset loaders
│   │   └── dataset.py
│   ├── training/       # Training utilities
│   │   └── train.py
│   ├── utils/          # Helper functions
│   │   └── transforms.py
│   └── cli.py          # Command-line interface
├── configs/            # Hydra configs (coming soon)
├── examples/           # Jupyter notebooks
├── tests/             # Unit tests
├── scripts/           # Utility scripts
├── weights/           # Model checkpoints
├── pyproject.toml     # Project metadata
└── README.md
```

## Benchmarks

Tested on NVIDIA GTX 1050 Ti (4GB VRAM):

| Pipeline Stage | v1.0 (TF 1.13)    | v2.0 (PyTorch)    |
| -------------- | ----------------- | ----------------- |
| Pose Detection | ~800ms (OpenPose) | ~20ms (MediaPipe) |
| Classification | ~200ms            | ~10ms             |
| **Total**      | **~1400ms**       | **~30ms**         |

**Result: 47x faster inference**

## Use Cases

### 1. Original: General Action Recognition

- Human activity monitoring
- Video surveillance
- Sports analytics

### 2. Autonomous Vehicles (New!)

```python
# Pedestrian action prediction for AVs
from hac import ActionPredictor

predictor = ActionPredictor(device='cuda')

# Predict pedestrian intent
result = predictor.predict_image(pedestrian_crop)

if result['action']['top_class'] in ['running', 'walking']:
    # High risk - pedestrian might cross
    alert_vehicle()
```

### 3. Edge Deployment

```python
# Optimize for Jetson/edge devices
model = create_model(
    model_name='mobilenetv3_small_100',
    num_classes=40
)

# Export to ONNX
torch.onnx.export(model, dummy_input, 'model.onnx')
```

## Migration from v1.0

If you're using the old TF 1.13 version:

```python
# Old (v1.0)
from tf_pose.estimator import TfPoseEstimator
estimator = TfPoseEstimator(...)
humans = estimator.inference(image)

# New (v2.0)
from hac import ActionPredictor
predictor = ActionPredictor()
result = predictor.predict_image(image)
```

## Performance Tips

1. **Batch processing**: Use DataLoader for multiple images
2. **GPU inference**: Set `device='cuda'` (4-5x faster)
3. **Model selection**: Use MobileNetV3 for real-time (<50ms)
4. **Pose detection**: Disable if only action classification needed
5. **TorchScript**: Export model for production

```python
# TorchScript export
model = ActionClassifier(...)
model.eval()
traced = torch.jit.trace(model, example_input)
traced.save('model_traced.pt')
```

## Roadmap

- [x] Core inference pipeline
- [x] Training scripts
- [x] CLI tools
- [x] Gradio demo
- [ ] Hydra configs
- [ ] ONNX export
- [ ] TensorRT optimization
- [ ] Video temporal models (SlowFast, X3D)
- [ ] Pedestrian trajectory prediction
- [ ] NuScenes/Waymo integration
- [ ] Pre-trained weights on HuggingFace

## Contributing

Contributions welcome! Areas of interest:

- Pre-trained model weights
- New datasets integration
- Temporal models for video
- Edge deployment optimizations
- Pedestrian action prediction for AVs

## Citation

If you use this work, please cite:

```bibtex
@software{saksena2025hac,
  author = {Saksena, Saumya Kumaar},
  title = {Human Action Classification v2.0},
  year = {2025},
  url = {https://github.com/dronefreak/human-action-classification}
}
```

Original v1.0 references:

- OpenPose: [Arxiv](https://arxiv.org/abs/1812.08008)
- Stanford 40 Actions: [Dataset](http://vision.stanford.edu/Datasets/40actions.html)

## License

Apache License - see [LICENSE](../LICENSE) file for details.

## Acknowledgments

- **MediaPipe**: Google's pose estimation
- **timm**: Ross Wightman's model library
- **Stanford 40**: Dataset creators
- **Original repo users**: 233+ stars and counting!

---

**Author**: Saumya Kumaar Saksena
**Contact**: [GitHub](https://github.com/dronefreak)
**Original Repo**: [human-action-classification](https://github.com/dronefreak/human-action-classification)
