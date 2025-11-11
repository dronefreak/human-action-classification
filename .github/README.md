# Human Action Classification v2.0 🚀

[![🐍 Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white&style=for-the-badge)](https://www.python.org/downloads/)
[![🔥 PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white&style=for-the-badge)](https://pytorch.org/)
[![🤗 HuggingFace Models](https://img.shields.io/badge/HuggingFace-Models-FFD21E?logo=huggingface&logoColor=yellow&style=for-the-badge)](https://huggingface.co/dronefreak/human-action-classification-stanford40)
[![⚖️ License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green?logo=open-source-initiative&logoColor=white&style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)

**The fastest way to classify human actions in images and video.** MediaPipe pose estimation + PyTorch CNNs with **90 FPS real-time performance**. Zero C++ compilation, pure Python.

> **v2.0**: Complete PyTorch rewrite — **47x faster** than the original TF 1.13 version • [Legacy code →](https://github.com/dronefreak/human-action-classification/tree/legacy)

---

## 🎬 Demo

<!-- TODO: Add demo GIF here showing: input image → skeleton overlay → action prediction -->
<!-- Example: ![Demo](assets/demo.gif) -->

<table>
  <tr>
    <td align="center">
      <img src="../assets/gardening.jpg" width="280"/><br/>
      <b>Gardening</b><br/>
      <!-- Confidence: 94% -->
    </td>
    <td align="center">
      <img src="../assets/looking_through_a_telescope.jpg" width="280"/><br/>
      <b>Looking Through a Telescope</b><br/>
      <!-- Confidence: 91% -->
    </td>
  </tr>
</table>

## 🎮 Try It Now

```bash
# Launch web demo in 10 seconds
pip install -e ".[demo]" && hac-demo
```

<!-- TODO: Add these when ready
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](your-colab-link)
[![🤗 HuggingFace Space](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](your-space-link)
-->

## ⚡ Why v2.0?

✅ **90 FPS real-time** (11ms per frame on RTX 4070 Super)
✅ **Zero C++ compilation** — pure Python `pip install`
✅ **100+ architectures** — swap any timm model
✅ **88.5% accuracy** on Stanford40 dataset

<details>
<summary>📊 Detailed Comparison: v1.0 vs v2.0</summary>

| Feature                   | v1.0 (TF 1.13)                  | v2.0 (PyTorch)                  |
| ------------------------- | ------------------------------- | ------------------------------- |
| **Inference Speed**       | 1.4s per image                  | ~0.011s per image (127x faster) |
| **Installation**          | SWIG + C++ compilation required | Pure Python, pip install        |
| **Pose Estimation**       | tf-pose-estimation (OpenPose)   | MediaPipe (production-ready)    |
| **Framework**             | TensorFlow 1.13                 | PyTorch 2.0+                    |
| **Models**                | Fixed MobileNet/Inception       | Any timm model (100+ options)   |
| **Dependency Management** | requirements.txt chaos          | pyproject.toml + Poetry         |
| **API**                   | Script-based                    | Clean Python API + CLI          |

</details>

## Quick Start

### Installation

```bash
# Quick start (recommended for most users)
pip install -e ".[demo]"

# Full installation with training tools
pip install -e ".[dev,demo,train]"
```

<details>
<summary>Using Conda? (Optional)</summary>

```bash
conda env create -f environment.yml
conda activate human-action-classification
pip install -e ".[demo]"
```

</details>

### Usage Examples

#### Web Interface (Easiest)

```bash
# Launch interactive demo
hac-demo

# Or with custom model
hac-demo --model weights/best.pth --share
```

#### Command Line

```bash
# Single image
hac-infer --image photo.jpg --model weights/best.pth

# Video processing
hac-infer --video video.mp4 --model weights/best.pth

# Real-time webcam
hac-infer --webcam --model weights/best.pth
```

#### Python API

```python
from hac import ActionPredictor

# Initialize predictor
predictor = ActionPredictor(
    model_path=None,  # Uses pretrained ImageNet backbone
    device='cuda',
    use_pose_estimation=True
)

# Predict from image
result = predictor.predict_image('person.jpg')

print(f"Pose: {result['pose']['class']}")
print(f"Action: {result['action']['top_class']}")
print(f"Confidence: {result['action']['top_confidence']:.2%}")
```

## 🎓 Training Your Own Models

### Quick Training

```bash
# 1. Download Stanford40 dataset from http://vision.stanford.edu/Datasets/40actions.html
# 2. Organize as: data/train/class1/*.jpg, data/val/class1/*.jpg
# 3. Train
python -m hac.training.train \
    --data_dir data/ \
    --model_name mobilenetv3_small_100 \
    --num_classes 40 \
    --epochs 50
```

<details>
<summary>📝 Advanced Training Options</summary>

### Python API

```python
from hac.training.train import Trainer
from hac.models.classifier import create_model
from hac.data.dataset import ActionDataset

# Create model (any timm architecture)
model = create_model(
    model_type='action',
    model_name='efficientnet_b0',
    num_classes=40,
    pretrained=True
)

# Setup datasets
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

### Custom Dataset

```bash
# Works with any image classification dataset
# Just organize as: data/train/your_class_1/*.jpg, data/val/your_class_1/*.jpg
python -m hac.training.train \
    --data_dir data/ \
    --model_name resnet18 \
    --num_classes YOUR_NUM_CLASSES \
    --batch_size 32 \
    --lr 1e-3
```

</details>

## Architecture

**Dual-stream pipeline:**

1. **MediaPipe Pose Detection** → Extract 33 body keypoints
2. **Rule-based Pose Classifier** → Classify as sitting/standing/lying (simple geometric rules)
3. **CNN Action Classifier** → Predict action from 40 classes (trained neural network)

Models can be swapped via timm integration - test 100+ architectures without code changes.

## Pretrained Models

| Model                 | Accuracy  | Speed (RTX 4070) | Best For      | Download                                                                                                                                                                                           |
| --------------------- | --------- | ---------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ResNet50**          | **88.5%** | 6ms              | Best Accuracy | [![🤗](https://img.shields.io/badge/↓-HuggingFace-FFD21E?logo=huggingface&logoColor=yellow)](https://huggingface.co/dronefreak/human-action-classification-stanford40/tree/main/resnet50)          |
| ResNet34              | 86.4%     | 5ms              | Balanced      | [![🤗](https://img.shields.io/badge/↓-HuggingFace-FFD21E?logo=huggingface&logoColor=yellow)](https://huggingface.co/dronefreak/human-action-classification-stanford40/tree/main/resnet34)          |
| **MobileNetV3-Large** | **82.1%** | **3ms**          | Edge/Mobile   | [![🤗](https://img.shields.io/badge/↓-HuggingFace-FFD21E?logo=huggingface&logoColor=yellow)](https://huggingface.co/dronefreak/human-action-classification-stanford40/tree/main/mobilenetv3_large) |
| ResNet18              | 82.3%     | 4ms              | Fast          | [![🤗](https://img.shields.io/badge/↓-HuggingFace-FFD21E?logo=huggingface&logoColor=yellow)](https://huggingface.co/dronefreak/human-action-classification-stanford40/tree/main/resnet18)          |
| MobileNetV3-Small     | 74.4%     | 2ms              | Ultra-Fast    | [![🤗](https://img.shields.io/badge/↓-HuggingFace-FFD21E?logo=huggingface&logoColor=yellow)](https://huggingface.co/dronefreak/human-action-classification-stanford40/tree/main/mobilenetv3_small) |

> All models trained on **Stanford40 Actions** (40 classes) • Licensed under [Apache 2.0](https://opensource.org/licenses/Apache-2.0)

<details>
<summary>Full Metrics Table (Precision, Recall, F1)</summary>

| Model             | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| ----------------- | -------- | --------------- | ------------ | -------- | ----------- |
| ResNet50          | 88.5%    | 0.887           | 0.885        | 0.884    | 0.884       |
| ResNet34          | 86.4%    | 0.869           | 0.864        | 0.862    | 0.862       |
| ResNet18          | 82.3%    | 0.821           | 0.823        | 0.818    | 0.818       |
| MobileNetV3-Large | 82.1%    | 0.822           | 0.821        | 0.817    | 0.817       |
| MobileNetV3-Small | 74.4%    | 0.738           | 0.744        | 0.735    | 0.735       |

</details>

### Use Any Model via timm

```python
# Lightweight (mobile/edge)
create_model('mobilenetv3_small_100', num_classes=40)
create_model('efficientnet_lite0', num_classes=40)

# Balanced (recommended)
create_model('efficientnet_b0', num_classes=40)
create_model('convnext_tiny', num_classes=40)

# Heavy (maximum accuracy)
create_model('efficientnet_b4', num_classes=40)
create_model('convnext_base', num_classes=40)
```

> [View all 100+ available models →](timm_model_families.md)

## Performance Benchmarks

**NVIDIA RTX 4070 Super (Latest):**
| Stage | Time | FPS |
|-------|------|-----|
| Pose Detection (MediaPipe) | ~5ms | - |
| Action Classification | ~6ms | - |
| **Total Pipeline** | **~11ms** | **90 FPS** |

**NVIDIA GTX 1050 Ti (Legacy):**
| Stage | Time | FPS |
|-------|------|-----|
| Pose Detection | ~20ms | - |
| Action Classification | ~10ms | - |
| **Total Pipeline** | **~30ms** | **33 FPS** |

**v1.0 (TF 1.13 + OpenPose) - GTX 1050 Ti:**
| Stage | Time |
|-------|------|
| Pose Detection (OpenPose) | ~800ms |
| Action Classification | ~200ms |
| **Total Pipeline** | **~1400ms (0.7 FPS)** |

> **Result:** v2.0 is **47x faster** on same hardware, **127x faster** on modern GPUs

## Real-World Applications

**Fitness & Sports** — Analyze workout form, track exercise reps, coach technique
**Autonomous Vehicles** — Predict pedestrian crossing intent, detect distracted behavior
**Healthcare** — Fall detection for elderly care, patient mobility monitoring
**Gaming & VR** — Control games with body movements, motion capture
**Workplace Safety** — Detect unsafe working postures, PPE compliance
**Content Creation** — Auto-tag video content, generate highlights, behavior analytics

<details>
<summary>Code Example: Pedestrian Intent Detection</summary>

```python
from hac import ActionPredictor

predictor = ActionPredictor(device='cuda')

# Analyze pedestrian behavior near crosswalk
result = predictor.predict_image(pedestrian_crop)
action = result['action']['top_class']
confidence = result['action']['top_confidence']

# High-risk actions
if action in ['running', 'walking'] and confidence > 0.8:
    # Pedestrian likely to cross
    vehicle.alert_driver()
    vehicle.reduce_speed()

# Medium-risk actions
elif action in ['looking_through_a_telescope', 'texting_message']:
    # Distracted pedestrian
    vehicle.monitor_closely()
```

</details>

<details>
<summary>Code Example: Export for Edge Devices</summary>

```python
from hac.models.classifier import create_model
import torch

# Create lightweight model
model = create_model('mobilenetv3_small_100', num_classes=40)
model.load_state_dict(torch.load('weights/best.pth'))
model.eval()

# Export to ONNX for Jetson/mobile
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    input_names=['image'],
    output_names=['action'],
    dynamic_axes={'image': {0: 'batch'}}
)
```

</details>

---

<details>
<summary>Migrating from v1.0?</summary>

### Quick Migration Guide

```python
# Old (v1.0 - TF 1.13)
from tf_pose.estimator import TfPoseEstimator
estimator = TfPoseEstimator(...)
humans = estimator.inference(image)

# New (v2.0 - PyTorch)
from hac import ActionPredictor
predictor = ActionPredictor()
result = predictor.predict_image(image)
```

**Key Changes:**

- ✅ No more SWIG/C++ compilation
- ✅ 127x faster inference
- ✅ Simpler API with `pip install`
- ✅ Models via HuggingFace Hub

See [legacy branch](https://github.com/dronefreak/human-action-classification/tree/legacy) for v1.0 code.

</details>

<details>
<summary>Performance Optimization Tips</summary>

### Speed Up Inference

```python
# 1. Use GPU (4-5x faster)
predictor = ActionPredictor(device='cuda')

# 2. Disable pose detection if not needed
predictor = ActionPredictor(use_pose_estimation=False)

# 3. Use lighter models for real-time
model = create_model('mobilenetv3_small_100', num_classes=40)  # <10ms

# 4. Batch processing for videos
results = [predictor.predict_image(frame) for frame in frames]

# 5. Export to TorchScript for production
model.eval()
traced = torch.jit.trace(model, example_input)
traced.save('model_traced.pt')
```

### Memory Optimization

- Use MobileNet models for edge devices
- Reduce input resolution if accuracy allows
- Enable mixed precision training with `torch.cuda.amp`

</details>

---

## Roadmap

**Completed**

- Core inference pipeline with MediaPipe + PyTorch
- Training scripts with modern PyTorch trainer
- CLI tools and Gradio web demo
- Pretrained weights on HuggingFace Hub
- Support for 100+ model architectures via timm

**In Progress**

- ONNX export utilities
- TensorRT optimization guides
- Example Jupyter notebooks

**Planned**

- HuggingFace Spaces demo
- Google Colab notebook
- Mobile deployment guides (iOS/Android)
- Raspberry Pi benchmarks

## Contributing

**High-impact contributions wanted:**

- [ ] **HuggingFace Space demo** — Deploy interactive web demo
- [ ] **Colab notebook** — One-click demo for users
- [ ] **Mobile benchmarks** — Test on iOS/Android devices
- [ ] **Raspberry Pi guide** — Edge deployment tutorial
- [ ] **ROS2 integration** — Robotics/AV integration example
- [ ] **Additional datasets** — Train on Kinetics/UCF-101
- [ ] **Video demo GIF** — Show real-time inference

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions.

**Have other ideas?** Open an issue or PR!

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
