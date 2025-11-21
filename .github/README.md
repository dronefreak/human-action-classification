# Human Action Classification 🎬

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white&style=for-the-badge)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white&style=for-the-badge)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Models-2%20Published-FFD21E?style=for-the-badge)](https://huggingface.co/dronefreak)
[![License](https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge)](LICENSE)

**State-of-the-art action recognition** for both **images** and **videos**. From pose-based image classification to temporal 3D CNNs for video understanding.

![Demo](../docs/demo.gif)

> 🎯 **Two complete pipelines:** Single-frame pose classification (90 FPS) + Video temporal modeling (87% UCF-101)

---

## 🎬 What's New

### Video Action Recognition (NEW!) 🔥

- **87.05% accuracy** on UCF-101 with MC3-18
- **3D CNN models** (R3D-18, MC3-18) for temporal understanding
- **Published on HuggingFace:** [MC3-18](https://huggingface.co/dronefreak/mc3-18-ucf101) | [R3D-18](https://huggingface.co/dronefreak/r3d-18-ucf101)
- Train your own with complete training pipeline

### Image Action Recognition

- **88.5% accuracy** on Stanford40 with ResNet50
- **90 FPS real-time** inference with MediaPipe + PyTorch
- **Pose-aware classification** with geometric reasoning
- **Published on HuggingFace:** [ResNet50](https://huggingface.co/dronefreak/human-action-classification-stanford40)

---

## 📊 Model Zoo

### 🎥 Video Models (Temporal - UCF-101)

| Model      | Accuracy   | Params | FPS | Dataset               | Download                                                                                               |
| ---------- | ---------- | ------ | --- | --------------------- | ------------------------------------------------------------------------------------------------------ |
| **MC3-18** | **87.05%** | 11.7M  | 30  | UCF-101 (101 classes) | [![HF](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/dronefreak/mc3-18-ucf101) |
| **R3D-18** | **83.80%** | 33.2M  | 40  | UCF-101 (101 classes) | [![HF](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/dronefreak/r3d-18-ucf101) |

**Input:** 16-frame clips @ 112×112
**Use case:** Action classification in video clips (sports, activities, human-object interaction)

<details>
<summary>📈 Comparison with Published Baselines</summary>

| Method | Published | This Repo  | Improvement |
| ------ | --------- | ---------- | ----------- |
| R3D-18 | 82.8%     | **83.8%**  | +1.0% ✅    |
| MC3-18 | 85.0%     | **87.05%** | +2.05% ✅   |

_Our models match or exceed original papers!_

</details>

### 🎥 Video Models (Temporal - HMDB51)

| Model      | Init         | Accuracy   | Train-Val Gap | Params | Dataset             | Download                                                                                                            |
| ---------- | ------------ | ---------- | ------------- | ------ | ------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **MC3-18** | Kinetics-400 | **56.34%** | 19%           | 11.7M  | HMDB51 (51 classes) | [![HF](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/dronefreak/mc3-18-hmdb51-kinetics)     |
| **MC3-18** | UCF-101      | **55.46%** | 13%           | 11.7M  | HMDB51 (51 classes) | [![HF](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/dronefreak/mc3-18-hmdb51-ucf-transfer) |

**Input:** 8-frame clips @ 112×112 (Kinetics init), 16-frame clips @ 112×112 (UCF-101 init)  
**Use case:** Human action recognition in short video clips  
**Note:** Reference baselines showing transfer learning tradeoffs. Kinetics initialization performs slightly better but overfits more. UCF-101 initialization shows better generalization despite lower accuracy.

### 🖼️ Image Models (Pose-based - Stanford40)

| Model             | Accuracy  | Speed | Dataset                 | Download                                                                                                                                                    |
| ----------------- | --------- | ----- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ResNet50**      | **88.5%** | 6ms   | Stanford40 (40 classes) | [![HF](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/dronefreak/human-action-classification-stanford40/tree/main/resnet50)          |
| ResNet34          | 86.4%     | 5ms   | Stanford40              | [![HF](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/dronefreak/human-action-classification-stanford40/tree/main/resnet34)          |
| MobileNetV3-Large | 82.1%     | 3ms   | Stanford40              | [![HF](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/dronefreak/human-action-classification-stanford40/tree/main/mobilenetv3_large) |
| ResNet18          | 82.3%     | 4ms   | Stanford40              | [![HF](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/dronefreak/human-action-classification-stanford40/tree/main/resnet18)          |

**Input:** Single RGB image @ 224×224
**Use case:** Real-time single-frame action classification (fitness, sports, daily activities)

---

## 🚀 Quick Start

### Installation

```bash
# Core library
pip install -e .

# With demo interface
pip install -e ".[demo]"

# Full installation (training + demo)
pip install -e ".[dev,demo,train]"
```

### Video Action Recognition (NEW!)

```python
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor

# Load video model
model = torch.hub.load('dronefreak/mc3-18-ucf101', 'model', pretrained=True)
model.eval()

# Prepare 16-frame clip (C, T, H, W)
transform = Compose([
    Resize((128, 171)),
    CenterCrop(112),
    ToTensor(),
    Normalize(mean=[0.43216, 0.394666, 0.37645],
              std=[0.22803, 0.22145, 0.216989])
])

# Inference
with torch.no_grad():
    output = model(video_tensor)  # (1, 101)
    prediction = output.argmax(dim=1)

print(f"Action: {ucf101_classes[prediction]}")
```

<details>
<summary>📝 Video Classification CLI</summary>

```bash
# Classify video clip
python -m hac.video.inference.predict \
    --video clip.mp4 \
    --model dronefreak/mc3-18-ucf101 \
    --num_frames 16

# Real-time webcam
python -m hac.video.inference.predict \
    --webcam \
    --model dronefreak/mc3-18-ucf101
```

</details>

### Image Action Recognition

```python
from hac import ActionPredictor

# Initialize with pose estimation
predictor = ActionPredictor(
    model_path=None,  # Uses pretrained ResNet50
    device='cuda',
    use_pose_estimation=True
)

# Predict from image
result = predictor.predict_image('person.jpg')

print(f"Pose: {result['pose']['class']}")
print(f"Action: {result['action']['top_class']}")
print(f"Confidence: {result['action']['top_confidence']:.2%}")
```

<details>
<summary>🖥️ Web Demo & CLI</summary>

```bash
# Launch interactive web demo
hac-demo

# Command line inference
hac-infer --image photo.jpg --model weights/best.pth

# Real-time webcam
hac-infer --webcam --model weights/best.pth
```

</details>

---

## 🎓 Training Your Own Models

### Video Models (UCF-101)

```bash
# 1. Download UCF-101
wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
unrar x UCF101.rar

# 2. Download official splits
wget https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
unzip UCF101TrainTestSplits-RecognitionTask.zip

# 3. Organize dataset
python scripts/split_ucf101.py \
    --source UCF-101/ \
    --output UCF-101-organized/ \
    --splits ucfTrainTestlist/ \
    --split_num 1

# 4. Train MC3-18
python -m hac.video.training.train \
    --data_dir UCF-101-organized/ \
    --model mc3_18 \
    --pretrained \
    --batch_size 32 \
    --epochs 200 \
    --output_dir outputs/mc3-18
```

**Expected results:**

- MC3-18: 87% accuracy (200 epochs, 6-7 hours on RTX 4070 Super)
- R3D-18: 84% accuracy (100 epochs, 3-4 hours)

<details>
<summary>📝 Advanced Video Training</summary>

```bash
# With augmentations
python -m hac.video.training.train \
    --data_dir UCF-101-organized/ \
    --model mc3_18 \
    --pretrained \
    --batch_size 32 \
    --epochs 200 \
    --lr 0.001 \
    --weight_decay 1e-4 \
    --mixup_alpha 0.2 \
    --cutmix_alpha 0.5 \
    --label_smoothing 0.1

# Try different models
--model r3d_18      # 83.8% accuracy
--model r2plus1d_18 # Alternative architecture
--model mc3_18      # 87% accuracy (best)
```

</details>

### Image Models (Stanford40)

```bash
# 1. Download Stanford40 dataset
# From: http://vision.stanford.edu/Datasets/40actions.html

# 2. Train
python -m hac.image.training.train \
    --data_dir data/ \
    --model_name resnet50 \
    --num_classes 40 \
    --epochs 50 \
    --batch_size 32
```

---

## 🏗️ Architecture

### Video Pipeline (3D CNN)

```
Video Clip (16 frames)
    ↓
Frame Preprocessing (112×112)
    ↓
3D CNN (MC3-18 / R3D-18)
    ↓
Temporal Convolutions
    ↓
Action Classification (101 classes)
```

**Key features:**

- Spatiotemporal convolutions
- Temporal modeling across 16 frames
- Pretrained on Kinetics-400
- Fine-tuned on UCF-101

### Image Pipeline (Pose + 2D CNN)

```
Image
    ↓
MediaPipe Pose Detection → 33 keypoints
    ↓
Pose Classifier → sitting/standing/lying
    ↓
2D CNN (ResNet50) → Action features
    ↓
Action Classification (40 classes)
```

**Key features:**

- Dual-stream: pose + appearance
- Real-time 90 FPS inference
- Geometric pose reasoning
- Any timm model backbone

---

## 📈 Performance Benchmarks

### Video Models (NVIDIA RTX 4070 Super)

| Model  | Inference Time | FPS | Batch Size |
| ------ | -------------- | --- | ---------- |
| MC3-18 | 33ms           | 30  | 1          |
| R3D-18 | 25ms           | 40  | 1          |

### Image Models (NVIDIA RTX 4070 Super)

| Pipeline Stage | Time     | FPS    |
| -------------- | -------- | ------ |
| MediaPipe Pose | 5ms      | -      |
| ResNet50 CNN   | 6ms      | -      |
| **Total**      | **11ms** | **90** |

> **Comparison:** v1.0 (TF 1.13 + OpenPose) = 1400ms → v2.0 (PyTorch + MediaPipe) = 11ms (**127× faster**)

---

## 🎯 Use Cases

### Video Understanding

- **Sports analysis** - Classify basketball, soccer, swimming actions
- **Surveillance** - Detect abnormal behavior in videos
- **Fitness tracking** - Recognize workout exercises
- **Content moderation** - Auto-tag video content

### Real-time Image Classification

- **Fitness coaching** - Analyze workout form
- **Healthcare** - Fall detection, mobility monitoring
- **Autonomous vehicles** - Pedestrian intent prediction
- **Gaming/VR** - Body-based game controls

---

## 🔬 Datasets

### Supported Datasets

| Dataset        | Classes | Videos | Use Case         | Models Available       |
| -------------- | ------- | ------ | ---------------- | ---------------------- |
| **UCF-101**    | 101     | 13,320 | Video temporal   | MC3-18, R3D-18 ✅      |
| **Stanford40** | 40      | 9,532  | Image pose-based | ResNet50, MobileNet ✅ |
| Kinetics-400   | 400     | 306K   | Pretraining      | -                      |

### UCF-101 Classes

101 human actions including:

- **Sports:** Basketball, Soccer, Swimming, Tennis, Volleyball
- **Music:** Playing Drums, Guitar, Piano, Violin
- **Activities:** Cooking, Gardening, Typing, Writing
- **Body motion:** Walking, Running, Jumping, Lunging

[Full list →](https://www.crcv.ucf.edu/data/UCF101.php)

### Stanford40 Classes

40 common human activities:

- applauding, climbing, cooking, cutting_trees, drinking
- fishing, gardening, playing_guitar, pouring_liquid, etc.

[Full list →](http://vision.stanford.edu/Datasets/40actions.html)

---

## 📚 Documentation

- [Model Zoo](timm_model_families.md) - All available models
- [Deployement](DEPLOYMENT.md) - Deployement documentation
- [Contributing](CONTRIBUTING.md) - How to contribute

---

## 🗺️ Roadmap

### ✅ Completed

- [x] Image classification with pose estimation
- [x] Video classification with 3D CNNs
- [x] Published models on HuggingFace
- [x] Training pipelines for both modalities
- [x] CLI and Python API
- [x] Web demo with Gradio

### 🚧 In Progress

- [ ] Two-stream fusion (spatial + temporal)
- [ ] Real-time video demo
- [ ] HuggingFace Spaces deployment
- [ ] ONNX export for production

### 📋 Planned

- [ ] Mobile deployment guides
- [ ] TensorRT optimization
- [ ] Additional datasets (Kinetics, AVA)
- [ ] Multi-person action detection
- [ ] Action localization in videos

---

## 🤝 Contributing

We welcome contributions! High-impact areas:

- **🎥 Video demos** - Create GIFs/videos showing real-time inference
- **📱 Mobile deployment** - iOS/Android guides
- **🚀 Model improvements** - Train on Kinetics, optimize architectures
- **📖 Documentation** - Tutorials, examples, notebooks
- **🐛 Bug fixes** - Always appreciated!

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines.

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@software{saksena2025hac,
  author = {Saksena, Saumya Kumaar},
  title = {Human Action Classification: Image and Video Understanding},
  year = {2025},
  url = {https://github.com/dronefreak/human-action-classification}
}
```

### Model Citations

**Video Models (MC3-18, R3D-18):**

```bibtex
@inproceedings{tran2018closer,
  title={A closer look at spatiotemporal convolutions for action recognition},
  author={Tran, Du and Wang, Heng and Torresani, Lorenzo and Ray, Jamie and LeCun, Yann and Paluri, Manohar},
  booktitle={CVPR},
  year={2018}
}
```

**Datasets:**

- UCF-101: [Paper](https://arxiv.org/abs/1212.0402)
- Stanford40: [Website](http://vision.stanford.edu/Datasets/40actions.html)

---

## 🙏 Acknowledgments

- **MediaPipe** - Google's pose estimation framework
- **timm** - Ross Wightman's model library
- **PyTorch** - Deep learning framework
- **UCF-101 & Stanford40** - Dataset creators
- **Original repo contributors** - 233+ stars!

---

## 📧 Contact

**Author:** Saumya Kumaar Saksena
**GitHub:** [@dronefreak](https://github.com/dronefreak)
**Models:** [HuggingFace](https://huggingface.co/dronefreak)

---

## 📜 License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

---

<div align="center">

**⭐ Star this repo if it helped you!**

[![GitHub stars](https://img.shields.io/github/stars/dronefreak/human-action-classification?style=social)](https://github.com/dronefreak/human-action-classification)
[![GitHub forks](https://img.shields.io/github/forks/dronefreak/human-action-classification?style=social)](https://github.com/dronefreak/human-action-classification/fork)

</div>
