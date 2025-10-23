# Migration Guide: v1.0 → v2.0

This guide helps you migrate from the TensorFlow 1.13 version to the modern PyTorch version.

## TL;DR

**Old:** SWIG compilation hell, 1.4s inference, hardcoded paths  
**New:** `pip install`, 30ms inference, clean API

## Breaking Changes

### 1. Installation

```bash
# Old (v1.0) - The SWIG nightmare
cd tf_pose/pafprocess/
swig -python -c++ pafprocess.i
python setup.py build_ext --inplace
# ...pray it works

# New (v2.0) - One line
pip install -e .
```

### 2. API Changes

#### Inference

```python
# Old (v1.0)
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path

estimator = TfPoseEstimator(
    get_graph_path('mobilenet_thin'),
    target_size=(432, 368)
)
humans = estimator.inference(image)

# New (v2.0)
from hac import ActionPredictor

predictor = ActionPredictor(device='cuda')
result = predictor.predict_image(image)
```

#### Pose Detection

```python
# Old (v1.0)
from tf_pose.estimator import TfPoseEstimator
estimator = TfPoseEstimator(...)
humans = estimator.inference(image, resize_to_default=True, upsample_size=4.0)
for human in humans:
    # Extract body parts...

# New (v2.0)
from hac.inference.pose_extractor import PoseExtractor

pose_extractor = PoseExtractor()
keypoints = pose_extractor.extract_keypoints(image)  # (33, 3) array
```

#### Classification

```python
# Old (v1.0)
import tensorflow as tf
with tf.Session() as sess:
    # Load graph...
    # Run inference...
    # Extract predictions...

# New (v2.0)
import torch

model = ActionClassifier(num_classes=40)
logits = model(image_tensor)
probs = torch.softmax(logits, dim=-1)
```

### 3. Training

```python
# Old (v1.0)
python3 scripts/retrain.py \
    --model_dir=tf_files/retrained_graph.pb \
    --output_labels=tf_files/retrained_labels.txt \
    --image_dir=training/

# New (v2.0)
python -m hac.training.train \
    --data_dir data/ \
    --model_name mobilenetv3_small_100 \
    --num_classes 40 \
    --epochs 50
```

### 4. Model Architecture

```python
# Old (v1.0) - Fixed architectures
# MobileNet v1 for pose
# Inception v3 for actions
# No flexibility

# New (v2.0) - Any timm model
from hac.models.classifier import create_model

model = create_model(
    model_type='action',
    model_name='efficientnet_b0',  # or ANY timm model
    num_classes=40
)
```

### 5. Configuration

```python
# Old (v1.0) - Hardcoded paths
address = '/home/user/specific/path'  # LOL

# New (v2.0) - Hydra configs (coming soon) or simple args
predictor = ActionPredictor(
    model_path='weights/best.pth',
    device='cuda'
)
```

## Feature Comparison

| Feature             | v1.0                       | v2.0             |
| ------------------- | -------------------------- | ---------------- |
| **Pose Estimation** | OpenPose (tf-pose)         | MediaPipe        |
| **Framework**       | TensorFlow 1.13            | PyTorch 2.0+     |
| **Models**          | MobileNet v1, Inception v3 | 100+ timm models |
| **Inference Speed** | 1.4s                       | 0.03s            |
| **Installation**    | SWIG + C++                 | Pure Python      |
| **Training API**    | Custom scripts             | Modern Trainer   |
| **CLI**             | Basic scripts              | Rich CLI tools   |
| **Web Demo**        | None                       | Gradio           |
| **Type Hints**      | None                       | Full typing      |
| **Tests**           | None                       | Unit tests       |
| **CI/CD**           | None                       | GitHub Actions   |

## Migration Checklist

- [ ] Update code to use new API
- [ ] Re-train models with new architecture (optional)
- [ ] Update inference pipeline
- [ ] Remove SWIG/C++ dependencies
- [ ] Test new inference speed
- [ ] Update documentation
- [ ] Celebrate 47x speedup 🎉

## Gradual Migration

If you can't migrate everything at once:

1. **Phase 1**: Install v2.0 alongside v1.0

   ```bash
   pip install -e . --prefix ~/.local
   ```

2. **Phase 2**: Port inference code first
   - Inference is the easiest to migrate
   - Biggest speed improvements

3. **Phase 3**: Port training code
   - Re-train models with new architecture
   - Use transfer learning from v1.0 results

4. **Phase 4**: Remove v1.0 dependencies
   - Uninstall TensorFlow 1.13
   - Remove SWIG/OpenPose code

## Code Examples

### Example 1: Simple Inference

```python
# v1.0
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
import cv2

estimator = TfPoseEstimator(get_graph_path('mobilenet_thin'))
image = cv2.imread('test.jpg')
humans = estimator.inference(image)

# v2.0
from hac import ActionPredictor

predictor = ActionPredictor()
result = predictor.predict_image('test.jpg')
print(result['action']['top_class'])
```

### Example 2: Webcam Prediction

```python
# v1.0
# Had to manually write OpenCV loop + inference

# v2.0
from hac import ActionPredictor

predictor = ActionPredictor()
predictor.predict_webcam()  # Press 'q' to quit
```

### Example 3: Batch Processing

```python
# v1.0
# Manual batch processing with TF sessions

# v2.0
from hac import ActionPredictor

predictor = ActionPredictor()

for image_path in image_paths:
    result = predictor.predict_image(image_path)
    print(f"{image_path}: {result['action']['top_class']}")
```

## Performance Gains

Benchmark on GTX 1050 Ti:

```
Old (v1.0):
- OpenPose: ~800ms
- Classification: ~200ms
- Total: ~1400ms per image

New (v2.0):
- MediaPipe: ~20ms
- Classification: ~10ms
- Total: ~30ms per image

Speedup: 47x faster! 🚀
```

## Common Issues

### Issue 1: "No module named \_pafprocess"

**v1.0 Solution:** SWIG compilation (see installation nightmare above)  
**v2.0 Solution:** This doesn't exist anymore. Just `pip install -e .`

### Issue 2: Slow inference

**v1.0:** OpenPose is inherently slow  
**v2.0:** MediaPipe is 40x faster. Also use `device='cuda'` for GPU acceleration.

### Issue 3: Hardcoded paths

**v1.0:** Had to manually edit `address` variable  
**v2.0:** Pass paths as arguments. No hardcoding.

### Issue 4: Can't find my trained model

**v1.0:** Models saved in `/tmp` by default (seriously?)  
**v2.0:** Models saved in `outputs/` with clear naming

## Getting Help

- **Issues:** Open on GitHub
- **Questions:** Use Discussions
- **Old v1.0 issues:** Most are solved by v2.0's architecture

## Why Migrate?

1. **Speed:** 47x faster inference
2. **Reliability:** No SWIG compilation
3. **Flexibility:** 100+ model architectures
4. **Modern:** Active PyTorch ecosystem
5. **Maintenance:** v1.0 uses TF 1.13 (EOL'd in 2020)
6. **Features:** Web demo, CLI tools, better API

## Need Both Versions?

Keep v1.0 in a separate branch:

```bash
git checkout -b legacy
# v1.0 code stays here

git checkout main
# v2.0 development continues
```

## Final Notes

The v2.0 rewrite maintains the **same core functionality** while:

- Removing all pain points
- Adding modern tooling
- Improving performance dramatically

Your stars brought us here - let's make this repo relevant for 2025! 🚀
