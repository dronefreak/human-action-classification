# Contributing to Human Action Classification v2.0

First off - thanks for considering contributing! This repo went from 233 stars on TF 1.13 code to a modern PyTorch rewrite, and your contributions can keep it relevant.

## Quick Links

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guide](#style-guide)

## Code of Conduct

**Be cool. Don't be a jerk.** That's it.

## How Can I Contribute?

### 1. Pre-trained Weights (HIGH PRIORITY)

The repo currently has no pre-trained weights on Stanford 40. This is the #1 contribution opportunity:

```bash
# Train on Stanford 40
python -m hac.training.train \
    --data_dir stanford40/ \
    --model_name mobilenetv3_small_100 \
    --epochs 50

# Upload to HuggingFace Hub
# Submit PR with download link
```

### 2. Dataset Integration

Add loaders for other action recognition datasets:

- NTU RGB+D
- Kinetics-400/700
- UCF-101
- HMDB-51
- AVA (for AV applications)

### 3. Temporal Models

Current implementation is single-frame. Add:

- SlowFast networks
- X3D models
- Video transformers (VideoMAE, TimeSformer)
- LSTM/GRU on pose sequences

### 4. Edge Optimization

- ONNX export scripts
- TensorRT optimization
- Quantization (INT8)
- Mobile deployment (CoreML, TFLite)
- Benchmark on Jetson/Pi

### 5. Documentation

- More example notebooks
- Video tutorials
- API documentation (Sphinx)
- Performance benchmarks

### 6. Testing

We're at like 10% test coverage. Help us get to 80%:

- Unit tests
- Integration tests
- Benchmark tests
- Edge case handling

## Development Setup

```bash
# Clone repo
git clone https://github.com/dronefreak/human-action-classification.git
cd human-action-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in dev mode
pip install -e ".[dev,demo,train]"

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## Project Structure

```
src/hac/
├── models/          # Add new architectures here
├── inference/       # Inference pipeline
├── data/           # Dataset loaders
├── training/       # Training utilities
└── utils/          # Helper functions

tests/              # Add tests here
examples/           # Add notebooks here
scripts/            # Utility scripts
configs/            # Hydra configs (coming soon)
```

## Pull Request Process

1. **Fork the repo** and create your branch from `main`

   ```bash
   git checkout -b feature/awesome-feature
   ```

2. **Make your changes**
   - Write clean, readable code
   - Add docstrings
   - Update README if needed

3. **Test your changes**

   ```bash
   # Run tests
   pytest tests/ -v

   # Check formatting
   black src/
   isort src/

   # Type checking
   mypy src/ --ignore-missing-imports
   ```

4. **Commit with clear messages**

   ```bash
   git commit -m "Add SlowFast model for temporal action recognition"
   ```

5. **Push and create PR**

   ```bash
   git push origin feature/awesome-feature
   ```

6. **Fill out PR template** with:
   - What you changed
   - Why you changed it
   - How to test it
   - Screenshots/benchmarks if applicable

## Style Guide

### Python Style

We mostly follow [Black](https://black.readthedocs.io/) with:

- Line length: 100
- Type hints where appropriate
- Docstrings for public functions

```python
def predict_action(
    image: np.ndarray,
    model: nn.Module,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Predict action from image.

    Args:
        image: Input image in BGR format
        model: PyTorch model
        device: Device to run inference on

    Returns:
        Dictionary with predictions
    """
    # Implementation
    pass
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
feat: add SlowFast model support
fix: correct pose classification threshold
docs: update installation instructions
test: add unit tests for pose extractor
perf: optimize MediaPipe inference
refactor: simplify training loop
```

### Documentation

- **Docstrings**: Google style
- **README**: Keep it concise, move details to docs/
- **Comments**: Explain WHY, not WHAT

## What We're Looking For

### High Priority

- Pre-trained weights on Stanford 40
- Temporal models (SlowFast, X3D)
- ONNX/TensorRT export
- More comprehensive tests

### Medium Priority

- Additional dataset loaders
- Hydra config integration
- Training on more datasets
- Pedestrian prediction for AVs

### Nice to Have

- Web UI improvements
- More example notebooks
- Performance optimizations
- Edge deployment examples

## What We're NOT Looking For

- Breaking changes without discussion
- Dependencies on deprecated packages
- Code without tests
- PRs that don't follow style guide

## Getting Help

- **Questions**: Open a Discussion
- **Bugs**: Open an Issue
- **Features**: Open an Issue to discuss first

## Recognition

Contributors get:

- Listed in README
- Shoutout in releases
- Good karma
- Street cred in the CV community

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thanks for making this repo better! 🚀

**Remember**: This repo has 233+ stars because people found it useful. Let's keep that momentum going with modern, maintainable code.
