# Contributing

Thanks for considering a contribution! Bug fixes, tests, documentation, new architectures and deployment guides are all welcome.

## Development setup

```bash
git clone https://github.com/dronefreak/human-action-classification.git
cd human-action-classification
python -m venv .venv && source .venv/bin/activate   # Python 3.10-3.12
pip install -e ".[dev,demo]"
pre-commit install   # optional: runs black/isort/flake8 on commit
```

> MediaPipe is pinned below 0.10.30, which removed the `mp.solutions` pose API the image pipeline uses.

## Project layout

```
src/hac/
├── common/      # shared: class-name lists, metrics, transforms, safetensors conversion
├── image/       # single-frame pipeline (Stanford40): data, models (timm), training, inference
├── video/       # clip pipeline (UCF-101/HMDB51): data + UCF-101 prep, 3D-CNN models, training, inference
└── cli.py       # hac-infer / hac-train / hac-demo entry points
tests/           # CPU-only pytest suite on synthetic data (no downloads)
scripts/         # standalone demos
examples/        # notebooks
```

Training entry points: `python -m hac.image.training.train --help` and `python -m hac.video.training.train --help`.

## Before opening a PR

```bash
pytest tests/                    # all tests must pass (CPU, ~15 s)
black src/ tests/                # formatting, line length 88
isort src/ tests/                # import order (black profile)
flake8 src/ tests/               # lint
```

CI runs the same checks on Python 3.10, 3.11 and 3.12 for every PR to `master`/`dev`.

Please also:

- add or update tests for behaviour you change (keep them CPU-only and download-free);
- keep checkpoints self-describing: trainers store `config` and `class_names`, and predictors rely on them;
- never reorder the lists in `hac/common/class_names.py`; index order must match the sorted class directories the released models were trained on;
- update the README if user-facing behaviour changes.

## Pull requests

1. Branch from `master`: `git checkout -b fix/short-description`.
2. Keep PRs focused; open an issue first for large changes or new dependencies.
3. Use [Conventional Commits](https://www.conventionalcommits.org/) messages, e.g. `fix: correct UCF-101 class order`, `feat: add X3D backbone`, `docs: clarify HMDB51 setup`, `test: cover video predictor`.
4. In the PR description, say what changed, why, and how it was tested (include metrics for model changes).

## Style

- Black formatting (88 columns), isort with the black profile.
- Type hints and Google-style docstrings on public functions.
- Comments explain _why_, not _what_.

## Where help is most useful

- Additional backbones or datasets for the video pipeline (e.g. X3D, Kinetics).
- Export and deployment (ONNX, TensorRT, quantization) with benchmarks.
- Reproducing and reporting results on other splits (UCF-101 splits 2/3, HMDB51).
- Documentation and example notebooks.

## Questions and bugs

Open a [GitHub issue](https://github.com/dronefreak/human-action-classification/issues). For security problems, follow [SECURITY.md](SECURITY.md) instead of filing a public issue.

By contributing, you agree that your contributions are licensed under the [Apache License 2.0](../LICENSE).
