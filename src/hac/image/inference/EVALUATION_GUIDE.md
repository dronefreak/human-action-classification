# Model Evaluation Guide

Complete guide for evaluating trained models and computing comprehensive metrics.

## Quick Start

### Evaluate Single Model

```bash
python evaluate.py \
    --checkpoint outputs/resnet34_best.pth \
    --data_dir stanford40/ \
    --split test \
    --output_dir evaluation_results/resnet34
```

This creates:

```
evaluation_results/resnet34/
├── metrics.json                      # All metrics in JSON
├── metrics.yaml                      # All metrics in YAML
├── metrics.txt                       # Human-readable report
├── confusion_matrix.npy              # NumPy array
├── confusion_matrix.png              # Visualization
├── confusion_matrix_normalized.png   # Normalized version
└── per_class_metrics.png            # Top/bottom classes
```

### Evaluate Multiple Models

```bash
python batch_evaluate.py \
    --checkpoints outputs/resnet34_best.pth \
                 outputs/mobilenet_best.pth \
                 outputs/efficientnet_best.pth \
    --data_dir stanford40/ \
    --split test
```

Creates comparison table:

```
batch_evaluation/
├── resnet34_best/
│   └── (all metrics)
├── mobilenet_best/
│   └── (all metrics)
├── efficientnet_best/
│   └── (all metrics)
├── comparison.csv       # CSV table
└── comparison.md        # Markdown table
```

## Metrics Computed

### 1. Overall Metrics

- **Accuracy**: Overall classification accuracy
- **Macro Average**: Unweighted mean per class
  - Precision
  - Recall
  - F1-Score
- **Weighted Average**: Weighted by support
  - Precision
  - Recall
  - F1-Score

### 2. Per-Class Metrics

For each of the 40 classes:

- Precision
- Recall
- F1-Score
- Support (number of samples)

### 3. Confusion Matrix

- Raw counts
- Normalized by row (recall)
- Visualizations

### 4. Error Analysis

- Top 10 most confident wrong predictions
- Shows which classes are confused

## Output Formats

### JSON (metrics.json)

```json
{
  "checkpoint_path": "outputs/resnet34_best.pth",
  "model_name": "resnet34",
  "accuracy": 86.0,
  "classification_report": {
    "overall_accuracy": 86.0,
    "macro_avg": {
      "precision": 0.8542,
      "recall": 0.8501,
      "f1_score": 0.8521
    },
    "weighted_avg": {
      "precision": 0.8598,
      "recall": 0.8600,
      "f1_score": 0.8599
    },
    "per_class": {
      "applauding": {
        "precision": 0.95,
        "recall": 0.92,
        "f1_score": 0.935,
        "support": 120
      },
      ...
    }
  },
  "worst_predictions": [
    {
      "sample_index": 1234,
      "true_label": "cooking",
      "predicted_label": "washing_dishes",
      "confidence": 0.95
    },
    ...
  ]
}
```

### YAML (metrics.yaml)

Same structure as JSON, but in YAML format for better readability.

### TXT (metrics.txt)

Human-readable report:

```
============================================================
MODEL EVALUATION RESULTS
============================================================

Evaluated at: 2025-10-26T12:00:00
Checkpoint: resnet34_best.pth
Model: resnet34
Dataset split: test
Number of samples: 5532

------------------------------------------------------------
OVERALL METRICS
------------------------------------------------------------
Accuracy: 86.00%

Macro Average:
  Precision: 0.8542
  Recall:    0.8501
  F1-Score:  0.8521

Weighted Average:
  Precision: 0.8598
  Recall:    0.8600
  F1-Score:  0.8599

------------------------------------------------------------
TOP 5 BEST PERFORMING CLASSES
------------------------------------------------------------
1. applauding
   Precision: 0.9500 | Recall: 0.9200 | F1: 0.9350
2. jumping
   Precision: 0.9400 | Recall: 0.9100 | F1: 0.9250
...

------------------------------------------------------------
BOTTOM 5 WORST PERFORMING CLASSES
------------------------------------------------------------
1. fixing_a_car
   Precision: 0.7200 | Recall: 0.6800 | F1: 0.7000
...

------------------------------------------------------------
TOP 10 MOST CONFIDENT WRONG PREDICTIONS
------------------------------------------------------------
1. True: cooking | Predicted: washing_dishes | Confidence: 0.9500
2. True: reading | Predicted: phoning | Confidence: 0.9200
...
```

## Command Line Options

### evaluate.py

| Argument        | Required | Default            | Description                |
| --------------- | -------- | ------------------ | -------------------------- |
| `--checkpoint`  | Yes      | -                  | Path to model checkpoint   |
| `--data_dir`    | Yes      | -                  | Path to dataset directory  |
| `--split`       | No       | test               | Dataset split (test/val)   |
| `--output_dir`  | No       | evaluation_results | Output directory           |
| `--batch_size`  | No       | 32                 | Batch size                 |
| `--num_workers` | No       | 4                  | DataLoader workers         |
| `--device`      | No       | cuda               | Device (cuda/cpu)          |
| `--model_name`  | No       | -                  | Override model name        |
| `--num_classes` | No       | -                  | Override number of classes |

### batch_evaluate.py

| Argument            | Required | Default          | Description                |
| ------------------- | -------- | ---------------- | -------------------------- |
| `--checkpoints`     | Yes\*    | -                | List of checkpoint paths   |
| `--checkpoints_dir` | Yes\*    | -                | Directory with checkpoints |
| `--data_dir`        | Yes      | -                | Path to dataset            |
| `--split`           | No       | test             | Dataset split              |
| `--output_dir`      | No       | batch_evaluation | Base output directory      |
| `--batch_size`      | No       | 32               | Batch size                 |
| `--device`          | No       | cuda             | Device                     |

\*One of `--checkpoints` or `--checkpoints_dir` required

## Use Cases

### 1. After Training - Evaluate Best Model

```bash
# Just finished training
python -m hac.training.train --data_dir stanford40/ --epochs 200

# Now evaluate
python evaluate.py \
    --checkpoint outputs/best.pth \
    --data_dir stanford40/ \
    --split test
```

### 2. Compare Multiple Training Runs

```bash
python batch_evaluate.py \
    --checkpoints outputs/exp1_best.pth \
                 outputs/exp2_best.pth \
                 outputs/exp3_best.pth \
    --data_dir stanford40/
```

### 3. Evaluate All Models in Directory

```bash
python batch_evaluate.py \
    --checkpoints_dir outputs/ \
    --data_dir stanford40/
```

### 4. Evaluate on Validation Set (Not Test)

```bash
python evaluate.py \
    --checkpoint outputs/resnet34_best.pth \
    --data_dir stanford40/ \
    --split val  # Use validation set
```

### 5. CPU-Only Evaluation

```bash
python evaluate.py \
    --checkpoint best.pth \
    --data_dir stanford40/ \
    --device cpu
```

## Interpreting Results

### Accuracy

- Overall percentage of correct predictions
- Good for balanced datasets
- **Your target**: 85%+

### Precision

- Of all predicted as class X, how many were actually X?
- Important when false positives are costly
- **Formula**: TP / (TP + FP)

### Recall

- Of all actual class X, how many did we find?
- Important when false negatives are costly
- **Formula**: TP / (TP + FN)

### F1-Score

- Harmonic mean of precision and recall
- Best single metric for classification
- **Formula**: 2 _ (precision _ recall) / (precision + recall)

### Macro vs Weighted Average

- **Macro**: Treats all classes equally (good for imbalanced data analysis)
- **Weighted**: Weights by class frequency (reflects real-world performance)

### Confusion Matrix

- Rows: True labels
- Columns: Predicted labels
- Diagonal: Correct predictions
- Off-diagonal: Confusions between classes

**Example**: High value at (cooking, washing_dishes) means model often confuses these.

## Using Metrics for HuggingFace Upload

### Extract Key Metrics

```bash
# Evaluate
python evaluate.py --checkpoint best.pth --data_dir stanford40/

# Extract metrics
cat evaluation_results/metrics.json | jq '.accuracy'
# Output: 86.0

cat evaluation_results/metrics.json | jq '.classification_report.macro_avg.f1_score'
# Output: 0.8521
```

### Add to Model Card (README.md)

```markdown
## Performance

Evaluated on Stanford 40 Actions test set (5,532 samples):

| Metric          | Score      |
| --------------- | ---------- |
| **Accuracy**    | **86.00%** |
| Macro F1        | 0.8521     |
| Weighted F1     | 0.8599     |
| Macro Precision | 0.8542     |
| Macro Recall    | 0.8501     |

### Per-Class Performance

Top 5 classes:

- Applauding: F1 = 0.935
- Jumping: F1 = 0.925
- Running: F1 = 0.918
- Waving hands: F1 = 0.912
- Drinking: F1 = 0.905

Bottom 5 classes:

- Fixing a car: F1 = 0.700
- Fixing a bike: F1 = 0.710
- Cutting trees: F1 = 0.725
  ...
```

## Troubleshooting

### Error: "No config found in checkpoint"

```bash
# Specify model manually
python evaluate.py \
    --checkpoint best.pth \
    --data_dir stanford40/ \
    --model_name resnet34 \
    --num_classes 40
```

### Error: "CUDA out of memory"

```bash
# Reduce batch size
python evaluate.py \
    --checkpoint best.pth \
    --data_dir stanford40/ \
    --batch_size 16
```

### Error: "Dataset split not found"

```bash
# Check your dataset structure
ls stanford40/
# Should have: train/ val/ test/

# Or use different split
--split val
```

### Plots not generating

```bash
# Install required packages
pip install matplotlib seaborn scikit-learn
```
