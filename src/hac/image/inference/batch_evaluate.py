#!/usr/bin/env python3
"""Batch evaluation script to evaluate multiple models at once.

Creates a comparison table of all models.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd


def run_evaluation(
    checkpoint: str, data_dir: str, split: str, output_dir: str, **kwargs
):
    """Run evaluation script for a single model."""
    cmd = [
        "python",
        "src/hac/image/inference/evaluate.py",
        "--checkpoint",
        checkpoint,
        "--data_dir",
        data_dir,
        "--split",
        split,
        "--output_dir",
        output_dir,
    ]

    # Add optional arguments
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])

    print(f"\nEvaluating: {checkpoint}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error evaluating {checkpoint}:")
        print(result.stderr)
        return None

    return output_dir


def load_metrics(output_dir: str) -> Dict:
    """Load metrics from evaluation output directory."""
    metrics_path = Path(output_dir) / "metrics.json"

    if not metrics_path.exists():
        return None

    with open(metrics_path, "r") as f:
        return json.load(f)


def create_comparison_table(results: List[Dict], output_path: str):
    """Create comparison table of all models."""

    rows = []
    for result in results:
        if result is None:
            continue

        report = result.get("classification_report", {})

        row = {
            "Model": result.get("model_name", "unknown"),
            "Accuracy (%)": f"{result.get('accuracy', 0):.2f}",
            "Macro Precision": f"{report.get('macro_avg', {}).get('precision', 0):.4f}",
            "Macro Recall": f"{report.get('macro_avg', {}).get('recall', 0):.4f}",
            "Macro F1": f"{report.get('macro_avg', {}).get('f1_score', 0):.4f}",
            "Weighted F1": f"{report.get('weighted_avg', {}).get('f1_score', 0):.4f}",
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by accuracy
    df["_acc_sort"] = df["Accuracy (%)"].astype(float)
    df = df.sort_values("_acc_sort", ascending=False).drop("_acc_sort", axis=1)

    # Save as CSV
    csv_path = f"{output_path}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Comparison table saved: {csv_path}")

    # Save as markdown
    md_path = f"{output_path}.md"
    with open(md_path, "w") as f:
        f.write("# Model Comparison\n\n")
        f.write(df.to_markdown(index=False))
    print(f"✓ Comparison table saved: {md_path}")

    # Print to console
    print("\n" + "=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100 + "\n")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluate multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all models in a directory
  python batch_evaluate.py \\
      --checkpoints_dir outputs/ \\
      --data_dir stanford40/ \\
      --split test

  # Evaluate specific models
  python batch_evaluate.py \\
      --checkpoints outputs/resnet34_best.pth \\
                    outputs/mobilenet_best.pth \\
                    outputs/efficientnet_best.pth \\
      --data_dir stanford40/ \\
      --split test
        """,
    )

    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--checkpoints", nargs="+", help="List of checkpoint paths to evaluate"
    )
    group.add_argument(
        "--checkpoints_dir",
        help="Directory containing checkpoints (evaluates all .pth files)",
    )

    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="batch_evaluation",
        help="Base output directory",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Get list of checkpoints
    if args.checkpoints_dir:
        checkpoints_dir = Path(args.checkpoints_dir)
        checkpoints = list(checkpoints_dir.glob("*.pth"))
        checkpoints = [
            str(cp) for cp in checkpoints if "best" in cp.name or "final" in cp.name
        ]
    else:
        checkpoints = args.checkpoints

    if not checkpoints:
        print("Error: No checkpoints found!")
        return

    print(f"Found {len(checkpoints)} checkpoints to evaluate")

    # Evaluate each model
    results = []
    for checkpoint in checkpoints:
        checkpoint_name = Path(checkpoint).stem
        model_output_dir = Path(args.output_dir) / checkpoint_name

        # Run evaluation
        output_dir = run_evaluation(
            checkpoint=checkpoint,
            data_dir=args.data_dir,
            split=args.split,
            output_dir=str(model_output_dir),
            batch_size=args.batch_size,
            device=args.device,
        )

        # Load results
        if output_dir:
            metrics = load_metrics(output_dir)
            if metrics:
                results.append(metrics)

    # Create comparison table
    if results:
        comparison_path = Path(args.output_dir) / "comparison"
        create_comparison_table(results, str(comparison_path))
    else:
        print("\nNo results to compare!")


if __name__ == "__main__":
    main()
