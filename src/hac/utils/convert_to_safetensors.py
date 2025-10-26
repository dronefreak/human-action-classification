#!/usr/bin/env python3
"""Convert all model checkpoints to SafeTensors format."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert_checkpoint(input_path: str, output_path: str = None):
    """Convert PyTorch checkpoint to SafeTensors."""

    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.parent / "model.safetensors"
    else:
        output_path = Path(output_path)

    print(f"Converting {input_path} → {output_path}")

    # Load checkpoint
    checkpoint = torch.load(input_path, map_location="cpu")

    # Extract state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Save as SafeTensors
    save_file(state_dict, str(output_path))

    # Get file sizes
    input_size = input_path.stat().st_size / 1024**2  # MB
    output_size = output_path.stat().st_size / 1024**2  # MB

    print(f"✓ Converted: {input_size:.1f}MB → {output_size:.1f}MB")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch checkpoints to SafeTensors"
    )
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint files to convert")
    parser.add_argument("--output_dir", type=str, help="Output directory")

    args = parser.parse_args()

    for checkpoint in args.checkpoints:
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "model.safetensors"
        else:
            output_path = None

        convert_checkpoint(checkpoint, output_path)


if __name__ == "__main__":
    main()
