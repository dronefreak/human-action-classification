#!/usr/bin/env python3
"""Quick demo script to test the inference pipeline."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from hac import ActionPredictor


def main():
    parser = argparse.ArgumentParser(description="Quick demo of action classification")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")
    parser.add_argument("--model", type=str, help="Path to model weights")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Human Action Classification v2.0 - Quick Demo")
    print("=" * 60)

    # Initialize predictor
    print(f"\nInitializing predictor on {args.device}...")

    predictor = ActionPredictor(
        model_path=args.model, device=args.device, use_pose_estimation=True
    )

    print("✓ Predictor ready!\n")

    if args.webcam:
        print("Starting webcam mode...")
        print("Press 'q' to quit\n")
        predictor.predict_webcam()

    elif args.image:
        print(f"Processing image: {args.image}\n")

        result = predictor.predict_image(args.image, return_pose=True, top_k=5)

        # Display results
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)

        if "pose" in result:
            print(f"\n📍 Pose: {result['pose']['class'].upper()}")

        print("\n🎬 Top 5 Action Predictions:")
        for i, pred in enumerate(result["action"]["predictions"], 1):
            bar = "█" * int(pred["confidence"] * 50)
            print(f"  {i}. {pred['class']:30s} {pred['confidence']:.3f} {bar}")

        print("\n" + "=" * 60)

    else:
        print("Error: Must specify --image or --webcam")
        print("\nExamples:")
        print("  python simple_demo.py --image test.jpg")
        print("  python simple_demo.py --webcam")
        print("  python simple_demo.py --image test.jpg --model weights/best.pth")


if __name__ == "__main__":
    main()
