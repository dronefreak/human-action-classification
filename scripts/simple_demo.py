#!/usr/bin/env python3
"""Quick demo script to test the inference pipeline with visual output."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
import torch

from hac import ActionPredictor


def draw_keypoints_skeleton(
    image: np.ndarray, keypoints: np.ndarray, confidence_threshold: float = 0.3
):
    """Draw pose keypoints with colored skeleton on image."""

    # COCO skeleton connections
    skeleton = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),  # head
        (0, 5),
        (0, 6),
        (5, 7),
        (7, 9),  # left arm
        (6, 8),
        (8, 10),  # right arm
        (5, 6),
        (5, 11),
        (6, 12),  # torso
        (11, 12),
        (11, 13),
        (13, 15),  # left leg
        (12, 14),
        (14, 16),  # right leg
    ]

    # Colors (BGR)
    line_color = (0, 255, 0)  # green
    point_color = (0, 0, 255)  # red

    # Draw skeleton lines
    for start_idx, end_idx in skeleton:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]

            if (
                start_point[2] > confidence_threshold
                and end_point[2] > confidence_threshold
            ):
                start_pos = (int(start_point[0]), int(start_point[1]))
                end_pos = (int(end_point[0]), int(end_point[1]))
                cv2.line(image, start_pos, end_pos, line_color, 2, cv2.LINE_AA)

    # Draw keypoints
    for kpt in keypoints:
        if kpt[2] > confidence_threshold:
            x, y = int(kpt[0]), int(kpt[1])
            cv2.circle(image, (x, y), 4, point_color, -1, cv2.LINE_AA)
            cv2.circle(image, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)

    return image


def create_simple_overlay(image: np.ndarray, action_text: str, confidence: float):
    """Create simple text overlay that scales with image size."""
    h, w = image.shape[:2]

    # Scale text based on image size
    font_scale = w / 1000.0  # Adaptive font size
    thickness = max(1, int(font_scale * 2))

    # Format text
    display_text = f"{action_text}: {confidence*100:.1f}%"

    # Get text size
    text_size = cv2.getTextSize(
        display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )[0]

    # Position at top-left with padding
    padding = 10
    text_x = padding
    text_y = text_size[1] + padding

    # Semi-transparent background
    overlay = image.copy()
    box_coords = (
        (text_x - 5, text_y - text_size[1] - 5),
        (text_x + text_size[0] + 5, text_y + 5),
    )
    cv2.rectangle(overlay, box_coords[0], box_coords[1], (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # Draw text with shadow
    cv2.putText(
        image,
        display_text,
        (text_x + 1, text_y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        display_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    return image


def save_visualization(image_path: str, result: dict, output_path: str = None):
    """Create and save simple visualization."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = image.shape[:2]

    # Draw keypoints if available
    if "pose" in result and "keypoints" in result["pose"]:
        keypoints = result["pose"]["keypoints"]
        if keypoints is not None and len(keypoints) > 0:
            image = draw_keypoints_skeleton(image, keypoints)

    # Add simple text overlay
    main_prediction = result["action"]["predictions"][0]
    action_text = main_prediction["class"].replace("_", " ").title()
    confidence = main_prediction["confidence"]

    image = create_simple_overlay(image, action_text, confidence)

    # Determine output path
    if output_path is None:
        input_path = Path(image_path)
        output_path = input_path.parent / f"{input_path.stem}_result{input_path.suffix}"
    else:
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".jpg")

    # Save
    output_path = str(output_path)
    if not output_path.lower().endswith((".jpg", ".jpeg", ".png")):
        output_path += ".jpg"

    cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Quick demo of action classification with visual output"
    )
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")
    parser.add_argument("--model", type=str, help="Path to model weights")
    parser.add_argument("--output", type=str, help="Output path for result image")
    parser.add_argument(
        "--top_k", type=int, default=3, help="Number of top predictions to show"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--no_pose", action="store_true", help="Disable pose estimation"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Human Action Classification v2.0 - Demo")
    print("=" * 60)

    # Initialize predictor
    print(f"\nInitializing predictor on {args.device}...")

    predictor = ActionPredictor(
        model_path=args.model, device=args.device, use_pose_estimation=not args.no_pose
    )

    print("✓ Predictor ready!\n")

    if args.webcam:
        print("Starting webcam mode...")
        print("Press 'q' to quit\n")
        predictor.predict_webcam()

    elif args.image:
        print(f"Processing image: {args.image}\n")

        result = predictor.predict_image(args.image, return_pose=True, top_k=args.top_k)

        # Display results in terminal
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)

        if "pose" in result:
            print(f"\n📍 Pose: {result['pose']['class'].upper()}")

        print(f"\n🎬 Top {args.top_k} Action Predictions:")
        for i, pred in enumerate(result["action"]["predictions"][: args.top_k], 1):
            bar = "█" * int(pred["confidence"] * 50)
            print(f"  {i}. {pred['class']:30s} {pred['confidence']:.3f} {bar}")

        print("\n" + "=" * 60)

        # Create and save visualization
        print("\n📸 Creating visualization...")
        output_path = save_visualization(
            args.image,
            result,
            output_path=args.output,
        )

        print(f"\n🎨 Done! View the result: {output_path}")
        print("=" * 60)

    else:
        print("Error: Must specify --image or --webcam")
        print("\nExamples:")
        print("  python simple_demo.py --image test.jpg")
        print("  python simple_demo.py --image test.jpg --output result.jpg")
        print("  python simple_demo.py --image test.jpg --model weights/best.pth")
        print("  python simple_demo.py --webcam")


if __name__ == "__main__":
    main()
