"""Command-line interface for human action classification."""

import argparse
import sys

import cv2
import torch

from hac.image.inference.predictor import ActionPredictor

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _add_image_model_args(parser):
    """Architecture hints for image weights that carry no config (.safetensors)."""
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="timm architecture if the weights have no embedded config "
        "(e.g. resnet50 for a .safetensors file)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of classes if the weights have no embedded config",
    )


def infer():
    """CLI for inference."""
    parser = argparse.ArgumentParser(
        description="Run inference on images/videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Single image (pose-aware image pipeline)
            hac-infer --image path/to/image.jpg --model weights/resnet50.pth

            # Video (3D-CNN video pipeline, e.g. MC3-18/R3D-18)
            hac-infer --video path/to/video.mp4 --model weights/mc3-18-ucf101.pth

            # Webcam (pose-aware image pipeline)
            hac-infer --webcam --model weights/resnet50.pth
                """,
    )

    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument(
        "--device", type=str, default=DEFAULT_DEVICE, help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of top predictions"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Frames sampled per clip (video only; use 8 for the "
        "Kinetics-400-initialized HMDB51 model, 16 for the UCF-101 models)",
    )
    _add_image_model_args(parser)

    args = parser.parse_args()

    # Validate inputs
    if not (args.image or args.video or args.webcam):
        parser.error("Must specify one of: --image, --video, or --webcam")

    # --video uses the dedicated 3D-CNN video pipeline (MC3-18/R3D-18), which
    # is a different model family from --image/--webcam's pose-aware 2D pipeline.
    if args.video:
        from hac.video.inference.predictor import VideoPredictor

        print(f"Loading video model from: {args.model}")
        predictor = VideoPredictor(
            model_path=args.model, num_frames=args.num_frames, device=args.device
        )

        print(f"\nProcessing video: {args.video}")
        result = predictor.predict_video(args.video, top_k=args.top_k)

        print(f"\nVideo: {result['video_path']}")
        print(f"Frames: {result['num_frames']} (sampled {result['sampled_frames']})")
        print(f"\nTop {args.top_k} Action Predictions:")
        for i, pred in enumerate(result["predictions"], 1):
            print(f"  {i}. {pred['class']}: {pred['confidence']:.3f}")
        return

    # --image / --webcam use the pose-aware image pipeline
    print(f"Loading model from: {args.model}")
    predictor = ActionPredictor(
        model_path=args.model,
        device=args.device,
        use_pose_estimation=True,
        model_name=args.model_name,
        num_classes=args.num_classes,
    )

    if args.image:
        print(f"\nProcessing image: {args.image}")
        result = predictor.predict_image(args.image, top_k=args.top_k)

        # Print results
        if "pose" in result:
            print(f"\nPose: {result['pose']['class']}")

        print(f"\nTop {args.top_k} Action Predictions:")
        for i, pred in enumerate(result["action"]["predictions"], 1):
            print(f"  {i}. {pred['class']}: {pred['confidence']:.3f}")

    elif args.webcam:
        print("\nStarting webcam prediction...")
        print("Press 'q' to quit")
        predictor.predict_webcam()


def train():
    """CLI for training."""
    print("Training is run through the module entry points:")
    print("  Image (e.g. Stanford40): python -m hac.image.training.train --help")
    print("  Video (e.g. UCF-101):    python -m hac.video.training.train --help")


def demo():
    """CLI for launching Gradio demo."""
    try:
        import gradio as gr
    except ImportError:
        print("Error: gradio not installed")
        print("Install with: pip install 'human-action-classification[demo]'")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Launch Gradio demo")
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument(
        "--device", type=str, default=DEFAULT_DEVICE, help="Device (cuda/cpu)"
    )
    _add_image_model_args(parser)

    args = parser.parse_args()

    # Initialize predictor
    predictor = ActionPredictor(
        model_path=args.model,
        device=args.device,
        use_pose_estimation=True,
        model_name=args.model_name,
        num_classes=args.num_classes,
    )

    def predict_interface(image):
        """Gradio interface function (Gradio supplies RGB; predictor wants BGR)."""
        if image is None:
            return None, "Please upload an image."
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result = predictor.predict_image(image_bgr, return_pose=True, top_k=5)

        # Format output
        output_text = ""

        if "pose" in result:
            output_text += f"**Pose:** {result['pose']['class']}\n\n"

        output_text += "**Top 5 Actions:**\n"
        for i, pred in enumerate(result["action"]["predictions"], 1):
            output_text += f"{i}. {pred['class']}: {pred['confidence']:.3f}\n"

        # Return annotated image (back to RGB for display) and text
        if "pose_image" in result:
            output_image = cv2.cvtColor(result["pose_image"], cv2.COLOR_BGR2RGB)
        else:
            output_image = image

        return output_image, output_text

    # Create Gradio interface
    interface = gr.Interface(
        fn=predict_interface,
        inputs=gr.Image(type="numpy", label="Upload Image"),
        outputs=[
            gr.Image(type="numpy", label="Pose Detection"),
            gr.Markdown(label="Predictions"),
        ],
        title="Human Action Classification",
        description="Upload an image to classify human actions and poses.",
        examples=[
            # Add example images if available
        ],
        theme=gr.themes.Soft(),
    )

    print(f"\nLaunching demo on port {args.port}")
    print(f"Local URL: http://localhost:{args.port}")

    interface.launch(server_port=args.port, share=args.share, server_name="0.0.0.0")


if __name__ == "__main__":
    # This allows running subcommands directly
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "infer":
            sys.argv.pop(1)
            infer()
        elif command == "train":
            sys.argv.pop(1)
            train()
        elif command == "demo":
            sys.argv.pop(1)
            demo()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: infer, train, demo")
    else:
        print("Usage: hac [infer|train|demo] [options]")
