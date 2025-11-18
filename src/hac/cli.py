"""Command-line interface for human action classification."""

import argparse
import sys

from hac.image.inference.predictor import ActionPredictor


def infer():
    """CLI for inference."""
    parser = argparse.ArgumentParser(
        description="Run inference on images/videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Single image
            hac-infer --image path/to/image.jpg --model weights/best.pth

            # Video
            hac-infer --video path/to/video.mp4 --model weights/best.pth

            # Webcam
            hac-infer --webcam --model weights/best.pth
                """,
    )

    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of top predictions"
    )

    args = parser.parse_args()

    # Validate inputs
    if not (args.image or args.video or args.webcam):
        parser.error("Must specify one of: --image, --video, or --webcam")

    # Initialize predictor
    print(f"Loading model from: {args.model}")
    predictor = ActionPredictor(
        model_path=args.model, device=args.device, use_pose_estimation=True
    )

    # Run inference
    if args.image:
        print(f"\nProcessing image: {args.image}")
        result = predictor.predict_image(args.image, top_k=args.top_k)

        # Print results
        if "pose" in result:
            print(f"\nPose: {result['pose']['class']}")

        print(f"\nTop {args.top_k} Action Predictions:")
        for i, pred in enumerate(result["action"]["predictions"], 1):
            print(f"  {i}. {pred['class']}: {pred['confidence']:.3f}")

    elif args.video:
        print(f"\nProcessing video: {args.video}")
        result = predictor.predict_video(args.video)

        print(f"\nVideo: {result['video_path']}")
        print(f"Total frames: {result['total_frames']}")
        print(f"Sampled frames: {result['sampled_frames']}")
        print(f"\nPredicted Action: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")

    elif args.webcam:
        print("\nStarting webcam prediction...")
        print("Press 'q' to quit")
        predictor.predict_webcam()


def train():
    """CLI for training."""
    print("Training CLI")
    print("Please use: python -m hac.training.train --help")
    print("Or import and use the Trainer class directly")


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

    args = parser.parse_args()

    # Initialize predictor
    predictor = ActionPredictor(
        model_path=args.model, device="cuda", use_pose_estimation=True
    )

    def predict_interface(image):
        """Gradio interface function."""
        result = predictor.predict_image(image, return_pose=True, top_k=5)

        # Format output
        output_text = ""

        if "pose" in result:
            output_text += f"**Pose:** {result['pose']['class']}\n\n"

        output_text += "**Top 5 Actions:**\n"
        for i, pred in enumerate(result["action"]["predictions"], 1):
            output_text += f"{i}. {pred['class']}: {pred['confidence']:.3f}\n"

        # Return annotated image and text
        output_image = result.get("pose_image", image)

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
    print("Local URL: http://localhost:{args.port}")

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
