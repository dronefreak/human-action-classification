#!/usr/bin/env python3
"""Video action predictor for inference."""


import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from hac.video.models.classifier import Video3DCNN


class VideoPredictor:
    """Predictor for video action recognition."""

    def __init__(self, model_path, num_frames=16, spatial_size=112, device="cuda"):
        """Initialize video predictor.

        Args:
            model_path: Path to model checkpoint
            num_frames: Number of frames to sample
            spatial_size: Spatial size for frames
            device: Device (cuda/cpu)
        """
        self.num_frames = num_frames
        self.spatial_size = spatial_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()

        # Transform
        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 171)),
                transforms.CenterCrop(spatial_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
                ),
            ]
        )

        # Class names (UCF-101)
        self.class_names = self._get_ucf101_classes()

    def _load_model(self, model_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        # Get config
        config = checkpoint.get("config", {})
        num_classes = config.get("num_classes", 101)
        model_name = config.get("model_name", "r3d_18")

        # Create model
        model = Video3DCNN(
            num_classes=num_classes, model_name=model_name, pretrained=False
        )

        # Load weights
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)

        model = model.to(self.device)
        model.eval()

        print(f"✓ Loaded model: {model_name} ({num_classes} classes)")

        return model

    def _get_ucf101_classes(self):
        """Get UCF-101 class names."""
        classes = [
            "ApplyEyeMakeup",
            "ApplyLipstick",
            "Archery",
            "BabyCrawling",
            "BalanceBeam",
            "BandMarching",
            "BaseballPitch",
            "Basketball",
            "BasketballDunk",
            "BenchPress",
            "Biking",
            "Billiards",
            "BlowDryHair",
            "BlowingCandles",
            "BodyWeightSquats",
            "Bowling",
            "BoxingPunchingBag",
            "BoxingSpeedBag",
            "BreastStroke",
            "BrushingTeeth",
            "CleanAndJerk",
            "CliffDiving",
            "CricketBowling",
            "CricketShot",
            "CuttingInKitchen",
            "Diving",
            "Drumming",
            "Fencing",
            "FieldHockeyPenalty",
            "FloorGymnastics",
            "FrisbeeCatch",
            "FrontCrawl",
            "GolfSwing",
            "Haircut",
            "Hammering",
            "HammerThrow",
            "HandstandPushups",
            "HandstandWalking",
            "HeadMassage",
            "HighJump",
            "HorseRace",
            "HorseRiding",
            "HulaHoop",
            "IceDancing",
            "JavelinThrow",
            "JugglingBalls",
            "JumpingJack",
            "JumpRope",
            "Kayaking",
            "Knitting",
            "LongJump",
            "Lunges",
            "MilitaryParade",
            "Mixing",
            "MoppingFloor",
            "Nunchucks",
            "ParallelBars",
            "PizzaTossing",
            "PlayingCello",
            "PlayingDaf",
            "PlayingDhol",
            "PlayingFlute",
            "PlayingGuitar",
            "PlayingPiano",
            "PlayingSitar",
            "PlayingTabla",
            "PlayingViolin",
            "PoleVault",
            "PommelHorse",
            "PullUps",
            "Punch",
            "PushUps",
            "Rafting",
            "RockClimbingIndoor",
            "RopeClimbing",
            "Rowing",
            "SalsaSpin",
            "ShavingBeard",
            "Shotput",
            "SkateBoarding",
            "Skiing",
            "Skijet",
            "SkyDiving",
            "SoccerJuggling",
            "SoccerPenalty",
            "StillRings",
            "SumoWrestling",
            "Surfing",
            "Swing",
            "TableTennisShot",
            "TaiChi",
            "TennisSwing",
            "ThrowDiscus",
            "TrampolineJumping",
            "Typing",
            "UnevenBars",
            "VolleyballSpiking",
            "WalkingWithDog",
            "WallPushups",
            "WritingOnBoard",
            "YoYo",
        ]
        return classes

    def _load_video(self, video_path):
        """Load video and extract frames."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames extracted from {video_path}")

        return frames

    def _sample_frames(self, frames):
        """Sample num_frames uniformly from video."""
        total_frames = len(frames)

        if total_frames < self.num_frames:
            # Repeat frames if video too short
            indices = np.arange(total_frames)
            indices = np.tile(indices, int(np.ceil(self.num_frames / total_frames)))
            indices = indices[: self.num_frames]
        else:
            # Uniform sampling
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)

        return [frames[i] for i in indices]

    def predict_video(self, video_path, top_k=5):
        """Predict action from video.

        Args:
            video_path: Path to video file
            top_k: Number of top predictions to return

        Returns:
            Dictionary with predictions
        """
        # Load and sample frames
        frames = self._load_video(video_path)
        frames = self._sample_frames(frames)

        # Transform frames
        transformed_frames = []
        for frame in frames:
            frame_pil = Image.fromarray(frame)
            frame_tensor = self.transform(frame_pil)
            transformed_frames.append(frame_tensor)

        # Stack: (T, C, H, W) → (C, T, H, W)
        video_tensor = torch.stack(transformed_frames).permute(1, 0, 2, 3)
        video_tensor = video_tensor.unsqueeze(0)  # Add batch dim: (1, C, T, H, W)
        video_tensor = video_tensor.to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(video_tensor)
            probs = torch.softmax(outputs, dim=1)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs[0], top_k)

        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append(
                {
                    "class": self.class_names[idx],
                    "class_idx": int(idx),
                    "confidence": float(prob),
                }
            )

        return {
            "video_path": str(video_path),
            "num_frames": len(frames),
            "sampled_frames": self.num_frames,
            "top_class": predictions[0]["class"],
            "top_confidence": predictions[0]["confidence"],
            "predictions": predictions,
        }

    def predict_batch(self, video_paths, batch_size=8):
        """Predict actions for multiple videos.

        Args:
            video_paths: List of video paths
            batch_size: Batch size for processing

        Returns:
            List of prediction dictionaries
        """
        results = []

        for i in range(0, len(video_paths), batch_size):
            batch_paths = video_paths[i : i + batch_size]

            for video_path in batch_paths:
                result = self.predict_video(video_path)
                results.append(result)

        return results


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Video action prediction")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Model checkpoint"
    )
    parser.add_argument("--video_path", type=str, required=True, help="Video file")
    parser.add_argument("--top_k", type=int, default=5, help="Top K predictions")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    # Create predictor
    predictor = VideoPredictor(model_path=args.model_path, device=args.device)

    # Predict
    result = predictor.predict_video(args.video_path, top_k=args.top_k)

    # Print results
    print(f"\n{'='*60}")
    print(f"Video: {result['video_path']}")
    print(f"Frames: {result['num_frames']} (sampled {result['sampled_frames']})")
    print(f"\nTop prediction: {result['top_class']} ({result['top_confidence']:.1%})")
    print(f"\nTop {args.top_k} predictions:")
    for i, pred in enumerate(result["predictions"], 1):
        print(f"  {i}. {pred['class']}: {pred['confidence']:.1%}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
