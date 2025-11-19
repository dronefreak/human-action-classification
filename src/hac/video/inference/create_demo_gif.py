#!/usr/bin/env python3
"""Generate demo GIF showing multiple video predictions in a grid.

Creates a professional visualization with video clips and predictions.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from hac.video.inference.predictor import VideoPredictor


class DemoGIFCreator:
    """Create demo GIF with grid of video predictions."""

    def __init__(
        self,
        model_path,
        grid_size=(2, 3),  # (rows, cols) = 6 videos
        clip_size=(320, 240),  # Size of each video clip
        fps=10,
        device="cuda",
    ):
        """Initialize demo GIF creator.

        Args:
            model_path: Path to trained model
            grid_size: Grid layout (rows, cols)
            clip_size: Size of each video in grid (width, height)
            fps: Output FPS
            device: Device for inference
        """
        self.predictor = VideoPredictor(model_path, device=device)
        self.grid_rows, self.grid_cols = grid_size
        self.clip_width, self.clip_height = clip_size
        self.fps = fps

        # Calculate output size
        self.output_width = self.clip_width * self.grid_cols
        self.output_height = self.clip_height * self.grid_rows

        # Load font for text
        try:
            self.font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
            )
            self.font_small = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
            )
        except:
            self.font = ImageFont.load_default()
            self.font_small = ImageFont.load_default()

    def load_video_clip(self, video_path, max_frames=30):
        """Load video and sample frames."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        frame_count = 0
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize to clip size
            frame = cv2.resize(frame, (self.clip_width, self.clip_height))
            frames.append(frame)
            frame_count += 1

        cap.release()

        # Loop if too short
        if len(frames) < max_frames and len(frames) > 0:
            while len(frames) < max_frames:
                frames.extend(frames[: max_frames - len(frames)])

        return frames[:max_frames]

    def add_text_overlay(self, frame, class_name, confidence, position="bottom"):
        """Add text overlay to frame."""
        frame_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame_pil)

        # Text
        text_main = f"{class_name}"
        text_conf = f"{confidence:.1%}"

        if position == "bottom":
            # Draw semi-transparent background
            bg_height = 40
            overlay = Image.new("RGBA", frame_pil.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle(
                [
                    (0, self.clip_height - bg_height),
                    (self.clip_width, self.clip_height),
                ],
                fill=(0, 0, 0, 180),
            )
            frame_pil = Image.alpha_composite(
                frame_pil.convert("RGBA"), overlay
            ).convert("RGB")
            draw = ImageDraw.Draw(frame_pil)

            # Dynamically adjust font size if text too long
            current_font = self.font
            text_bbox = draw.textbbox((0, 0), text_main, font=current_font)
            text_width = text_bbox[2] - text_bbox[0]

            # If text too wide, try smaller font or truncate
            max_width = self.clip_width - 10  # Leave 5px margin on each side

            if text_width > max_width:
                # Try smaller font first
                try:
                    smaller_font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
                    )
                    text_bbox = draw.textbbox((0, 0), text_main, font=smaller_font)
                    text_width = text_bbox[2] - text_bbox[0]

                    if text_width <= max_width:
                        current_font = smaller_font
                    else:
                        # Still too long, truncate text
                        while text_width > max_width and len(text_main) > 3:
                            text_main = text_main[:-4] + "..."
                            text_bbox = draw.textbbox(
                                (0, 0), text_main, font=smaller_font
                            )
                            text_width = text_bbox[2] - text_bbox[0]
                        current_font = smaller_font
                except:
                    # Truncate if can't load smaller font
                    while text_width > max_width and len(text_main) > 3:
                        text_main = text_main[:-4] + "..."
                        text_bbox = draw.textbbox((0, 0), text_main, font=current_font)
                        text_width = text_bbox[2] - text_bbox[0]

            # Draw text centered
            text_x = max(
                5, (self.clip_width - text_width) // 2
            )  # Ensure at least 5px margin
            text_y = self.clip_height - bg_height + 5

            # Class name (white)
            draw.text(
                (text_x, text_y), text_main, fill=(255, 255, 255), font=current_font
            )

            # Confidence (green)
            conf_y = text_y + 18
            conf_bbox = draw.textbbox((0, 0), text_conf, font=self.font_small)
            conf_width = conf_bbox[2] - conf_bbox[0]
            conf_x = max(5, (self.clip_width - conf_width) // 2)
            draw.text(
                (conf_x, conf_y), text_conf, fill=(0, 255, 0), font=self.font_small
            )

        return np.array(frame_pil)

    def create_grid_frame(self, video_frames_list, predictions, frame_idx):
        """Create a single frame with grid of videos."""
        # Create empty canvas
        canvas = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)

        # Place each video in grid
        for video_idx, (video_frames, pred) in enumerate(
            zip(video_frames_list, predictions)
        ):
            row = video_idx // self.grid_cols
            col = video_idx % self.grid_cols

            # Get frame from this video at frame_idx
            frame = video_frames[frame_idx % len(video_frames)]

            # Add text overlay
            frame = self.add_text_overlay(
                frame, pred["top_class"], pred["top_confidence"]
            )

            # Place in grid
            y_start = row * self.clip_height
            y_end = (row + 1) * self.clip_height
            x_start = col * self.clip_width
            x_end = (col + 1) * self.clip_width

            canvas[y_start:y_end, x_start:x_end] = frame

        return canvas

    def create_demo_gif(
        self, video_paths, output_path="demo.gif", num_frames=60, add_header=True
    ):
        """Create demo GIF from multiple videos.

        Args:
            video_paths: List of video file paths
            output_path: Output GIF path
            num_frames: Number of frames in output GIF
            add_header: Add title header

        Returns:
            Path to created GIF
        """
        print(f"\n{'='*60}")
        print("Creating Demo GIF")
        print(f"{'='*60}")
        print(f"Videos: {len(video_paths)}")
        print(f"Grid: {self.grid_rows}×{self.grid_cols}")
        print(f"Output: {output_path}")
        print(f"{'='*60}\n")

        # Limit to grid size
        video_paths = video_paths[: self.grid_rows * self.grid_cols]

        # Load all videos
        print("Loading videos...")
        video_frames_list = []
        for video_path in tqdm(video_paths):
            frames = self.load_video_clip(video_path, max_frames=num_frames)
            video_frames_list.append(frames)

        # Get predictions
        print("\nGetting predictions...")
        predictions = []
        for video_path in tqdm(video_paths):
            pred = self.predictor.predict_video(video_path, top_k=1)
            predictions.append(pred)

        # Create frames
        print("\nGenerating frames...")
        output_frames = []

        for frame_idx in tqdm(range(num_frames)):
            # Create grid frame
            grid_frame = self.create_grid_frame(
                video_frames_list, predictions, frame_idx
            )

            # Add header if requested
            if add_header:
                grid_frame = self.add_header(grid_frame)

            output_frames.append(Image.fromarray(grid_frame))

        # Save as GIF
        print(f"\nSaving GIF to {output_path}...")
        output_frames[0].save(
            output_path,
            save_all=True,
            append_images=output_frames[1:],
            duration=int(1000 / self.fps),
            loop=0,
            optimize=True,
        )

        print(f"✓ Demo GIF created: {output_path}")
        print(f"  Size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")
        print(f"  Frames: {num_frames}")
        print(f"  FPS: {self.fps}")
        print(f"{'='*60}\n")

        return output_path

    def add_header(self, frame):
        """Add title header to frame."""
        header_height = 80
        new_frame = np.zeros(
            (frame.shape[0] + header_height, frame.shape[1], 3), dtype=np.uint8
        )

        # Dark background for header
        new_frame[:header_height] = [30, 30, 40]

        # Add frame below header
        new_frame[header_height:] = frame

        # Add text
        frame_pil = Image.fromarray(new_frame)
        draw = ImageDraw.Draw(frame_pil)

        # Title
        title = "UCF-101 Action Recognition - MC3-18"
        try:
            title_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28
            )
        except:
            title_font = self.font

        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (new_frame.shape[1] - title_width) // 2

        draw.text((title_x, 20), title, fill=(255, 255, 255), font=title_font)

        # Subtitle
        subtitle = "Real-time video action classification"
        subtitle_bbox = draw.textbbox((0, 0), subtitle, font=self.font_small)
        subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
        subtitle_x = (new_frame.shape[1] - subtitle_width) // 2

        draw.text(
            (subtitle_x, 52), subtitle, fill=(200, 200, 200), font=self.font_small
        )

        return np.array(frame_pil)


def collect_videos(video_dir: str | Path, exts=("mp4", "avi")):
    video_dir = Path(video_dir)
    if not video_dir.exists():
        raise ValueError(f"Directory does not exist: {video_dir}")

    paths = []
    for ext in exts:
        paths.extend(video_dir.rglob(f"*.{ext}"))

    if not paths:
        raise ValueError(f"No video files found in: {video_dir}")

    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Create demo GIF with video predictions"
    )

    # Model
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
    )

    # Videos
    parser.add_argument("--video_dir", type=str, help="Directory with sample videos")
    parser.add_argument("--video_paths", nargs="+", help="List of specific video paths")

    # Output
    parser.add_argument(
        "--output", type=str, default="demo.gif", help="Output GIF path"
    )

    # Grid settings
    parser.add_argument(
        "--grid_rows", type=int, default=2, help="Number of rows in grid"
    )
    parser.add_argument(
        "--grid_cols", type=int, default=3, help="Number of columns in grid"
    )

    # Video settings
    parser.add_argument(
        "--clip_width", type=int, default=320, help="Width of each video clip"
    )
    parser.add_argument(
        "--clip_height", type=int, default=240, help="Height of each video clip"
    )
    parser.add_argument("--fps", type=int, default=10, help="Output FPS")
    parser.add_argument(
        "--num_frames", type=int, default=60, help="Number of frames in output"
    )

    # Options
    parser.add_argument("--no_header", action="store_true", help="Disable title header")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Get video paths
    if args.video_paths:
        video_paths = [Path(p) for p in args.video_paths]
    elif args.video_dir:
        video_paths = collect_videos(args.video_dir)
    else:
        raise ValueError("Provide either --video_dir or --video_paths")

    print(f"Found {len(video_paths)} videos")

    # Create GIF creator
    creator = DemoGIFCreator(
        model_path=args.model_path,
        grid_size=(args.grid_rows, args.grid_cols),
        clip_size=(args.clip_width, args.clip_height),
        fps=args.fps,
        device=args.device,
    )

    # Create demo
    creator.create_demo_gif(
        video_paths=video_paths,
        output_path=args.output,
        num_frames=args.num_frames,
        add_header=not args.no_header,
    )


if __name__ == "__main__":
    main()
