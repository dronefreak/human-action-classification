#!/usr/bin/env python3
"""Generate a demo GIF: a grid of video clips, each labelled with its prediction.

If a clip's parent directory is a known class name (e.g. ``UCF-101/YoYo/x.avi``)
it is used as ground truth: tiles are marked correct (green) or wrong (red, with
the true label shown) and the header reports how many clips were classified
correctly, so the banner shows honest results rather than cherry-picked ones.

Example:
    python -m hac.video.inference.create_demo_gif \\
        --model_path mc318-ufc101-split-1.pth --video_dir sample_videos/ \\
        --grid_rows 4 --grid_cols 6 --output docs/demo.gif
"""

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from hac.video.inference.predictor import VideoPredictor

FONT_BOLD = "DejaVuSans-Bold.ttf"
FONT_REGULAR = "DejaVuSans.ttf"
FONT_DIRS = ("/usr/share/fonts/truetype/dejavu",)

GREEN = (88, 214, 141)
RED = (240, 98, 98)
WHITE = (255, 255, 255)
GREY = (190, 190, 200)
HEADER_BG = (24, 26, 34)


def _load_font(name: str, size: int):
    """Load a TrueType font by name, falling back to PIL's default font."""
    candidates = [name] + [str(Path(d) / name) for d in FONT_DIRS]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return right - left


def _fit_text(draw, text: str, max_width: int, font_name: str, sizes: Sequence[int]):
    """Shrink through ``sizes`` until ``text`` fits, then ellipsize at the smallest."""
    font = None
    for size in sizes:
        font = _load_font(font_name, size)
        if _text_width(draw, text, font) <= max_width:
            return text, font
    while len(text) > 1 and _text_width(draw, text + "…", font) > max_width:
        text = text[:-1]
    return text + "…", font


class DemoGIFCreator:
    """Create a demo GIF with a grid of video predictions."""

    def __init__(
        self,
        model_path,
        grid_size=(2, 3),
        clip_size=(320, 240),
        fps=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_frames_model=16,
    ):
        """Initialize demo GIF creator.

        Args:
            model_path: Path to a trained video checkpoint
            grid_size: Grid layout (rows, cols)
            clip_size: Size of each tile (width, height)
            fps: Output GIF frame rate; clips are sampled at this real-time rate
            device: Device for inference
            num_frames_model: Frames per clip fed to the model
        """
        self.predictor = VideoPredictor(
            model_path, num_frames=num_frames_model, device=device
        )
        self.grid_rows, self.grid_cols = grid_size
        self.clip_width, self.clip_height = clip_size
        self.fps = fps
        self.output_width = self.clip_width * self.grid_cols
        self.output_height = self.clip_height * self.grid_rows

    # ------------------------------------------------------------------ loading
    def load_video_clip(self, video_path, max_frames=30):
        """Read up to ``max_frames`` frames sampled at ``self.fps`` (real-time)."""
        cap = cv2.VideoCapture(str(video_path))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        stride = max(1, round(src_fps / self.fps))

        frames, index = [], 0
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if index % stride == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(cv2.resize(frame, (self.clip_width, self.clip_height)))
            index += 1
        cap.release()

        if not frames:
            raise ValueError(f"Could not read frames from {video_path}")
        while len(frames) < max_frames:  # loop short clips
            frames.extend(frames[: max_frames - len(frames)])
        return frames[:max_frames]

    def ground_truth(self, video_path) -> Optional[str]:
        """Parent directory name if it is one of the model's classes."""
        name = Path(video_path).parent.name
        return name if name in self.predictor.class_names else None

    # ---------------------------------------------------------------- rendering
    def make_label_overlay(self, pred: dict, truth: Optional[str]) -> Image.Image:
        """Static RGBA overlay with the prediction (and truth if wrong)."""
        w, h = self.clip_width, self.clip_height
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        correct = truth is None or pred["top_class"] == truth
        lines = 2 if correct else 3
        line_h = max(14, h // 9)
        band_h = line_h * lines + 8
        draw.rectangle([(0, h - band_h), (w, h)], fill=(0, 0, 0, 185))

        accent = GREEN if correct else RED
        if truth is not None:  # coloured edge marks correct / wrong at a glance
            draw.rectangle([(0, 0), (w - 1, h - 1)], outline=accent + (255,), width=3)

        max_w = w - 10
        sizes = [line_h, line_h - 2, line_h - 4]
        y = h - band_h + 4

        name, font = _fit_text(draw, pred["top_class"], max_w, FONT_BOLD, sizes)
        draw.text(((w - _text_width(draw, name, font)) // 2, y), name, WHITE, font)
        y += line_h

        mark = "" if truth is None else ("✓ " if correct else "✗ ")
        conf = f"{mark}{pred['top_confidence']:.0%}"
        font = _load_font(FONT_BOLD, line_h - 3)
        draw.text(((w - _text_width(draw, conf, font)) // 2, y), conf, accent, font)
        y += line_h

        if not correct:
            gt, font = _fit_text(
                draw, f"true: {truth}", max_w, FONT_REGULAR, [s - 2 for s in sizes]
            )
            draw.text(((w - _text_width(draw, gt, font)) // 2, y), gt, GREY, font)
        return overlay

    def make_header(self, title: str, subtitle: str) -> np.ndarray:
        """Static header strip rendered once."""
        height = max(56, self.output_width // 14)
        header = Image.new("RGB", (self.output_width, height), HEADER_BG)
        draw = ImageDraw.Draw(header)
        title_text, title_font = _fit_text(
            draw, title, self.output_width - 20, FONT_BOLD, [height // 2 - 2, 20, 16]
        )
        sub_text, sub_font = _fit_text(
            draw, subtitle, self.output_width - 20, FONT_REGULAR, [height // 4, 12]
        )
        title_y = height // 10
        draw.text(
            ((self.output_width - _text_width(draw, title_text, title_font)) // 2,
             title_y),
            title_text, WHITE, title_font,
        )  # fmt: skip
        draw.text(
            ((self.output_width - _text_width(draw, sub_text, sub_font)) // 2,
             title_y + height // 2),
            sub_text, GREY, sub_font,
        )  # fmt: skip
        return np.array(header)

    def create_grid_frame(self, clips: List[List[np.ndarray]], overlays, frame_idx):
        """Compose one output frame from the tiles at ``frame_idx``."""
        canvas = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
        for i, (frames, overlay) in enumerate(zip(clips, overlays)):
            tile = Image.fromarray(frames[frame_idx % len(frames)]).convert("RGBA")
            tile = np.array(Image.alpha_composite(tile, overlay).convert("RGB"))
            r, c = divmod(i, self.grid_cols)
            canvas[
                r * self.clip_height : (r + 1) * self.clip_height,
                c * self.clip_width : (c + 1) * self.clip_width,
            ] = tile
        return canvas

    # -------------------------------------------------------------------- main
    def create_demo_gif(
        self,
        video_paths,
        output_path="demo.gif",
        num_frames=30,
        add_header=True,
        title="Video Action Recognition",
        subtitle=None,
        max_colors=128,
    ):
        """Render the grid GIF and return its path."""
        video_paths = list(video_paths)[: self.grid_rows * self.grid_cols]
        print(f"Rendering {len(video_paths)} clips → {output_path}")

        predictions, truths, clips = [], [], []
        for path in tqdm(video_paths, desc="Predicting"):
            predictions.append(self.predictor.predict_video(path, top_k=1))
            truths.append(self.ground_truth(path))
            clips.append(self.load_video_clip(path, max_frames=num_frames))

        overlays = [self.make_label_overlay(p, t) for p, t in zip(predictions, truths)]

        labelled = [(p, t) for p, t in zip(predictions, truths) if t is not None]
        if subtitle is None:
            subtitle = "Top-1 prediction per clip"
            if labelled:
                n_ok = sum(p["top_class"] == t for p, t in labelled)
                subtitle += (
                    f" · {n_ok}/{len(labelled)} correct (✓ / ✗ vs. folder label)"
                )
        header = self.make_header(title, subtitle) if add_header else None

        frames = []
        for idx in tqdm(range(num_frames), desc="Composing"):
            frame = self.create_grid_frame(clips, overlays, idx)
            if header is not None:
                frame = np.vstack([header, frame])
            frames.append(Image.fromarray(frame))

        # One shared palette for all frames: smaller file and no colour flicker
        palette_src = Image.fromarray(
            np.vstack([np.asarray(f) for f in frames[:: max(1, num_frames // 6)]])
        ).quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT)
        frames = [
            f.quantize(palette=palette_src, dither=Image.Dither.NONE) for f in frames
        ]
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / self.fps),
            loop=0,
            optimize=True,
        )
        size_mb = Path(output_path).stat().st_size / 1024**2
        print(f"✓ Saved {output_path} ({size_mb:.1f} MB, {num_frames} frames)")
        return output_path


def collect_videos(video_dir: Union[str, Path], exts=("mp4", "avi")):
    """Recursively collect video files, sorted for reproducible grids."""
    video_dir = Path(video_dir)
    if not video_dir.exists():
        raise ValueError(f"Directory does not exist: {video_dir}")
    paths = sorted(p for ext in exts for p in video_dir.rglob(f"*.{ext}"))
    if not paths:
        raise ValueError(f"No video files found in: {video_dir}")
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Create a demo GIF with video predictions"
    )
    parser.add_argument("--model_path", required=True, help="Video model checkpoint")
    parser.add_argument("--video_dir", help="Directory with sample videos")
    parser.add_argument("--video_paths", nargs="+", help="Specific video paths")
    parser.add_argument("--output", default="demo.gif", help="Output GIF path")
    parser.add_argument("--grid_rows", type=int, default=2)
    parser.add_argument("--grid_cols", type=int, default=3)
    parser.add_argument("--clip_width", type=int, default=320)
    parser.add_argument("--clip_height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=10, help="Output FPS (real-time)")
    parser.add_argument("--num_frames", type=int, default=30, help="Frames in GIF")
    parser.add_argument(
        "--model_frames", type=int, default=16, help="Frames per clip for the model"
    )
    parser.add_argument("--max_colors", type=int, default=128, help="GIF palette")
    parser.add_argument("--title", default="Video Action Recognition")
    parser.add_argument("--subtitle", default=None, help="Override the subtitle")
    parser.add_argument("--no_header", action="store_true", help="Disable header")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    if args.video_paths:
        video_paths = [Path(p) for p in args.video_paths]
    elif args.video_dir:
        video_paths = collect_videos(args.video_dir)
    else:
        parser.error("Provide either --video_dir or --video_paths")

    creator = DemoGIFCreator(
        model_path=args.model_path,
        grid_size=(args.grid_rows, args.grid_cols),
        clip_size=(args.clip_width, args.clip_height),
        fps=args.fps,
        device=args.device,
        num_frames_model=args.model_frames,
    )
    creator.create_demo_gif(
        video_paths=video_paths,
        output_path=args.output,
        num_frames=args.num_frames,
        add_header=not args.no_header,
        title=args.title,
        subtitle=args.subtitle,
        max_colors=args.max_colors,
    )


if __name__ == "__main__":
    main()
