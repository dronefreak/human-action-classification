#!/usr/bin/env python3
"""Video preprocessing utility for UCF-101 dataset.

Extracts frames from videos for efficient training.
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import cv2
from tqdm import tqdm


def video_to_frames(video_path, output_dir, target_fps=None, quality=95):
    """Extract frames from a single video.

    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        target_fps: Target FPS for frame extraction (None = all frames)
        quality: JPEG quality (1-100)

    Returns:
        Number of frames extracted
    """
    try:
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"  ⚠️  Failed to open: {video_path.name}")
            return 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        _ = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame interval
        if target_fps and target_fps < fps:
            frame_interval = int(fps / target_fps)
        else:
            frame_interval = 1

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame at interval
            if frame_count % frame_interval == 0:
                frame_path = output_dir / f"frame_{saved_count:06d}.jpg"
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                saved_count += 1

            frame_count += 1

        cap.release()
        return saved_count

    except Exception as e:
        print(f"  ❌ Error processing {video_path.name}: {e}")
        return 0


def process_video_wrapper(args):
    """Wrapper for multiprocessing."""
    video_path, output_dir, target_fps, quality = args
    num_frames = video_to_frames(video_path, output_dir, target_fps, quality)
    return video_path.name, num_frames


def process_ucf101(
    ucf_dir, output_dir, target_fps=10, quality=95, num_workers=4, video_ext=".avi"
):
    """Process entire UCF-101 dataset to frames.

    Args:
        ucf_dir: UCF-101 root directory
        output_dir: Output directory for frames
        target_fps: Target FPS for extraction
        quality: JPEG quality (1-100)
        num_workers: Number of parallel workers
        video_ext: Video file extension

    Returns:
        Dictionary with processing statistics
    """
    ucf_dir = Path(ucf_dir)
    output_dir = Path(output_dir)

    if not ucf_dir.exists():
        raise ValueError(f"Dataset directory not found: {ucf_dir}")

    # Collect all videos
    all_videos = []
    for class_dir in sorted(ucf_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        videos = list(class_dir.glob(f"*{video_ext}"))
        all_videos.extend(
            [
                (
                    video_path,
                    output_dir / class_dir.name / video_path.stem,
                    target_fps,
                    quality,
                )
                for video_path in videos
            ]
        )

    print(f"\n{'='*60}")
    print("UCF-101 Video to Frames Extraction")
    print(f"{'='*60}")
    print(f"Input directory: {ucf_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total videos: {len(all_videos)}")
    print(f"Target FPS: {target_fps}")
    print(f"JPEG quality: {quality}")
    print(f"Workers: {num_workers}")
    print(f"{'='*60}\n")

    # Process videos in parallel
    stats = {
        "total_videos": len(all_videos),
        "processed": 0,
        "failed": 0,
        "total_frames": 0,
        "classes": {},
    }

    if num_workers > 1:
        # Parallel processing
        with mp.Pool(num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(process_video_wrapper, all_videos),
                    total=len(all_videos),
                    desc="Processing videos",
                )
            )
    else:
        # Sequential processing
        results = []
        for args in tqdm(all_videos, desc="Processing videos"):
            results.append(process_video_wrapper(args))

    # Collect statistics
    for video_name, num_frames in results:
        if num_frames > 0:
            stats["processed"] += 1
            stats["total_frames"] += num_frames

            # Extract class name
            class_name = video_name.split("_")[1]  # e.g., v_ApplyEyeMakeup_g01_c01.avi
            if class_name not in stats["classes"]:
                stats["classes"][class_name] = {"videos": 0, "frames": 0}
            stats["classes"][class_name]["videos"] += 1
            stats["classes"][class_name]["frames"] += num_frames
        else:
            stats["failed"] += 1

    # Print summary
    print(f"\n{'='*60}")
    print("Processing Complete!")
    print("{'='*60}")
    print(f"Total videos: {stats['total_videos']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total frames extracted: {stats['total_frames']:,}")
    print(
        f"Average frames per video: {stats['total_frames'] / max(stats['processed'], 1):.1f}"  # noqa: E501
    )
    print(f"Number of classes: {len(stats['classes'])}")
    print(f"{'='*60}\n")

    # Save statistics
    stats_file = output_dir / "extraction_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to: {stats_file}")

    return stats


def verify_extraction(frames_dir, min_frames=8):
    """Verify extracted frames and identify problematic videos.

    Args:
        frames_dir: Directory with extracted frames
        min_frames: Minimum expected frames per video

    Returns:
        Dictionary with verification results
    """
    frames_dir = Path(frames_dir)

    print(f"\n{'='*60}")
    print("Verifying Extracted Frames")
    print(f"{'='*60}")

    issues = {"missing_frames": [], "empty_dirs": []}

    total_videos = 0
    total_frames = 0

    for class_dir in sorted(frames_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        for video_dir in class_dir.iterdir():
            if not video_dir.is_dir():
                continue

            total_videos += 1
            frames = list(video_dir.glob("frame_*.jpg"))
            num_frames = len(frames)
            total_frames += num_frames

            if num_frames == 0:
                issues["empty_dirs"].append(str(video_dir))
            elif num_frames < min_frames:
                issues["missing_frames"].append(
                    {"path": str(video_dir), "frames": num_frames}
                )

    print(f"Total videos: {total_videos}")
    print(f"Total frames: {total_frames:,}")
    print(f"Average frames per video: {total_frames / max(total_videos, 1):.1f}")
    print("\nIssues found:")
    print(f"  Empty directories: {len(issues['empty_dirs'])}")
    print(f"  Videos with < {min_frames} frames: {len(issues['missing_frames'])}")

    if issues["empty_dirs"]:
        print("\nEmpty directories (first 5):")
        for path in issues["empty_dirs"][:5]:
            print(f"  - {path}")

    if issues["missing_frames"]:
        print("\nVideos with few frames (first 5):")
        for item in issues["missing_frames"][:5]:
            print(f"  - {item['path']}: {item['frames']} frames")

    print(f"{'='*60}\n")

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from UCF-101 videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract at 10 FPS with 4 workers
  python preprocess.py --input UCF-101/ --output UCF-101-frames/ --fps 10 --workers 4

  # Extract all frames with high quality
  python preprocess.py --input UCF-101/ --output UCF-101-frames/ --quality 100

  # Verify extraction
  python preprocess.py --verify UCF-101-frames/
        """,
    )

    # Input/output
    parser.add_argument("--input", type=str, help="Input UCF-101 directory")
    parser.add_argument("--output", type=str, help="Output directory for frames")

    # Processing options
    parser.add_argument(
        "--fps", type=int, default=10, help="Target FPS for extraction (default: 10)"
    )
    parser.add_argument(
        "--quality", type=int, default=95, help="JPEG quality 1-100 (default: 95)"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--ext", type=str, default=".avi", help="Video file extension (default: .avi)"
    )

    # Verification
    parser.add_argument(
        "--verify", type=str, help="Verify extracted frames in directory"
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=8,
        help="Minimum frames for verification (default: 8)",
    )

    args = parser.parse_args()

    # Verify mode
    if args.verify:
        verify_extraction(args.verify, args.min_frames)
        return

    # Extract mode
    if not args.input or not args.output:
        parser.error("--input and --output are required for extraction")

    # Process dataset
    _ = process_ucf101(
        ucf_dir=args.input,
        output_dir=args.output,
        target_fps=args.fps,
        quality=args.quality,
        num_workers=args.workers,
        video_ext=args.ext,
    )

    # Auto-verify after extraction
    print("\nRunning verification...")
    verify_extraction(args.output, min_frames=args.fps)


if __name__ == "__main__":
    main()
