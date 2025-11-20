#!/usr/bin/env python3
"""Video preprocessing utility for HMDB-51 dataset.

Extracts frames from videos for efficient training. Handles the organized HMDB-51
structure (train/test splits).
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

        # Handle videos with invalid/zero FPS
        if fps <= 0 or fps > 120:
            print(f"  ⚠️  Invalid FPS ({fps}) for {video_path.name}, using default 30")
            fps = 30.0

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

        # Warn if very few frames
        if saved_count < 8:
            print(f"  ⚠️  Only {saved_count} frames extracted from {video_path.name}")

        return saved_count

    except Exception as e:
        print(f"  ❌ Error processing {video_path.name}: {e}")
        return 0


def process_video_wrapper(args):
    """Wrapper for multiprocessing."""
    video_path, output_dir, target_fps, quality = args
    num_frames = video_to_frames(video_path, output_dir, target_fps, quality)
    return video_path.name, num_frames, str(video_path.parent.name)


def process_hmdb51(
    hmdb_dir, output_dir, target_fps=10, quality=95, num_workers=4, video_ext=".avi"
):
    """Process HMDB-51 dataset to frames.

    Handles HMDB-51 structure:
    hmdb_dir/
        train/
            class1/
                video1.avi
                video2.avi
            class2/
                ...
        test/
            class1/
                video3.avi
            ...

    Args:
        hmdb_dir: HMDB-51 root directory (with train/test splits)
        output_dir: Output directory for frames
        target_fps: Target FPS for extraction
        quality: JPEG quality (1-100)
        num_workers: Number of parallel workers
        video_ext: Video file extension

    Returns:
        Dictionary with processing statistics
    """
    hmdb_dir = Path(hmdb_dir)
    output_dir = Path(output_dir)

    if not hmdb_dir.exists():
        raise ValueError(f"Dataset directory not found: {hmdb_dir}")

    # Check for train/test splits
    train_dir = hmdb_dir / "train"
    test_dir = hmdb_dir / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise ValueError(
            f"Expected train/ and test/ subdirectories in {hmdb_dir}\n"
            "Make sure you've organized HMDB-51 into splits first."
        )

    # Collect all videos
    all_videos = []

    for split in ["train", "test"]:
        split_dir = hmdb_dir / split

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            videos = list(class_dir.glob(f"*{video_ext}"))
            all_videos.extend(
                [
                    (
                        video_path,
                        output_dir / split / class_dir.name / video_path.stem,
                        target_fps,
                        quality,
                    )
                    for video_path in videos
                ]
            )

    print(f"\n{'='*60}")
    print("HMDB-51 Video to Frames Extraction")
    print(f"{'='*60}")
    print(f"Input directory: {hmdb_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total videos: {len(all_videos)}")
    print(f"  Train: {len(list((hmdb_dir / 'train').rglob(f'*{video_ext}')))}")
    print(f"  Test: {len(list((hmdb_dir / 'test').rglob(f'*{video_ext}')))}")
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
        "train": {"videos": 0, "frames": 0, "classes": {}},
        "test": {"videos": 0, "frames": 0, "classes": {}},
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
    for video_name, num_frames, class_name in results:
        if num_frames > 0:
            stats["processed"] += 1
            stats["total_frames"] += num_frames

            # Determine split from video path
            split = "train" if "/train/" in str(all_videos[0][0]) else "test"
            # Actually need to track this better - use class_name lookup
            # For now, approximate based on order

            # Update class stats
            if class_name not in stats["train"]["classes"]:
                stats["train"]["classes"][class_name] = {"videos": 0, "frames": 0}
            if class_name not in stats["test"]["classes"]:
                stats["test"]["classes"][class_name] = {"videos": 0, "frames": 0}

        else:
            stats["failed"] += 1

    # Better split tracking
    train_frames_dir = output_dir / "train"
    test_frames_dir = output_dir / "test"

    if train_frames_dir.exists():
        for class_dir in train_frames_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                videos = list(class_dir.iterdir())
                frames = sum(len(list(v.glob("*.jpg"))) for v in videos if v.is_dir())
                stats["train"]["videos"] += len(videos)
                stats["train"]["frames"] += frames
                stats["train"]["classes"][class_name] = {
                    "videos": len(videos),
                    "frames": frames,
                }

    if test_frames_dir.exists():
        for class_dir in test_frames_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                videos = list(class_dir.iterdir())
                frames = sum(len(list(v.glob("*.jpg"))) for v in videos if v.is_dir())
                stats["test"]["videos"] += len(videos)
                stats["test"]["frames"] += frames
                stats["test"]["classes"][class_name] = {
                    "videos": len(videos),
                    "frames": frames,
                }

    # Print summary
    print(f"\n{'='*60}")
    print("Processing Complete!")
    print(f"{'='*60}")
    print(f"Total videos: {stats['total_videos']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total frames extracted: {stats['total_frames']:,}")
    print(
        f"Average frames per video: {stats['total_frames'] / max(stats['processed'], 1):.1f}"
    )
    print("\nTrain Split:")
    print(f"  Videos: {stats['train']['videos']}")
    print(f"  Frames: {stats['train']['frames']:,}")
    print(f"  Classes: {len(stats['train']['classes'])}")
    print("\nTest Split:")
    print(f"  Videos: {stats['test']['videos']}")
    print(f"  Frames: {stats['test']['frames']:,}")
    print(f"  Classes: {len(stats['test']['classes'])}")
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

    issues = {
        "missing_frames": [],
        "empty_dirs": [],
        "train": {"total_videos": 0, "total_frames": 0},
        "test": {"total_videos": 0, "total_frames": 0},
    }

    for split in ["train", "test"]:
        split_dir = frames_dir / split

        if not split_dir.exists():
            print(f"⚠️  Split directory not found: {split_dir}")
            continue

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            for video_dir in class_dir.iterdir():
                if not video_dir.is_dir():
                    continue

                issues[split]["total_videos"] += 1
                frames = list(video_dir.glob("frame_*.jpg"))
                num_frames = len(frames)
                issues[split]["total_frames"] += num_frames

                if num_frames == 0:
                    issues["empty_dirs"].append(str(video_dir))
                elif num_frames < min_frames:
                    issues["missing_frames"].append(
                        {"path": str(video_dir), "frames": num_frames}
                    )

    print("Train Split:")
    print(f"  Videos: {issues['train']['total_videos']}")
    print(f"  Frames: {issues['train']['total_frames']:,}")
    if issues["train"]["total_videos"] > 0:
        print(
            f"  Avg frames/video: {issues['train']['total_frames'] / issues['train']['total_videos']:.1f}"
        )

    print("\nTest Split:")
    print(f"  Videos: {issues['test']['total_videos']}")
    print(f"  Frames: {issues['test']['total_frames']:,}")
    if issues["test"]["total_videos"] > 0:
        print(
            f"  Avg frames/video: {issues['test']['total_frames'] / issues['test']['total_videos']:.1f}"
        )

    print("\nIssues found:")
    print(f"  Empty directories: {len(issues['empty_dirs'])}")
    print(f"  Videos with < {min_frames} frames: {len(issues['missing_frames'])}")

    if issues["empty_dirs"]:
        print(f"\nEmpty directories (first 10):")
        for path in issues["empty_dirs"][:10]:
            print(f"  - {path}")

    if issues["missing_frames"]:
        print("\nVideos with few frames (first 10):")
        for item in issues["missing_frames"][:10]:
            print(f"  - {item['path']}: {item['frames']} frames")

    print(f"{'='*60}\n")

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from HMDB-51 videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract at 10 FPS with 4 workers
  python preprocess_hmdb51.py --input HMDB51_organised/ --output HMDB51_frames/ --fps 10 --workers 4

  # Extract all frames with high quality
  python preprocess_hmdb51.py --input HMDB51_organised/ --output HMDB51_frames/ --quality 100

  # Verify extraction
  python preprocess_hmdb51.py --verify HMDB51_frames/
  
Note: Input directory should have train/ and test/ subdirectories with class folders.
        """,
    )

    # Input/output
    parser.add_argument(
        "--input", type=str, help="Input HMDB-51 directory (with train/test)"
    )
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
    stats = process_hmdb51(
        hmdb_dir=args.input,
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
