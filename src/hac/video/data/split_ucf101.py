#!/usr/bin/env python3
"""Organize UCF-101 dataset using official train/test splits.

Uses the predefined trainlist and testlist files from UCF-101.
"""

import argparse
import json
import shutil
from pathlib import Path

from tqdm import tqdm


def parse_split_file(split_file, has_labels=True):
    """Parse UCF-101 split file.

    Args:
        split_file: Path to train/test split file
        has_labels: Whether file contains class labels (trainlist) or not (testlist)

    Returns:
        List of (video_path, class_id) tuples if has_labels,
        otherwise list of video_paths
    """
    entries = []

    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if has_labels:
                # trainlist format: "ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi 1"
                parts = line.split()
                video_path = parts[0]
                class_id = int(parts[1])
                entries.append((video_path, class_id))
            else:
                # testlist format: "ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi"
                video_path = line
                entries.append(video_path)

    return entries


def get_class_mapping(source_dir):
    """Create class name to index mapping from directory structure.

    Args:
        source_dir: UCF-101 source directory

    Returns:
        Dictionary mapping class names to indices (0-based)
    """
    source_dir = Path(source_dir)
    classes = sorted([d.name for d in source_dir.iterdir() if d.is_dir()])
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    return class_to_idx, classes


def organize_split(
    source_dir, output_dir, split_file, split_name, has_labels=True, symlink=False
):
    """Organize videos according to split file.

    Args:
        source_dir: Source UCF-101 directory (videos or frames)
        output_dir: Output directory
        split_file: Path to split file (trainlist/testlist)
        split_name: Name of split (train/test)
        has_labels: Whether split file has labels
        symlink: Use symlinks instead of copying

    Returns:
        Statistics dictionary
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # Parse split file
    entries = parse_split_file(split_file, has_labels)

    print(f"\nOrganizing {split_name} split...")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Videos: {len(entries)}")

    # Statistics
    stats = {
        "split": split_name,
        "total": len(entries),
        "copied": 0,
        "missing": 0,
        "classes": {},
    }

    missing_files = []

    # Process each entry
    for entry in tqdm(entries, desc=f"Organizing {split_name}"):
        if has_labels:
            video_rel_path, class_id = entry
        else:
            video_rel_path = entry

        # Get class name from path
        class_name = video_rel_path.split("/")[0]
        video_name = Path(video_rel_path).name

        # Source path
        source_video = source_dir / class_name / video_name

        # Handle both .avi videos and extracted frames
        is_frames = False
        if not source_video.exists():
            # Try frames directory (stem without extension)
            source_video = source_dir / class_name / video_name.replace(".avi", "")
            if source_video.exists() and source_video.is_dir():
                is_frames = True

        if not source_video.exists():
            stats["missing"] += 1
            missing_files.append(str(video_rel_path))
            continue

        # Output path
        output_class_dir = output_dir / split_name / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        if is_frames:
            # Copy/link frame directory
            output_path = output_class_dir / source_video.name
        else:
            # Copy/link video file
            output_path = output_class_dir / video_name

        # Copy or symlink
        if output_path.exists():
            stats["copied"] += 1
        elif symlink:
            output_path.symlink_to(source_video.absolute())
            stats["copied"] += 1
        else:
            if is_frames:
                shutil.copytree(source_video, output_path)
            else:
                shutil.copy2(source_video, output_path)
            stats["copied"] += 1

        # Track class statistics
        if class_name not in stats["classes"]:
            stats["classes"][class_name] = 0
        stats["classes"][class_name] += 1

    # Print summary
    print(f"\n{split_name.upper()} Split Summary:")
    print(f"  Total entries: {stats['total']}")
    print(f"  Successfully organized: {stats['copied']}")
    print(f"  Missing files: {stats['missing']}")
    print(f"  Classes: {len(stats['classes'])}")

    if missing_files and len(missing_files) <= 10:
        print("\nMissing files:")
        for f in missing_files:
            print(f"  - {f}")
    elif missing_files:
        print("\nMissing files (first 10):")
        for f in missing_files[:10]:
            print(f"  - {f}")
        print(f"  ... and {len(missing_files) - 10} more")

    return stats


def create_split_structure(
    source_dir, output_dir, splits_dir, split_num=1, symlink=False
):
    """Create train/test split structure using official UCF-101 splits.

    Args:
        source_dir: Source UCF-101 directory (videos or frames)
        output_dir: Output directory
        splits_dir: Directory containing trainlist and testlist files
        split_num: Split number (1, 2, or 3)
        symlink: Use symlinks instead of copying

    Returns:
        Combined statistics
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    splits_dir = Path(splits_dir)

    # Check split files
    train_file = splits_dir / f"trainlist{split_num:02d}.txt"
    test_file = splits_dir / f"testlist{split_num:02d}.txt"

    if not train_file.exists():
        raise FileNotFoundError(f"Train split file not found: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test split file not found: {test_file}")

    print(f"\n{'='*60}")
    print("UCF-101 Dataset Split Organization")
    print(f"{'='*60}")
    print(f"Using split: {split_num}")
    print(f"Train split file: {train_file}")
    print(f"Test split file: {test_file}")
    print(f"Method: {'Symlinks' if symlink else 'Copy'}")
    print(f"{'='*60}")

    # Get class mapping
    class_to_idx, classes = get_class_mapping(source_dir)

    print(f"\nFound {len(classes)} classes in source directory")

    # Organize train split
    train_stats = organize_split(
        source_dir=source_dir,
        output_dir=output_dir,
        split_file=train_file,
        split_name="train",
        has_labels=True,
        symlink=symlink,
    )

    # Organize test split
    test_stats = organize_split(
        source_dir=source_dir,
        output_dir=output_dir,
        split_file=test_file,
        split_name="test",
        has_labels=False,
        symlink=symlink,
    )

    # Combined statistics
    combined_stats = {
        "split_num": split_num,
        "train": train_stats,
        "test": test_stats,
        "total_videos": train_stats["total"] + test_stats["total"],
        "total_organized": train_stats["copied"] + test_stats["copied"],
        "total_missing": train_stats["missing"] + test_stats["missing"],
        "classes": classes,
        "class_to_idx": class_to_idx,
    }

    # Save statistics
    stats_file = output_dir / f"split{split_num:02d}_stats.json"
    with open(stats_file, "w") as f:
        json.dump(combined_stats, f, indent=2)

    # Save class mapping
    classes_file = output_dir / "classes.txt"
    with open(classes_file, "w") as f:
        for class_name in classes:
            f.write(f"{class_name}\n")

    print(f"\n{'='*60}")
    print("Organization Complete!")
    print(f"{'='*60}")
    print(f"Total videos: {combined_stats['total_videos']}")
    print(f"Successfully organized: {combined_stats['total_organized']}")
    print(f"Missing: {combined_stats['total_missing']}")
    print(f"\nTrain: {train_stats['copied']} videos")
    print(f"Test: {test_stats['copied']} videos")
    print("\nOutput structure:")
    print(f"  {output_dir}/")
    print("    ├── train/")
    print("    │   ├── ApplyEyeMakeup/")
    print("    │   ├── Archery/")
    print("    │   └── ... (101 classes)")
    print("    └── test/")
    print("        ├── ApplyEyeMakeup/")
    print("        └── ...")
    print(f"\nStatistics saved to: {stats_file}")
    print(f"Class names saved to: {classes_file}")
    print(f"{'='*60}\n")

    return combined_stats


def verify_split(output_dir, split_name="train"):
    """Verify organized split.

    Args:
        output_dir: Output directory
        split_name: Split to verify (train/test)
    """
    output_dir = Path(output_dir)
    split_dir = output_dir / split_name

    if not split_dir.exists():
        print(f"Split directory not found: {split_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Verifying {split_name.upper()} split")
    print(f"{'='*60}")

    classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    total_items = 0

    for class_name in classes:
        class_dir = split_dir / class_name
        items = list(class_dir.iterdir())
        total_items += len(items)

    print(f"Classes: {len(classes)}")
    print(f"Total items: {total_items}")
    print(f"Average items per class: {total_items / len(classes):.1f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Organize UCF-101 using official train/test splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Organize using split 1 (copy files)
  python split_ucf101.py \\
      --source UCF-101/ \\
      --output UCF-101-organized/ \\
      --splits ucfTrainTestlist/ \\
      --split_num 1

  # Organize extracted frames using symlinks (faster, less space)
  python split_ucf101.py \\
      --source UCF-101-frames/ \\
      --output UCF-101-split/ \\
      --splits ucfTrainTestlist/ \\
      --split_num 1 \\
      --symlink

  # Verify organization
  python split_ucf101.py \\
      --verify UCF-101-organized/ \\
      --split_name train

Download splits from:
https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
        """,
    )

    # Input/output
    parser.add_argument(
        "--source", type=str, help="Source UCF-101 directory (videos or frames)"
    )
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument(
        "--splits", type=str, help="Directory with trainlist/testlist files"
    )

    # Split options
    parser.add_argument(
        "--split_num",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Split number to use (1, 2, or 3)",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks instead of copying (faster, saves space)",
    )

    # Verification
    parser.add_argument("--verify", type=str, help="Verify organized directory")
    parser.add_argument(
        "--split_name",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Split to verify",
    )

    args = parser.parse_args()

    # Verify mode
    if args.verify:
        verify_split(args.verify, args.split_name)
        return

    # Organize mode
    if not all([args.source, args.output, args.splits]):
        parser.error("--source, --output, and --splits are required")

    # Create split structure
    _ = create_split_structure(
        source_dir=args.source,
        output_dir=args.output,
        splits_dir=args.splits,
        split_num=args.split_num,
        symlink=args.symlink,
    )

    # Auto-verify
    print("\nVerifying train split...")
    verify_split(args.output, "train")

    print("\nVerifying test split...")
    verify_split(args.output, "test")


if __name__ == "__main__":
    main()
