import argparse
import shutil
from pathlib import Path

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}


def is_video_file(path: Path):
    return path.suffix.lower() in VIDEO_EXTENSIONS


def organize_dataset(root_dir, split_dir, output_dir, split):
    """Organize HMDB-51 dataset according to official splits.

    Split indicators:
        0 = unused (skip)
        1 = training set
        2 = test set
    """
    root_dir = Path(root_dir)
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)
    assert split in [1, 2, 3], "Split must be 1, 2, or 3"

    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    classes = [d for d in root_dir.iterdir() if d.is_dir()]

    # Statistics
    stats = {"total": 0, "train": 0, "test": 0, "unused": 0, "missing": 0}

    print(f"\n{'='*60}")
    print(f"Organizing HMDB-51 Split {split}")
    print(f"{'='*60}\n")

    for cls_dir in classes:
        cls_name = cls_dir.name

        cls_train_dir = train_dir / cls_name
        cls_test_dir = test_dir / cls_name
        cls_train_dir.mkdir(exist_ok=True)
        cls_test_dir.mkdir(exist_ok=True)

        split_file = split_dir / f"{cls_name}_test_split{split}.txt"
        if not split_file.exists():
            print(f"⚠️  Missing split file: {split_file}")
            continue

        with split_file.open() as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue

                filename, indicator = parts
                indicator = int(indicator)
                stats["total"] += 1

                # Skip unused videos (indicator = 0)
                if indicator == 0:
                    stats["unused"] += 1
                    continue

                # Find source file
                src_path = cls_dir / filename

                # Try any extension fallback
                if not src_path.exists():
                    # Attempt to resolve missing extension (AVI/MP4 mismatch)
                    alt = None
                    for ext in VIDEO_EXTENSIONS:
                        candidate = cls_dir / (
                            filename.replace(".avi", ext).replace(".mp4", ext)
                        )
                        if candidate.exists():
                            alt = candidate
                            break
                    if alt:
                        src_path = alt
                    else:
                        print(f"⚠️  File not found: {cls_name}/{filename}")
                        stats["missing"] += 1
                        continue

                # Copy to appropriate directory
                # indicator 1 = train, indicator 2 = test
                if indicator == 1:
                    dst_dir = cls_train_dir
                    stats["train"] += 1
                elif indicator == 2:
                    dst_dir = cls_test_dir
                    stats["test"] += 1
                else:
                    print(f"⚠️  Invalid indicator {indicator} for {filename}")
                    continue

                shutil.copy2(src_path, dst_dir / src_path.name)

    print(f"\n{'='*60}")
    print("Organization Complete")
    print(f"{'='*60}")
    print(f"Total videos in split files: {stats['total']}")
    print(f"  ├─ Train (indicator=1): {stats['train']}")
    print(f"  ├─ Test (indicator=2): {stats['test']}")
    print(f"  ├─ Unused (indicator=0): {stats['unused']} (skipped)")
    print(f"  └─ Missing files: {stats['missing']}")
    print(
        f"\nUsage rate: {(stats['train'] + stats['test']) / stats['total'] * 100:.1f}%"
    )
    print(f"Unused rate: {stats['unused'] / stats['total'] * 100:.1f}%")
    print(f"{'='*60}\n")

    verify_split(root_dir, split_dir, output_dir, split)


def verify_split(root_dir, split_dir, output_dir, split):
    """Verify that organized dataset matches split files."""
    print("\n" + "=" * 60)
    print("Verifying Split Consistency")
    print("=" * 60 + "\n")

    root_dir = Path(root_dir)
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)

    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    success = True
    total_errors = 0

    classes = [d for d in root_dir.iterdir() if d.is_dir()]

    for cls_dir in classes:
        cls_name = cls_dir.name

        split_file = split_dir / f"{cls_name}_test_split{split}.txt"
        if not split_file.exists():
            continue

        train_expected = set()
        test_expected = set()

        # Parse split file (skip indicator=0)
        with split_file.open() as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue

                filename, indicator = parts
                indicator = int(indicator)

                # Skip unused videos
                if indicator == 0:
                    continue

                # Add to expected set
                if indicator == 1:
                    train_expected.add(filename)
                elif indicator == 2:
                    test_expected.add(filename)

        # Get actual files
        cls_train_dir = train_dir / cls_name
        cls_test_dir = test_dir / cls_name

        train_actual = {
            f.name for f in cls_train_dir.glob("*") if f.is_file() and is_video_file(f)
        }

        test_actual = {
            f.name for f in cls_test_dir.glob("*") if f.is_file() and is_video_file(f)
        }

        # Compare
        train_missing = train_expected - train_actual
        train_extra = train_actual - train_expected
        test_missing = test_expected - test_actual
        test_extra = test_actual - test_expected

        if train_missing or train_extra:
            success = False
            total_errors += len(train_missing) + len(train_extra)
            print(f"❌ Train mismatch in class: {cls_name}")
            if train_missing:
                print(
                    f"   Missing ({len(train_missing)}): {list(train_missing)[:3]}..."
                )
            if train_extra:
                print(f"   Extra ({len(train_extra)}): {list(train_extra)[:3]}...")

        if test_missing or test_extra:
            success = False
            total_errors += len(test_missing) + len(test_extra)
            print(f"❌ Test mismatch in class: {cls_name}")
            if test_missing:
                print(f"   Missing ({len(test_missing)}): {list(test_missing)[:3]}...")
            if test_extra:
                print(f"   Extra ({len(test_extra)}): {list(test_extra)[:3]}...")

    print("\n" + "=" * 60)
    if success:
        print("✅ Verification PASSED: All splits match expected lists!")
    else:
        print(f"❌ Verification FAILED: {total_errors} mismatches found")
        print("   Check file extensions and missing videos")
    print("=" * 60 + "\n")


def print_split_statistics(root_dir, split_dir):
    """Print statistics about the split files."""
    root_dir = Path(root_dir)
    split_dir = Path(split_dir)

    print("\n" + "=" * 60)
    print("Split Statistics")
    print("=" * 60 + "\n")

    for split_num in [1, 2, 3]:
        stats = {"train": 0, "test": 0, "unused": 0}

        classes = [d for d in root_dir.iterdir() if d.is_dir()]

        for cls_dir in classes:
            cls_name = cls_dir.name
            split_file = split_dir / f"{cls_name}_test_split{split_num}.txt"

            if not split_file.exists():
                continue

            with split_file.open() as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue

                    indicator = int(parts[1])
                    if indicator == 0:
                        stats["unused"] += 1
                    elif indicator == 1:
                        stats["train"] += 1
                    elif indicator == 2:
                        stats["test"] += 1

        total = stats["train"] + stats["test"] + stats["unused"]
        print(f"Split {split_num}:")
        print(f"  Train (1):  {stats['train']:>5} ({stats['train']/total*100:>5.1f}%)")
        print(f"  Test (2):   {stats['test']:>5} ({stats['test']/total*100:>5.1f}%)")
        print(
            f"  Unused (0): {stats['unused']:>5} ({stats['unused']/total*100:>5.1f}%)"
        )
        print(f"  Total:      {total:>5}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organize HMDB-51 dataset according to official splits"
    )
    parser.add_argument(
        "--root", required=True, help="Root directory with class folders"
    )
    parser.add_argument(
        "--split_dir", required=True, help="Directory containing split files"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for organized dataset"
    )
    parser.add_argument(
        "--split",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="Split number (1, 2, or 3)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print statistics, don't organize",
    )

    args = parser.parse_args()

    if args.stats_only:
        print_split_statistics(args.root, args.split_dir)
    else:
        organize_dataset(args.root, args.split_dir, args.output, args.split)
