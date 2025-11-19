import argparse
import shutil
from pathlib import Path

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}


def is_video_file(path: Path):
    return path.suffix.lower() in VIDEO_EXTENSIONS


def organize_dataset(root_dir, split_dir, output_dir, split):
    root_dir = Path(root_dir)
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)
    assert split in [1, 2, 3], "Split must be 1, 2, or 3"

    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    classes = [d for d in root_dir.iterdir() if d.is_dir()]

    for cls_dir in classes:
        cls_name = cls_dir.name

        cls_train_dir = train_dir / cls_name
        cls_test_dir = test_dir / cls_name
        cls_train_dir.mkdir(exist_ok=True)
        cls_test_dir.mkdir(exist_ok=True)

        split_file = split_dir / f"{cls_name}_test_split{split}.txt"
        if not split_file.exists():
            print(f"Warning: Missing split file: {split_file}")
            continue

        with split_file.open() as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue

                filename, is_train = parts
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
                        print(f"Warning: {filename} not found in {cls_name}")
                        continue

                dst_dir = cls_train_dir if is_train == "1" else cls_test_dir
                shutil.copy2(src_path, dst_dir / src_path.name)

    print(f"Dataset organized in {output_dir}")
    verify_split(root_dir, split_dir, output_dir, split)


def verify_split(root_dir, split_dir, output_dir, split):
    print("\nVerifying split consistency...")

    root_dir = Path(root_dir)
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)

    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    success = True

    classes = [d for d in root_dir.iterdir() if d.is_dir()]

    for cls_dir in classes:
        cls_name = cls_dir.name

        split_file = split_dir / f"{cls_name}_test_split{split}.txt"
        if not split_file.exists():
            print(f"Warning: Missing split file for verification: {split_file}")
            continue

        train_expected = set()
        test_expected = set()

        with split_file.open() as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue

                filename, is_train = parts
                target = train_expected if is_train == "1" else test_expected

                # Normalize expected names by extension presence
                target.add(filename)

        # Actual files
        cls_train_dir = train_dir / cls_name
        cls_test_dir = test_dir / cls_name

        train_actual = {
            f.name for f in cls_train_dir.glob("*") if f.is_file() and is_video_file(f)
        }

        test_actual = {
            f.name for f in cls_test_dir.glob("*") if f.is_file() and is_video_file(f)
        }

        # Compare
        if train_expected != train_actual:
            success = False
            print(f"[ERROR] Train mismatch ({cls_name})")
            print("  Missing:", train_expected - train_actual)
            print("  Extra:", train_actual - train_expected)

        if test_expected != test_actual:
            success = False
            print(f"[ERROR] Test mismatch ({cls_name})")
            print("  Missing:", test_expected - test_actual)
            print("  Extra:", test_actual - test_expected)

    if success:
        print("Verification PASSED: all splits match expected lists.")
    else:
        print("Verification FAILED: see mismatch details above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", type=int, required=True)
    args = parser.parse_args()

    organize_dataset(args.root, args.split_dir, args.output, args.split)
