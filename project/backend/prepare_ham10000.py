import os
import csv
import argparse
import shutil
from typing import Dict

HAM_LABEL_MAP: Dict[str, str] = {
    "nv": "nv",
    "mel": "mel",
    "bcc": "bcc",
    "akiec": "akiec",
    "bkl": "bkl",
    "df": "df",
    "vasc": "vasc",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def prepare(images_dir: str, metadata_csv: str, output_dir: str, no_lesion_dir: str | None = None) -> None:
    ensure_dir(output_dir)
    for cls in list(HAM_LABEL_MAP.values()) + ["no_lesion"]:
        ensure_dir(os.path.join(output_dir, cls))

    # Read metadata: columns include image_id, dx
    with open(metadata_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get("image_id")
            dx = row.get("dx")  # lesion diagnosis label
            if not image_id or not dx:
                continue
            cls = HAM_LABEL_MAP.get(dx)
            if not cls:
                continue
            # HAM images are typically JPG named <image_id>.jpg
            src_jpg = os.path.join(images_dir, f"{image_id}.jpg")
            src_png = os.path.join(images_dir, f"{image_id}.png")
            src = src_jpg if os.path.exists(src_jpg) else src_png if os.path.exists(src_png) else None
            if not src:
                continue
            dst = os.path.join(output_dir, cls, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    # Add no_lesion images if provided
    if no_lesion_dir and os.path.isdir(no_lesion_dir):
        dst_dir = os.path.join(output_dir, "no_lesion")
        for name in os.listdir(no_lesion_dir):
            src = os.path.join(no_lesion_dir, name)
            if not os.path.isfile(src):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                continue
            dst = os.path.join(dst_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    print(f"Prepared dataset at: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare HAM10000 into class folders with optional no_lesion images")
    parser.add_argument("--images_dir", required=True, help="Path to HAM10000 images folder")
    parser.add_argument("--metadata_csv", required=True, help="Path to HAM10000_metadata.csv")
    parser.add_argument("--output_dir", required=True, help="Output directory with class subfolders")
    parser.add_argument("--no_lesion_dir", default=None, help="Optional directory of non-skin/normal images for no_lesion class")
    args = parser.parse_args()
    prepare(args.images_dir, args.metadata_csv, args.output_dir, args.no_lesion_dir)
