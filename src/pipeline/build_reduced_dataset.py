"""
build_reduced_dataset.py
-------------------------
Builds the reduced RPC (Retail Product Checkout) dataset subset used by the
PRODEXA project, within the assignment's 12-hour scope.

This script is intentionally built ON TOP OF the preprocessing module
already written by the "Image Acquisition & Preprocessing" member
(src/preprocessing.py). It reuses, unmodified:

    - stratified_sample()   -> balanced per-meta-category sampling from the
                                exemplar (train2019) COCO annotations
    - batch_preprocess()    -> resize + denoise + lighting normalization,
                                applied to every sampled exemplar image

It adds the parts specific to *dataset reduction* that preprocessing.py does
not cover:

    1. Copying the sampled RAW exemplar images into data/dataset/<category>/
       (kept alongside the preprocessed copies in data/processed/<category>/,
       so both the "before" and "after" preprocessing images are available
       for the demo).
    2. Sampling a smaller set of EASY-difficulty checkout images from the
       val2019 / test2019 split, for the detection + segmentation + full
       end-to-end pipeline demo (per the project's documented scope: checkout
       images are only used at "easy" clutter level, since higher clutter
       involves heavy occlusion which conflicts with the assignment's
       "clearly separated products preferred" limitation).
    3. Writing a *trimmed* COCO annotations file containing only the
       sampled checkout images, so ground-truth boxes are still usable by
       the detection/segmentation module without shipping the full ~20GB
       annotation file.
    4. Writing data/dataset_subset.txt: an exact manifest of every file used,
       for GitHub/demo reproducibility (the assignment explicitly asks for
       this and for GitHub history to reflect real, separable contributions).

Expected input layout (this is how the RPC dataset unzips):

    <raw-dir>/
        train2019/                     exemplar images
        val2019/                       checkout images
        test2019/                      checkout images
        instances_train2019.json
        instances_val2019.json
        instances_test2019.json

Usage
-----
    python -m src.pipeline.build_reduced_dataset \\
        --raw-dir /path/to/retail_product_checkout \\
        --out-dir data \\
        --images-per-category 80 \\
        --checkout-images 150

Run from the repository root so that `from src.preprocessing import ...`
resolves correctly.
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.preprocessing import stratified_sample, batch_preprocess  # noqa: E402


# Candidate field names that different releases / mirrors of the RPC
# annotation files have used to record checkout-image clutter level.
LEVEL_KEY_CANDIDATES = ["level", "difficulty", "mode", "clutter_level", "diff"]


def load_coco(path):
    with open(path, "r") as f:
        return json.load(f)


def detect_level_key(images):
    """
    Find which key (if any) in the COCO 'images' entries records the
    easy/medium/hard clutter level, since this has varied across dataset
    mirrors. Returns None if no candidate key is present.
    """
    if not images:
        return None
    for key in LEVEL_KEY_CANDIDATES:
        if key in images[0]:
            return key
    return None


def sample_checkout_subset(annotations_path, level="easy", num_images=150, seed=42):
    """
    Sample `num_images` checkout images tagged with the given clutter level
    from a COCO-format val/test annotations file.

    Unlike stratified_sample() in preprocessing.py (which assumes one
    category per exemplar image), checkout images can contain many product
    instances/categories each, so this samples at the image level using
    whatever difficulty field the annotation file provides.

    Returns:
        chosen_images: list of COCO 'image' dicts that were sampled
        trimmed_coco: a COCO-format dict (images/annotations/categories)
                      containing only the sampled images, so the file stays
                      small but ground-truth boxes remain usable
    """
    coco = load_coco(annotations_path)
    images = coco["images"]

    level_key = detect_level_key(images)
    if level_key is None:
        print(
            "  [!] Could not find an explicit difficulty field "
            f"(tried {LEVEL_KEY_CANDIDATES}) in {os.path.basename(annotations_path)}. "
            "Sampling from all images instead -- please spot-check a few of "
            "the copied images to confirm they are low-clutter/'easy' scenes."
        )
        candidates = images
    else:
        candidates = [
            img for img in images
            if str(img.get(level_key, "")).lower() == level
        ]
        print(f"  Found {len(candidates)} images with {level_key} == '{level}'")

    random.seed(seed)
    chosen = random.sample(candidates, min(num_images, len(candidates)))
    chosen_ids = {img["id"] for img in chosen}

    trimmed_annotations = [
        ann for ann in coco["annotations"] if ann["image_id"] in chosen_ids
    ]
    trimmed_coco = {
        "images": chosen,
        "annotations": trimmed_annotations,
        "categories": coco["categories"],
    }
    return chosen, trimmed_coco


def copy_files(filenames, source_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    copied = 0
    for fname in filenames:
        src = os.path.join(source_dir, fname)
        dst = os.path.join(out_dir, fname)
        try:
            shutil.copy2(src, dst)
            copied += 1
        except FileNotFoundError:
            print(f"    [!] missing file, skipped: {fname}")
    return copied, len(filenames)


def build_classification_subset(args, manifest_rows):
    train_dir = os.path.join(args.raw_dir, "train2019")
    train_ann = os.path.join(args.raw_dir, "instances_train2019.json")

    print(f"\n=== Exemplar / classification subset ===")
    print(f"Sampling {args.images_per_category} images per meta-category from {train_ann} ...")
    sample_records = stratified_sample(
        train_ann, images_per_category=args.images_per_category, seed=args.seed
    )
    print(f"  -> {len(sample_records)} images total across all meta-categories")

    by_category = defaultdict(list)
    for r in sample_records:
        by_category[r["supercategory"]].append(r["filename"])

    dataset_dir = os.path.join(args.out_dir, "dataset")
    processed_dir = os.path.join(args.out_dir, "processed")

    for supercategory, filenames in sorted(by_category.items()):
        raw_out = os.path.join(dataset_dir, supercategory)
        copied, total = copy_files(filenames, train_dir, raw_out)
        print(f"  [{supercategory}] raw copied: {copied}/{total} -> {raw_out}")

        if not args.skip_preprocessing:
            processed_out = os.path.join(processed_dir, supercategory)
            success, total = batch_preprocess(filenames, train_dir, processed_out)
            print(f"  [{supercategory}] preprocessed: {success}/{total} -> {processed_out}")

    for r in sample_records:
        manifest_rows.append(
            f"{r['filename']}\texemplar\t{r['supercategory']}\t{r['category_name']}\t-"
        )
    return sample_records


def build_checkout_subset(args, manifest_rows):
    checkout_dir = os.path.join(args.raw_dir, args.checkout_split)
    checkout_ann = os.path.join(args.raw_dir, f"instances_{args.checkout_split}.json")

    print(f"\n=== Easy-checkout subset (detection / segmentation / full pipeline) ===")
    if not os.path.exists(checkout_ann):
        print(
            f"  [!] {checkout_ann} not found -- skipping checkout subset.\n"
            f"      Pass --raw-dir pointing at the extracted RPC dataset root "
            f"to include it."
        )
        return []

    print(f"Sampling {args.checkout_images} '{args.checkout_level}' images from {checkout_ann} ...")
    chosen_images, trimmed_coco = sample_checkout_subset(
        checkout_ann,
        level=args.checkout_level,
        num_images=args.checkout_images,
        seed=args.seed,
    )

    filenames = [img["file_name"] for img in chosen_images]
    checkout_out_dir = os.path.join(args.out_dir, "checkout_easy")
    copied, total = copy_files(filenames, checkout_dir, checkout_out_dir)
    print(f"  copied: {copied}/{total} -> {checkout_out_dir}")

    annotations_out_dir = os.path.join(args.out_dir, "annotations")
    os.makedirs(annotations_out_dir, exist_ok=True)
    trimmed_path = os.path.join(
        annotations_out_dir,
        f"instances_{args.checkout_split}_{args.checkout_level}_subset.json",
    )
    with open(trimmed_path, "w") as f:
        json.dump(trimmed_coco, f)
    print(f"  trimmed annotations -> {trimmed_path} "
          f"({len(trimmed_coco['images'])} images, {len(trimmed_coco['annotations'])} boxes)")

    for img in chosen_images:
        manifest_rows.append(f"{img['file_name']}\tcheckout\t-\t-\t{args.checkout_level}")
    return chosen_images


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the reduced RPC dataset subset for PRODEXA."
    )
    parser.add_argument(
        "--raw-dir", default="data/raw",
        help="Path to the extracted full RPC dataset "
             "(contains train2019/, val2019/, test2019/, instances_*.json)",
    )
    parser.add_argument(
        "--out-dir", default="data",
        help="Output root, matches PRODEXA's data/ layout",
    )
    parser.add_argument(
        "--images-per-category", type=int, default=80,
        help="Exemplar images sampled per meta-category (classification subset)",
    )
    parser.add_argument(
        "--checkout-split", default="test2019", choices=["val2019", "test2019"],
        help="Which checkout split to sample easy images from",
    )
    parser.add_argument(
        "--checkout-level", default="easy", choices=["easy", "medium", "hard"],
    )
    parser.add_argument(
        "--checkout-images", type=int, default=150,
        help="Number of checkout images to sample",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-preprocessing", action="store_true",
        help="Only copy raw exemplar files, skip the resize/denoise/normalize step",
    )
    parser.add_argument(
        "--skip-checkout", action="store_true",
        help="Only build the classification subset, skip checkout images",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    manifest_rows = []
    build_classification_subset(args, manifest_rows)
    if not args.skip_checkout:
        build_checkout_subset(args, manifest_rows)

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "dataset_subset.txt")
    with open(manifest_path, "w") as f:
        f.write("filename\tsplit\tsupercategory\tcategory_name\tcheckout_level\n")
        for row in manifest_rows:
            f.write(row + "\n")

    print(f"\nManifest written to {manifest_path} ({len(manifest_rows)} files total)")
    print("Done.")


if __name__ == "__main__":
    main()
