"""Convert HW3 .tif dataset to COCO JSON format with RLE masks.

Usage:
    python preprocessing/prepare_coco.py \
        --data_root datasets/hw3 \
        --output_dir datasets/hw3/annotations \
        --val_ratio 0.15 \
        --seed 42
"""

import argparse
import json
import os
import random

import numpy as np
import tifffile
from pycocotools import mask as mask_utils


CLASS_TO_ID = {
    "class1": 1,
    "class2": 2,
    "class3": 3,
    "class4": 4,
}

CATEGORIES = [
    {"id": 1, "name": "class1", "supercategory": "cell"},
    {"id": 2, "name": "class2", "supercategory": "cell"},
    {"id": 3, "name": "class3", "supercategory": "cell"},
    {"id": 4, "name": "class4", "supercategory": "cell"},
]


def load_image(image_path):
    """Load .tif image and convert RGBA -> RGB if needed."""
    img = tifffile.imread(image_path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def mask_to_rle(binary_mask):
    """Encode a binary mask as RLE using pycocotools."""
    binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(binary_mask)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def rle_to_bbox(rle):
    """Get bounding box [x, y, w, h] from RLE mask."""
    bbox = mask_utils.toBbox(rle).tolist()
    return bbox


def process_sample(image_id, image_name, image_dir, ann_id_start):
    """Process one training sample, returning image info and annotations."""
    sample_path = os.path.join(image_dir, image_name)
    image_path = os.path.join(sample_path, "image.tif")

    img = load_image(image_path)
    height, width = img.shape[:2]

    image_info = {
        "id": image_id,
        "file_name": f"{image_name}/image.tif",
        "height": height,
        "width": width,
    }

    annotations = []
    ann_id = ann_id_start

    for class_name, category_id in CLASS_TO_ID.items():
        mask_path = os.path.join(sample_path, f"{class_name}.tif")
        if not os.path.exists(mask_path):
            continue

        instance_mask = tifffile.imread(mask_path).astype(np.float64)
        instance_ids = np.unique(instance_mask)
        instance_ids = instance_ids[instance_ids != 0]

        for inst_id in instance_ids:
            binary_mask = (instance_mask == inst_id).astype(np.uint8)

            if binary_mask.sum() < 5:
                continue

            rle = mask_to_rle(binary_mask)
            bbox = rle_to_bbox(rle)
            area = float(mask_utils.area(rle))

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": rle,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

    return image_info, annotations, ann_id


def build_coco_json(image_names, image_dir, start_image_id=1):
    """Build full COCO-format dict for a list of image names."""
    images = []
    annotations = []
    ann_id = 1

    for i, name in enumerate(image_names):
        image_id = start_image_id + i
        print(f"  Processing [{i+1}/{len(image_names)}]: {name}")
        image_info, anns, ann_id = process_sample(
            image_id, name, image_dir, ann_id
        )
        images.append(image_info)
        annotations.extend(anns)

    return {
        "info": {"description": "HW3 Cell Instance Segmentation"},
        "categories": CATEGORIES,
        "images": images,
        "annotations": annotations,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert HW3 .tif dataset to COCO JSON format."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="datasets/hw3",
        help="Root directory containing train/ and test_release/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/hw3/annotations",
        help="Directory to save COCO JSON files",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Fraction of training data to use for validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_dir = os.path.join(args.data_root, "train")
    all_samples = sorted(os.listdir(train_dir))

    random.seed(args.seed)
    random.shuffle(all_samples)
    n_val = max(1, int(len(all_samples) * args.val_ratio))
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]

    print(f"Total samples : {len(all_samples)}")
    print(f"Train samples : {len(train_samples)}")
    print(f"Val samples   : {len(val_samples)}")

    print("\nBuilding train annotations...")
    train_coco = build_coco_json(train_samples, train_dir, start_image_id=1)
    train_out = os.path.join(args.output_dir, "train.json")
    with open(train_out, "w") as f:
        json.dump(train_coco, f)
    print(f"Saved: {train_out}")
    print(f"  Images     : {len(train_coco['images'])}")
    print(f"  Annotations: {len(train_coco['annotations'])}")

    print("\nBuilding val annotations...")
    val_coco = build_coco_json(
        val_samples, train_dir, start_image_id=len(train_samples) + 1
    )
    val_out = os.path.join(args.output_dir, "val.json")
    with open(val_out, "w") as f:
        json.dump(val_coco, f)
    print(f"Saved: {val_out}")
    print(f"  Images     : {len(val_coco['images'])}")
    print(f"  Annotations: {len(val_coco['annotations'])}")

    print("\nBuilding test image info...")
    id_map_path = os.path.join(args.data_root, "test_image_name_to_ids.json")
    with open(id_map_path, "r") as f:
        id_map = json.load(f)

    test_images = []
    for entry in id_map:
        test_images.append({
            "id": entry["id"],
            "file_name": entry["file_name"],
            "height": entry["height"],
            "width": entry["width"],
        })

    test_coco = {
        "info": {"description": "HW3 Cell Instance Segmentation - Test"},
        "categories": CATEGORIES,
        "images": test_images,
        "annotations": [],
    }
    test_out = os.path.join(args.output_dir, "test.json")
    with open(test_out, "w") as f:
        json.dump(test_coco, f)
    print(f"Saved: {test_out}")
    print(f"  Images: {len(test_coco['images'])}")

    print("\nDone! Annotation files ready for training.")


if __name__ == "__main__":
    main()
