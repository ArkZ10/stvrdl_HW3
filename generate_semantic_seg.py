"""Generate semantic segmentation maps from instance masks for HTC training.

For each training image, creates a single-channel semantic seg map where:
- 0 = background
- 1 = any cell (binary foreground)

Saved as PNG files alongside the images.
"""

import os
import numpy as np
import tifffile
from PIL import Image


def generate_seg_map(sample_path):
    """Generate binary semantic seg map from instance masks."""
    img = tifffile.imread(os.path.join(sample_path, "image.tif"))
    h, w = img.shape[:2]
    seg_map = np.zeros((h, w), dtype=np.uint8)

    for cls in ["class1", "class2", "class3", "class4"]:
        mask_path = os.path.join(sample_path, f"{cls}.tif")
        if not os.path.exists(mask_path):
            continue
        mask = tifffile.imread(mask_path)
        seg_map[mask > 0] = 1

    return seg_map


def main():
    train_dir = "datasets/hw3/train"
    samples = sorted(os.listdir(train_dir))
    print(f"Generating semantic seg maps for {len(samples)} samples...")

    for i, name in enumerate(samples):
        sample_path = os.path.join(train_dir, name)
        seg_map = generate_seg_map(sample_path)
        out_path = os.path.join(sample_path, "seg.png")
        Image.fromarray(seg_map).save(out_path)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(samples)}] done")

    print("Done!")


if __name__ == "__main__":
    main()
