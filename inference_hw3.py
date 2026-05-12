"""Generate test-results.json for CodaBench submission.

Usage:
    python tools/inference_hw3.py \
        --config configs/nuhtc/nuhtc_fulldata_hw3.py \
        --checkpoint work_dirs/nuhtc_fulldata_hw3/epoch_30.pth \
        --output test-results.json \
        --score-thr 0.3
"""

import argparse
import json
import os
import sys

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "thirdparty", "mmdetection"
))

import mmcv
import nuhtc  # noqa: F401 registers custom modules
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from pycocotools import mask as mask_utils


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference and generate CodaBench submission file."
    )
    parser.add_argument(
        "--config",
        default="configs/nuhtc/nuhtc_fulldata_hw3.py",
        help="Path to mmdet config file",
    )
    parser.add_argument(
        "--checkpoint",
        default="work_dirs/nuhtc_fulldata_hw3/epoch_30.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        default="test-results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.3,
        help="Minimum confidence score threshold",
    )
    return parser.parse_args()


def main():
    """Run inference and save predictions in COCO result format."""
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.workers_per_gpu = 0
    cfg.data.samples_per_gpu = 1

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )

    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval()
    model = MMDataParallel(model, device_ids=[0])

    with open(cfg.data.test.ann_file, "r") as f:
        ann_data = json.load(f)
    fname_to_id = {img["file_name"]: img["id"] for img in ann_data["images"]}

    results = []
    prog_bar = mmcv.ProgressBar(len(dataset))

    with torch.no_grad():
        for data in data_loader:
            img_meta = data["img_metas"][0].data[0][0]
            filename = os.path.basename(img_meta["filename"])
            image_id = fname_to_id.get(filename)

            if image_id is None:
                prog_bar.update()
                continue

            result = model(return_loss=False, rescale=True, **data)
            bbox_results, mask_results = result[0]

            for cat_id, (bboxes, masks) in enumerate(
                zip(bbox_results, mask_results), start=1
            ):
                if bboxes is None or len(bboxes) == 0:
                    continue

                for bbox, mask in zip(bboxes, masks):
                    score = float(bbox[4])
                    if score < args.score_thr:
                        continue

                    x1, y1, x2, y2 = bbox[:4]
                    rle = mask_utils.encode(
                        mask.astype("uint8").copy(order="F")
                    )
                    rle["counts"] = rle["counts"].decode("utf-8")

                    results.append({
                        "image_id": image_id,
                        "category_id": cat_id,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": score,
                        "segmentation": rle,
                    })

            prog_bar.update()

    with open(args.output, "w") as f:
        json.dump(results, f)
    print(f"\nSaved {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
