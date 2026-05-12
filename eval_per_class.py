"""Per-class AP50 evaluation for NuHTC checkpoint."""

import argparse
import io
import json
import sys

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


CLASS_NAMES = ["class1", "class2", "class3", "class4"]


def evaluate_per_class(ann_file, result_file):
    coco_gt = COCO(ann_file)

    with open(result_file) as f:
        results = json.load(f)

    coco_dt = coco_gt.loadRes(results)

    print(f"{'Class':<12} {'AP50':>8} {'#GT':>8} {'#DT':>8}")
    print("-" * 38)

    overall_eval = COCOeval(coco_gt, coco_dt, "segm")
    overall_eval.params.iouThrs = [0.5]
    overall_eval.evaluate()
    overall_eval.accumulate()
    overall_eval.summarize()
    print(f"\nOverall AP50: {overall_eval.stats[0]:.4f}\n")

    for cat_id, cat_name in zip(sorted(coco_gt.getCatIds()), CLASS_NAMES):
        ann_ids = coco_gt.getAnnIds(catIds=[cat_id])
        n_gt = len(ann_ids)
        n_dt = sum(1 for r in results if r["category_id"] == cat_id)

        eval_ = COCOeval(coco_gt, coco_dt, "segm")
        eval_.params.iouThrs = [0.5]
        eval_.params.catIds = [cat_id]
        eval_.evaluate()
        eval_.accumulate()

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        eval_.summarize()
        sys.stdout = old_stdout

        ap50 = eval_.stats[0]
        print(f"{cat_name:<12} {ap50:>8.4f} {n_gt:>8} {n_dt:>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", default="datasets/hw3/annotations/val.json")
    parser.add_argument("--result", default="val_results.json")
    args = parser.parse_args()
    evaluate_per_class(args.ann, args.result)
