"""Evaluate a trained RT-DETR detector on the held-out test split."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate gesture detection on the test split")
    parser.add_argument("--data", default="data/rps/data.yaml")
    parser.add_argument("--weights", default="runs/detect/rps_baseline/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None, help="e.g. 0 for first GPU, cpu for CPU")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    data = Path(args.data).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    if not data.is_file():
        parser.error(f"Dataset YAML not found: {data}")
    if not weights.is_file():
        parser.error(f"Weights not found: {weights}. Train a model first or pass --weights.")
    if args.batch < 1 or args.imgsz < 1 or args.workers < 0:
        parser.error("batch and imgsz must be positive; workers must be non-negative")

    from ultralytics import RTDETR

    model = RTDETR(str(weights))
    metrics = model.val(data=str(data), split="test", imgsz=args.imgsz, batch=args.batch,
                        device=args.device, workers=args.workers)
    box = metrics.box
    print(f"\nTest: mAP50={box.map50:.4f}, mAP50-95={box.map:.4f}, "
          f"precision={box.mp:.4f}, recall={box.mr:.4f}")
    print("AP50 by class (classes present in test labels):")
    for class_id, ap in zip(box.ap_class_index, box.ap50):
        class_id = int(class_id)
        print(f"  {model.names[class_id]}: {ap:.4f}")


if __name__ == "__main__":
    main()
