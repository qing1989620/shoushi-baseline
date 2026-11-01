"""Train a small Rock/Paper/Scissors detection baseline with RT-DETR."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train an RT-DETR gesture detector")
    parser.add_argument("--data", default="data/rps/data.yaml", help="Dataset YAML")
    parser.add_argument("--model", default="rtdetr-l.pt", help="Pretrained weights or RT-DETR YAML")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None, help="e.g. 0 for first GPU, cpu for CPU")
    parser.add_argument("--degrees", type=float, default=0.0, help="Rotation augmentation in degrees; 0 disables it")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="rps_baseline")
    args = parser.parse_args()
    data = Path(args.data).expanduser().resolve()
    if not data.is_file():
        parser.error(f"Dataset YAML not found: {data}. Prepare the dataset first.")
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 1 or args.workers < 0:
        parser.error("epochs, batch and imgsz must be positive; workers must be non-negative")

    from ultralytics import RTDETR

    model = RTDETR(args.model)
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, workers=args.workers, project=str(Path(args.project).expanduser().resolve()),
                name=args.name, degrees=args.degrees, seed=42, deterministic=True, exist_ok=False)
    weights = Path(model.trainer.save_dir).resolve() / "weights" / "best.pt"
    print(f"\nBest weights: {weights}")
    print("Use this path with --weights in the camera and evaluation scripts.")


if __name__ == "__main__":
    main()
