"""Train a small Rock/Paper/Scissors detection baseline with RT-DETR."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train an RT-DETR gesture detector")
    parser.add_argument("--data", default="data/rps_real_full/data.yaml", help="Dataset YAML")
    parser.add_argument("--model", default="rtdetr-l.pt", help="Pretrained weights or RT-DETR YAML")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained weights on the dataset test split")
    parser.add_argument("--background-test", action="store_true", help="Measure false detections on no-hand background images")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold for --background-test")
    parser.add_argument("--weights", default="runs/detect/rps_final/weights/best.pt", help="Trained weights for --evaluate")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None, help="e.g. 0 for first GPU, cpu for CPU")
    parser.add_argument("--lr", type=float, default=0.0001, help="AdamW initial learning rate")
    parser.add_argument("--warmup", type=float, default=0.5, help="Warmup epochs")
    parser.add_argument("--degrees", type=float, default=0.0, help="Rotation augmentation in degrees; 0 disables it")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="rps_final")
    args = parser.parse_args()
    data = Path(args.data).expanduser().resolve()
    if not data.is_file():
        parser.error(f"Dataset YAML not found: {data}. Prepare the dataset first.")
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 1 or args.workers < 0 or not 0 <= args.conf <= 1 or args.lr <= 0 or args.warmup < 0:
        parser.error("epochs, batch and imgsz must be positive; workers must be non-negative")

    from ultralytics import RTDETR

    if args.evaluate and args.background_test:
        parser.error("Choose only one of --evaluate or --background-test")

    if args.background_test:
        weights = Path(args.weights).expanduser().resolve()
        if not weights.is_file():
            parser.error(f"Weights not found: {weights}")
        image_dir = data.parent / "images" / "background_test"
        images = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not images:
            parser.error(f"Background test images not found: {image_dir}")
        model = RTDETR(str(weights))
        count = boxes = 0
        for result in model.predict(source=[str(p) for p in images], stream=True, imgsz=args.imgsz, device=args.device, conf=args.conf, batch=args.batch, verbose=False):
            n = len(result.boxes)
            count += int(n > 0)
            boxes += n
        print(f"No-hand images={len(images)}, images with false detections={count}, rate={count / len(images):.4f}, false boxes={boxes}, conf={args.conf:.2f}")
        return

    if args.evaluate:
        weights = Path(args.weights).expanduser().resolve()
        if not weights.is_file():
            parser.error(f"Weights not found: {weights}")
        model = RTDETR(str(weights))
        metrics = model.val(data=str(data), split="test", imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers)
        box = metrics.box
        print(f"Test mAP50={box.map50:.4f}, mAP50-95={box.map:.4f}, precision={box.mp:.4f}, recall={box.mr:.4f}")
        for class_id, ap in zip(box.ap_class_index, box.ap50):
            print(f"  {model.names[int(class_id)]}: {ap:.4f}")
        return

    model = RTDETR(args.model)
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, workers=args.workers, project=str(Path(args.project).expanduser().resolve()),
                name=args.name, degrees=args.degrees, mosaic=0.0, optimizer="AdamW", lr0=args.lr, warmup_epochs=args.warmup, seed=42, deterministic=True, exist_ok=False)
    weights = Path(model.trainer.save_dir).resolve() / "weights" / "best.pt"
    print(f"\nBest weights: {weights}")
    print("Use this path with --weights in camera.py or train.py --evaluate.")


if __name__ == "__main__":
    main()
