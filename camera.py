"""摄像头实时检测：与 train.py 生成的 Ultralytics RT-DETR 权重直接配合使用。"""
import argparse
import json
import time
from pathlib import Path


def percentile(values, p):
    """线性插值百分位数；不依赖额外统计库。"""
    ordered = sorted(values)
    if not ordered:
        return None
    pos = (len(ordered) - 1) * p / 100
    left = int(pos)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (pos - left)


def camera_source(value):
    # 数字表示摄像头编号；也允许传入视频文件路径或视频流 URL。
    try:
        return int(value)
    except ValueError:
        return value


def main():
    parser = argparse.ArgumentParser(description="RT-DETR 摄像头手势检测")
    parser.add_argument("--weights", default="runs/detect/rps_final/weights/best.pt", help="训练得到的权重")
    parser.add_argument("--source", default="0", help="摄像头编号、视频路径或视频流地址，默认 0")
    parser.add_argument("--device", default=None, help="推理设备，如 cpu、0；默认自动选择")
    parser.add_argument("--conf", type=float, default=0.45, help="检测置信度阈值")
    parser.add_argument("--nms-iou", type=float, default=0.7, help="同一位置重复框的去重阈值；1 表示关闭")
    parser.add_argument("--imgsz", type=int, default=320, help="推理输入图像尺寸")
    parser.add_argument("--metrics", default="camera_metrics.json", help="退出时写入的性能统计文件")
    args = parser.parse_args()
    weights = Path(args.weights)
    if not weights.is_file():
        parser.error(f"找不到模型权重：{weights}。请先训练，或通过 --weights 指定 best.pt。")
    if not 0 <= args.conf <= 1 or not 0 < args.nms_iou <= 1 or args.imgsz <= 0:
        parser.error("--conf 必须在 0 到 1 之间，--nms-iou 在 0 到 1 之间且大于 0，--imgsz 必须大于 0。")
    try:
        import cv2
        from ultralytics import RTDETR
        from torchvision.ops import nms
    except ImportError as exc:
        parser.exit(1, f"缺少依赖：{exc}。请先按照 README 安装依赖。\n")

    gui_line = next((line.strip() for line in cv2.getBuildInformation().splitlines() if line.strip().startswith("GUI:")), "")
    if gui_line.split(":", 1)[-1].strip().upper() == "NONE":
        parser.exit(1, "当前 OpenCV 不支持窗口显示（GUI: NONE），请参阅 README 的摄像头故障排查。\n")

    model = RTDETR(str(weights))
    capture = cv2.VideoCapture(camera_source(args.source))
    if not capture.isOpened():
        capture.release()
        parser.exit(1, f"无法打开视频源 {args.source}，请检查摄像头权限和 --source。\n")

    durations = []
    inference_ms = []
    loop_start = time.perf_counter()
    last_ms = None
    window = "Gesture detection (Q/Esc to quit)"
    print("按 Q 或 Esc 退出；建议至少采集 300 帧后退出，以获得更有参考价值的统计。")
    print("计时边界：调用 read 前开始，到 imshow 返回结束；包括取帧、推理和绘制，不包括 waitKey。")
    try:
        while True:
            start = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                print("视频读取结束或取帧失败。")
                break
            options = {"source": frame, "conf": args.conf, "imgsz": args.imgsz, "verbose": False}
            if args.device is not None:
                options["device"] = args.device
            result = model.predict(**options)[0]
            inference_ms.append(float(result.speed["inference"]))
            # 按位置合并同一手部产生的高度重叠框，保留置信度最高的预测。
            if args.nms_iou < 1 and len(result.boxes) > 1:
                keep = nms(result.boxes.xyxy, result.boxes.conf, args.nms_iou)
                result.boxes = result.boxes[keep]
            # plot 保留模型自身的类别名称、每个目标的框和置信度。
            shown = result.plot()
            if last_ms is not None:
                text = f"FPS: {1000 / last_ms:.1f}  latency: {last_ms:.1f} ms (previous frame)"
                cv2.putText(shown, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, shown)
            last_ms = (time.perf_counter() - start) * 1000
            durations.append(last_ms)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
    except KeyboardInterrupt:
        print("\n用户中断。")
    finally:
        wall_seconds = time.perf_counter() - loop_start
        capture.release()
        cv2.destroyAllWindows()
        if durations:
            fps_values = [1000 / ms for ms in durations if ms > 0]
            summary = {
                "frames": len(durations),
                "recommended_min_frames": 300,
                "meets_recommended_minimum": len(durations) >= 300,
                "timing_boundary": "Before VideoCapture.read to after imshow returns, including capture, inference and drawing; excludes waitKey. imshow return is not physical screen presentation.",
                "latency_ms_mean": sum(durations) / len(durations),
                "latency_ms_p95": percentile(durations, 95),
                "fps_mean": len(durations) / wall_seconds,
                "wall_seconds_including_waitKey": wall_seconds,
                "inference_ms_mean": sum(inference_ms) / len(inference_ms),
                "inference_ms_p95": percentile(inference_ms, 95),
                "fps_p95_instantaneous": percentile(fps_values, 95),
                "weights": str(weights),
                "source": args.source,
                "device": args.device,
                "confidence": args.conf,
                "duplicate_box_iou": args.nms_iou,
                "image_size": args.imgsz,
            }
            metrics_path = Path(args.metrics)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"统计：{len(durations)} 帧，平均延迟 {summary['latency_ms_mean']:.2f} ms，P95 延迟 {summary['latency_ms_p95']:.2f} ms，平均 FPS {summary['fps_mean']:.2f}。")
            print(f"统计已保存至 {metrics_path}")
            if len(durations) < 300:
                print("注意：有效帧数不足 300，建议重新运行并采集至少 300 帧。")
        else:
            print("没有成功处理的帧，未生成性能统计。")


if __name__ == "__main__":
    main()
