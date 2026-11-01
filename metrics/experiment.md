# 本次基线训练与测试记录

- 硬件：NVIDIA GeForce RTX 4050 Laptop GPU（6 GB）；Python 3.12、PyTorch 2.7.1+cu118、Ultralytics 8.4.41。
- 数据：同源重建 128×128 图像，自动生成单手检测框。训练/验证/原始测试分别 2268/252/372 张，类别均衡；固定随机种子 42。训练、验证、测试各随机抽看了 60 张框，整体覆盖手部，有框包含袖口。抽检不是逐张验收，全部标签仍属于伪标注。
- 初始化：Ultralytics `rtdetr-l.pt` 预训练权重；真实微调命令：`python train.py --device 0 --epochs 5 --batch 4 --imgsz 320 --workers 0`。最佳权重为 `runs/detect/rps_baseline/weights/best.pt`，SHA-256：`43f02fea6a4649ab8c6fbecceeb8dcf1c9c3ccd525d53036decae8fa41998ee8`。训练逐轮结果见 `runs/detect/rps_baseline/results.csv`。
- 测试命令：`python evaluate.py --device 0 --imgsz 320 --batch 4 --workers 0`，使用保留的 372 张原始测试图及自动框。

| 指标 | 实测 |
|---|---:|
| mAP50 | 0.8022 |
| mAP50–95 | 0.5081 |
| Precision | 0.8855 |
| Recall | 0.8532 |
| Rock AP50 | 0.9950 |
| Scissors AP50 | 0.8434 |
| Paper AP50 | 0.5681 |

测试集与训练集均依赖自动框，指标不是经人工标注核准的真实场景性能。测试集由原始提供的 test 包重建，尚未核验与任务书指定 `hand.tar` 一致；不同组的数据未统一时不可直接比较。Paper 类明显较弱。已用真实摄像头读取 5 帧并验证训练权重加载及输出格式，但未验证可见窗口、实际出拳识别准确率、多人效果或任务书要求的 300 帧真实性能；这些需要用户现场测试并如实报告。
