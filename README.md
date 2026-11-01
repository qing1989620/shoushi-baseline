# 石头剪刀布目标检测 Baseline

本项目实现任务书的基础流程：同源分类图像 → 生成待核查的检测框 → 微调带 Transformer 解码器的 RT-DETR → 固定测试集评估 → 用训练出的权重进行实时摄像头检测。类别固定为 `0 Rock`、`1 Scissors`、`2 Paper`。检测器可输出多个框，但单手图像训练不保证多人、复杂环境或实时速度达到任务目标，需要实测。

## 1. 环境

建议 Python 3.10–3.12，Windows/Linux；训练推荐 NVIDIA GPU。先按 [PyTorch 官方安装指引](https://pytorch.org/get-started/locally/)安装匹配显卡和驱动的 PyTorch，然后：

```powershell
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`True` 表示 PyTorch 能使用 CUDA。CPU 可用，但 RT-DETR 训练和推理会慢。Windows 可将下列命令中的 `python` 替换成虚拟环境的 Python。

## 2. 数据与标注

任务书指定 Laurence Moroney 数据经虚谷号缩放并打包的 `hand.tar`，参见 [教程](https://mc.dfrobot.com.cn/article-361023.html) 与 [镜像仓库](https://gitee.com/xiezuoru/vvBoard-docs)。该包是图像分类数据，**不含检测框**。当前仓库中的数据不是核验过的 `hand.tar`：公开仓库当前版本未找到该文件，因此使用 [TensorFlow 数据集构建器](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/rock_paper_scissors/rock_paper_scissors_dataset_builder.py) 所指同源原图，将其缩放至 128×128 重建。尚不能证明与指定压缩包逐文件一致；若社团统一发放原包，须重新转换并统一评估。来源、SHA-256、数量见 `data/provenance.json`，生成方法见 `reconstruct_dataset.py`。原始训练源 2520 张、原始测试集 372 张。

本项目采用图像边缘的背景颜色估计前景，再取最大连通区域生成一个矩形框。生成结果只是伪标注；必须抽样检查三类及三个数据划分，修正有误的框，并在报告中说明方法和质量。基于这些框计算的检测指标并不等同于人工真实标注指标。

仓库已包含准备好的 `data/rps/data.yaml`（路径相对于 YAML，适合克隆后使用）。可直接查看 `data/rps/manifest.csv` 和图像/标签。重新生成时首先准备好如下结构（`data/raw/hand`）。也可以下载 `data/provenance.json` 中列出的两个 ZIP 到 `data/raw/downloads/`，分别命名为 `rps.zip` 和 `rps-test-set.zip`，核对 SHA-256 后执行 `python reconstruct_dataset.py`：

```text
hand/
  rps/{rock,scissors,paper}/...
  rps-test-set/{rock,scissors,paper}/...
```

保留原始 `rps-test-set` 作为 test，按类别、种子 42 将 `rps` 的 90%/10% 划分为 train/val。对于没有既定测试集的数据不能直接套用本脚本；这符合任务书对既定测试集的优先规则。原包内其它目录不会混入训练或测试。脚本会检查跨训练源和测试源的完全相同图像并停止，避免数据泄漏。不会执行线下图像增强。输出目录必须不存在或为空，避免覆盖人工修正。

```powershell
python prepare_dataset.py --source data/raw/hand --output data/rps_rebuilt
python review_labels.py --data data/rps_rebuilt --split train --output review_train.jpg
python review_labels.py --data data/rps_rebuilt --split val --output review_val.jpg
python review_labels.py --data data/rps_rebuilt --split test --output review_test.jpg
```

重建后训练/评估时用 `--data data/rps_rebuilt/data.yaml` 指向新数据。检查拼图中的框并按文件名前缀在 `data/rps_rebuilt/images/<划分>` 找到原图，在对应 `labels/<划分>/<同名>.txt` 修改 YOLO 格式标签 `类别 中心x 中心y 宽 高`（坐标归一化到 0–1），可使用 LabelImg 等标注工具。`manifest.csv` 的 `review_flag=1` 只是提醒，不代表其他框都正确；建议按类别与划分抽样，保存检查数和错误数。若生成中断，请处理错误后使用新的空输出目录重跑。

## 3. 训练

首次运行会由 Ultralytics 下载 COCO 预训练的 `rtdetr-l.pt`；随后在本数据上进行真实微调。请记录框架版本、预训练权重来源、训练参数和日志；下载失败时自行放置官方权重并使用 `--model 路径`。

```powershell
python train.py --device 0 --epochs 50 --batch 4 --imgsz 320
```

默认输出 `runs/detect/rps_baseline/weights/best.pt`。再次运行时框架会创建 `rps_baseline2` 等新目录，**以终端打印的 Best weights 路径为准**，通过后续 `--weights` 指定。显存不足可降低 `--batch`；CPU 使用 `--device cpu`。训练参数与曲线保存在运行目录；固定随机种子不保证不同硬件完全一致。

对照实验例子：保持其他参数一致，第二次训练加入 `--degrees 30 --name rps_rotate`，与默认 `--degrees 0` 比较，评估两个模型并如实记录；Ultralytics 的其他默认训练增强仍然存在，不应将框架默认增强当成自己的创新。

## 4. 测试集评估

```powershell
python evaluate.py --device 0 --weights runs/detect/rps_baseline/weights/best.pt
```

输出固定 test 集的 mAP50、mAP50–95、Precision、Recall、各类 AP50 和框架生成的图表。需先核查测试集伪框；除模型选择所用验证集以外，不要依据 test 反复调参。自采困难场景与统一来源测试结果分开报告。

## 5. 摄像头交互

```powershell
python camera.py --source 0 --device 0 --weights runs/detect/rps_baseline/weights/best.pt
```

每帧显示每个检测目标的类别、置信度和框，`Q` 或 `Esc` 退出。摄像头打不开可试 `--source 1`，或关闭占用摄像头的软件。通过 `--conf` 调整置信度阈值、`--imgsz` 调整输入尺寸。真实运行至少 300 帧后退出，`camera_metrics.json` 记录取帧到 `imshow` 返回的平均/P95耗时、模型推理耗时和总帧数/墙钟时间的 FPS；不包含 `waitKey` 在延迟内，且 `imshow` 返回并不保证物理屏幕已刷新。显示的单帧 FPS/延迟为上一帧值，不能代替统计。首次加载、预热和取帧会影响结果，报告须说明硬件和计时方法。视频文件也可通过 `--source 文件路径` 传入。

## 6. 已训练权重与测试结果

仓库包含 RTX 4050 上经过 5 轮真实微调的默认权重。独立测试集 mAP50 为 **0.8022**，其中 Rock / Scissors / Paper 的 AP50 分别为 **0.9950 / 0.8434 / 0.5681**。完整训练设置、实验结果及局限见 [实验记录](metrics/experiment.md)。这些是同源重建图像和伪标注上的结果，不代表真实摄像头和多人场景效果。

## 7. 验收与局限

实际录制 60 秒以上，逐类切换并测试至少 2 手、尝试 3 手以上，记录漏检、重复框和遮挡。任务书还要求正面、左侧、右侧、俯视和 ±30° 六组以及至少三类困难场景的量化测试、模型改进对照、报告和真实演示视频；这些均须现场采样、逐项记录，代码不自动保证完成。训练图主要是白底单手，与真实多手环境存在差异：如识别不佳，优先改善标注并补充独立采集与标注的训练图，务必记录来源和划分，绝不把测试素材混进训练。
