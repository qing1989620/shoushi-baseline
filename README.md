# 石头剪刀布手势检测 Baseline

一套可供学生学习的 RT-DETR 目标检测基线：使用 `train.py` 训练模型，用 `camera.py` 调用训练好的模型进行实时手势检测。类别固定为 `0 Rock（石头）`、`1 Scissors（剪刀）`、`2 Paper（布）`。检测画面可以输出多个目标的框、类别与置信度，实际识别效果取决于手势、环境和硬件。

## 文件说明

- `train.py`：模型训练和测试集评估。
- `camera.py`：摄像头实时检测。
- `data/rps_real_full/`：已标注训练、验证和测试数据，详细来源见其中 `PROVENANCE.md`。
- `runs/detect/rps_final/weights/best.pt`：运行训练后生成的模型（仓库不提供训练好的手势权重）。
- `算法方向方案.pdf`：项目任务书，包含数据来源及仓库链接。

## 安装

建议 Python 3.10–3.12；有 NVIDIA 显卡时先安装与驱动匹配的 [PyTorch](https://pytorch.org/get-started/locally/)，然后运行：

```powershell
python -m pip install -r requirements.txt
```

若 Windows 上提示 `OpenCV GUI: NONE` 或 `cvShowImage`，可能是全局环境的多个 OpenCV 包发生冲突。可以用独立环境复用已安装的依赖，并单独安装支持窗口的 OpenCV：

```powershell
python -m venv --system-site-packages .venv-camera
.\.venv-camera\Scripts\python.exe -m pip install --ignore-installed --no-deps opencv-python==4.13.0.92
```

后续命令中的 `python` 可替换为 `.\.venv-camera\Scripts\python.exe`。

## 训练与评估

**仓库不包含训练好的手势模型，请先训练，再运行摄像头。**

```powershell
python train.py --device 0 --epochs 50 --batch 4 --imgsz 320 --workers 0
python train.py --evaluate --device 0 --workers 0
# 测试无手背景的误框比例（默认阈值 0.45）
python train.py --background-test --device 0 --workers 0
# 评估公开场景参考集（与统一来源 test 分开报告）
python train.py --evaluate --data data/rps_real_full/real_test.yaml --device 0 --workers 0
```

首次训练会从 Ultralytics 下载通用预训练权重 `rtdetr-l.pt` 作为初始化，再使用本项目数据训练；这不是仓库提供的已训练手势模型。训练输出路径由程序打印；已有同名目录时会创建新目录。用新权重评估或打开摄像头时，添加 `--weights 新权重路径`。训练默认禁用 Mosaic 以降低内存压力；旋转增强可用 `--degrees 30` 指定。

## 摄像头识别

训练结束后运行以下命令，默认加载 `runs/detect/rps_final/weights/best.pt`：

```powershell
python camera.py --device 0
```

如训练时生成了其他路径，请用 `--weights 训练输出的best.pt路径` 指定。按 Q 或 Esc 退出；摄像头编号不是 0 时使用 `--source 1`。无 NVIDIA 显卡可使用 `--device cpu`。默认置信度阈值为 `0.45`，并对重叠框进行去重：降低 `--conf` 可能减少漏检，但也可能增加误报；遮挡导致不同手的框高度重叠时，去重可能合并两个目标，可试 `--nms-iou 1` 关闭。程序退出后记录帧数、FPS、纯推理耗时及取帧至 `imshow` 返回的耗时，建议采集至少 300 帧。该时间不含 `waitKey`，`imshow` 返回也不等于屏幕实际完成刷新。

## 数据说明

数据集结合 Laurence Moroney 的石头剪刀布图像及 [Rock Paper Scissors 2 公开检测数据](https://universe.roboflow.com/rock-paper-scissors-2/rock-paper-scissors-bwev7)（[Hugging Face 镜像](https://huggingface.co/datasets/Xuban/rock-paper-scissors-roboflow)，发布者标注 CC BY 4.0）。原指定 `hand.tar` 未能获得，因此项目中的 128×128 同源重建图不能声称与该压缩包逐文件一致。其检测框由背景分割生成，仍是需要人工检查的伪标注；公开数据标签也可能存在错误。

训练集共 3,537 张，其中统一来源手势 2,268 张、公开来源手势 1,085 张、无手背景增强图 184 张（来自 23 张不同原图）；验证集为 130 张公开来源图；`test` 为 372 张统一来源保留测试图，`official_val` 为其原验证划分；`real_test` 为 93 张公开数据参考图。16 张不同于训练背景的无手图保留在 `background_test`，用于测量误框；公开图基于近似编号分组，并排除了部分与参考集近似的图片；该方法不能保证不同人的视频或近重复完全隔离，参考集含相似场景与连续帧，其评测不代表新用户摄像头的效果。来源、许可、划分与清单详见 `data/rps_real_full/PROVENANCE.md` 和 `manifest.csv`。现场采集的视频应与训练数据按人员和拍摄场次分开，且分别报告成绩。

无手图来自 [COCO 2017](https://cocodataset.org/) / [COCO128](https://docs.ultralytics.com/datasets/detect/coco128/)；原始 Flickr 图片链接、COCO 元数据标注的 CC BY 2.0 / CC BY-SA 2.0 许可、变换方式和哈希记录在 `data/rps_real_full/background_sources.csv`。COCO 不拥有图片版权，使用前须遵守各图片许可。

请按任务书实际测试不同方向、多人、复杂背景及 300 帧性能。公开数据和训练增强不保证能适应每台摄像头或每个人的姿势。

## 已知问题与学生改进方向

实际摄像头测试中，用户反馈模型会把人物等非手势目标误识别为手势。当前效果仅作为 baseline，仍需学生自行采集并标注不同人物、背景、距离和角度的数据，分析误检、漏检及类别混淆。包含真实手势的图片须标注所有可见手势，不能简单作为无手背景；无手画面可作为负样本。训练和测试应按人员、场景或拍摄批次划分，并用未参与训练及调参的新场景分别报告误框率和手势召回率。

## 本次模型测试

以下为开发期间一次训练的历史参考结果，**仓库不提供该手势模型权重**；重新训练得到的结果可能不同。该实验在此前 8 轮 RT-DETR 模型基础上加入无手背景样本微调 4 轮，使用 RTX 4050 Laptop GPU、320 输入、batch 4、AdamW、学习率 0.0001 和 `warmup=0.5`。

| 测试 | 本次结果 |
|---|---|
| 同源保留测试集：372 张、伪标注框 | mAP50 **0.9870**；mAP50–95 **0.7359** |
| 公开场景参考集：93 张、149 个框 | mAP50 **0.7199**；mAP50–95 **0.4022** |
| 保留的无手背景：16 张，默认置信度 0.45 | **1 / 16** 张出现误框（6.25%） |
| 另一来源绿底真人手势：每类 60 张，阈值 0.45 | 最高置信框的整图类别正确率 **72.2%**（130 / 180） |

修复前，COCO128 中筛出的 65 张无手参考图有 **55 / 65** 张出现误框；同一批图、相同阈值 0.25 下，本次模型为 **6 / 65**。默认阈值 0.45 下为 **4 / 65**。这批图片在开发时已查看和用于阈值分析，不属于严格盲测；16 张保留背景图包含在其中，数量也较少。

在 130 张含手验证图上，按类别和 IoU≥0.5 匹配，默认阈值 0.45 的召回为 **182 / 213（85.4%）**；修复前、原阈值 0.25 的召回为 **194 / 213（91.1%）**。减少误框存在一定漏检代价，实际情况随场景而变。上表的 mAP 使用评估接口的原始检测结果；摄像头端另做重叠框去重。

绿底图片来自 [Julien de la Bruère-Terreault 的 Rock Paper Scissors 数据集](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)（来源标注 CC BY-SA 4.0），每类按文件名字典序取前 60 张；它只有图像级标签，没有检测框，不能与检测 mAP 比较。其背景和姿势较单一。公开场景参考集参与过近重复筛除，也不能视为严格独立于训练来源。真实摄像头测试只验证了窗口与模型加载，实际无手误框、多人效果和不同环境仍需现场测试；模型不能保证任何场景都不误识别。
