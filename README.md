# 石头剪刀布手势检测 Baseline

一套可供学生学习的 RT-DETR 目标检测基线：使用 `train.py` 训练模型，用 `camera.py` 调用训练好的模型进行实时手势检测。类别固定为 `0 Rock（石头）`、`1 Scissors（剪刀）`、`2 Paper（布）`。检测画面可以输出多个目标的框、类别与置信度，实际识别效果取决于手势、环境和硬件。

## 文件说明

- `train.py`：模型训练和测试集评估。
- `camera.py`：摄像头实时检测。
- `data/rps_real_full/`：已标注训练、验证和测试数据，详细来源见其中 `PROVENANCE.md`。
- `runs/detect/rps_final/weights/best.pt`：训练好的模型。
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

## 摄像头识别

```powershell
python camera.py --device 0
```

按 Q 或 Esc 退出；摄像头编号不是 0 时使用 `--source 1`。无 NVIDIA 显卡可使用 `--device cpu`；降低 `--conf` 会增加检出，同时可能产生更多误报。程序退出后记录帧数、FPS、纯推理耗时及取帧至 `imshow` 返回的耗时，建议采集至少 300 帧。该时间不含 `waitKey`，`imshow` 返回也不等于屏幕实际完成刷新。

## 训练与评估

```powershell
python train.py --device 0 --epochs 50 --batch 4 --imgsz 320 --workers 0
python train.py --evaluate --device 0 --workers 0
# 评估公开场景参考集（与统一来源 test 分开报告）
python train.py --evaluate --data data/rps_real_full/real_test.yaml --device 0 --workers 0
```

首次训练会下载 Ultralytics 的 `rtdetr-l.pt` 预训练权重。训练输出路径由程序打印；已有同名目录时会创建新目录。用新权重评估或打开摄像头时，添加 `--weights 新权重路径`。训练默认禁用 Mosaic 以降低内存压力；旋转增强可用 `--degrees 30` 指定。

## 数据说明

数据集结合 Laurence Moroney 的石头剪刀布图像及 [Rock Paper Scissors 2 公开检测数据](https://universe.roboflow.com/rock-paper-scissors-2/rock-paper-scissors-bwev7)（[Hugging Face 镜像](https://huggingface.co/datasets/Xuban/rock-paper-scissors-roboflow)，发布者标注 CC BY 4.0）。原指定 `hand.tar` 未能获得，因此项目中的 128×128 同源重建图不能声称与该压缩包逐文件一致。其检测框由背景分割生成，仍是需要人工检查的伪标注；公开数据标签也可能存在错误。

训练集共 3,353 张，其中统一来源 2,268 张、公开来源 1,085 张；验证集为 130 张公开来源图；`test` 为 372 张统一来源保留测试图，`official_val` 为其原验证划分；`real_test` 为 93 张公开数据参考图。公开图基于近似编号分组，并排除了部分与参考集近似的图片；该方法不能保证不同人的视频或近重复完全隔离，参考集含相似场景与连续帧，其评测不代表新用户摄像头的效果。来源、许可、划分与清单详见 `data/rps_real_full/PROVENANCE.md` 和 `manifest.csv`。现场采集的视频应与训练数据按人员和拍摄场次分开，且分别报告成绩。

请按任务书实际测试不同方向、多人、复杂背景及 300 帧性能。公开数据和训练增强不保证能适应每台摄像头或每个人的姿势。

## 本次模型测试

仓库中的权重从 `rtdetr-l.pt` 开始微调 8 轮，使用 RTX 4050 Laptop GPU、输入 320、batch 4、`workers=0`。实际测试结果：

| 测试 | 结果 |
|---|---|
| 同源保留测试集（372 张，自动框） | mAP50 **0.9950**；mAP50–95 **0.7594** |
| 公开场景参考集（93 张，149 个框） | mAP50 **0.7996**；mAP50–95 **0.4717**；Rock / Scissors / Paper AP50 分别为 **0.7868 / 0.7202 / 0.8918** |
| 另一来源绿底真人手势（每类 60 张） | 最高置信框的整图类别正确率 **53.3%**；Rock / Scissors / Paper 分别为 **36.7% / 23.3% / 100%** |

绿底图片来自 [Julien de la Bruère-Terreault 的 Rock Paper Scissors 数据集](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)（来源标注 CC BY-SA 4.0），每类按文件名字典序取前 60 张，阈值 0.25；它只有图像级标签，没有检测框，不能与检测 mAP 比较。图片场景和姿势较单一，不能代表普遍摄像头准确率。公开场景参考集参与过近重复筛除，亦不能视为严格独立于训练来源。真实摄像头测试暂只验证窗口与模型加载，不代表已完成准确率和多人测量；本模型对某些手势方向仍会误判。
