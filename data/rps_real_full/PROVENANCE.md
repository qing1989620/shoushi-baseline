# 合并数据来源与划分

- 统一任务数据来自 Laurence Moroney RPS 图像的 128×128 重建；原图来源、压缩包 SHA-256 与限制详见 `data/provenance.json`。`course_` 开头是这一来源，统一分类图上的框为背景分割伪标注。
- 额外图片及 YOLO 框来自 [Xuban/rock-paper-scissors-roboflow](https://huggingface.co/datasets/Xuban/rock-paper-scissors-roboflow)，上游为 Rock Paper Scissors 2 的 [Roboflow 数据集](https://universe.roboflow.com/rock-paper-scissors-2/rock-paper-scissors-bwev7)，原发布者标为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。原始标注 0=Rock、1=Paper、2=Scissors；本版本调整为 0=Rock、1=Scissors、2=Paper。图像经复制并重命名；标注行类别被映射，坐标不变。`real_` 开头是该来源。
- 上游 train 1485 张；使用上游保留的 100 张公开参考图逐图计算 pHash 距离，若距离≤10，排除候选所在的前缀+连续 15 个编号的整组，共排除 18 组 / 270 张。24 个触发候选及距离见 `excluded_near_holdout.csv`。这是降低相似帧泄漏的启发式，不能保证人物、源视频或近重复完全隔离。
- 保留上游真实图 1215 张，按同样的编号组，以随机种子 42 抽组生成 130 张真实验证集；其余 1085 张加入统一训练集。总 train=3353，val=130。统一原始 val=252 存 `images/official_val`，原始 test=372 存 `images/test`，均未混入训练。
- `manifest.csv` 列出每张图片、来源、划分、近似组和 SHA-256。构建时外部评估集用于相似帧筛除，故它不能被当作严格全新且完全独立的最终测试。应另行采集新参与者和新场景评估。

- `images/real_test` / `labels/real_test` 是上述 100 张参考图中排除 7 张白底合成风格图后的 93 张（149 个框）；`real_test.yaml` 用于评估。它们参与过 pHash 近重复排除，同一人物和场景可能相关，仅供参考，不能视为完全独立的新用户测试。`real_test_manifest.csv` 记录筛选与哈希。
