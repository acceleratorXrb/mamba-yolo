# Baseline Guarantee

后续所有时序改动都必须满足以下约束：

1. 必须始终保留官方原始单帧 `Mamba-YOLO` 可直接启动的训练路径。
2. 启动官方原始版本时，必须满足：
   - `temporal=false`
   - `temporal_consistency=false`
   - 不依赖任何时序输入
   - 直接走原始 `DetectionModel`
3. 时序相关改动只能作用于：
   - `temporal=true` 的配置
   - `TemporalDetectionModel`
   - 时序数据链路
4. 单帧基线配置固定保留在：
   - [visdrone_vid_singleframe_dev.yaml](/home/easyai/桌面/mamba-yolo3/configs/train/visdrone_vid_singleframe_dev.yaml)
5. 单帧基线启动脚本固定保留在：
   - [run_visdrone_vid_singleframe_baseline.sh](/home/easyai/桌面/mamba-yolo3/scripts/run_visdrone_vid_singleframe_baseline.sh)

这意味着：

- 后续可以持续迭代时序版本
- 但任何时候都必须能回到“官方原始 Mamba-YOLO 单帧训练”做对照实验
