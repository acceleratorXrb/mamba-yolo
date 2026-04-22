# DEPLOYMENT

## 1. 拉取仓库

```bash
git clone <你的仓库地址>
cd mamba-yolo3
```

作用：获取项目源码并进入项目目录。

## 2. 一键准备训练前所需内容

```bash
bash scripts/deploy_project.sh all
```

作用：
- 创建项目内环境 `.conda/mambayolo`
- 按本机 `nvcc` 自动安装匹配的 `PyTorch`
- 编译 `selective_scan`
- 尝试准备 `UAVDT`
- 准备 `VisDrone-VID train/val/test-dev + toolkit`
- 准备 `YOLOFT-S` 本地对照线依赖与数据

说明：
- `UAVDT` 下载失败时不会阻塞 `VisDrone-VID`
- `UAVDT` 下载失败时也不会阻塞 `YOLOFT-S`
- `all` 执行完后，可以直接启动主工程训练或 `YOLOFT-S` 对照训练
- `all` 结束后会自动打印部署摘要，直接显示 `VisDrone-VID test-dev`、`YOLOFT`、`UAVDT`、官方 toolkit 运行时是否就绪

## 3. 单独准备 VisDrone-VID

```bash
bash scripts/deploy_project.sh prepare-visdrone
```

作用：只准备 `VisDrone-VID train/val/test-dev + toolkit`，不处理 `UAVDT`。

## 4. 启动 VisDrone-VID 时序训练

```bash
bash scripts/deploy_project.sh train-visdrone-temporal
```

作用：启动当前项目的时序版 `VisDrone-VID` 训练。

超参位置：

```bash
configs/train/visdrone_vid_temporal_dev.yaml
```

作用：修改 `epochs`、`batch`、`workers`、`optimizer`、`amp`、`val_interval`、时序参数和数据增强。

## 5. 启动 VisDrone-VID 单帧基线

```bash
bash scripts/deploy_project.sh train-visdrone-singleframe
```

作用：启动官方原始单帧 `Mamba-YOLO-T` 基线训练。

超参位置：

```bash
configs/train/visdrone_vid_singleframe_dev.yaml
```

作用：修改单帧基线训练超参。

## 6. 评测 VisDrone-VID

```bash
bash scripts/deploy_project.sh eval-visdrone
```

作用：对当前 `VisDrone-VID` 最佳权重做独立评测，输出：
- `AP`
- `AP50`
- `AP75`
- 分类别指标

## 7. 导出 VisDrone-VID 官方格式并在可用时调用官方 toolkit

```bash
bash scripts/deploy_project.sh eval-visdrone-official
```

作用：
- 对 `VisDrone-VID test-dev` 导出官方逐视频结果文本
- 如果本机安装了 `octave` 或 `matlab`，继续调用官方 `VID toolkit`
- 如果本机没有 `octave/matlab`，只导出官方格式结果，不会中断

输出目录：

```bash
output_dir/visdrone_vid_official/temporal_dev
```

关键输出：

```bash
output_dir/visdrone_vid_official/temporal_dev/test_official_txt
output_dir/visdrone_vid_official/temporal_dev/test_official_export_summary.json
output_dir/visdrone_vid_official/temporal_dev/test_official_metrics.json
```

说明：
- `test_official_metrics.json` 只有在本机可执行官方 toolkit 时才会生成
- 官方 toolkit 需要 `octave` 或 `matlab`

## 8. 单独准备 YOLOFT-S 对照线

```bash
bash scripts/deploy_project.sh prepare-yoloft
```

作用：
- 给当前项目环境补齐 `YOLOFT` 所需依赖
- 生成 `YOLOFT` 本地 `VisDrone-VID` 适配数据

说明：
- 如果你已经执行过 `bash scripts/deploy_project.sh all`，这里通常不需要再单独执行

## 9. 启动 YOLOFT-S 本地对照训练

```bash
bash scripts/deploy_project.sh train-yoloft-s
```

作用：启动 `third_party/YOLOFT` 下的 `YOLOFT-S` 本地对照训练。

对应文件：

```bash
third_party/YOLOFT/scripts/run_yoloft_s_visdrone_local.sh
third_party/YOLOFT/config/train/orige_stream_visdrone_local.yaml
third_party/YOLOFT/config/visdrone2019VID_local_10cls.yaml
```

作用：
- 启动脚本
- YOLOFT 本地训练超参
- YOLOFT 本地数据配置

## 10. 启动 UAVDT 时序训练

```bash
bash scripts/deploy_project.sh train-uavdt-temporal
```

作用：启动 `full UAVDT` 时序训练。

超参位置：

```bash
configs/train/uavdt_full_benchmark_temporal_dev_img640_adamw.yaml
```

作用：修改 `UAVDT` 时序训练超参。

## 11. 导出 UAVDT 官方 DET 结果

```bash
bash scripts/deploy_project.sh export-uavdt-det
```

作用：导出 `UAVDT` 官方 `DET toolkit` 所需结果格式。

## 12. 主脚本完整命令列表

```bash
bash scripts/deploy_project.sh setup-env
bash scripts/deploy_project.sh prepare-uavdt
bash scripts/deploy_project.sh prepare-visdrone
bash scripts/deploy_project.sh prepare-yoloft
bash scripts/deploy_project.sh prepare-data
bash scripts/deploy_project.sh train-visdrone-singleframe
bash scripts/deploy_project.sh train-visdrone-temporal
bash scripts/deploy_project.sh train-uavdt-temporal
bash scripts/deploy_project.sh train-yoloft-s
bash scripts/deploy_project.sh eval-visdrone
bash scripts/deploy_project.sh eval-visdrone-official
bash scripts/deploy_project.sh export-uavdt-det
bash scripts/deploy_project.sh all
```

作用：这是当前仓库所有可直接执行的部署/训练/评测入口。
