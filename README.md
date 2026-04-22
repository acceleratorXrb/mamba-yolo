

#毕业设计要求
* 主要研究内容

第一，设计时空混合架构，将YOLO的CNN骨干与Mamba模块深度融合，CNN提取空间局部特征，Mamba处理时序连续帧间的长程依赖，捕捉小目标的运动轨迹与上下文关联。第二，开发自适应扫描机制，针对无人机俯拍视角中目标分布稀疏的特点，让Mamba动态调整扫描路径，聚焦潜在目标区域，提升计算效率。第三，构建多尺度状态传递网络，在特征金字塔（FPN）中嵌入轻量化Mamba单元，增强不同尺度特征间的信息融合，专门优化极小目标的特征表示。第四，建立完整的评估体系，在VisDrone、UAVDT等无人机数据集上，从检测精度（特别是小目标AP）、时序一致性、推理速度等多维度验证模型效能，并进行详实的模块消融实验。
* 目标和要求 

一、理论基础：深入理解状态空间模型（SSM）的选择性扫描机制、YOLO架构及视频目标检测特性；二、实验规范：采用固定随机种子，在相同硬件环境下进行3次重复实验取均值，与SOTA方法公平对比；三、分析深度：必须进行模块消融实验与可视化分析（如Mamba状态热力图），阐明性能增益来源；四、工程实现：提供完整训练/推理代码、详细配置文件及模型权重，确保可复现性；五、创新性：在混合架构设计、时序特征融合或扫描策略上体现明确创新点。
* 特色

第一，场景强驱动：摒弃通用检测框架的宽泛优化，精准锚定“无人机视频流”与“小目标”两大核心约束，使研究问题尖锐，技术设计有的放矢。第二，时空双优架构：创新性地将擅长局部感知的CNN与擅长长序列建模的Mamba有机融合，构建“空间提取-时序推理”双支路协同的混合骨干网络，同时提升单帧识别精度与帧间关联一致性。第三，动态稀疏感知：充分利用无人机视频背景相对静止、目标稀疏的特点，设计自适应的选择性扫描机制，让模型计算资源智能聚焦于运动区域，实现精度与效率的平衡。第四，端到端可部署：整个研究以实际部署为导向，模型设计兼顾精度与速度，提供从训练、验证到轻量化部署（如TensorRT）的完整技术链，确保研究成果不仅停留在论文，更具备落地应用的生命力。

* 成果价值 

学术上，首次将Mamba架构系统性地应用于无人机视频目标检测领域，探索了状态空间模型处理视觉时序信号的潜力，为“CNN+SSM”这一新兴架构范式提供了重要的实证案例与性能基准。其提出的自适应扫描与多尺度状态传递机制，可为视频理解、长序列分析等研究方向提供新思路。工程上，直接针对无人机巡检、安防监控、智慧交通等现实场景中“小、快、密”目标检测的痛点，提供精度更高、时序更稳的解决方案。所开发的模型与优化策略，能有效提升现有无人机视觉系统的自动化水平，具有明确的产业转化前景。生态上，开源高质量的代码、训练配置与模型权重，能推动无人机视觉与高效架构研究社区的发展，促进相关技术的快速迭代与落地。

## 当前训练命令与超参位置

### 一次性准备环境与数据

```bash
bash scripts/deploy_project.sh all
```

作用：
- 创建项目内环境 `.conda/mambayolo`
- 安装主工程依赖
- 编译 `selective_scan`
- 准备 `VisDrone-VID`
- 准备 `YOLOFT-S` 本地适配数据
- 尝试准备 `UAVDT`

### 1. VisDrone-VID 时序版 Mamba-YOLO

训练命令：

```bash
bash scripts/deploy_project.sh train-visdrone-temporal
```

超参修改位置：

```bash
configs/train/visdrone_vid_temporal_dev.yaml
```

主要可改项：
- `epochs`
- `batch`
- `imgsz`
- `workers`
- `optimizer`
- `lr0`
- `lrf`
- `weight_decay`
- `warmup_epochs`
- `amp`
- `val`
- `val_interval`
- `temporal`
- `temporal_stride`
- `temporal_clip_length`
- `temporal_consistency`
- `temporal_consistency_weight`

### 2. VisDrone-VID 单帧版官方原始 Mamba-YOLO

训练命令：

```bash
bash scripts/deploy_project.sh train-visdrone-singleframe
```

超参修改位置：

```bash
configs/train/visdrone_vid_singleframe_dev.yaml
```

主要可改项：
- `epochs`
- `batch`
- `imgsz`
- `workers`
- `optimizer`
- `lr0`
- `lrf`
- `weight_decay`
- `warmup_epochs`
- `amp`
- `val`
- `val_interval`
- 各类增强参数

### 3. UAVDT 时序版 Mamba-YOLO

训练命令：

```bash
bash scripts/deploy_project.sh train-uavdt-temporal
```

超参修改位置：

```bash
configs/train/uavdt_full_benchmark_temporal_dev_img640_adamw.yaml
```

主要可改项：
- `epochs`
- `batch`
- `imgsz`
- `workers`
- `optimizer`
- `lr0`
- `lrf`
- `weight_decay`
- `warmup_epochs`
- `amp`
- `val`
- `val_interval`
- `temporal`
- `temporal_stride`
- `temporal_clip_length`
- `temporal_consistency`
- `temporal_consistency_weight`

### 4. YOLOFT-S 对照模型（VisDrone-VID）

训练命令：

```bash
bash scripts/deploy_project.sh train-yoloft-s
```

训练超参修改位置：

```bash
third_party/YOLOFT/config/train/orige_stream_visdrone_local.yaml
```

这里控制：
- `epochs`
- `batch`
- `imgsz`
- `workers`
- `optimizer`
- `lr0`
- `lrf`
- `weight_decay`
- `warmup_epochs`
- `amp`
- `val`
- `val_interval`
- `save_period`
- 各类增强参数

数据与视频切分相关参数位置：

```bash
third_party/YOLOFT/config/visdrone2019VID_local_10cls.yaml
```

这里控制：
- `path`
- `train`
- `val`
- `test`
- `labels_dir`
- `images_dir`
- `split_length`
- `match_number`
- `interval`
- `rho`
- `names`

模型结构位置：

```bash
third_party/YOLOFT/config/yoloft/yoloft-S.yaml
```

### 评测命令

主工程 VisDrone-VID 本地评测：

```bash
bash scripts/deploy_project.sh eval-visdrone
```

主工程 VisDrone-VID 官方格式导出 / 官方 toolkit 评测：

```bash
bash scripts/deploy_project.sh eval-visdrone-official
```

YOLOFT-S 训练前准备：

```bash
bash scripts/deploy_project.sh prepare-yoloft
```

### 常用超参作用说明

下面这些超参在主工程 `Mamba-YOLO` 和 `YOLOFT` 中基本都存在，含义相近。

- `epochs`
  - 总训练轮数。越大训练越久，通常上限更高，但也更容易过拟合。
- `batch`
  - 每次迭代的样本数。越大通常吞吐更高，但更占显存。
- `imgsz`
  - 输入分辨率。越大通常对小目标更友好，但训练和推理都会更慢、更占显存。
- `workers`
  - dataloader 的并行加载进程数。太小可能喂不满 GPU，太大也可能引入额外开销。
- `optimizer`
  - 优化器类型。常见是 `AdamW` 或 `SGD/auto`，会影响收敛速度和稳定性。
- `lr0`
  - 初始学习率。过大容易震荡，过小会收敛太慢。
- `lrf`
  - 最终学习率系数，决定学习率从初始值衰减到什么水平。
- `weight_decay`
  - 权重衰减，控制正则化强度。过小容易过拟合，过大可能压制学习。
- `warmup_epochs`
  - 预热轮数。训练初期逐步升高学习率，减少前几轮不稳定。
- `warmup_bias_lr`
  - bias 参数的预热学习率，主要影响训练初期分类头和回归头的稳定性。
- `amp`
  - 自动混合精度。打开后通常更省显存、更快，但部分自定义算子可能不稳定。
- `val`
  - 是否在训练过程中做验证。
- `val_interval`
  - 每多少个 `epoch` 验证一次。越小越容易及时看到指标，但总训练时间更长。
- `save_period`
  - 每隔多少个 `epoch` 额外保存一次权重。
- `fraction`
  - 只使用数据集的某个比例。适合 smoke test，不适合正式实验。

主工程时序版额外常见超参：

- `temporal`
  - 是否启用时序模型。`false` 时走官方原始单帧 `Mamba-YOLO`。
- `temporal_stride`
  - 邻帧采样步长。`1` 表示取相邻帧，值越大表示时间间隔越大。
- `temporal_clip_length`
  - 时序 clip 长度。当前主线一般用 `3` 帧。
- `temporal_consistency`
  - 是否启用时序一致性损失。
- `temporal_consistency_weight`
  - 时序一致性损失权重。过大可能压制检测主任务，过小可能不起作用。

YOLOFT 数据与视频切分相关超参：

- `split_length`
  - 每个视频段切分长度。影响流式训练时单个子视频的长度。
- `match_number`
  - 邻近帧匹配/取样数量，影响时序关联强度。
- `interval`
  - 帧采样间隔。值越大表示跨得越远。
- `rho`
  - YOLOFT 数据流构造时的额外时序控制参数，影响样本组织方式。
- `labels_dir`
  - 标签目录名。
- `images_dir`
  - 图像目录名。
- `names`
  - 数据集类别名称列表。

常见调参建议：

- 想提速：
  - 优先调 `batch`、`amp`、`val_interval`
- 想提高小目标效果：
  - 优先观察 `imgsz`、学习率和训练轮数
- 想保证时序稳定：
  - 优先观察 `temporal_clip_length`、`temporal_stride`、`temporal_consistency_weight`
- 想做 smoke test：
  - 用更小的 `epochs`、`batch`、`workers`，并把 `fraction` 设成小值
