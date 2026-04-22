# METHOD

## 1. 文档目的

本文档专门说明当前项目相对**官方原始 `Mamba-YOLO`** 的模型级、数据级、损失级和评测级改动。  
重点不是泛泛描述，而是回答 4 个问题：

1. 官方 `Mamba-YOLO` 被改了哪些部分
2. 每一处改动想解决什么问题
3. 代码具体落在什么位置
4. 当前实现与论文目标模型之间的关系是什么

---

## 2. 官方原始 `Mamba-YOLO` 的基线形态

官方原始版本的核心特征是：

- 单帧输入
- 官方 `Mamba-YOLO-T/L` 主干与颈部
- 官方 `Detect` 检测头
- 标准 Ultralytics 检测训练与验证流程

在本项目中，始终保留了**官方原始单帧 baseline 可单独启动**的能力。  
这条基线训练入口通过脚本保留：

- [scripts/run_visdrone_vid_singleframe_baseline.sh](/home/easyai/桌面/mamba-yolo3/scripts/run_visdrone_vid_singleframe_baseline.sh)
- [scripts/deploy_project.sh](/home/easyai/桌面/mamba-yolo3/scripts/deploy_project.sh)

也就是说，当前仓库不是直接“覆盖官方实现”，而是在官方实现之上增加了一条**可开关的时序增强路径**。

---

## 3. 总体改进思路

相对官方原始 `Mamba-YOLO`，当前版本的核心目标是：

- 从单帧检测器扩展为视频检测器
- 保持官方主干和检测头尽量不被破坏
- 用最小侵入方式引入：
  - 三帧时序输入
  - 空间/时序双支路融合
  - 多尺度时序状态传递
  - 时序一致性约束
  - `VisDrone-VID` 的 COCO 风格评测

从方法角度看，当前版本可概括为：

> 在官方 `Mamba-YOLO` 骨架上，引入了三帧视频输入、显式空间主支路与时序副支路、跨尺度状态传播以及时序一致性正则，从而将原始单帧检测模型扩展为面向视频流的时空联合检测框架。

---

## 4. 数据链路改动

### 4.1 新增时序数据集

新增类：

- [official-mamba-yolo/ultralytics/data/dataset.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/data/dataset.py)
  - `YOLOTemporalDataset`

主要作用：

- 把原来的单帧样本扩展为奇数长度时序 clip
- 当前默认使用：
  - `temporal_clip_length = 3`
  - `temporal_stride = 1`
- 为每个样本生成：
  - `temporal_imgs`
  - `temporal_valid`
  - `sequence_id`

其中：

- `temporal_imgs`：保存 `t-1, t, t+1`
- `temporal_valid`：标记哪些邻帧是真实存在的，哪些只是回退填充

代码位置：

- [official-mamba-yolo/ultralytics/data/dataset.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/data/dataset.py)

### 4.2 修复时序增强错位

时序模型最关键的问题之一，是中心帧和邻帧必须经过**同一套增强**。  
如果中心帧做了仿射变换，而邻帧没做同样变换，时序信息会直接失真。

为此新增了专门的时序增强构造：

- [official-mamba-yolo/ultralytics/data/augment.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/data/augment.py)
  - `temporal_v8_transforms`

当前策略：

- 保留 `RandomPerspective`
- 保留 `RandomHSV`
- 保留 `RandomFlip`
- 去掉会破坏跨帧对应关系的：
  - `Mosaic`
  - `MixUp`
  - `Albumentations`

这样做的原因是：

- 单帧检测增强追求样本多样性
- 时序检测增强更强调帧间对齐

---

## 5. 模型主链改动

### 5.1 新增时序模型入口

新增类：

- [official-mamba-yolo/ultralytics/nn/tasks.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/nn/tasks.py)
  - `TemporalDetectionModel`

作用：

- 保留官方 backbone + neck + detect head
- 在 `Detect` 之前插入时序融合模块
- 在 `temporal=true` 时替换原始单帧检测模型

核心设计原则：

- **不重写官方 Detect head**
- **不破坏单帧 baseline**
- **时序增强只在 `temporal=true` 时启用**

### 5.2 多帧前向逻辑

核心函数：

- [official-mamba-yolo/ultralytics/nn/tasks.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/nn/tasks.py)
  - `_forward_to_detect`
  - `_predict_temporal_clip`

工作流程：

1. 对当前帧提取官方检测特征
2. 对邻帧提取同尺度检测特征
3. 把每个尺度的三帧特征堆叠成 `[B, T, C, H, W]`
4. 对每个尺度单独进行时序融合
5. 最终仍交回官方 `Detect` 头输出

也就是说：

- 当前模型不是“检测头后做后处理”
- 而是“检测头前做时序特征增强”

---

## 6. 时序模块改动

所有核心时序模块位于：

- [official-mamba-yolo/ultralytics/nn/modules/mamba_yolo.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/nn/modules/mamba_yolo.py)

### 6.1 `TemporalStateTransfer`

作用：

- 从三帧 clip 中提取时序状态
- 通过扫描式状态聚合得到 `temporal_state`

特点：

- 支持 `mean / weighted / scan`
- 当前默认：
  - `fusion_mode = scan`
- 在扫描时显式使用 `temporal_valid`
  - 避免缺失邻帧的占位图污染主时序状态

为什么要它：

- 这是时序支路的核心
- 负责把三帧信息压缩成当前帧可用的时序表征

代码位置：

- [official-mamba-yolo/ultralytics/nn/modules/mamba_yolo.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/nn/modules/mamba_yolo.py)
  - `class TemporalStateTransfer`

### 6.2 `SpatialTemporalFusionBlock`

作用：

- 显式实现“空间主支路 + 时序副支路”的门控融合

输入：

- `spatial_feat`：当前帧空间特征
- `temporal_feat`：由 `TemporalStateTransfer` 得到的时序特征

输出：

- `fused`
- `fusion_gate`

为什么新增这一层：

- 之前的时序增强更偏“隐式融合”
- 不够符合论文里“空间提取-时序推理双支路协同”的表述
- 现在把两条分支的职责显式拆开

代码位置：

- [official-mamba-yolo/ultralytics/nn/modules/mamba_yolo.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/nn/modules/mamba_yolo.py)
  - `class SpatialTemporalFusionBlock`

### 6.3 `AdaptiveSparseGuide`

作用：

- 生成稀疏引导图 `guide`

引导信号由三部分组成：

1. 帧间差分 `diff`
2. 当前特征显著性 `saliency`
3. 可选上一帧检测热图 `prev_det_heatmap`

为什么保留：

- 小目标和局部变化区域更依赖显式引导
- 这条引导图后续会用于稀疏扫描与时序细化

代码位置：

- [official-mamba-yolo/ultralytics/nn/modules/mamba_yolo.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/nn/modules/mamba_yolo.py)
  - `class AdaptiveSparseGuide`

### 6.4 `SparseGuidedTemporalScan`

作用：

- 用 `guide` 对融合特征做稀疏强调

作用理解：

- 不是对整张特征图平均处理
- 而是突出运动明显、响应显著的区域

代码位置：

- [official-mamba-yolo/ultralytics/nn/modules/mamba_yolo.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/nn/modules/mamba_yolo.py)
  - `class SparseGuidedTemporalScan`

### 6.5 `TemporalGuidedXSSBlock`

作用：

- 在时序状态注入后，再经过一个 `XSSBlock`
- 对融合结果进行进一步结构化建模

理解方式：

- 这一步不是简单卷积后结束
- 而是让时序状态继续进入 Mamba 风格模块

代码位置：

- [official-mamba-yolo/ultralytics/nn/modules/mamba_yolo.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/nn/modules/mamba_yolo.py)
  - `class TemporalGuidedXSSBlock`

### 6.6 `TemporalFusionScale`

作用：

- 组织单个尺度上的完整时序融合流程

它把以下模块串起来：

1. `TemporalStateTransfer`
2. `SpatialTemporalFusionBlock`
3. `AdaptiveSparseGuide`
4. `SparseGuidedTemporalScan`
5. `TemporalGuidedXSSBlock`

也就是说，一个尺度上的完整逻辑是：

- 当前帧特征 -> 空间支路
- clip 特征 -> 时序支路
- 双支路门控融合
- 稀疏引导增强
- 时序指导下的 XSS 细化

代码位置：

- [official-mamba-yolo/ultralytics/nn/modules/mamba_yolo.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/nn/modules/mamba_yolo.py)
  - `class TemporalFusionScale`

### 6.7 `MultiScaleTemporalStateBlock`

作用：

- 实现跨尺度时序状态传递

逻辑：

- 从粗尺度向细尺度做 top-down 传播
- 高层时序状态指导低层特征

为什么新增：

- 这是当前实现里最接近论文目标中“多尺度状态传递网络”的部分
- 让高层时序上下文不仅停留在高层，而能下传到小目标更敏感的细粒度层

代码位置：

- [official-mamba-yolo/ultralytics/nn/modules/mamba_yolo.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/nn/modules/mamba_yolo.py)
  - `class MultiScaleTemporalStateBlock`
- [official-mamba-yolo/ultralytics/nn/tasks.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/nn/tasks.py)
  - `_predict_temporal_clip`

---

## 7. 可选检测先验分支

当前实现里还保留了一条可开关的“上一帧检测先验”路径：

- `_decode_frame_detections`
- `_build_prev_det_heatmaps`

代码位置：

- [official-mamba-yolo/ultralytics/nn/tasks.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/nn/tasks.py)

用途：

- 把上一帧检测框 rasterize 成热图
- 作为 `AdaptiveSparseGuide` 的额外先验输入

当前状态：

- 默认关闭：
  - `temporal_prev_det_prior = false`

关闭原因：

- 训练期需要额外解码和 `NMS`
- 明显拖慢训练
- 早期先验噪声较大，可能反向污染 guide

因此当前仓库里：

- 这条分支代码仍然保留
- 但默认不启用

---

## 8. 损失函数改动

### 8.1 新增 `temporal_consistency_loss`

位置：

- [official-mamba-yolo/ultralytics/utils/loss.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/utils/loss.py)
  - `temporal_consistency_loss`

作用：

- 对 `center_feat` 和 `temporal_state` 做特征级一致性约束

当前形式：

- 使用 cosine distance
- 并且只在“稳定区域”上约束

稳定区域定义：

- 用 `guide` 反向构造 `stable_mask`
- 运动变化大的位置少约束
- 避免过度压制真实运动目标

还会结合：

- `temporal_valid`

作用：

- 只有真实邻帧存在时，这个损失才参与
- 避免无效邻帧占位图引入伪监督

整体损失变为：

- `box_loss`
- `cls_loss`
- `dfl_loss`
- `temporal_consistency_loss`

---

## 9. 配置系统改动

新增/扩展的重要配置项：

- `temporal`
- `temporal_stride`
- `temporal_clip_length`
- `temporal_consistency`
- `temporal_consistency_weight`
- `temporal_prev_det_prior`
- `temporal_multiscale_state`
- `visdrone_vid_coco_metrics`

代码位置：

- [official-mamba-yolo/ultralytics/cfg/default.yaml](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/cfg/default.yaml)
- [official-mamba-yolo/ultralytics/cfg/__init__.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/cfg/__init__.py)

这些配置项的作用是：

- 让官方单帧 baseline 和时序版共享同一套训练入口
- 通过开关切换，不需要分裂成两个完全不同的工程

---

## 10. 训练器与验证器改动

### 10.1 训练器

位置：

- [official-mamba-yolo/ultralytics/models/yolo/detect/train.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/models/yolo/detect/train.py)

主要改动：

- 扩展 loss 名称
- 允许记录：
  - `temporal_consistency_loss`
- 在 `VisDrone-VID` 场景下允许把额外指标写入 `results.csv`

### 10.2 验证器

位置：

- [official-mamba-yolo/ultralytics/models/yolo/detect/val.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/models/yolo/detect/val.py)

主要改动：

- 新增 `VisDrone-VID` 的 COCO 风格评测适配
- 额外输出：
  - `AP`
  - `AP50`
  - `AP75`
- 额外输出每类：
  - `P`
  - `R`
  - `mAP50`
  - `mAP50-95`
  - `AP`
  - `AP50`
  - `AP75`

输出文件：

- `visdrone_vid_val_per_class_metrics.json`

---

## 11. 数据集与评测脚本改动

### 11.1 新增 `VisDrone-VID` 数据准备脚本

- [scripts/download_visdrone_vid.py](/home/easyai/桌面/mamba-yolo3/scripts/download_visdrone_vid.py)
- [scripts/prepare_visdrone_vid.py](/home/easyai/桌面/mamba-yolo3/scripts/prepare_visdrone_vid.py)

作用：

- 下载、解压、整理 `VisDrone-VID`
- 转成适合当前时序数据集读取的格式

### 11.2 新增 `VisDrone-VID` 评测脚本

- [scripts/evaluate_visdrone_vid.py](/home/easyai/桌面/mamba-yolo3/scripts/evaluate_visdrone_vid.py)

作用：

- 离线计算 `AP / AP50 / AP75`
- 输出分类别指标

### 11.3 新增 `UAVDT` 官方 DET 导出脚本

- [scripts/evaluate_uavdt_official_det.py](/home/easyai/桌面/mamba-yolo3/scripts/evaluate_uavdt_official_det.py)

作用：

- 导出官方 `UAVDT DET toolkit` 所需结果格式

---

## 12. 当前实现与论文目标模型的关系

当前实现已经完成了：

- 三帧输入
- 时序数据对齐增强
- 空间/时序双支路融合
- 多尺度状态传递
- 时序一致性损失
- `VisDrone-VID` 正式指标评测

但需要实事求是地说：

当前版本仍然更接近一个**工程上可运行的研究原型**，而不是论文最终版的完全终态。

主要原因：

- 仍然以“官方 backbone/neck + 时序增强模块”方式实现
- 不是从 backbone 最早层就重构出完全独立的双支路主干
- 上一帧检测先验分支目前默认关闭
- 还有进一步收敛结构与正式 benchmark 的空间

因此，论文里更稳妥的写法是：

- 把当前版本描述为：
  - 在官方 `Mamba-YOLO` 基础上构建的时空增强检测框架
- 而不是声称：
  - 已经完全替代官方骨干为全新主干

---

## 13. 推荐论文表述

可直接使用的简化表述如下：

### 13.1 总体表述

> 本文在官方 `Mamba-YOLO` 基础上，引入面向视频流目标检测的时空增强机制。具体而言，首先构建以当前帧特征为主的空间分支与以跨帧状态聚合为主的时序分支；随后通过门控融合模块实现空间-时序协同表征；最后在特征金字塔层面引入自顶向下的多尺度时序状态传播，并结合时序一致性约束共同提升视频检测性能。

### 13.2 关键创新点表述

> 与原始单帧 `Mamba-YOLO` 相比，本文方法主要改进包括：  
> 1. 构建三帧输入的视频检测数据链路；  
> 2. 设计显式空间主支路与时序副支路的门控融合模块；  
> 3. 引入多尺度时序状态传递机制；  
> 4. 通过特征级时序一致性损失增强跨帧稳定性；  
> 5. 在 `VisDrone-VID` 上补充 COCO 风格 `AP / AP50 / AP75` 与分类别评测。

---

## 14. 关键代码入口总览

- 时序数据集：
  - [official-mamba-yolo/ultralytics/data/dataset.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/data/dataset.py)

- 时序增强：
  - [official-mamba-yolo/ultralytics/data/augment.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/data/augment.py)

- 时序模型入口：
  - [official-mamba-yolo/ultralytics/nn/tasks.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/nn/tasks.py)

- 时序核心模块：
  - [official-mamba-yolo/ultralytics/nn/modules/mamba_yolo.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/nn/modules/mamba_yolo.py)

- 时序一致性损失：
  - [official-mamba-yolo/ultralytics/utils/loss.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/utils/loss.py)

- `VisDrone-VID` 验证器：
  - [official-mamba-yolo/ultralytics/models/yolo/detect/val.py](/home/easyai/桌面/mamba-yolo3/official-mamba-yolo/ultralytics/models/yolo/detect/val.py)

- 数据准备脚本：
  - [scripts/download_visdrone_vid.py](/home/easyai/桌面/mamba-yolo3/scripts/download_visdrone_vid.py)
  - [scripts/prepare_visdrone_vid.py](/home/easyai/桌面/mamba-yolo3/scripts/prepare_visdrone_vid.py)
  - [scripts/download_uavdt_official.py](/home/easyai/桌面/mamba-yolo3/scripts/download_uavdt_official.py)
  - [scripts/prepare_uavdt_full.py](/home/easyai/桌面/mamba-yolo3/scripts/prepare_uavdt_full.py)

---

## 15. 当前最重要的认识

当前仓库最重要的价值，不是“已经得到最终最优结果”，而是：

- 已经把官方 `Mamba-YOLO` 成功改造成一个**可训练、可验证、可对比、可复现**的视频检测研究框架
- 后续无论继续调结构、调超参、补实验、写论文，都已经有一个清晰稳定的代码基础

这也是本文档存在的意义：

- 防止后续忘记每一处改动是做什么的
- 保证论文描述、代码实现、实验流程三者能对得上
