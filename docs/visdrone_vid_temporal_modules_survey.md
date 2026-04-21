# VisDrone-VID 时序模块方案调研

更新时间：2026-04-20

目的：整理视频目标检测，尤其是 `VisDrone-VID` / 无人机视频场景里常见的时序建模方案，记录论文、核心思路、是否有公开代码，以及这些方法对当前 `Mamba-YOLO` 时序改造有什么参考价值。

## 1. 先说结论

视频目标检测里的时序模块，主流基本可以归为 5 类：

1. 光流/显式对齐后做特征聚合
2. 非光流的序列特征聚合或记忆库
3. Transformer 式时空注意力
4. RNN/GRU/递归式时序更新
5. 输出级 temporal consistency / 历史框先验 / tubelet linking

如果只看“最适合你当前项目继续演化”的方向，优先级我建议是：

1. `SELSA / MEGA / EOVOD`
2. `TransVOD`
3. `TA-GRU / RMA-Net` 这类无人机场景化时序模块

原因：

- 你的当前实现已经有多尺度特征融合和时序状态传递，和 `SELSA / MEGA / EOVOD` 的思路最接近。
- 如果继续加复杂模块，最自然的是把“跨帧记忆聚合”做得更强，而不是立刻切到全新范式。
- `VisDrone-VID` 是视频流，小目标多、遮挡多、模糊多，纯单帧检测不够，时序聚合和历史先验通常有价值。

## 2. VisDrone-VID 官方口径

`VisDrone-VID` 官方更常用的指标是：

- `AP`
- `AP50`
- `AP75`
- `AR1`
- `AR10`
- `AR100`
- `AR500`

官方 challenge 论文：

- `VisDrone-VID2019: The Vision Meets Drone`
- 链接：<https://ziming-liu.github.io/visdrone/visdronevid_paper.pdf>

官方工具：

- `VisDrone2018-VID-toolkit`
- 链接：<https://github.com/VisDrone/VisDrone2018-VID-toolkit>

数据集主页：

- `VisDrone-Dataset`
- 链接：<https://github.com/VisDrone/VisDrone-Dataset>

## 3. 主流视频检测方法与时序方案

### 3.1 FGFA

- 论文：`Flow-Guided Feature Aggregation for Video Object Detection`
- 链接：<https://arxiv.org/abs/1703.10025>
- 代码：<https://github.com/msracver/Flow-Guided-Feature-Aggregation>

核心思路：

- 先用光流把邻帧特征对齐到当前帧
- 再对齐后加权聚合特征
- 本质是“先对齐，再融合”

适合借鉴的点：

- 时序特征直接相加之前要先尽量对齐
- 如果不做对齐，邻帧信息容易把小目标抹糊

对当前项目的启发：

- 你现在没有显式运动补偿
- 如果后续要强化时序模块，最经典的强化点就是加轻量对齐，而不是只做堆叠融合

### 3.2 SELSA

- 论文：`Sequence Level Semantics Aggregation for Video Object Detection`
- 论文链接：<https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Sequence_Level_Semantics_Aggregation_for_Video_Object_Detection_ICCV_2019_paper.pdf>
- 代码：<https://github.com/happywu/Sequence-Level-Semantics-Aggregation>

核心思路：

- 不只看相邻帧，而是把整段视频当成一个语义集合
- 当前帧目标特征和整段序列中语义相近的目标特征做聚合

适合借鉴的点：

- 不是只做 `t-1, t, t+1`
- 可以做“序列级语义检索/聚合”

对当前项目的启发：

- 你的三帧方案偏局部时序
- 如果效果上限不高，可以往“更长时域 memory + 相似性聚合”走

### 3.3 MEGA

- 论文：`Memory Enhanced Global-Local Aggregation for Video Object Detection`
- 论文链接：<https://arxiv.org/abs/2003.12063>
- 代码：<https://github.com/Scalsol/mega.pytorch>

核心思路：

- 同时建模局部邻域和长程 memory
- 当前帧既看近邻，也看远处历史关键帧

适合借鉴的点：

- local + global 两级时间上下文
- memory bank 不需要每一帧都参与，能兼顾效果和开销

对当前项目的启发：

- 这是你当前版本最值得借鉴的路线之一
- 你可以在三帧局部 clip 外，再加一个轻量的历史特征缓存

### 3.4 EOVOD

- 论文：`Efficient One-stage Video Object Detection by Exploiting Temporal Consistency`
- 论文链接：<https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950001.pdf>
- 代码：<https://github.com/guanxiongsun/EOVOD>

核心思路：

- 不走很重的光流或大 memory
- 直接利用上一帧检测结果，给当前帧提供位置和尺度先验
- 强调 temporal consistency 和高效率

适合借鉴的点：

- 对一阶段检测器很友好
- 可以把上一帧框/中心热区变成当前帧的显式引导

对当前项目的启发：

- 你现在已经有 `guide` 概念
- 最容易继续强化的方向，就是把历史检测结果也接入 guide，而不是只用帧差

### 3.5 TransVOD

- 论文：`TransVOD: End-to-End Video Object Detection with Spatial-Temporal Transformers`
- 论文链接：<https://arxiv.org/abs/2201.05047>
- 代码：<https://github.com/SJTU-LuHe/TransVOD>

核心思路：

- 用时空 Transformer 在 query / feature 两侧做跨帧交互
- 强化 long-range 依赖建模

适合借鉴的点：

- 如果后面要强调“更强的时序依赖建模”，Transformer 是标准做法

对当前项目的启发：

- 你的 `Mamba` 路线本质上也在追求高效长序列建模
- 可借鉴的不是“整个改成 DETR”，而是跨帧 attention 的组织方式

### 3.6 TA-GRU（无人机视频）

- 论文：`Object Detection in Drone Video with Temporal Attention Gated Recurrent Unit Based on Transformer`
- 链接：<https://www.mdpi.com/2504-446X/7/7/466>

核心思路：

- 用 `TA-GRU` 递归传播时序特征
- 用 `DeformAlign` 做帧间对齐
- 再用 temporal attention / fusion 做聚合

代码情况：

- 我没有查到稳定可用的公开 GitHub 仓库
- 公开评审信息里还提到作者给过失效代码链接，因此不能把它当现成代码基线

对当前项目的启发：

- “GRU 状态传播 + 对齐 + 融合” 这套结构非常适合写成毕业设计叙事
- 如果你以后想把 `TemporalStateTransfer` 做得更明确，`GRU-like update` 是一条可解释路线

### 3.7 RMA-Net（无人机视频）

- 论文：`Object detection in drone video based on recurrent motion attention`
- 链接：<https://www.sciencedirect.com/science/article/pii/S0167865524001375>

核心思路：

- 用 recurrent motion attention 建模长时域运动信息
- 强调在 drone vision 场景下同时兼顾精度和速度

代码情况：

- 当前未查到明确官方公开仓库

对当前项目的启发：

- 你现在的 `guide` 仍偏局部帧差
- RMA 这类方法提示你可以把“运动注意力”做成递归累积，而不是瞬时差分

### 3.8 STAM on YOLOX（微小无人机）

- 论文：`A video object detector with Spatio-Temporal Attention Module for micro UAV detection`
- 链接：<https://www.sciencedirect.com/science/article/pii/S0925231224007446>

核心思路：

- 在 YOLOX 中插入 `Spatio-Temporal Attention Module`
- 强化微小 UAV 目标在视频中的时空可分辨性

代码情况：

- 当前未查到明确官方公开仓库

对当前项目的启发：

- 这是“在现有单帧 YOLO 检测器里直接插时空模块”的典型做法
- 和你当前改 `Mamba-YOLO` 的范式非常接近

### 3.9 TransVisDrone（无人机到无人机）

- 论文：`TransVisDrone: Spatio-Temporal Transformer for Vision-based Drone-to-Drone Detection in Aerial Videos`
- 论文链接：<https://arxiv.org/abs/2210.08423>
- 项目页：<https://tusharsangam.github.io/TransVisDrone-project-page/>
- 代码：<https://github.com/tusharsangam/TransVisDrone>

核心思路：

- 用 `CSPDarkNet-53` 抽空间特征
- 用 `Video Swin Transformer` 建模时空依赖

适合借鉴的点：

- 这是更贴近“无人机场景”的时空 Transformer 方案
- 但任务更偏 drone-to-drone 检测，不完全等价于 VisDrone-VID 通用目标检测

## 4. 这些方案怎么归纳成可操作设计

### 4.1 最常见的时序模块设计套路

1. `对齐 + 聚合`
- 代表：`FGFA`
- 关键词：光流、warp、邻帧加权融合

2. `局部 clip + 长程 memory`
- 代表：`MEGA`
- 关键词：local context、global memory、关键帧缓存

3. `序列级语义聚合`
- 代表：`SELSA`
- 关键词：跨帧相似目标聚合、sequence-level semantics

4. `历史结果先验 / temporal consistency`
- 代表：`EOVOD`
- 关键词：上一帧框先验、低成本 temporal consistency

5. `时空 attention / Transformer`
- 代表：`TransVOD`、`TransVisDrone`
- 关键词：跨帧 attention、query interaction

6. `递归状态更新`
- 代表：`TA-GRU`、`RMA-Net`
- 关键词：GRU、recurrent attention、motion memory

### 4.2 对当前 Mamba-YOLO 项目最有参考价值的实现方向

如果继续改你当前的 `Mamba-YOLO`，最合理的 3 条路线是：

1. `EOVOD 风格`
- 在当前 `guide` 基础上接入上一帧预测结果
- 成本最低，和一阶段检测头最兼容

2. `MEGA / SELSA 风格`
- 在当前三帧局部融合之外，再加轻量 memory bank
- 能解释成“局部-全局联合时序聚合”

3. `TA-GRU 风格`
- 把 `TemporalStateTransfer` 从当前 scan/mix，改成更明确的 gated recurrent update
- 论文表述会更清晰

## 5. 哪些方法有公开代码

### 明确有公开代码

- `FGFA`
  - <https://github.com/msracver/Flow-Guided-Feature-Aggregation>
- `SELSA`
  - <https://github.com/happywu/Sequence-Level-Semantics-Aggregation>
- `MEGA`
  - <https://github.com/Scalsol/mega.pytorch>
- `EOVOD`
  - <https://github.com/guanxiongsun/EOVOD>
- `TransVOD`
  - <https://github.com/SJTU-LuHe/TransVOD>
- `TransVisDrone`
  - <https://github.com/tusharsangam/TransVisDrone>

### 目前未查到稳定官方代码

- `TA-GRU`
- `RMA-Net`
- `STAM on YOLOX`

## 6. 对毕业设计可直接引用的总结

可以把现有视频检测时序模块总结成下面这段：

> 现有视频目标检测方法通常通过跨帧特征对齐、时序特征聚合、长时记忆建模以及时空注意力机制来提升检测性能。经典方法如 FGFA 通过光流对齐邻帧特征后再进行融合，SELSA 和 MEGA 进一步将时序上下文扩展到序列级语义聚合和全局记忆增强。近年来，EOVOD 强调以较低代价利用上一帧检测结果和 temporal consistency 提升一阶段视频检测性能；TransVOD 和 TransVisDrone 则采用时空 Transformer 显式建模跨帧依赖；面向无人机视频场景的 TA-GRU 与 RMA-Net 则通过递归状态更新和运动注意力增强视频中的小目标检测。

## 7. 对当前项目的直接建议

如果你下一步要把 `Mamba-YOLO` 的时序模块继续做强，而不是完全另起炉灶，建议按这个顺序尝试：

1. 在现有 `guide` 中加入上一帧预测框先验，走 `EOVOD` 风格
2. 给当前三帧局部融合增加轻量历史 memory，走 `MEGA` 风格
3. 把 `TemporalStateTransfer` 改造成更明确的 gated recurrent update，走 `TA-GRU / RMA-Net` 风格

## 8. 参考链接汇总

- VisDrone-VID challenge paper:
  - <https://ziming-liu.github.io/visdrone/visdronevid_paper.pdf>
- VisDrone-VID toolkit:
  - <https://github.com/VisDrone/VisDrone2018-VID-toolkit>
- VisDrone dataset:
  - <https://github.com/VisDrone/VisDrone-Dataset>
- FGFA paper:
  - <https://arxiv.org/abs/1703.10025>
- FGFA code:
  - <https://github.com/msracver/Flow-Guided-Feature-Aggregation>
- SELSA paper:
  - <https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Sequence_Level_Semantics_Aggregation_for_Video_Object_Detection_ICCV_2019_paper.pdf>
- SELSA code:
  - <https://github.com/happywu/Sequence-Level-Semantics-Aggregation>
- MEGA paper:
  - <https://arxiv.org/abs/2003.12063>
- MEGA code:
  - <https://github.com/Scalsol/mega.pytorch>
- EOVOD paper:
  - <https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950001.pdf>
- EOVOD code:
  - <https://github.com/guanxiongsun/EOVOD>
- TransVOD paper:
  - <https://arxiv.org/abs/2201.05047>
- TransVOD code:
  - <https://github.com/SJTU-LuHe/TransVOD>
- TA-GRU paper:
  - <https://www.mdpi.com/2504-446X/7/7/466>
- RMA-Net paper:
  - <https://www.sciencedirect.com/science/article/pii/S0167865524001375>
- TransVisDrone paper:
  - <https://arxiv.org/abs/2210.08423>
- TransVisDrone project:
  - <https://tusharsangam.github.io/TransVisDrone-project-page/>
- TransVisDrone code:
  - <https://github.com/tusharsangam/TransVisDrone>
