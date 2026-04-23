# Mamba-YOLO 时序模块可执行改造方案

更新时间：2026-04-20

## 1. 当前代码现状

当前时序实现已经具备这些基础能力：

- 数据侧：
  - `YOLOTemporalDataset` 能按 `t-1, t, t+1` 组织三帧输入
  - 文件位置：`mamba-yolo-temporal/ultralytics/data/dataset.py`
- 模型侧：
  - `TemporalDetectionModel` 在检测头前对三个检测尺度分别做时序融合
  - 文件位置：`mamba-yolo-temporal/ultralytics/nn/tasks.py`
- 模块侧：
  - `TemporalStateTransfer`
  - `AdaptiveSparseGuide`
  - `SparseGuidedTemporalScan`
  - `TemporalGuidedXSSBlock`
  - 文件位置：`mamba-yolo-temporal/ultralytics/nn/modules/mamba_yolo.py`
- 损失侧：
  - 已经有轻量 `temporal_consistency_loss`

也就是说，现在这套代码不是“没有时序”，而是：

- 已经有三帧输入
- 已经有局部时序融合
- 已经有基于帧差的 guide
- 已经有 feature-level consistency

但它仍然存在 3 个明显上限：

1. 只做局部 `3` 帧 clip，没有长时记忆
2. guide 只来自帧差和显著性，没有历史检测结果先验
3. `TemporalStateTransfer` 更像对称聚合，不够强调方向性时序传播

## 2. 调研后最适合当前项目的路线

结合已有代码和公开视频检测方案，最适合你当前项目继续做强的路线不是推倒重来，而是：

### 路线选择

采用：

- `EOVOD` 的历史检测结果先验
- `MEGA` 的轻量局部-全局记忆思想
- 保留当前 `Mamba-YOLO` 的多尺度时序融合骨架

不优先采用：

- `FGFA` 风格完整光流对齐
- `TransVOD` 风格全量时空 Transformer 重构

原因：

- 你的当前代码已经有三帧局部融合和 guide 机制，和 `EOVOD/MEGA` 最兼容
- 光流路线改动太重，工程风险高
- Transformer 重构会把项目从“改造 Mamba-YOLO”变成“另起一套视频检测器”

## 3. 最终建议方案

方案名称建议：

- `Mamba-YOLO-TGMM`
- 全称可写为：`Temporal Guided Memory Mamba-YOLO`

核心组成：

1. 局部三帧多尺度融合
2. 历史检测先验引导
3. 轻量长期记忆聚合
4. 一致性正则

## 4. 具体怎么改

### 阶段 A：历史检测先验引导

目标：

- 把当前 `guide` 从“纯帧差引导”升级为“帧差 + 历史检测先验引导”

对应参考：

- `EOVOD`

当前问题：

- `AdaptiveSparseGuide` 只用 `current-prev` 差分和当前特征显著性
- 它不知道上一帧模型到底“看到了什么目标”

改法：

1. 在训练和验证时缓存前一帧预测框
2. 把前一帧预测框 rasterize 成一个低分辨率 heatmap
3. 把这个 heatmap 作为 `AdaptiveSparseGuide` 的额外输入
4. guide 从当前的 `2` 通道输入改成：
   - `diff`
   - `saliency`
   - `prev_det_heatmap`

代码落点：

- `mamba-yolo-temporal/ultralytics/nn/modules/mamba_yolo.py`
  - 改 `AdaptiveSparseGuide`
- `mamba-yolo-temporal/ultralytics/nn/tasks.py`
  - 在 `TemporalDetectionModel._predict_temporal_clip()` 里收集中心帧或前一帧的预测先验

实现建议：

- 第一版不要直接拿 NMS 后离散框全量参与
- 先只取上一帧 top-k 高置信框，生成一个简化热图
- 热图大小直接对齐每个检测尺度

收益：

- guide 不再只看运动变化
- 会显式偏向“上一帧出现过目标”的区域

风险：

- 训练初期预测不稳定，先验可能脏

规避方式：

- 训练前若干 epoch 只用原始 guide
- 之后再渐进启用历史先验分支

### 阶段 B：加入轻量长期记忆

目标：

- 在当前 `3` 帧 clip 外增加长时历史信息

对应参考：

- `MEGA`
- `SELSA`

当前问题：

- 当前只看 `t-1, t, t+1`
- 一旦短时邻帧质量不好，就没有更稳定的历史补充

改法：

1. 在 `TemporalDetectionModel` 内为每个尺度维护一个短历史 memory queue
2. 每次前向时，把当前中心特征写入 memory
3. 下一次前向时，从 memory 中取最近 `K` 个历史特征做一次轻量聚合
4. 将 memory 聚合结果与当前 `TemporalStateTransfer` 输出再融合

代码落点：

- `mamba-yolo-temporal/ultralytics/nn/tasks.py`
  - 给 `TemporalDetectionModel` 增加 `memory_bank`
- `mamba-yolo-temporal/ultralytics/nn/modules/mamba_yolo.py`
  - 新增 `TemporalMemoryAggregator`

实现建议：

- 第一版不要做复杂 key-value attention
- 直接做：
  - cosine similarity weighting
  - 或者 learned scalar weighting

推荐参数：

- 每尺度 memory 长度：`4` 或 `6`
- 只在验证和推理阶段保留完整 memory
- 训练阶段先做局部 memory 模拟，不做跨 batch 持久缓存

收益：

- 解决当前 clip 太短的问题
- 遮挡和运动模糊情况下更稳

风险：

- 训练和推理的 memory 定义不一致，容易引入分布差异

规避方式：

- 第一版先做“单 sample 内伪历史聚合”
- 确认收益后，再加真正跨时间 persistent memory

### 阶段 C：把状态传播改成更明确的递归更新

目标：

- 让 `TemporalStateTransfer` 更像“时间状态传播”，而不是偏对称聚合

对应参考：

- `TA-GRU`
- `RMA-Net`

当前问题：

- 当前 `scan + 双向均衡 + 全局平均` 设计太对称
- 对前后帧顺序的敏感性不够强

改法：

1. 保留当前 scan 框架，但把状态更新写成显式门控形式：
   - `h_t = z_t * proposal_t + (1-z_t) * h_{t-1}`
2. proposal 不只来自当前帧，还来自 guide 修正后的候选状态
3. 中心帧输出时，优先使用前向状态和后向状态的非对称组合，而不是简单均值

代码落点：

- `mamba-yolo-temporal/ultralytics/nn/modules/mamba_yolo.py`
  - 重写 `TemporalStateTransfer._scan_memory()`

实现建议：

- 第一版只改状态更新公式，不改整个接口
- 保持输入输出 shape 完全不变

收益：

- 强化真正的时序建模能力
- 让“前后顺序”更重要

风险：

- 改动核心模块，最容易引入不稳定

规避方式：

- 必须先做单元探针验证，再上完整训练

## 5. 不建议现在做的事

1. 不建议立刻上光流
- 开销大
- 工程链路复杂
- 对毕业设计当前阶段不划算

2. 不建议重写成 Transformer 视频检测器
- 会偏离当前 `Mamba-YOLO` 路线
- 对照实验也会变得不公平

3. 不建议继续加更多损失项优先于结构改造
- 当前主要瓶颈更像时序利用不充分
- 不是 loss 数量不够

## 6. 推荐实施顺序

严格按下面顺序做：

### 第一步

做“历史检测先验 guide”

原因：

- 改动最小
- 与当前代码最兼容
- 最容易在短期内看到收益

### 第二步

做“轻量 memory 聚合”

原因：

- 能补当前三帧局限
- 比彻底重写状态传播更稳

### 第三步

再做“递归状态传播重构”

原因：

- 这是最深的一步
- 需要在前两步已经确认链路稳定后再做

## 7. 每一步如何验证

### A 阶段验证

要看：

- `guide` 热图是否明显覆盖上一帧目标区域
- 单步前向时，`guide_mean/min/max` 是否正常
- 对比无先验版本，验证集 `AP50` 是否提升

### B 阶段验证

要看：

- memory 聚合结果是否非零且有限
- 遮挡/模糊场景下预测是否更稳
- `AP` 和 `AP75` 是否提升

### C 阶段验证

要看：

- 前后帧交换时，输出差异是否明显增大
- 时序方向敏感性是否增强
- 后期 epoch 指标是否不再过早见顶

## 8. 成功标准

达到下面这几个条件，就可以认为方案有效：

1. `VisDrone-VID` 上 `AP / AP50 / AP75` 至少有一项稳定提升
2. `UAVDT` 上 `mAP50 / mAP50-95` 不退化
3. 训练中没有新出现的 `NaN` / 不稳定爆炸
4. 可视化里邻帧连续目标的漏检率下降

## 9. 最终建议

如果只选一条最值得现在立刻落地的方案：

- **先做“历史检测先验 guide”**

原因：

- 与当前 `AdaptiveSparseGuide` 最匹配
- 工程代价低
- 论文表达清晰
- 最容易形成“相比原版 Mamba-YOLO 的时序创新点”

## 10. 后续执行建议

建议后续直接按下面节奏推进：

1. 实现阶段 A
2. 做短程序探针验证
3. 跑 `VisDrone-VID` smoke
4. 跑 `VisDrone-VID` dev
5. 如果有效，再实现阶段 B

这样风险最低，结论也最干净。

## 11. 当前已实现记录

### 2026-04-20：阶段 A / v1

已经开始落地“历史检测先验 guide”。

当前实现方式：

1. 从上一帧特征金字塔经 `Detect` 头解出简化检测结果
2. 只保留 top-k 高置信检测
3. 将上一帧检测框栅格化成每个尺度上的低分辨率热图
4. 把热图作为 `AdaptiveSparseGuide` 的第三路输入

现在的 `guide` 输入已经从：

- `diff`
- `saliency`

升级成：

- `diff`
- `saliency`
- `prev_det_heatmap`

当前这版的定位：

- 这是 `EOVOD` 风格启发下的最小可执行版本
- 目标是先验证“历史检测先验”是否有效
- 不是最终的完整时序系统

### 2026-04-20：阶段 A / v1 探针验证结果

已经用真实 `VisDrone-VID` 验证 batch 做过一次不经过完整训练的前向探针。

探针目标：

1. 确认 `prev_det_heatmap` 不是全零
2. 确认 guide 数值稳定
3. 确认无效前帧会被 `temporal_valid` 正确屏蔽

实测结果：

- `temporal_valid = [[0, 1, 1], [1, 1, 1]]`
- 尺度 0：
  - `prev_det_heatmap shape = (2, 1, 48, 84)`
  - `sum = 5.145442`
  - `max = 0.001281`
  - `guide mean/min/max = 0.484132 / 0.358048 / 0.637255`
- 尺度 1：
  - `prev_det_heatmap shape = (2, 1, 24, 42)`
  - `sum = 1.286647`
  - `max = 0.001281`
  - `guide mean/min/max = 0.498889 / 0.361816 / 0.638811`
- 尺度 2：
  - `prev_det_heatmap shape = (2, 1, 12, 21)`
  - `sum = 0.321805`
  - `max = 0.001281`
  - `guide mean/min/max = 0.495981 / 0.357659 / 0.644530`

结论：

- `prev_det_heatmap` 在真实 batch 上已经是非零的
- 新增的“历史检测先验”分支不是空接线
- guide 数值分布正常，没有出现 NaN / Inf

工程判断：

- 阶段 A / v1 已经达到“可继续进入 smoke/dev 训练验证”的状态
- 如果后续收益不明显，下一步优先考虑：
  - 给历史先验增加 warmup 渐进启用
  - 提高先验热图表达能力，而不是回退这个分支

如果这版验证有效，再继续接：

- 阶段 B：轻量 memory
- 阶段 C：递归状态传播重构
