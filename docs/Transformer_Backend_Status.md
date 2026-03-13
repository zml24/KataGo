# KataGo Transformer 后端工作汇总

截至 2026-03-13，这一轮工作的目标是：

1. 为 `KataGo_Transformer/nano/model.py` 定义的 Transformer 增加原生模型导出格式。
2. 在 `KataGo` 里增加可本地验证的 Transformer 推理基线。
3. 建立 `Torch / EIGEN / METAL / 未来 CUDA` 的统一对照链路。
4. 在上 CUDA 机器前，尽量把结构性问题和明显数值错误先收掉。

## 当前结论

- `EIGEN` Transformer 路径已经能本地构建并跑通。
- `METAL` Transformer 路径已经能本地构建并跑通。
- 已经不只是比较少量输出，而是支持比较：
  - `raw`
  - `rawFull`
  - `postprocess`
  - `nnOutput`
- 当前参考基线是 `Torch CPU FP32`。
- 还没有和 `Torch CPU FP32 default attention` 完全对齐。
- 当前 best-known 结果下，最大误差仍然在 `rawFull.seki`。
- 从误差形态看，当前更像是数值路径差异，而不是权重映射、输出布局、symmetry 这类结构性错误。

## 已实现内容

### 1. 原生 Transformer 模型格式

在 `KataGo_Transformer` 侧新增了原生导出脚本，支持把 `nano/model.py` checkpoint 导出成 `.kgtr/.kgtr.gz`。

相关文件：

- `KataGo_Transformer/nano/export_cuda.py`
- `KataGo/cpp/neuralnet/transformerdesc.h`
- `KataGo/cpp/neuralnet/transformerdesc.cpp`

当前导出格式支持：

- `version=15`
- `stem=cnn1/cnn3/cnn5`
- `ape=none/per_pos/d4`
- `rpe=rope`
- full heads 导出
- `scorebelief` 导出

当前明确不支持：

- `gab`
- `rpb`
- `stem_d4`
- `stem_norm`

其中 `stem_norm` 不是“先忽略”，而是已经在导出阶段显式拒绝，避免静默数值错误。

### 2. C++ Transformer reference core

新增了纯 C++ 的 Transformer 推理核心，用于：

- `EIGEN` backend
- `METAL` backend 的本地 reference 路径
- raw/full dump

相关文件：

- `KataGo/cpp/neuralnet/transformerinference.h`
- `KataGo/cpp/neuralnet/transformerinference.cpp`

支持的输出包括：

- subset raw：
  - `policy`
  - `policyPass`
  - `value`
  - `scoreValue`
  - `ownership`
- full raw：
  - `policy(6)`
  - `policyPass(6)`
  - `value(3)`
  - `misc(10)`
  - `moreMisc(8)`
  - `ownership`
  - `scoring`
  - `futurePos`
  - `seki`
  - `scoreBelief`

### 3. EIGEN / METAL 后端接入

相关文件：

- `KataGo/cpp/neuralnet/eigenbackend.cpp`
- `KataGo/cpp/neuralnet/metalbackend.cpp`
- `KataGo/cpp/neuralnet/metalbackend.h`
- `KataGo/cpp/neuralnet/nninterface.h`

已经完成的内容：

- `getTransformerRawOutputs(...)` 可以导出 subset raw 和 full raw。
- `runtransformerdump` 可以把固定样例位置的输入特征、raw/full raw、`NNOutput` 一次性 dump 到 JSON。
- `METAL` 路径中 Transformer policy 布局读取错误已经修复。
- `METAL` 编译相关的 Swift Optional 和 module cache 问题已经收掉。

### 4. 对照工具链

相关文件：

- `KataGo/cpp/tests/testtransformer.cpp`
- `KataGo/cpp/tests/tests.h`
- `KataGo/cpp/command/runtests.cpp`
- `KataGo/cpp/main.cpp`
- `KataGo_Transformer/nano/compare_backend_dump.py`

当前支持：

- 固定样例 full dump
- 单样例过滤
- `Torch CPU FP32 default` 对照
- `Torch CPU FP32 explicit attention` 对照

单样例 dump 用法：

```bash
katago runtransformerdump MODEL.kgtr.gz OUTPUT.json 0 canary_opening_18
```

Torch 对照用法：

```bash
python3 compare_backend_dump.py \
  --checkpoint CHECKPOINT.ckpt \
  --dump-json OUTPUT.json \
  --output-json REPORT.json \
  --attention-mode default
```

或：

```bash
python3 compare_backend_dump.py \
  --checkpoint CHECKPOINT.ckpt \
  --dump-json OUTPUT.json \
  --output-json REPORT.json \
  --attention-mode explicit
```

## 本地验证

### 使用的 checkpoint

本地验证用的是：

- `/Volumes/WenshuSpace/下载/checkpoint-s6553600000.ckpt`

导出的模型文件是：

- `/tmp/checkpoint-s6553600000.kgtr.gz`

### EIGEN 构建

已经解决 `Eigen3` 缺失问题，并成功构建：

- build 目录：`/tmp/katago-eigen-build`
- 可执行文件：`/tmp/katago-eigen-build/katago`

### METAL 构建

已经解决：

- Swift / SDK 组合问题
- module cache 写权限问题
- Transformer policy 布局 bug

构建产物：

- `/tmp/katago-metal-build/katago`

## 当前 best-known 数值结果

当前最佳已验证状态，是在 C++ reference core 中保留第一轮 `double` 累加改动后得到的结果。

对照目标：

- `Torch CPU FP32 default`

结果：

- `symmetry=0`：`maxAbsErr = 0.00012111663818359375`
- `symmetry=3`：`maxAbsErr = 0.0001277923583984375`

对应报告：

- `/tmp/eigen_transformer_compare_full_default.json`
- `/tmp/eigen_transformer_compare_full_sym3.json`

误差最大项：

- `rawFull.seki`

当前还没有做到：

- 与 `Torch CPU FP32 default` 完全 bitwise 对齐

## 已验证并回退的失败方案

下面这些方向都实际试过了，但效果更差，所以已经回退，不在当前代码里保留：

### 1. 更改 attention 输出累加方式

曾尝试继续修改：

- `attnOut` 累加
- `meanRows` 累加

结果：

- `maxAbsErr` 从 `1.211e-4` 回升到 `1.430e-4`

结论：

- 这不是正确方向，已回退。

### 2. 在 macOS 上把矩阵乘法切到 Accelerate / CBLAS

结果：

- 速度明显变快
- 但 `maxAbsErr` 回升到 `2.055e-4`

结论：

- 不适合作为 correctness baseline
- 已回退

## 当前判断

当前已经可以排除掉一批更危险的问题：

- policy / value / ownership 布局错位
- symmetry 输入输出没接好
- `METAL` 读取 Transformer policy 时的索引错误
- full head 权重导出缺失
- `scorebelief` 缺失
- 明显的权重转置方向错误

当前更可能的剩余问题是：

- 手写 reference attention 路径与 `Torch default SDPA` 的数值路径仍有差异
- 误差会在 `seki` 这一路被放大

因此现在最合理的理解是：

- 还没有完全对齐
- 但剩余问题更像数值实现细节，而不是结构性接错

## CUDA 状态

CUDA 侧已经接入了不少内容，但这台本地 Mac 不能做最终验证，因为没有：

- `nvcc`
- `cuda.h`

当前已完成：

- CUDA backend 的 Transformer 主链路接入
- full raw 输出相关结构接入
- 对称性相关问题已补
- 固定棋盘限制前移

当前未完成：

- CUDA 机器上的真实构建验证
- CUDA 与 `Torch / EIGEN` 的 full raw 数值对照

## 建议的下一步

### 1. 先把 reference 路径继续往 Torch CPU FP32 靠拢

优先做：

- layer-wise dump
- 定位第一层开始偏离的位置
- 专盯 `seki` 前的 trunk / final norm / value spatial split

### 2. 在 CUDA 机器上使用同一套对照链路

建议流程：

1. 用现有 `.kgtr.gz` 模型跑 `runtransformerdump`
2. 用 `compare_backend_dump.py` 分别对 `torch default` 和 `torch explicit` 出两份报告
3. 同时对照：
   - `CUDA vs Torch`
   - `CUDA vs EIGEN`
   - `CUDA vs METAL`

### 3. 不要把当前状态误判为“已经完全对齐”

当前更准确的说法应该是：

- 基线已经搭好了
- 本地 `EIGEN` / `METAL` 都能跑
- full raw / postprocess / nnOutput 都能比较
- 结构性错误已经排掉很多
- 但 `Torch CPU FP32 default` 还没有完全对齐

