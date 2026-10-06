# experiment/results · 实验结果与历史数据

本目录保存 M1 跨基座实验的**原始模型轨迹**，以及 T-195 前代码口径的历史件。口径说明统一见仓库根 [CALIBRATION.md](../../CALIBRATION.md)。

## 1. M1 四模型完整轨迹（2026-09-09，证据档 E1）

24 题冻结题集，A（闭环回灌）/ B（开环不回灌）各题逐轮，bootstrap 固定种子。每个目录含一个 `trajectories.jsonl`，每行是一轮（题 × 臂 × 轮）：

| 目录 | 模型 | 量化/后端 |
|---|---|---|
| `m1_qwen3-14b/` | Qwen3-14B-Instruct | Q4_K_M，GPU |
| `m1_qwen3-vl-8b/` | Qwen3-VL-8B-Instruct | Q4_K_M，GPU |
| `m1_qwen2.5-7b/` | Qwen2.5-7B-Instruct | Q4_K_M，GPU |
| `m1_gemma-3-12b/` | Gemma-3-12B-Instruct | Q4_K_M，GPU |

主要字段：`题id`、`臂`、`round_k`（0–3）、`correct`（模型方向是否与客观方向一致）、`答对`、`z箱`（clear/fuzzy/below）、`claim`（模型结构化判定）、`verdict`、`一致率`、`repeat_n`、`feedback条数`。

**口径标注**：M1 执行于 T-195（2026-09-28）之前，轨迹内 `中心化z` 为旧代码口径（正区间，present ⟺ z>+1）。但 `correct`/`答对` 布尔、错误率、收敛轮 τfb（轮）等**结论口径无关**，可直接分析。

独立复算（不依赖原始运行环境）：

```powershell
# A/B 错误率、τfb 拟合、bootstrap、配对差分（--set 路径不存在时自动回退用轨迹内 z箱）
python ..\analysis\r4_m1_stats.py --trajectories .\m1_qwen3-14b\trajectories.jsonl --set __none__.jsonl
```

## 2. M0 缺口（诚实声明）

M0（2026-09-04，Qwen2.5-7B Q4 在 CPU 上，早期题集）的**原始过程文件疑似在早期快速迭代中被覆盖、未完整归档**。经多轮检索（遍历全部 r4_rounds* 目录、按 M0 错误率特征匹配、检索 T51 证据），未找到 M0 完整轨迹；`experiments/r4_rounds` 经核实是 2026-09-03 的 mock 数据，非 M0。

- M0 的权威数字（A 臂 1.00→1.00→0.75→0.50、B 恒 1.00、τ_fb≈5.8 轮 R²=0.99 等）以**论文 v1.3 附录 B** 为准；
- M0 定位为早期探索性 **E0** 实验，样本小、仅作数量级；
- **M1 四模型完整轨迹可独立复算**，承担主要证据（E1）；
- 本仓库**不伪造 M0 轨迹**。T-51（`experiments/llama_cpp_spike/T51_block1_EVIDENCE.md`，未入仓库）可佐证当时已观测到的 present 假阳性偏置。

## 3. legacy_code_tier/（旧代码口径历史件，仅溯源）

T-195 前代码口径的原始件，**不是当前复现实验输入**：

- `ev/EV_*.json`：转换前代码口径证据锚（模谱 γ 字段即 g=λ_im，失稳为正）；
- `anchors/{id}.anchor.json`：裁决锚（`c_result.eigen_pairs` 第三元为 λ_im）；
- `README.md`：口径对照。

## 4. 如何产生新结果

当前物理口径的端到端复现用仓库 `experiment/run_closed_loop.py`（自包含，仅需本目录 harness 编译出的 r3-claim 与本地模型），不依赖已归档的 r3_engine。用法见该脚本顶部说明。

## 5. 端到端复现记录（2026-10-05，CPU，物理口径）

在本地 CPU（llama.cpp Release 构建、Qwen2.5-7B-Instruct Q4_K_M）上用 run_closed_loop.py 复现，全程 T-195 物理口径（present ⟺ z<−1）：

- **`repro_check/`（ZCL001，A/D 臂）**：clear present 题。A 臂读证据锚首轮即判对（unstable_mode_count=1、present、关键模符号断言"正"），D 臂无锚题面亦判对；r3-claim → llama-cli → Claim → 标定判据 → 判分 → 轨迹的链路全程贯通。
- **`repro_fuzzy/`（ZFA002、ZFP001，A 臂）**：
  - ZFP001（客观 present）首轮即对；
  - **ZFA002（客观 absent）首轮误判 present，k1、k2 维持 present，k3 经 3 条规则反馈翻转为 absent**（value 1→0，关键证据符号断言"正"→"负"）——错误经确定性反馈收敛，且模型对 γ 符号的判断同步改变，不是蒙对类别。
  - 三态统计把 ZFA002 记为"不收敛"，仅因翻转发生在末轮 k3、没有 k4 构成"连续两轮对"；轨迹本身显示其末轮已正确。

这组复现表明：仓库可在不依赖原始 r3_engine 的条件下，从冻结题集端到端重跑，并重现"无锚自由漂移 / 闭环错误收敛"的对照。
