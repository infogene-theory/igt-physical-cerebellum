# 数据来源与许可说明（DATA_SOURCES）

本文件逐项说明仓库中**每一类数据从哪来、如何得到、能否重新获取、用什么许可**，
并如实声明缺口。符号口径以 [CALIBRATION.md](CALIBRATION.md) 为唯一权威；
复现实验的步骤见 [experiment/run_closed_loop.py](experiment/run_closed_loop.py)。

本仓库**不含任何个人隐私数据，也不含第三方专有数据**：题集为作者合成，
实验轨迹为作者在本地机器上自行运行所得。

---

## 1. 总览

| 数据 | 位置 | 来源性质 | 是否随仓库 | 许可 |
|---|---|---|---|---|
| 24 题合成时序 | [`bench/*.csv`](bench) | 作者用脚本合成（确定性 seed） | 是 | 见代码许可 |
| 24 题证据锚 EV | [`bench/EV_*.json`](bench) | 谱计算纯函数 + 标定，作者生成 | 是 | 见代码许可 |
| 题谱图 | [`bench/spectra/*.png`](bench/spectra) | 作者由数据绘制 | 是 | CC BY 4.0 |
| 波形生成器 | [`bench/r4_inject_synth.py`](bench/r4_inject_synth.py) | 作者原创，仅依赖 Python 标准库 | 是 | BSL 1.1 |
| 题集编排（归档） | [`bench/r4_z_gen.py`](bench/r4_z_gen.py) | 作者原创，归档副本（见其顶部说明） | 是 | BSL 1.1 |
| 标定查表 | [`experiment/analysis/r4_epsilon_table.py`](experiment/analysis) | 作者实测统计（n=500/500） | 是 | BSL 1.1 |
| M1 四模型轨迹 | [`experiment/results/m1_*/`](experiment/results) | 作者本地实验（2026-09-09） | 是 | CC BY 4.0（数据/记录） |
| 代码口径历史件 | [`experiment/results/legacy_code_tier/`](experiment/results) | 转换前归档，作者保留 | 是 | 同上 |
| 论文 | [`paper/`](paper) | 作者撰写 | 是 | CC BY 4.0 |
| 配图 | [`figures/`](figures) | 作者绘制 | 是 | CC BY 4.0 |
| 模型权重 | 不随仓库 | 第三方开源模型，需自行下载 | 否 | 各模型自带许可 |
| 推理引擎 llama.cpp | 不随仓库 | 开源项目，需自行安装 | 否 | MIT |
| M0 原始过程文件 | 未完整保留 | 早期实验（见 §4 缺口声明） | 否 | — |

> 许可简称：**BSL 1.1**＝Business Source License 1.1（[根 LICENSE](LICENSE)，
> Change Date 2028-04-13 起转为 Apache License 2.0）；**CC BY 4.0**＝Creative Commons
> Attribution 4.0（[LICENSE-DOCS.md](LICENSE-DOCS.md)）。

---

## 2. 冻结题集（bench/）如何生成

24 题全部为**合成的多通道时序**，不含真实测量值。生成方法与参数公开：

- 采样：每段 **240 帧**、采样间隔 dt = 1 s；
- 信号：在基线（CPU 约 30%、内存约 45%）上叠加角频率 **ω = 0.9 rad/s** 的振荡，
  振荡幅值按 e^(a·τ) 演化（a 为增长率/衰减率，按难度分箱取值，网格见 r4_z_gen.py）；
- 噪声：叠加由线性同余生成器（LCG，一种确定性伪随机数算法）产生的白噪声，
  主档幅度 σn = 1.0，**seed 公开、可逐字节复现**；
- CSV 为 8 列：`ts_ms,cpu_pct,mem_pct,disk_used_pct,net_kb_s,proc_cnt,load1,标签`；
- 24 题经 **SHA-256 哈希去重**，字节互不相同（无伪重复）；
- 分箱（v2.1）：clear 6 / fuzzy_present 3 / fuzzy_absent 6 / fuzzy_stuck 3 / below 6。

**重新生成波形（自包含、可直接运行，仅用 Python 标准库）：**

```bash
python bench/r4_inject_synth.py --kind divergent --out test.csv \
    --frames 240 --start 0 --seed 11001 --a 0.00022 --omega 0.9 --noise-amp 1.0
```

从波形到证据锚（EV）的谱计算，其纯函数逻辑已由开源的 **igt_core**
（[`igt_core/src/complex_dmd.rs`](igt_core/src) 等）提供；标定常数由
[`experiment/analysis/r4_epsilon_table.py`](experiment/analysis/r4_epsilon_table.py)
给出（n_cal = n_val = 500 次实测统计）。[`bench/r4_z_gen.py`](bench/r4_z_gen.py)
是当时一键造题的原始编排，**但其调用的 r3_engine.exe 未开源，故在本仓库内不能直接运行**
（详见该文件顶部说明）——我们保留它以完整公开"题目如何被选出"，权威可复现路径是
冻结产物 + igt_core + run_closed_loop.py。

---

## 3. M1 四模型实验轨迹（experiment/results/）

- 实验日期：2026-09-09；作者在本地 CPU 上运行；
- 四个模型（权重需自行从其官方发布渠道下载，许可为各模型自带条款）：
  - **Qwen3-14B**（量化 Q3）、**Qwen3-VL-8B**（Q4）、
  - **Qwen2.5-7B-Instruct**（Q4）、**gemma-3-12B**（Q4）；
- 推理后端：**llama.cpp** 的 llama-cli（开源，MIT，https://github.com/ggerganov/llama.cpp ）；
- 设计：24 题冻结题集，A（闭环）/B（开环）臂每题独立重复 3 次，bootstrap 固定种子 1000 次；
- 字段口径：轨迹中的历史 z 为**旧代码口径**，但判定对错的 `correct` 布尔量与口径无关，
  故错误率、τfb（轮）结论有效；转换前的代码口径 EV 与裁决锚已归档至
  [`legacy_code_tier/`](experiment/results/legacy_code_tier)，换算见 CALIBRATION.md。

---

## 4. 诚实缺口声明

1. **M0（2026-09-04，首轮 E0）原始过程文件未完整归档。** 早期迭代中过程记录疑似被覆盖，
   本仓库不提供 M0 的逐轮轨迹，也**不伪造**；M0 的汇总数字（A 臂 1.00→0.50、B 臂恒 1.00、
   τfb≈5.8 轮）以论文 v1.3 附录 B 为准。M1 的完整轨迹可独立复算。
2. **真机受控负载数据**（CPU 8.8%→64.9% 平台，论文 §8.1）属"进行中、不计入证据"，
   原始 CSV 未随本仓库提供；该现象（体制切换而非段内失稳）在论文中如实记录。
3. **r3_engine.exe 未开源。** 本仓库开源的是确定性核心 igt_core 与自包含编排，
   不含整个 igt_engine / igt_agent。
4. **模型权重与 llama.cpp 不随仓库分发**，需按其各自许可自行下载、安装。
5. 全部定量结论仅在论文声明的精确适用域内成立（系统失稳窄任务 × 中心化 z 证据锚 ×
   本地开源模型），不外推为普适定理，也不宣称"已消灭幻觉"。

---

## 5. 引用与署名

若使用本仓库的数据或结论，请按 [LICENSE-DOCS.md](LICENSE-DOCS.md) 的方式署名：

> 郑达（Damon），IGT Physical Cerebellum，
> https://github.com/infogene-theory/igt-physical-cerebellum ，CC BY 4.0（文档/数据）；
> 代码遵循 BSL 1.1（2028-04-13 起 Apache-2.0）。
