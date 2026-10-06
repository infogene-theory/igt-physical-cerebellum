# -*- coding: utf-8 -*-
"""T-37 (R4 Wave0) · 持续失稳合成注入生成器（可复现、断网、带 ground-truth 起始时刻）

两类持续失稳信号（修正 R3 阶跃注入复 EDMD 失稳模恒 0）:
  (a) divergent     振幅 e^{a·τ} 增长振荡发散（|λ|>1 → **γ<0**；★T-195 新口径：γ=−λ_im，失稳 ⟺ γ<0）
  (b) accelerating  单调加速爬升（τ^p, p>2 二阶导恒正）
≥2 通道同步（cpu/mem 同源, N=2 满足复 EDMD）；起始帧 t0 已知; t0 前平稳段(γ≤0 期望)。
尺度防溢出: e^{a·τmax}·amp 控制在 ~10×; τ^p 用 (τ/τmax)^p 归一化。

用法:
  python scripts/r4_inject_synth.py --kind divergent --a 0.05 --frames 60 --start 20 --out x.csv
  python scripts/r4_inject_synth.py --kind accelerating --power 3 --frames 60 --start 20 --out x.csv
输出 CSV 布局与 sys_harvest 一致（8 列; load1=0.7cpu+0.3mem 派生不入谱）。
"""
import argparse
import math
import os
import random

TS0 = 1700000000000


def 写csv(cpu_f, mem_f, n帧, 起始帧, out):
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("ts_ms,cpu_pct,mem_pct,disk_used_pct,net_kb_s,proc_cnt,load1,标签\n")
        for t in range(n帧):
            cpu = max(0.5, min(98.0, cpu_f(t)))
            mem = max(0.5, min(98.0, mem_f(t)))
            load1 = 0.7 * cpu + 0.3 * mem
            label = "注入" if t >= 起始帧 else "平稳"
            f.write("{},{:.3f},{:.3f},77.28,0,{},{:.3f},{}\n".format(
                TS0 + t * 1000, cpu, mem, 100 + t // 2, load1, label))
    print(f"[ok] {os.path.basename(out)}: n={n帧} 起始帧={起始帧}")


def 平稳波动(seed, n帧, 起始帧, amp=0.3):
    rng = random.Random(seed)
    return [amp * rng.random() for _ in range(n帧)]


def 全段噪声(seed, n, cpu_amp=0.25, mem_amp=0.10):
    """LCG 白噪声(无周期)叠加到全段——使 --seed 对 start=0 段也生效, 逐题字节不同可复现"""
    state = (seed & 0xFFFFFFFF) | 1
    cpu, mem = [], []
    for _ in range(n):
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        v = ((state >> 33) / 2.0 ** 31) - 1.0
        cpu.append(cpu_amp * v)
        mem.append(mem_amp * v)
    return cpu, mem


def 构建(a):
    n = a.frames
    s0 = a.start
    平 = 平稳波动(a.seed, n, s0)
    if a.noise_amp > 0:
        # T-47 ③ 强噪声: 白噪声幅覆盖(与信号主形态同量级考验 DMD 提取/噪声地板判别)
        噪cpu, 噪mem = 全段噪声(a.seed, n, a.noise_amp, a.noise_amp * 0.4)
    else:
        噪cpu, 噪mem = 全段噪声(a.seed, n)
    if a.kind == "divergent":
        # 幅度自动缩放保证峰值在 cap 内且短窗(前 30%)已有可检增长:
        # amp = 目标峰值 / e^{a·注入长}  → 峰值 ≈ 基线 + 目标峰值
        注入长 = max(1, n - s0)
        cpu_amp = 25.0 / math.exp(a.a * 注入长)
        mem_amp = 10.0 / math.exp(a.a * 注入长)
        # cpu = 基线 + amp·e^{a·τ}·sin(ω·τ), τ=t−s0; mem = 基线 + 同源(不同幅度保持 N=2)
        def cpu_f(t):
            return 30.0 + (平[t] * 0.5 if t < s0 else cpu_amp * math.exp(a.a * (t - s0)) * math.sin(a.omega * (t - s0)))
        def mem_f(t):
            return 45.0 + (平[t] * 0.4 if t < s0 else mem_amp * math.exp(a.a * (t - s0)) * math.sin(a.omega * (t - s0) + 0.3))
    elif a.kind == "accelerating":  # 单调指数加速爬升 e^{a·τ}(归一化), 无振荡——指数增长是 EDMD 本征模(|μ|=e^{a·Δt}>1)
        注入长 = max(1, n - s0)
        def cpu_f(t):
            if t < s0:
                return 30.0 + 平[t] * 0.5
            τ = t - s0
            return 30.0 + 25.0 * (math.exp(a.a * τ) - 1.0) / (math.exp(a.a * 注入长) - 1.0)
        def mem_f(t):
            if t < s0:
                return 45.0 + 平[t] * 0.4
            τ = t - s0
            return 45.0 + 10.0 * (math.exp(a.a * τ) - 1.0) / (math.exp(a.a * 注入长) - 1.0)
    else:  # stable: 平稳对照段(无注入)——高频稳态正弦(短窗内覆盖多周期, 无局部单调长片段), 期望 γ≤0
        def cpu_f(t):
            return 30.0 + 1.5 * math.sin(1.1 * t) + 0.8 * math.sin(2.3 * t + 0.5) + 0.15 * 平[t]
        def mem_f(t):
            return 45.0 + 1.0 * math.cos(0.9 * t) + 0.5 * math.cos(1.9 * t) + 0.1 * 平[t]
    # 全段叠加 LCG seed 噪声（确定性可复现; 幅值 << 信号主形态, 不改变类别判据）
    cpu0, mem0 = cpu_f, mem_f
    cpu_f = lambda t: cpu0(t) + 噪cpu[t]
    mem_f = lambda t: mem0(t) + 噪mem[t]
    写csv(cpu_f, mem_f, n, s0, a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["divergent", "accelerating", "stable"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--start", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--a", type=float, default=0.05)
    ap.add_argument("--omega", type=float, default=0.9)
    ap.add_argument("--power", type=float, default=3.0)
    # T-47 ③ 强噪声边界: 覆盖全段 LCG 白噪声幅(cpu; mem=0.4×), 缺省 0=用原默认(0.25/0.10)
    ap.add_argument("--noise-amp", type=float, default=0.0)
    a = ap.parse_args()
    if a.start >= a.frames - 10:
        ap.error("--start 需至少留 10 帧注入段")
    构建(a)


if __name__ == "__main__":
    main()
