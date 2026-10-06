# -*- coding: utf-8 -*-
r"""T-55 · M1 第一档(Qwen3VL-8B GPU)统计（PM 口径钉死, 独立读 trajectories 现算）

① 主体：A/B 臂全体口径错误率 p_err(k)（分母=全体 23/24 题=剔除 arm_fail；停轮后沿用末态；
   严禁 survivor 口径）→ 拟合 τfb: d_fb(k)=1−e^(−2k/τfb)（等价 p_err(k)=p0·e^(−2k/τ)，
   OLS ln p_err = ln p0 − (2/τ)k；p=0 点剔除）→ 报点估 + R² + **跨题 bootstrap(B≥1000,
   固定种子)中位数与 2.5/97.5 百分位**（repeat 内无方差，CI 只能靠跨题重采样——PM 条）。
② A/B 逐 k 反升率(错→对 or 对→错的轮间翻转)与配对差分 Δ(k)=p_err,B−p_err,A。
③ C/D 按 z 箱对/错表与 C−D 差。
④ 全程标注 E1、n、限域。
用法: python -X utf8 scripts/r4_m1_stats.py --trajectories <path> --set <z_set.jsonl> [--out <md>]
"""
import argparse
import io
import json
import math
import os
import random
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def 拟合τ(points):
    """points=[(k,p)] p∈(0,1]; ln p = ln p0 − (2/τ)k; 返回 (τ, p0, R², 点数) 或 None"""
    pts = [(k, p) for k, p in points if p > 0 and p < 1]
    if len(pts) < 2:
        return None
    xs = [math.log(p) for _, p in pts]
    ks = [float(k) for k, _ in pts]
    n = len(pts)
    mk = sum(ks) / n
    ml = sum(xs) / n
    sxy = sum((ks[i] - mk) * (xs[i] - ml) for i in range(n))
    sxx = sum((ks[i] - mk) ** 2 for i in range(n))
    if abs(sxx) < 1e-12:
        return None
    b = sxy / sxx  # = −2/τ
    a = ml - b * mk
    if b >= 0:
        return None  # 需下降
    tau = -2.0 / b
    p0 = math.exp(a)
    ss_tot = sum((xs[i] - ml) ** 2 for i in range(n))
    ss_res = sum((xs[i] - (a + b * ks[i])) ** 2 for i in range(n))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"tau": tau, "p0": p0, "R2": r2, "n": n}


def 题集期望(path):
    d = {}
    if os.path.exists(path):
        for ln in io.open(path, encoding="utf-8"):
            j = json.loads(ln)
            d[j.get("id")] = {"z箱": (j.get("真值") or {}).get("z箱"),
                              "期望": (j.get("期望") or {}).get("direction")}
    return d


def 主():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", default="experiments/r4_rounds_z21_full_8b_gpu/trajectories.jsonl")
    ap.add_argument("--set", default="experiments/r4_bench_z21/z_set.jsonl")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="M1档1: Qwen3VL-8B-Instruct Q4_K_M GPU(ngl99), E1, 方向二值窄任务")
    a = ap.parse_args()
    rows = [json.loads(x) for x in io.open(a.trajectories, encoding="utf-8") if x.strip()]
    期望 = 题集期望(a.set)
    by = {}
    for r in rows:
        by.setdefault(r.get("臂"), {}).setdefault(r.get("题id"), []).append(r)
    K = 4  # k0..k3
    lines = [f"# T-55 M1 第一档统计（{a.tag}）", ""]

    def 臂序列(臂):
        """返回 {题: {k0..k3 末态沿用 correct}, arm_fail 题剔}}"""
        out = {}
        失败题 = set()
        for tid, rs in (by.get(臂) or {}).items():
            rs = sorted(rs, key=lambda x: x.get("round_k") or 0)
            末 = None
            cc = {}
            for x in rs:
                if x.get("failed"):
                    失败题.add(tid)
                    continue
                k = x.get("round_k")
                c = x.get("correct")
                if k is not None and c is not None:
                    cc[k] = bool(c)
                    末 = bool(c)
            # 停轮后沿用末态
            for k in range(K):
                if k not in cc:
                    cc[k] = 末
            if tid not in 失败题:
                out[tid] = cc
        return out, 失败题

    # ② A/B 错误率 + τ
    pA, failA = 臂序列("A")
    pB, failB = 臂序列("B")
    nA, nB = len(pA), len(pB)  # PM 口径: A 剔 arm_fail=23, B 全 24, 各自分母
    lines.append(f"## A/B 全体口径错误率（A 分母={nA}(剔 arm_fail: {failA}); B 分母={nB}(剔: {failB})）")
    errA, errB = {}, {}
    for k in range(K):
        errA[k] = sum(1 for c in pA.values() if c.get(k) is False) / nA
        errB[k] = sum(1 for c in pB.values() if c.get(k) is False) / nB
    lines.append(f"- A 错误率: " + "  ".join(f"k{k}={errA[k]:.3f}" for k in range(K)))
    lines.append(f"- B 错误率: " + "  ".join(f"k{k}={errB[k]:.3f}" for k in range(K)))
    lines.append(f"- 配对差分 Δ=B−A: " + "  ".join(f"k{k}={errB[k]-errA[k]:+.3f}" for k in range(K)))

    # 拟合 A（全体口径）
    fitA = 拟合τ([(k, errA[k]) for k in range(K)])
    rng = random.Random(20260908)
    taus = []
    ids = list(pA.keys())
    for _ in range(1000):
        samp = [pA[t] for t in rng.choices(ids, k=len(ids))]
        e = {k: sum(1 for c in samp if c.get(k) is False) / len(samp) for k in range(K)}
        f = 拟合τ([(k, e[k]) for k in range(K)])
        if f:
            taus.append(f["tau"])
    taus.sort()
    if fitA:
        lines.append(f"## τfb 拟合（A 臂全体口径, p_err=p0·e^(−2k/τ)）")
        lines.append(f"- 点估: τ={fitA['tau']:.2f} 轮, p0={fitA['p0']:.3f}, R²={fitA['R2']:.3f}, 拟合点数={fitA['n']}")
        if taus:
            med = taus[len(taus) // 2]
            lo = taus[int(len(taus) * 0.025)]
            hi = taus[int(len(taus) * 0.975)]
            lines.append(f"- 跨题 bootstrap(B=1000, 种子固定): τ 中位={med:.2f} 轮, 95%CI=[{lo:.2f}, {hi:.2f}]（n_boot 成功 {len(taus)}）")
        lines.append("- 注: k1/k2 同为 %.3f 是平台、非光滑指数; 若 R² 低/CI 发散如实报, 不硬给点估" % errA[1])

    # ③ 反升率与配对
    lines.append("## A/B 逐 k 轮间反升(错→对)（全体题口径, 反升=由错转对的题数/各自分母）")
    for k in range(1, K):
        upA = sum(1 for c in pA.values() if c.get(k - 1) is False and c.get(k) is True) / nA
        upB = sum(1 for c in pB.values() if c.get(k - 1) is False and c.get(k) is True) / nB
        lines.append(f"- k{k-1}→k{k}: A 反升率 {upA:.3f}, B 反升率 {upB:.3f}")

    # ④ C/D 按 z 箱(clear/fuzzy/below, 期望 z 箱) + C−D
    lines.append("## C/D 按 z 箱对错（单轮, 答对字段）与 C−D 差")
    箱序 = ["clear", "fuzzy", "below"]
    cD = {}
    for 臂 in ("C", "D"):
        对 = {}
        for tid, rs in (by.get(臂) or {}).items():
            x = rs[-1]
            z = (期望.get(tid) or {}).get("z箱") or x.get("z箱")
            对.setdefault(z, [0, 0])
            对[z][0] += 1 if x.get("答对") else 0
            对[z][1] += 1
        cD[臂] = 对
    lines.append("| z箱 | C 对/总 | D 对/总 | C−D |")
    lines.append("|---|---|---|---|")
    for z in 箱序:
        c = cD["C"].get(z, [0, 0])
        d = cD["D"].get(z, [0, 0])
        cd = (c[0] / c[1] if c[1] else 0) - (d[0] / d[1] if d[1] else 0)
        lines.append(f"| {z} | {c[0]}/{c[1]} | {d[0]}/{d[1]} | {cd:+.2f} |")
    lines.append("")
    lines.append(f"标注: {a.tag}; n(题): A={nA}(剔 arm_fail {failA}) / B={nB}(剔 {failB}) / C=D=24; repeat 内无方差, CI 仅跨题。")
    out = a.out or (os.path.splitext(a.trajectories)[0] + "_m1_stats.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    主()
