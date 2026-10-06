# -*- coding: utf-8 -*-
"""T-48 块3 z_set v2 · 按实测中心化 z 三箱重造 R4 合成题(240 帧, ω=0.9, core EDMD 唯一证据源)

z_set v2.1 归箱边界(PM 2026-09-04 smoke9 repeat3 实测裁定, 判据/ε_c/core 一律不动):
  clear          z < −1.3         期望 present, k0 即收(不进 τfb 拟合)   # ★T-195 随迁：口径翻转 ⇒ 负侧
  fuzzy_present  1.28 ≤ z ≤ 1.3   期望 present(补漏报方向; 下沿 1.28 由实测定: 1.28 k0 直接对)
  fuzzy_absent   0.5 ≤ z ≤ 0.9    期望 absent(压假阳性方向, τfb 收敛桶主来源)
  fuzzy_stuck    1.0 ≤ z < 1.28   期望 present 但**7B 反馈不可纠带·保守漏报**:
                                  单列只记翻转率、不进 τfb 收敛桶, 留作 M1(换模型)对照点
  below          z < 0.5          期望 absent(控 present 假阳性负样本)
方向不对称实测由来: z1.02 弱 present 四轮反馈(含末轮完整 Verdict)全判 absent 撬不动、
repeat3 一致率 1.0 非随机; z0.69 absent 侧 k0/k1 假阳性→k2 转 absent、k3 锁定(收敛)。
出题参数 σn=1.0 主档; 归箱与方向全按**实测中心化 z**(弃不落箱/错向题)。

用法: python -X utf8 scripts/r4_z_gen.py [--out-dir experiments/r4_bench_z21] [--seed-base 11000]
输出: {out-dir}/z_set.jsonl + 每题 CSV/anchor/EV[增导标定 b/ε_c/z/z箱] + EVIDENCE.md。

★ 仓库归档说明（igt-physical-cerebellum）：本文件是原始题集编排脚本的逐字归档副本，
  用于公开“24 题如何被选出”的完整逻辑。它在本仓库内**不能直接运行**——其中
  “组锚/生成EV”调用的是 r3_engine.exe（anchor/evidence 子命令，**未随本仓库开源**）。
  该两步所用谱计算纯函数已由本仓库开源的 igt_core（complex_dmd 等）提供，标定查表见
  experiment/analysis/r4_epsilon_table.py；波形本体可用同目录 r4_inject_synth.py 从 seed
  独立生成。实验复现请用 experiment/run_closed_loop.py；数据来源与口径见
  ../DATA_SOURCES.md 与 ../CALIBRATION.md。
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
ENGINE = os.path.join(ROOT, "igt_engine", "target", "debug", "r3_engine.exe")
SYNTH = os.path.join(ROOT, "scripts", "r4_inject_synth.py")
PY = sys.executable
import r4_epsilon_table as 标定  # noqa: E402

_环境 = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
帧 = 240
ω0 = 0.9


def 生成段(名, 目录, seed, a, σn):
    out = os.path.join(目录, f"{名}.csv")
    subprocess.run([PY, SYNTH, "--kind", "divergent", "--out", out, "--start", "0",
                    "--seed", str(seed), "--frames", str(帧), "--a", str(a), "--omega", str(ω0),
                    "--noise-amp", str(σn)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   env=_环境, check=True)
    return out


def 组锚(csv):
    out = csv.replace(".csv", ".anchor.json")
    subprocess.run([ENGINE, "anchor", "--csv", csv, "--out", out, "--baseline", "true"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=_环境, check=True)
    return out, json.load(open(out, encoding="utf-8"))


def 主带γ(锚_obj, 目标="失稳"):
    """★ T-195 选模按目标分（理论 171 §一.2「不可一律 λ_im max」）：
    目标="失稳"（默认，判据量用途）⇒ **λ_im 最大（＝γ 最小）** = 最接近失稳／最不衰减；
    目标="稳定" ⇒ γ 最大（λ_im 最负）。全模谱第三元为**新口径 γ**（稳定>0，失稳 ⟺ γ<0）。"""
    全 = [(m[0], m[1], m[2]) for m in ((锚_obj.get("c_result") or {}).get("全模谱", []))]
    带 = [m for m in 全 if abs(m[1] - ω0) < 0.25]
    if not 带:
        return None, None
    m = (min(带, key=lambda x: x[2]) if 目标 == "失稳" else max(带, key=lambda x: x[2]))
    return float(m[2]), float(m[1])


def 生成EV(锚路径, ev路径, b, ec, z, 箱名, σn):
    subprocess.run([ENGINE, "evidence", "--anchor", 锚路径, "--out", ev路径],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=_环境, check=True)
    with open(ev路径, encoding="utf-8") as f:
        ev = json.load(f)
    ev["标定"] = {"γ_H0中心b": b, "ε_c": ec, "中心化z": round(z, 3),
                "z箱": 箱名, "σn": σn, "窗长帧": 帧, "ω": ω0,
                "口径": "显著性=中心化z=(γ_EDMD−b)/ε_c, 相对零分布判显著而非相对0; 阈|z|>1↔误报5%"}
    with open(ev路径, "w", encoding="utf-8") as f:
        json.dump(ev, f, ensure_ascii=False, indent=1)


def sha256_自检(目录):
    hs = {}
    for f in sorted(x for x in os.listdir(目录) if x.endswith(".csv")):
        h = hashlib.sha256(open(os.path.join(目录, f), "rb").read()).hexdigest()
        hs.setdefault(h, []).append(f)
    重复 = {h: fs for h, fs in hs.items() if len(fs) > 1}
    if 重复:
        raise RuntimeError(f"SHA256 去重失败: {list(重复.values())[:3]}")
    return len(hs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiments/r4_bench_z21")
    ap.add_argument("--seed-base", type=int, default=11000)
    a = ap.parse_args()
    目录 = os.path.join(ROOT, a.out_dir)
    os.makedirs(目录, exist_ok=True)
    for f in os.listdir(目录):
        if f.endswith((".csv", ".json", ".jsonl", ".md")):
            os.remove(os.path.join(目录, f))
    条目们, 日志 = [], []
    seq = 0
    裁决词表 = ["失稳", "稳态", "慢化方向", "增长模", "健康", "风险", "present", "absent",
             "higher", "lower", "tier", "失稳模", "eigen_pairs", "Verdict", "supported", "contradicted"]

    def 收(子类, 前缀, n, 条件):
        """扫 (a 网格, seed) 直到收够 n; 条件(实测z, σn)→(box, 方向) 由调用方定收否。
        弃题文件清理; 每段一 unique seed。"""
        nonlocal seq
        已收 = 0
        a网格, σn_ = 子类["a网格"], 子类["σn"]
        for _ in range(800):
            av = a网格[seq % len(a网格)]
            seed = a.seed_base + seq * 7 + 1
            seq += 1
            名 = f"{前缀}{已收 + 1:03d}"
            try:
                生成段(名, 目录, seed, av, σn_)
                锚路径, 锚_obj = 组锚(os.path.join(目录, f"{名}.csv"))
            except subprocess.CalledProcessError as e:
                日志.append(f"{名} 段失败: {e}")
                continue
            γ, ω = 主带γ(锚_obj)
            if γ is None:
                日志.append(f"{名} 弃: 主带无模")
                继续弃(名)
                continue
            c = 标定.查表(σn_, 帧, ω0)
            z = (γ - c["b"]) / c["ε_c"]
            box, 方向, 子类名 = 条件(z)
            if box is None:
                日志.append(f"{名} 弃(a={av} σn={σn_}) z={z:.2f} γ={γ:+.6f}")
                继续弃(名)
                continue
            生成EV(锚路径, os.path.join(目录, f"EV_{名}.json"), c["b"], c["ε_c"], z, box, σn_)
            条目们.append({
                "id": f"R4-{名}", "类": f"z箱·{子类名}", "tag": "boundary",
                "csv": f"{名}.csv", "seed": seed,
                "参数": {"a": av, "σn": σn_, "frames": 帧, "omega": ω0},
                "期望": {"observable": "unstable_mode_count", "direction": 方向, "单位": "count"},
                "真值": {"主带γ": round(γ, 7), "主带ω": round(ω, 4), "中心化z": round(z, 3),
                       "z箱": box, "裸γ<0": γ < 0},   # ★T-195 新口径：失稳 ⟺ γ<0
                "标定": {"b": c["b"], "ε_c": c["ε_c"], "σn": σn_, "frames": 帧, "ω": ω0,
                       "误报率": c["误报率"]},
            })
            日志.append(f"{名} OK {box}({方向}) z={z:.2f} γ={γ:+.6f} (a={av} σn={σn_})")
            已收 += 1
            if 已收 >= n:
                return
        raise RuntimeError(f"{前缀} 800 次未收够 n={n}——需调 a 网格/σn")

    def 继续弃(名):
        for suf in (".csv", ".anchor.json"):
            p = os.path.join(目录, 名 + suf)
            if os.path.exists(p):
                os.remove(p)

    def clear条件(z):
        # ★ T-195：失稳 present ⟺ z 显著偏负（z<−1.3 对应旧 z>+1.3，口径翻转）
        return ("clear", "present", "clear") if z < -1.3 else (None, None, None)

    def fuzzyP条件(z):
        # present 侧下沿 z≥1.28(PM v2.1: 1.28 k0 直接对; <1.28 的弱 present 撬不动)
        return ("fuzzy", "present", "fuzzy_present") if -1.3 <= z <= -1.28 else (None, None, None)

    def fuzzyA条件(z):
        # absent 侧上沿 z≤0.9(PM v2.1: 压假阳性方向, 收敛桶主来源)
        return ("fuzzy", "absent", "fuzzy_absent") if -0.9 <= z <= -0.5 else (None, None, None)

    def stuck条件(z):
        # 反馈不可纠带(保守漏报): z∈(0.9,1.28) 的 present 题, 单列只记翻转率、不进 τfb 收敛桶, 留作 M1 对照
        return ("fuzzy", "present", "fuzzy_stuck") if -1.28 < z <= -1.0 else (None, None, None)

    def below条件(z):
        return ("below", "absent", "below") if z > -0.5 else (None, None, None)

    # 子类网格(σn=1.0 主档; z 单题散布≈0.5 → 细网格多 seed 密集扫)
    seq = 0
    收({"名": "clear", "σn": 1.0, "a网格": [2.2e-4, 2.8e-4, 3.5e-4, 4.5e-4, 6e-4, 8e-4, 1.1e-3, 1.5e-3]},
       "ZCL", 6, clear条件)
    收({"名": "fuzzy_present", "σn": 1.0, "a网格": [1.25e-4, 1.3e-4, 1.35e-4, 1.4e-4, 1.45e-4]},
       "ZFP", 3, fuzzyP条件)
    收({"名": "fuzzy_absent", "σn": 1.0, "a网格": [6e-5, 6.5e-5, 7e-5, 7.5e-5, 8e-5, 8.5e-5, 9e-5, 9.5e-5]},
       "ZFA", 6, fuzzyA条件)
    收({"名": "fuzzy_stuck", "σn": 1.0, "a网格": [1.0e-4, 1.05e-4, 1.1e-4, 1.15e-4, 1.2e-4, 1.25e-4]},
       "ZFS", 3, stuck条件)
    收({"名": "below", "σn": 1.0, "a网格": [2e-5, 3e-5, 4e-5, 5e-5, 6e-5]},
       "ZBL", 6, below条件)
    n_csv = sha256_自检(目录)
    集路径 = os.path.join(目录, "z_set.jsonl")
    with open(集路径, "w", encoding="utf-8") as f:
        for e in 条目们:
            f.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")
    def 计(名):
        return sum(1 for e in 条目们 if 名 in e["类"])
    with open(os.path.join(目录, "EVIDENCE.md"), "w", encoding="utf-8") as f:
        f.write("# T-48 块3 z_set v2.1 EVIDENCE\n\n")
        f.write(f"- 题数 {len(条目们)}; 目录 {a.out_dir}\n")
        f.write(f"- 子类: clear {计('clear')} / fuzzy_present {计('fuzzy_present')} / "
                f"fuzzy_absent {计('fuzzy_absent')} / fuzzy_stuck {计('fuzzy_stuck')} / below {计('below')}\n")
        f.write("- 归箱边界(v2.1; ★T-195 取负侧): clear z<−1.3(present, k0 即收) / fuzzy_present −1.3~−1.28(present) / "
                "fuzzy_absent 0.5~0.9(absent, 收敛桶主来源) / fuzzy_stuck 1.0~1.28(present 但 7B 反馈不可纠带, "
                "只记翻转率不进 τfb) / below z<0.5(absent)\n")
        f.write("- 方向不对称由来(PM smoke9 repeat3): z1.02 弱 present 四轮反馈撬不动、z1.28 k0 直接对、"
                "z0.69 absent 侧假阳性→k2 转对(收敛)\n")
        f.write(f"- 帧={帧} ω={ω0}; 标定查表 r4_epsilon_table(v10 实测 n_cal=n_val=500); "
                f"SHA256 去重 {n_csv}\n\n## 生成校验日志\n\n```\n" + "\n".join(日志) + "\n```\n")
    print(f"[ok] z_set v2.1 题集 {len(条目们)} 条 → {集路径}")


if __name__ == "__main__":
    main()
