# -*- coding: utf-8 -*-
"""T-41 块4 (R4-v2 分析) · 消费 T-41 块3 轨迹(trajectories.jsonl) → R4_RESULT_v2 骨架(不覆盖 v1)

按 PM 放行口径, 五段:
  ① 三态占比(A/B per tag): 未启动(题过简)/收敛/不收敛(反馈无效或过难) → 喂 T-43 调难度
  ② 收敛桶内拟合 τ_fb: p_err(k)=p_err(0)·e^(−2k/τ_fb)(τ 单位=轮), 报点估/R²/95%CI/sample_n
  ③ 主效应配对差分 Δ(k)=p_err,B(k)−p_err,A(k)(同题集配对), 并报 D 臂盲答先验(对照 iv 基准)
  ④ C 错锚臂复现对照 iii(反向率 per tag); D 创造臂对照 ii(无物理 F 不硬判, 文本审计占位)
  ⑤ 汇出 R4_RESULT_v2_{date}.json/.md(trajectories 同目录), 不覆盖 v1 文件

拟合只用收敛桶(corrects 满足块2 三态分类=收敛), 未启动/不收敛单列, 严禁拿全部轮次硬拟合。
运行: python -X utf8 scripts/r4_fit.py --trajectories <path> [--set <题集 jsonl>]
      python -X utf8 scripts/r4_fit.py --self-test   (合成数据验拟合恢复)
"""
import argparse
import json
import math
import os
import sys
from datetime import date
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
os.environ.setdefault("PYTHONUTF8", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def 三态分类(corrects, Kmax):
    """与 r4_rounds 完全同口径(块2 冻结): 未启动=k0 即对; 收敛=k0 错且连续 2 轮对且其后不翻错(k*=第二轮); 余=不收敛"""
    if not corrects:
        return "不收敛", None
    if corrects[0]:
        return "未启动", None
    for i in range(len(corrects) - 1):
        if corrects[i] and corrects[i + 1]:
            if all(corrects[i + 2:]):
                return "收敛", i + 1
    return "不收敛", None


def 拟合τ_fb(points):
    """points=[(k, p_err(k))] 且 p∈(0,1]; 线性回归 ln p = ln p0 − (2/τ_fb)·k。
    返回 dict 或 {ok:False, reason}。p 取 ln(p) 仅 p=0 点剔除(无信息)。"""
    pts = [(float(k), float(p)) for k, p in points if p > 0 and math.isfinite(p)]
    n = len(pts)
    if n < 2:
        return {"ok": False, "reason": f"有效拟合点 {n} < 2(收敛桶内错误轮不足, 无法拟合)"}
    xs = [k for k, _ in pts]
    ys = [math.log(p) for _, p in pts]
    xbar, ybar = sum(xs) / n, sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    if sxx == 0:
        return {"ok": False, "reason": "拟合点全部同轮(sxx=0)"}
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    yhat = [intercept + slope * x for x in xs]
    sse = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    sst = sum((y - ybar) ** 2 for y in ys)
    r2 = 1.0 - (sse / sst) if sst > 0 else (1.0 if sse == 0 else 0.0)
    if slope >= 0:
        return {"ok": False, "reason": f"斜率 {slope:.4f} ≥ 0(收敛桶错误率未随轮下降或平台), 无衰减可拟合"}
    # 95% CI: slope ± 1.96·se
    se2 = sse / (n - 2) if n > 2 else 0.0
    se_slope = math.sqrt(se2 / sxx) if n > 2 else float("nan")
    τ = -2.0 / slope
    if math.isfinite(se_slope) and se_slope > 0:
        lo_slope = slope + 1.96 * se_slope
        hi_slope = slope - 1.96 * se_slope
        # slope<0 → lo_slope(更近 0)对应 τ 上界; hi_slope 更负对应 τ 下界
        τ_lo = -2.0 / lo_slope if lo_slope < 0 else None
        τ_hi = -2.0 / hi_slope if hi_slope < 0 else None
        # T-44 修1: 负斜率反演后 τ_lo > τ_hi, 按数值升序输出 [下界,上界]
        ci = sorted([τ_lo, τ_hi]) if (τ_lo is not None and τ_hi is not None) else None
    else:
        ci = None
    p0 = math.exp(intercept)
    return {"ok": True, "n_points": n, "tau_fb": τ, "tau_fb_ci": ci,
            "r2": r2, "intercept_ln": intercept, "p_err0_est": p0}


def 载rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def tag_of(rid):
    p = rid.split("-")
    if len(p) >= 2 and p[1].upper() in ("U", "S", "C", "K"):
        return p[1].upper()
    return "?"


def 分组(rows):
    """(题id, 臂) → {round_k: row}"""
    g = defaultdict(dict)
    for r in rows:
        g[(r.get("题id"), r.get("臂"))][r.get("round_k")] = r
    return g


def 每题correct序列(g):
    """per (题id,臂): 按 k 排 correct 列表(剔除 None 与 failed 轮)"""
    out = {}
    for key, ks in g.items():
        seq = []
        for k in sorted(ks):
            v = ks[k].get("correct")
            if v is not None and not ks[k].get("failed"):
                seq.append(v)
        out[key] = seq
    return out


def 主体(traj_p, out_dir, out_date):
    rows = 载rows(traj_p)
    g = 分组(rows)
    seqs = 每题correct序列(g)
    事实id集 = sorted({r["题id"] for r in rows if r.get("题id", "").startswith("R4-") and not r.get("题id", "").startswith("R4-K")})
    K题id集 = sorted({r["题id"] for r in rows if r.get("题id", "").startswith("R4-K")})

    arms = [r["臂"] for r in rows]
    per_arm_rows = {a: sum(1 for x in arms if x == a) for a in sorted(set(arms))}

    # ---- ① 三态占比 (A/B per tag) ----
    三态 = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # 臂→tag→态→n
    收敛轮list = defaultdict(list)
    for (题id, 臂) in seqs:
        if 臂 not in ("A", "B"):
            continue
        cs = seqs[(题id, 臂)]
        if not cs:
            continue
        态, kstar = 三态分类(cs, 99)
        三态[臂][tag_of(题id)][态] += 1
        if 态 == "收敛":
            收敛轮list[臂].append(kstar)

    # ---- ② 收敛桶拟合 τ_fb (A 臂 态=收敛 题, per k 聚 p_err) ----
    收敛题 = {}
    for (题id, 臂) in seqs:
        if 臂 == "A":
            cs = seqs[(题id, 臂)]
            态, _ = 三态分类(cs, 99)
            if 态 == "收敛":
                收敛题[题id] = g[(题id, "A")]
    fit_by_tag = {}
    for tag in sorted({tag_of(i) for i in 收敛题}):
        题ids = [i for i in 收敛题 if tag_of(i) == tag]
        # per k: 该 tag 收敛桶题 at k 有 correct 观测 → p_err(k)
        ks = sorted({k for i in 题ids for k in 收敛题[i] if 收敛题[i][k].get("correct") is not None})
        pts = []
        for k in ks:
            obs = [收敛题[i][k] for i in 题ids if k in 收敛题[i] and 收敛题[i][k].get("correct") is not None]
            if not obs:
                continue
            err = sum(1 for o in obs if o["correct"] is False) / len(obs)
            pts.append((k, err))
        fit = 拟合τ_fb(pts)
        fit["sample_n"] = len(题ids)
        fit_by_tag[tag] = fit

    # ---- ③ 配对差分 Δ(k)=p_err,B(k)−p_err,A(k) (A 收敛桶同题集配对) + D 先验 ----
    Δ_by_tag = {}
    D先验 = {}
    for tag in sorted({tag_of(i) for i in 收敛题}):
        题ids = sorted(i for i in 收敛题 if tag_of(i) == tag)
        # A p_err per k(收敛桶), B p_err per k(同题集, B 全 Kmax)
        pa, pb = {}, {}
        for i in 题ids:
            for k, r in g.get((i, "A"), {}).items():
                if r.get("correct") is not None:
                    pa.setdefault(k, []).append(r["correct"])
            for k, r in g.get((i, "B"), {}).items():
                if r.get("correct") is not None:
                    pb.setdefault(k, []).append(r["correct"])
        rows_ = []
        for k in sorted(set(pa) & set(pb)):
            ea = sum(1 for c in pa[k] if c is False) / len(pa[k])
            eb = sum(1 for c in pb[k] if c is False) / len(pb[k])
            rows_.append({"k": k, "p_err_A": round(ea, 4), "p_err_B": round(eb, 4),
                          "Δ": round(eb - ea, 4)})
        Δ_by_tag[tag] = rows_
        # D 盲答先验(k0 错率, 对照 iv 基准)
        d0 = []
        for i in 题ids:
            r0 = g.get((i, "D"), {}).get(0)
            if r0 and r0.get("答对") is not None:
                d0.append(r0["答对"])
        if d0:
            D先验[tag] = {"D_k0_对率": round(sum(1 for x in d0 if x is True) / len(d0), 4),
                          "D_k0_错率": round(sum(1 for x in d0 if x is False) / len(d0), 4),
                          "n": len(d0)}
        else:
            D先验[tag] = {"D_k0_对率": None, "D_k0_错率": None, "n": 0}

    # ---- ④ C 错锚反向率(对照 iii) + D 创造(对照 ii) ----
    C反向 = defaultdict(lambda: [0, 0])  # tag → [反向(n=题答对False), 总]
    C非反向 = defaultdict(lambda: [0, 0])
    for 题id in 事实id集:
        r0 = g.get((题id, "C"), {}).get(0)
        if not r0:
            continue
        对 = r0.get("答对")
        tag = tag_of(题id)
        if 对 is False:
            C反向[tag][0] += 1
            C反向[tag][1] += 1
        elif 对 is True:
            C非反向[tag][0] += 1
            C非反向[tag][1] += 1
    D创造 = []
    for 题id in K题id集:
        r0 = g.get((题id, "D"), {}).get(0)
        if not r0:
            continue
        txt = (r0.get("模型文本") or "").strip()
        D创造.append({"题id": 题id, "文本长": len(txt),
                      "占位": "(mock 无文本" in txt or "失败" in txt[:40],
                      "无硬判词": any(w in txt for w in ("无法", "不能", "不足以", "不确定", "需更多", "缺乏"))})
    # ---- T-45 改3 副指标: 类别对但证据锚定错(★T-195 新口径: 引**正**模/越界/未引**负**模)计数 ----
    锚定错 = defaultdict(lambda: defaultdict(int))  # tag → 细类
    类别错 = defaultdict(int)  # tag → verdict contradicted 轮数(真类别错)
    正确轮 = defaultdict(int)
    for (题id, 臂) in g:
        if 臂 not in ("A", "B"):
            continue
        tag = tag_of(题id)
        for k, row in g[(题id, 臂)].items():
            if row.get("failed"):
                continue
            v, ev_ok = row.get("verdict"), row.get("evidence_ok")
            if v == "supported" and ev_ok is True:
                正确轮[tag] += 1
            elif v == "supported" and ev_ok is False:
                rea = row.get("evidence_reason") or ""
                if "越界" in rea:
                    锚定错[tag]["越界索引"] += 1
                elif "负γ模支撑" in rea:   # ★T-195: 随 T-194 engine reason 文案（present 须由负γ模支撑）
                    锚定错[tag]["引正模/未引负模"] += 1
                elif "不符" in rea:
                    锚定错[tag]["符号断言不符"] += 1
                else:
                    锚定错[tag]["其它"] += 1
            elif v == "contradicted":
                类别错[tag] += 1
    # ---- 汇总 ----
    result = {
        "meta": {"date": out_date, "trajectories": traj_p,
                 "per_arm_rows": per_arm_rows,
                 "事实题数": len(事实id集), "K题数": len(K题id集)},
        "三态占比": {臂: {tag: dict(cnt) for tag, cnt in 三态[臂].items()} for 臂 in ("A", "B")},
        "收敛轮list": {臂: sorted(set(收敛轮list[臂])) for 臂 in ("A", "B")},
        "τ_fb拟合_收敛桶A": fit_by_tag,
        "配对差分Δ": Δ_by_tag,
        "D_盲答先验": D先验,
        "C错锚反向率": {tag: (f"{a}/{b}" if b else "n/a") for tag, (a, b) in C反向.items()},
        "D创造臂文本": {"题数": len(D创造), "占位数": sum(1 for d in D创造 if d["占位"]),
                     "无硬判词数": sum(1 for d in D创造 if d["无硬判词"])},
        "副指标_类别对但证据锚定错": {tag: dict(cnt) for tag, cnt in 锚定错.items()},
        "副指标_类别错轮数": dict(类别错),
        "副指标_证据全对轮数": dict(正确轮),
    }
    json_p = os.path.join(out_dir, f"R4_RESULT_v2_{out_date}.json")
    md_p = os.path.join(out_dir, f"R4_RESULT_v2_{out_date}.md")
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    with open(md_p, "w", encoding="utf-8") as f:
        f.write(_md(result))
    print(f"[ok] R4_RESULT_v2 骨架已写 → {md_p}")
    return result


def _md(r):
    L = []
    L.append(f"# R4_RESULT_v2（R4-v2 收敛/差分/拟合骨架）\n\n- 日期：{r['meta']['date']}")
    L.append(f"- 数据：{r['meta']['trajectories']}（本报告数字全部由该文件现算，可复算）")
    L.append(f"- 本骨架**不证明/证伪 H7.3**：mock 状态机验证结构；真模型 v2 跑批后按同脚本复算成正式报告（不覆盖 v1）。\n")
    L.append("## 0. 跑批概况")
    L.append("| 臂 | 轨迹行 |")
    for a, n in sorted(r["meta"]["per_arm_rows"].items()):
        L.append(f"| {a} | {n} |")
    L.append("")
    L.append("## 1. 三态占比（A/B，喂 T-43 调难度）")
    for 臂 in ("A", "B"):
        if not r["三态占比"].get(臂):
            continue
        L.append(f"- **{臂} 臂**（未启动=题过简 / 收敛 / 不收敛=反馈无效或过难）:")
        for tag, cnt in sorted(r["三态占比"][臂].items()):
            L.append(f"  - {tag}: 未启动={cnt.get('未启动', 0)} 收敛={cnt.get('收敛', 0)} 不收敛={cnt.get('不收敛', 0)}")
    L.append("")
    L.append("## 2. τ_fb 拟合（只收敛桶 A；p_err(k)=p_err(0)·e^(−2k/τ_fb)，τ 单位=轮）")
    for tag, fit in sorted(r["τ_fb拟合_收敛桶A"].items()):
        if fit.get("ok"):
            ci = f"[{fit['tau_fb_ci'][0]:.2f}, {fit['tau_fb_ci'][1]:.2f}]" if fit.get("tau_fb_ci") else "n/a"
            L.append(f"- {tag}: τ_fb={fit['tau_fb']:.3f} 轮 (95%CI {ci}) R²={fit['r2']:.3f} "
                     f"p0≈{fit['p_err0_est']:.3f} sample_n={fit['sample_n']} 拟合点={fit['n_points']}")
        else:
            L.append(f"- {tag}: **N/A**（{fit.get('reason', '?')}）sample_n={fit.get('sample_n', 0)}")
    L.append("")
    L.append("## 3. 配对差分 Δ(k)=p_err,B(k)−p_err,A(k)（同题集；扣 D 盲答先验参考 §4）")
    for tag, rows_ in sorted(r["配对差分Δ"].items()):
        L.append(f"- **{tag}**: " + "；".join(
            f"k={x['k']} A={x['p_err_A']} B={x['p_err_B']} Δ={x['Δ']:+}" for x in rows_) or "(无配对轮)")
    L.append("")
    L.append("## 4. 对照基准")
    L.append("- **D 盲答先验**（对照 iv 基准，非 0.5）：" +
             "；".join(f"{t}: 对率={d['D_k0_对率']} 错率={d['D_k0_错率']}(n={d['n']})" for t, d in sorted(r["D_盲答先验"].items())))
    L.append("- **C 错锚反向率**（对照 iii 复现；反向=答对 False/被错锚带反）：" +
             "；".join(f"{t}: {v}" for t, v in sorted(r["C错锚反向率"].items())))
    L.append(f"- **D 创造臂**（对照 ii：无物理 F 不硬判）：{r['D创造臂文本']['题数']} 题，占位 {r['D创造臂文本']['占位数']}，"
             f"无硬判措辞 {r['D创造臂文本']['无硬判词数']}（真模型 v2 后此项有效）")
    L.append("")
    L.append("## 5. 副指标：类别对但证据锚定错（T-45；不被表面 correct 掩盖）")
    L.append("- 「正确」=verdict supported 且 证据判分对；「锚定错」=supported 但判分 false（★T-195：引正模/越界/未引负模）；「类别错」=contradicted。")
    for tag in sorted(set(list(r["副指标_类别错轮数"]) + list(r["副指标_类别对但证据锚定错"]))):
        L.append(f"- {tag}: 证据全对={r['副指标_证据全对轮数'].get(tag, 0)} | 锚定错=" +
                 " ".join(f"{k}={v}" for k, v in sorted(r["副指标_类别对但证据锚定错"].get(tag, {}).items())) +
                 f" | 类别错={r['副指标_类别错轮数'].get(tag, 0)}")
    L.append("")
    L.append("*拟合与三态口径同 r4_rounds 块2/块3（未启动/不收敛单列，严禁全轮次硬拟合）。*")
    return "\n".join(L)


def 自测():
    """合成收敛桶数据: τ_fb_true=3, p_err(k)=e^(−2k/3), k=0..3 → 拟合应恢复 τ≈3"""
    pts = [(0, 1.0), (1, math.exp(-2 / 3)), (2, math.exp(-4 / 3)), (3, math.exp(-2.0))]
    fit = 拟合τ_fb(pts)
    assert fit["ok"], fit
    assert abs(fit["tau_fb"] - 3.0) < 0.05, fit
    assert fit["r2"] > 0.99, fit
    # p=0 点剔除
    pts2 = [(0, 1.0), (1, math.exp(-2 / 3)), (2, 0.0), (3, 0.0)]
    f2 = 拟合τ_fb(pts2)
    assert f2["ok"] and abs(f2["tau_fb"] - 3.0) < 0.05, f2
    # 单点/平台/上升 → N/A
    assert not 拟合τ_fb([(0, 1.0)])["ok"]
    assert not 拟合τ_fb([(0, 1.0), (1, 1.0), (2, 1.0)])["ok"]  # 平台斜率0
    assert not 拟合τ_fb([(0, 0.2), (1, 0.5), (2, 0.9)])["ok"]  # 上升
    # 三态口径与块2 一致
    assert 三态分类([False, True, True], 4) == ("收敛", 2)
    assert 三态分类([True], 4) == ("未启动", None)
    assert 三态分类([False, True, True, False], 4) == ("不收敛", None)
    print("r4_fit 自测: 拟合恢复 τ≈3.00 单点/平台/上升 N/A 三态口径一致 全过")


def 主():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", default=os.path.join(ROOT, "experiments", "r4_rounds", "trajectories.jsonl"))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        自测()
        return
    out_dir = a.out_dir or os.path.dirname(os.path.abspath(a.trajectories))
    if not os.path.exists(a.trajectories):
        raise SystemExit(f"轨迹文件不存在: {a.trajectories}")
    # T-44 修2: 输出目录不自建会 FileNotFoundError
    os.makedirs(out_dir, exist_ok=True)
    主体(a.trajectories, out_dir, date.today().isoformat())


if __name__ == "__main__":
    主()
