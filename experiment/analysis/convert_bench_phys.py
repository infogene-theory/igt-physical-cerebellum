# -*- coding: utf-8 -*-
"""T-195 口径统一 · 把 bench 冻结题集证据锚从「代码口径」转换到「物理口径」（确定性镜像，可审计）。

【背景，诚实记录】
  M0（2026-09-04）/ M1（2026-09-09）实验执行于 T-195「γ 符号统一令」（2026-09-28）**之前**，
  故 bench 复制来的冻结证据锚 EV 用的是当时的代码口径：模谱第三元 g = λ_im（失稳 g>0）、
  标定中心化 z 为正区间（present ⟺ z>+1）。
  T-195 后 igt_core 与论文正文统一为物理口径：γ = −λ_im = −g（耗散率，稳定 γ>0、失稳 γ<0）、
  present ⟺ z<−1。
本脚本按 T-195 的**纯符号镜像规则**转换 EV（不重算、不拟合、不改变任何结论），
使 bench 与当前 igt_core / 论文口径一致。

【转换规则】（来源：γ 符号统一令 v1.0、理论侧 171 裁决、igt_core/src/complex_dmd.rs:46-48）
  模谱[].γ        : g → −g（物理 γ）
  标定.γ_H0中心b  : b → −b
  标定.中心化z    : z → −z
  保持不变        : ε_c、τ数值、对照τ数值、频率、ω、z箱名、谱质量、window、anchor_id
  说明：τ = 1/γ = −1/g 恒等，故 τ 数值不变；错误率 / 收敛轮 τfb（轮）等结论口径无关。

【历史件归档】
  转换前的代码口径 EV、裁决锚 {id}.anchor.json（c_result.eigen_pairs 第三元为 λ_im）
  备份到 experiment/results/legacy_code_tier/，并在其 README 标注口径。

用法:
  python convert_bench_phys.py --repo <仓库根>            # dry-run，只打印将做的改动
  python convert_bench_phys.py --repo <仓库根> --apply    # 执行转换与归档
  python convert_bench_phys.py --repo <仓库根> --selftest # 纯函数自测
"""
import argparse
import json
import os
import shutil
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def 读json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def 写json(p, o):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f, ensure_ascii=False, indent=1)
        f.write("\n")


def 物理方向(z):
    """物理口径客观方向：present ⟺ z<0 且 |z|>1（即 z<−1）；否则 absent。"""
    if z is None:
        return None
    return "present" if (z < 0.0 and abs(z) > 1.0) else "absent"


def 转换EV(o):
    """把单个代码口径 EV 深拷贝后转为物理口径。返回 (新对象, 变更字段清单)。"""
    原 = json.loads(json.dumps(o, ensure_ascii=False))
    新 = json.loads(json.dumps(o, ensure_ascii=False))
    变更 = []
    # ① 模谱 γ: g → −g
    for m in 新.get("模谱", []):
        if m.get("γ") is not None:
            m["γ"] = -float(m["γ"])
            变更.append("模谱[%d].γ 反号" % m.get("索引", 0))
    # ②③ 标定 b、z 反号；ε_c 与其余不变
    标 = 新.get("标定") or {}
    if 标:
        if 标.get("γ_H0中心b") is not None:
            标["γ_H0中心b"] = -float(标["γ_H0中心b"])
            变更.append("标定.b 反号")
        if 标.get("中心化z") is not None:
            标["中心化z"] = round(-float(标["中心化z"]), 3)
            变更.append("标定.z 反号")
        标["口径"] = ("物理口径（T-195 γ符号统一令）：λ=ω−iγ，γ=−λ_im，稳定 γ>0、失稳 γ<0；"
                  "中心化 z=(γ−b)/ε_c，present ⟺ z<−1，|z|≤1 判 absent；τ=1/γ。"
                  "（代码旧口径 g=−γ，present ⟺ z>+1，二者镜像等价。）")
        新["标定"] = 标
    新["口径版本"] = "物理口径 v2（T-195，2026-09-28）"
    # 不变量自检：ε_c、τ数值、频率 必须与原件逐字相等
    assert (新.get("标定") or {}).get("ε_c") == (原.get("标定") or {}).get("ε_c"), "ε_c 必须不变"
    assert 新.get("τ数值") == 原.get("τ数值"), "τ数值 必须不变"
    assert 新.get("对照τ数值") == 原.get("对照τ数值"), "对照τ数值 必须不变"
    assert 新.get("频率") == 原.get("频率"), "频率 必须不变"
    assert 新.get("anchor_id") == 原.get("anchor_id"), "anchor_id 必须不变"
    return 新, 变更


LEGACY_README = """# 代码口径历史件（T-195 之前）

本目录保存 M0 / M1 实验当时（T-195「γ 符号统一令」2026-09-28 **之前**）的**代码口径**原始件，
仅作历史溯源，**不是当前复现实验的输入**。当前复现请用仓库 `bench/` 内已转为物理口径的证据锚。

## 口径对照

| 量 | 代码口径（本目录，T-195 前） | 物理口径（当前 bench / core / 论文） |
|---|---|---|
| 模谱第三元 | g = λ_im，**失稳 g>0**、稳定 g<0 | γ = −λ_im = −g，**失稳 γ<0**、稳定 γ>0 |
| 中心化 z | 正区间，**present ⟺ z>+1** | 负区间，**present ⟺ z<−1** |
| 弛豫时间 | τ = −1/λ_im | τ = 1/γ（与左式**数值恒等**） |

镜像换算：γ = −g、b 反号、z 反号、ε_c 不变；τ 数值与错误率、收敛轮 τfb（轮）等**结论口径无关**。

## 内容

- `ev/`：转换前的代码口径证据锚 EV_{id}.json（模谱 γ 字段即 g=λ_im）。
- `anchors/`：裁决锚 {id}.anchor.json（`c_result.eigen_pairs` 第三元为 λ_im；含 right_vectors/c_result）。

理论来源：γ 符号统一令 v1.0、理论侧 171 裁决、igt_core/src/complex_dmd.rs:46-48。
转换脚本：`experiment/analysis/convert_bench_phys.py`。
"""


def 列题(bench):
    evs = sorted(n for n in os.listdir(bench) if n.startswith("EV_") and n.endswith(".json"))
    ids = [n[3:-5] for n in evs]
    return evs, ids


def 主():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="仓库根（含 bench/ 与 experiment/）")
    ap.add_argument("--apply", action="store_true", help="执行（缺省 dry-run）")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    bench = os.path.join(a.repo, "bench")
    legacy = os.path.join(a.repo, "experiment", "results", "legacy_code_tier")

    if a.selftest:
        return 自测(bench)

    evs, ids = 列题(bench)
    print("发现 EV %d 个：%s" % (len(evs), ", ".join(ids)))
    for ev名 in evs:
        tid = ev名[3:-5]
        原 = 读json(os.path.join(bench, ev名))
        新, 变更 = 转换EV(原)
        z新 = (新.get("标定") or {}).get("中心化z")
        print("  %s: %s；转后 z=%s → %s" % (tid, "、".join(变更), z新, 物理方向(z新)))
        裁决 = os.path.join(bench, "%s.anchor.json" % tid)
        if os.path.exists(裁决):
            print("      裁决锚 %s.anchor.json → 归档 legacy/anchors/" % tid)

    if not a.apply:
        print("\n[dry-run] 未写入。加 --apply 执行转换、备份与归档。")
        return 0

    os.makedirs(os.path.join(legacy, "ev"), exist_ok=True)
    os.makedirs(os.path.join(legacy, "anchors"), exist_ok=True)
    for ev名 in evs:
        tid = ev名[3:-5]
        p = os.path.join(bench, ev名)
        原 = 读json(p)
        写json(os.path.join(legacy, "ev", ev名), 原)          # 备份代码口径 EV
        新, _ = 转换EV(原)
        写json(p, 新)                                        # 写回物理口径 EV
        裁决 = os.path.join(bench, "%s.anchor.json" % tid)
        if os.path.exists(裁决):
            shutil.move(裁决, os.path.join(legacy, "anchors", "%s.anchor.json" % tid))
    写json_p = os.path.join(legacy, "README.md")
    with open(写json_p, "w", encoding="utf-8") as f:
        f.write(LEGACY_README)
    print("\n[apply] 已转换 %d 个 EV 为物理口径；代码口径件归档至 %s" % (len(evs), legacy))
    return 0


def 自测(bench):
    P = F = 0

    def T(名,  cond, 明细=""):
        nonlocal P, F
        if cond:
            P += 1
            print("  PASS %s" % 名)
        else:
            F += 1
            print("  FAIL %s  %s" % (名, 明细))

    # 用代码口径样例验证转换
    样例 = {
        "ZCL001": "present",   # clear
        "ZBL001": "absent",    # below
        "ZFA002": "absent",    # fuzzy_absent
        "ZFP001": "present",   # fuzzy_present
        "ZFS001": "present",   # fuzzy_stuck
    }
    print("【EV 代码口径 → 物理口径 转换自测】")
    for tid, 期望 in 样例.items():
        p = os.path.join(bench, "EV_%s.json" % tid)
        if not os.path.exists(p):
            T("%s 存在" % tid, False, "缺文件（可能已转换，自测请在 --apply 前对原始 bench 跑）")
            continue
        原 = 读json(p)
        新, 变更 = 转换EV(原)
        z新 = (新.get("标定") or {}).get("中心化z")
        方向 = 物理方向(z新)
        T("%s 转后方向=%s（z=%s）" % (tid, 期望, z新), 方向 == 期望,
          "得 %s" % 方向)
        # 主带模（ω≈0.9）γ：方向**唯一由中心化 z 判定**（相对零噪声地板），不要求裸 γ 符号——
        # 强信号 clear 失稳 γ<0（已指数增长）；fuzzy 临界 present 时 γ 可正（弱衰减、显著低于地板 b）。
        主带 = [m for m in 新.get("模谱", []) if abs(m["ω"] - 0.9) < 0.25]
        if 主带:
            g = 主带[0]["γ"]
            if tid == "ZCL001":
                T("%s clear 增长模 γ<0（已失稳）" % tid, g < 0, "γ=%s" % g)
            elif tid in ("ZFP001", "ZFS001"):
                T("%s 临界弱衰减：γ>0 但 z<−1（present 由 z 判，非裸符号）" % tid,
                  g > 0 and z新 < -1.0, "γ=%s z=%s" % (g, z新))
            else:
                T("%s 主带 γ 未显著低于地板（absent）" % tid, 方向 == "absent", "γ=%s" % g)
        # 不变量
        T("%s τ数值/ε_c/频率 不变" % tid,
          新.get("τ数值") == 原.get("τ数值")
          and (新.get("标定") or {}).get("ε_c") == (原.get("标定") or {}).get("ε_c")
          and 新.get("频率") == 原.get("频率"))
    print("\n%s：PASS=%d FAIL=%d" % ("全部通过" if F == 0 else "存在失败", P, F))
    return 0 if F == 0 else 1


if __name__ == "__main__":
    sys.exit(主())
