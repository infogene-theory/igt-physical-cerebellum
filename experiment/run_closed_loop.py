# -*- coding: utf-8 -*-
r"""IGT 物理小脑 · 自包含闭环编排（去 r3_engine 依赖，物理口径 T-195）。

这是论文《给大模型装物理小脑》核心实验的**可复现编排**：用冻结题集与一个本地大模型，
端到端跑出三条臂，轨迹字段与 M1 对齐，可直接喂 experiment/analysis/r4_m1_stats.py 分析。

三条臂：
  A 闭环回灌：k0 读「证据锚」判读；错则按三级梯度（规则 → 模态 → 完整统计）造反馈，逐轮回灌；
  B 开环对照：每轮同一证据锚、**不回灌**（默认 4 轮），暴露无反馈时错误不收敛；
  D 无锚零点：只给判读任务句、**不给谱数据**，作为开环幻觉基线（单轮）。

客观方向由 Python 从 bench 证据锚的标定 z **独立判定**（present ⟺ z<−1），不依赖 r3_engine；
模型采样通过 experiment/harness 编译出的 r3-claim.exe（它再以子进程调用 llama.cpp 的 llama-cli）。

口径：物理口径（λ=ω−iγ，γ=−λ_im，稳定 γ>0、失稳 γ<0），详见仓库根 CALIBRATION.md。

【与原始 r4_rounds.py 的差异，诚实标注】
  - 去掉 r3_engine / 本地 server / 多模态图片等依赖，只留 local-cli 一条路径；
  - B 臂统一跑 4 轮（原始 r4 对 clear 箱为 2 轮）；B 无反馈、方向恒定，不影响结论；
  - 单样本（不做逐轮 N 次聚合）；95% CI 由 r4_m1_stats 的 bootstrap 给出；
  - 不跑 C（错锚）臂。

用法（实际运行）:
  python run_closed_loop.py --claim-bin .\harness\target\debug\r3-claim.exe \
      --llama-cli C:\...\llama-cli.exe --model C:\...\model.gguf \
      --model-name Qwen2.5-7B-Instruct --quant Q4_K_M --arms A,B,D
  python run_closed_loop.py --selftest      # 纯函数自测（不需模型）
  python run_closed_loop.py --dry ...      # 只打印将执行的命令，不采样
"""
import argparse
import json
import os
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OBS = "unstable_mode_count"
DIR_ENUM = "present,absent"
无锚题面 = ("对一段系统负载观测（不提供谱数据）。请判断该系统段是否存在增长失稳模（处于失稳状态）。"
          "输出 ClaimForCheck：observable=unstable_mode_count，direction 判 present/absent。")


def 物理方向(z):
    """物理口径：present ⟺ z<0 且 |z|>1（z<−1）；否则 absent。"""
    if z is None:
        return None
    return "present" if (z < 0.0 and abs(z) > 1.0) else "absent"


def 三态分类(corrects, Kmax):
    """三态分桶：未启动=k0 即对；收敛=k0 错且存在连续 2 轮对、其后不翻错；其余不收敛。"""
    if not corrects:
        return "不收敛", None
    if corrects[0]:
        return "未启动", None
    for i in range(len(corrects) - 1):
        if corrects[i] and corrects[i + 1] and all(corrects[i + 2:]):
            return "收敛", i + 1
    return "不收敛", None


def 造反馈(claim, k, kmax, b, ec, z明细):
    """三级梯度反馈（标定题，物理口径）；只给统计判据、不泄目标方向。"""
    偏离 = ("系统统计判读与您不符。请按题面「判读规则(标定口径)」复核：显著性看中心化 z=(γ−b)/ε_c——"
            "|z|≤1 判 absent、z<−1 判 present（γ<0 侧为失稳）；裸 γ 符号不作为判据(EDMD 地板偏置已计入 b)")
    档 = "规则" if k == 0 else ("末轮完整" if k == kmax - 1 else "模态")
    文本 = "[反馈档:%s] %s" % (档, 偏离)
    if 档 == "规则":
        文本 += ("；统计判读标准: 失稳判定看主带模 γ 是否相对零噪声地板显著——"
                 "中心化 z=(γ−%0.2e)/%0.2e, |z|>1 才判 present（γ<0 侧＝失稳）; 裸 γ 为正不等于 absent"
                 "（EDMD 地板偏置已计入 b）" % (b, ec))
    elif 档 == "模态":
        文本 += ("；判据量(噪声地板口径): γ_H0 中心 b=%0.2e、ε_c=%0.2e——"
                 "|γ−b|≤ε_c 即 z∈[−1,1] 在零分布内(判 absent); 越出才显著偏负(present); "
                 "请据此复核您对主带模符号的引用是否与中心化判读一致" % (b, ec))
    else:
        文本 += "；完整统计裁决(仅末轮): %s" % (z明细 or "主带模缺失, 判读不成立")
    return {"轮次": k + 1, "observable": OBS,
            "模型方向": (claim or {}).get("direction", ""),
            "偏离说明": 文本, "单位": (claim or {}).get("unit", "")}


class 编排:
    def __init__(self, a):
        self.a = a
        here = os.path.dirname(os.path.abspath(__file__))
        if not a.bench:
            a.bench = os.path.normpath(os.path.join(here, "..", "bench"))
        if not a.claim_bin:
            a.claim_bin = os.path.normpath(os.path.join(here, "harness", "target", "debug", "r3-claim.exe"))
        self.题集 = self.加载题()
        tag = (a.out_name or ("run_%s" % time.strftime("%Y%m%d_%H%M%S")))
        self.out_dir = os.path.join(here, "results", tag)
        os.makedirs(self.out_dir, exist_ok=True)
        self.traj_path = os.path.join(self.out_dir, "trajectories.jsonl")
        self.摘要路径 = os.path.join(self.out_dir, "summary.json")
        self.已完成 = self.读断点() if not a.fresh else set()
        self.摘要 = []

    def 加载题(self):
        ids = set(x.strip() for x in (self.a.ids or "").split(",") if x.strip())
        题 = []
        for n in sorted(os.listdir(self.a.bench)):
            if not (n.startswith("EV_") and n.endswith(".json")):
                continue
            tid = n[3:-5]
            if ids and tid not in ids:
                continue
            ev路径 = os.path.join(self.a.bench, n)
            ev = json.load(open(ev路径, encoding="utf-8"))
            标 = ev["标定"]
            z, 箱 = 标["中心化z"], 标.get("z箱")
            γ, _, _, _, z明细 = self.z明细(ev)
            题.append({"id": tid, "ev路径": ev路径, "ev": ev,
                       "z": z, "箱": 箱, "客观": 物理方向(z),
                       "b": 标["γ_H0中心b"], "ec": 标["ε_c"],
                       "γ": γ, "z明细": z明细,
                       "kmax": 1 if 箱 == "clear" else 3})
        return 题

    @staticmethod
    def z明细(ev):
        标 = ev["标定"]
        b, ec, z, 箱 = 标["γ_H0中心b"], 标["ε_c"], 标["中心化z"], 标.get("z箱")
        主带 = [m for m in ev.get("模谱", []) if abs(m["ω"] - 0.9) < 0.25]
        γ = min((m["γ"] for m in 主带), default=None)
        方向 = 物理方向(z)
        明细 = "γ=%+.6f b=%+.6f ε_c=%.2e z=%+.3f 箱=%s → %s" % (γ if γ is not None else float("nan"), b, ec, z, 箱, 方向)
        return γ, z, 箱, 方向, 明细

    def 读断点(self):
        done = set()
        if os.path.exists(self.traj_path):
            for line in open(self.traj_path, encoding="utf-8"):
                try:
                    x = json.loads(line)
                except Exception:
                    continue
                if x.get("round_k") == 0:
                    done.add((x.get("题id"), x.get("臂")))
        return done

    def 写轨迹(self, o):
        o["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.traj_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    def 采样(self, 题, 反馈, d臂):
        a = self.a
        out临时 = os.path.join(self.out_dir, "_claim_%s_%d.json" % (题["id"], len(反馈)))
        cmd = [a.claim_bin]
        if d臂:
            cmd += ["--prompt-only", 无锚题面]
        else:
            cmd += ["--anchor", 题["ev路径"], "--target-observable", OBS, "--direction-enum", DIR_ENUM]
            if 反馈:
                cmd += ["--feedback", json.dumps(反馈, ensure_ascii=False)]
        cmd += ["--out", out临时, "--llama-cli", a.llama_cli or "", "--model", a.model or "",
                "--model-name", a.model_name, "--quant", a.quant,
                "--max-token", str(a.max_token), "--timeout", str(a.timeout)]
        if a.raw_fail_dir:
            cmd += ["--raw-fail-dir", a.raw_fail_dir]
        if a.dry:
            print("    [dry] " + " ".join('"%s"' % c for c in cmd))
            return {"dry": True}
        try:
            r = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace",
                               timeout=a.timeout + 40)
        except subprocess.TimeoutExpired:
            return {"failed": True, "fail_reason": "编排层超时"}
        if r.returncode != 0:
            return {"failed": True, "fail_reason": "r3-claim rc=%d: %s" %
                    (r.returncode, (r.stderr or "")[:300])}
        if not os.path.exists(out临时):
            return {"failed": True, "fail_reason": "无 claim 输出"}
        return json.load(open(out临时, encoding="utf-8"))

    @staticmethod
    def 精简claim(claim):
        return {"observable": claim.get("observable"), "direction": claim.get("direction"),
                "value": claim.get("value"), "unit": claim.get("unit")}

    def 记一轮(self, 题, 臂, k, 反馈, claim, failed=None):
        if failed:
            self.写轨迹({"题id": 题["id"], "类": "system", "臂": 臂, "round_k": k,
                         "feedback条数": len(反馈), "claim": None, "答对": None, "correct": None,
                         "failed": True, "reason": failed,
                         "中心化z": 题["z"], "z箱": 题["箱"], "z判方向": 题["客观"], "z明细": 题["z明细"]})
            return None, None
        correct = claim.get("direction") == 题["客观"]
        答对 = bool(correct and claim.get("observable") == OBS)
        self.写轨迹({
            "题id": 题["id"], "类": "system", "臂": 臂, "round_k": k,
            "feedback条数": len(反馈), "claim": self.精简claim(claim),
            "关键证据引用": claim.get("关键证据引用"),
            "verdict": None,
            "reason": "Python 标定口径判据（无 r3_engine）：%s" % 题["z明细"],
            "答对": 答对, "correct": correct, "evidence_ok": None, "evidence_reason": None,
            "中心化z": 题["z"], "z箱": 题["箱"], "z判方向": 题["客观"], "z明细": 题["z明细"],
            "repeat_n": 1, "correct_freq": 1.0 if correct else 0.0, "一致率": 1.0,
        })
        return correct, claim.get("direction")

    def 跑A(self, 题):
        反馈, corrects, 方向们 = [], [], []
        k = 0
        while k <= 题["kmax"]:
            claim = self.采样(题, 反馈, False)
            if claim.get("failed"):
                self.记一轮(题, "A", k, 反馈, None, claim["fail_reason"])
                break
            if claim.get("dry"):
                return
            correct, dirc = self.记一轮(题, "A", k, 反馈, claim)
            corrects.append(correct)
            方向们.append(dirc)
            if k == 0 and correct and 题["箱"] == "clear":
                break
            if 题["箱"] != "clear" and k >= 1 and corrects[-1] and corrects[-2]:
                break
            if k >= 题["kmax"]:
                break
            反馈.append(造反馈(claim, k, 题["kmax"], 题["b"], 题["ec"], 题["z明细"]))
            k += 1
        self.收摘要(题, "A", corrects, 方向们, 题["kmax"])

    def 跑B(self, 题):
        corrects, 方向们 = [], []
        for k in range(self.a.b_rounds):
            claim = self.采样(题, [], False)
            if claim.get("failed"):
                self.记一轮(题, "B", k, [], None, claim["fail_reason"])
                break
            if claim.get("dry"):
                return
            correct, dirc = self.记一轮(题, "B", k, [], claim)
            corrects.append(correct)
            方向们.append(dirc)
        self.收摘要(题, "B", corrects, 方向们, self.a.b_rounds - 1)

    def 跑D(self, 题):
        claim = self.采样(题, [], True)
        if claim.get("failed"):
            self.记一轮(题, "D", 0, [], None, claim["fail_reason"])
            self.收摘要(题, "D", [], [], 0)
            return
        if claim.get("dry"):
            return
        correct, dirc = self.记一轮(题, "D", 0, [], claim)
        self.收摘要(题, "D", [correct], [dirc], 0)

    def 收摘要(self, 题, 臂, corrects, 方向们, kmax):
        态, 收敛轮 = (三态分类(corrects, kmax) if corrects else ("不收敛", None))
        翻转率 = None
        if len(方向们) >= 2:
            翻转率 = round(sum(1 for i in range(1, len(方向们))
                               if 方向们[i] and 方向们[i] != 方向们[i - 1]) / (len(方向们) - 1), 3)
        self.摘要.append({"题": 题["id"], "臂": 臂, "态": 态, "收敛轮": 收敛轮,
                          "corrects": corrects, "z箱": 题["箱"], "客观": 题["客观"], "翻转率": 翻转率})

    def run(self):
        arms = [x.strip() for x in self.a.arms.split(",") if x.strip()]
        for 题 in self.题集:
            for 臂 in arms:
                if (题["id"], 臂) in self.已完成:
                    print("跳过(已完成) %s %s" % (题["id"], 臂))
                    continue
                print("【%s · %s】客观=%s z=%+.2f 箱=%s" %
                      (题["id"], 臂, 题["客观"], 题["z"], 题["箱"]))
                {"A": self.跑A, "B": self.跑B, "D": self.跑D}[臂](题)
        if not self.a.dry:
            json.dump(self.摘要, open(self.摘要路径, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            self.打印汇总()
            print("\n轨迹：%s\n摘要：%s" % (self.traj_path, self.摘要路径))

    def 打印汇总(self):
        for 臂 in [x.strip() for x in self.a.arms.split(",") if x.strip()]:
            rows = [s for s in self.摘要 if s["臂"] == 臂]
            if not rows:
                continue
            末轮错 = sum(1 for s in rows if s["corrects"] and s["corrects"][-1] is False)
            收敛 = sum(1 for s in rows if s["态"] == "收敛")
            未启动 = sum(1 for s in rows if s["态"] == "未启动")
            不收敛 = sum(1 for s in rows if s["态"] == "不收敛")
            print("臂 %s: n=%d 末轮错=%d | 未启动 %d / 收敛 %d / 不收敛 %d" %
                  (臂, len(rows), 末轮错, 未启动, 收敛, 不收敛))


def 自测():
    P = F = 0

    def T(名, cond, 明细=""):
        nonlocal P, F
        if cond:
            P += 1
            print("  PASS %s" % 名)
        else:
            F += 1
            print("  FAIL %s  %s" % (名, 明细))

    print("【物理方向判定】")
    for z, 期望 in [(-1.642, "present"), (-0.181, "absent"), (-0.854, "absent"),
                    (-1.295, "present"), (0.067, "absent"), (2.4, "absent"), (None, None)]:
        T("z=%s → %s" % (z, 期望), 物理方向(z) == 期望)
    print("【三态分类】")
    T("k0即对=未启动", 三态分类([True, True, True], 3)[0] == "未启动")
    T("k0错后连续对=收敛", 三态分类([False, False, True, True], 3) == ("收敛", 3))
    T("恒错=不收敛", 三态分类([False, False, False, False], 3)[0] == "不收敛")
    T("收敛后翻错=不收敛", 三态分类([False, True, True, False], 3)[0] == "不收敛")
    print("【反馈构造】")
    f0 = 造反馈({"direction": "absent", "unit": "count"}, 0, 3, 1.98e-4, 1.25e-4, "明细")
    T("k0=规则档、不泄方向", f0["偏离说明"].startswith("[反馈档:规则]")
      and "present" not in f0["偏离说明"].split("复核")[0])
    f2 = 造反馈({"direction": "present", "unit": "count"}, 2, 3, 1.98e-4, 1.25e-4, "Z明细")
    T("末轮=完整档含明细", f2["偏离说明"].startswith("[反馈档:末轮完整]") and "Z明细" in f2["偏离说明"])
    print("\n%s：PASS=%d FAIL=%d" % ("全部通过" if F == 0 else "存在失败", P, F))
    return 0 if F == 0 else 1


def 主():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench")
    ap.add_argument("--claim-bin")
    ap.add_argument("--llama-cli", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--arms", default="A,B,D")
    ap.add_argument("--b-rounds", type=int, default=4)
    ap.add_argument("--ids", default="")
    ap.add_argument("--out-name", default="")
    ap.add_argument("--max-token", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--raw-fail-dir", default="")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return 自测()
    编排(a).run()
    return 0


if __name__ == "__main__":
    sys.exit(主())
