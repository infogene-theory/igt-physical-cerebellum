# -*- coding: utf-8 -*-
"""T-48 块1 产物 · ε_c 标定表（core EDMD γ 判读的两个标定常数）

来源: v10 出题刻度总表实测（240 帧, ω=0.9, n_cal=500 / n_val=500 校准集与验证集 seed 段互不重叠,
      每 (a,σn) 格 n=300; 留言板 2026-09-04【IDE T-48 出题刻度总表 v10】）。

口径(PM 终裁 B ＋ **γ符号统一令 v1.0 随迁**): 显著性一律用**中心化** z = (γ_EDMD − 零分布中心 b) ÷ ε_c
—— 相对零分布判显著, 而非相对 0。

★★ **γ 口径（统一令 v1.0，T-195 随迁；`igt_core/src/complex_dmd.rs:46-48`）**:
  - 新口径 **γ = −λ_im ＝耗散率，稳定 γ>0；失稳 ⟺ γ<0**（旧口径 γ_old＝λ_im 稳定负）；
  - **地板 b 随口径翻转**：`b_new = −b_old`（**正偏**）；**ε_c 不变**（中心化后取 |·|，对镜像不变）；
  - **中心化 z 随之反号**（`z_new = −z_old`）⇒ **判定亦反转**：
    **失稳（present）⟺ z<0 且显著**（旧口径为 z>0）—— 判据与地板同源翻转，结论保持连续。
  - 依据：理论侧 171 §一.1/§一.3 裁决。

箱位(按实测 z, z_set v2 PM 2026-09-04 收紧; **T-195 随迁为负区间**):
  clear  清晰箱  z < −1.3    期望 present, k0 即收(一致率闸 ≥0.8)
  fuzzy  模糊箱  −1.3 ≤ z ≤ −0.5  方向各半: present 侧 z∈[−1.3,−1.28] / absent 侧 z∈[−0.9,−0.5]
  below  不可分箱 z > −0.5   期望 absent(控 present 假阳性负样本), 不卡 k0
"""
import json
import os

# (噪声幅 σn, 窗长帧数, 角频率 ω) → 零分布中心 b 与 ε_c(中心化后 |·| 的 95 分位)
# ★ T-195: b 随 γ 口径翻转取反（b_new = −b_old，正偏）；ε_c 不变。
标定表 = {
    (0.2, 240, 0.9): {"b": 2.64e-4, "ε_c": 6.5e-5, "误报率": 0.052, "误报CI": (0.033, 0.071)},
    (1.0, 240, 0.9): {"b": 1.98e-4, "ε_c": 1.25e-4, "误报率": 0.064, "误报CI": (0.043, 0.085)},
    (3.0, 240, 0.9): {"b": 1.76e-4, "ε_c": 8.8e-5, "误报率": 0.060, "误报CI": (0.039, 0.081)},
}

箱阈值 = {"clear": -1.3, "fuzzy下": -1.3, "fuzzy上": -0.5}
# z_set v2.1(PM 2026-09-04 smoke9 repeat3 实测): fuzzy 反馈**方向不对称**——
#   z1.02 弱 present(补漏报方向) k0~k3 四轮反馈含末轮完整 Verdict 均撬不动, repeat3 一致率 1.0 非随机;
#   z1.28 present k0 直接对; z0.69 absent(压假阳性方向) k0/k1 假阳性→k2 转 absent、k3 锁定(收敛)。
# 故: present 侧(补漏报)下沿抬到 z≥1.28; z∈(0.9,1.28) 的 present 题单列「7B 反馈不可纠带·保守漏报」
#     (仍出题, 只描述翻转率、不进 τfb 收敛桶, 留作 M1 换模型对照点);
#     absent 侧(压假阳性)保留为上沿 z≤0.9, 是收敛桶主来源。判据/ε_c/core 一律不动。
# ★ T-195 随迁（γ 口径翻转 ⇒ z_new = −z_old）：阈值全部取负，语义不变（present 侧 |z| 更大）。
不可纠下, 不可纠上 = -1.28, -0.9
present下沿 = -1.28
absent上沿 = -0.9


def 子类(z, 方向=None):
    """z_set v2.1 出题子类: clear / fuzzy_present / fuzzy_absent / fuzzy_stuck(不可纠带) / below
    （★ T-195：判据为**负区间**——失稳 present ⟺ z 更负）"""
    if z < 箱阈值["clear"]:
        return "clear"
    if z <= present下沿:
        return "fuzzy_present"
    if z > 箱阈值["fuzzy上"]:
        return "below"
    if z < 不可纠上 and (方向 or "present") == "present":
        return "fuzzy_stuck"
    return "fuzzy_absent"


def 查表(σn, frames=240, ω=0.9):
    """(σn, 窗长, 频率) → {b, ε_c, 误报率, 误报CI}; 表内无精确键则报错(标定外不许出题)"""
    k = (float(σn), int(frames), round(float(ω), 3))
    if k not in 标定表:
        raise KeyError(f"标定表无 ({σn}, {frames}, {ω}) 格——须先跑 v10 标定再出题, 禁止外推")
    return dict(标定表[k])


def 中心化z(γ, σn, frames=240, ω=0.9):
    """z = (γ_EDMD − 零分布中心 b) ÷ ε_c（PM B 口径；★T-195：γ 与 b 同为**新口径**，失稳 ⟺ z<0）"""
    c = 查表(σn, frames, ω)
    return (float(γ) - c["b"]) / c["ε_c"]


def 箱(z):
    """★ T-195 随迁：阈值取负（z<−1.3 clear；z≤−0.5 fuzzy；否则 below）"""
    if z < 箱阈值["clear"]:
        return "clear"
    if z <= 箱阈值["fuzzy上"]:
        return "fuzzy"
    return "below"


def 显著(z, 阈=1.0):
    """中心化 z 是否达显著(ε_c 按 |·| 95 分位定义 ⇒ 阈 1.0 对应误报 5%)"""
    return abs(z) > 阈


def 判读方向(γ, σn, frames=240, ω=0.9):
    """按中心化 z 判 present/absent（模型侧与出题侧同一口径; 不要求裸 γ）
    ★ T-195 随迁：**失稳 present ⟺ z<0 且显著**（γ<0 侧；旧口径为 z>0）——见文件头口径说明。"""
    z = 中心化z(γ, σn, frames, ω)
    return "present" if (z < 0 and 显著(z)) else "absent"


def 导出表():
    return [{"σn": k[0], "frames": k[1], "ω": k[2], **v} for k, v in 标定表.items()]


if __name__ == "__main__":
    for 行 in 导出表():
        print(行)
