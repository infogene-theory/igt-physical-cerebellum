# -*- coding: utf-8 -*-
r"""生成《给大模型装物理小脑》5 张配图（学术浅色风，中文标注）。

数字一律取自论文 v1.3 附录 B（权威），不自行编造：
  图1 耗散方向各向异性（概念）  图2 三端管线流程
  图3 M0 闭环 vs 开环           图4 M1 四模型末轮错误率与 τfb
  图5 六失效条件
输出：仓库 figures/figN_*.png（dpi=170）
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["axes.edgecolor"] = "#c9d0d9"
plt.rcParams["axes.linewidth"] = 0.9

INK = "#1f2a3a"; BLUE = "#274b8f"; BLUE_L = "#8aa6d6"; GOLD = "#c08a2e"
GREY = "#9aa3b0"; RED = "#b3402f"; GREEN = "#3a7d5d"; BG_BOX = "#f3f6fb"


def 框(ax, x, y, w, h, text, fc=BG_BOX, ec=BLUE, fs=11, tc=INK, sub=None, sub_fs=8.6):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                       fc=fc, ec=ec, lw=1.4, mutation_aspect=1)
    ax.add_patch(b)
    if sub:
        ax.text(x + w / 2, y + h * 0.62, text, ha="center", va="center",
                fontsize=fs, color=tc, weight="bold")
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=sub_fs, color="#5a6675")
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, weight="bold")


def 箭头(ax, p1, p2, color=INK, lw=1.6, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle="arc3,rad=%s" % rad))


# ---------- 图1：耗散方向各向异性 ----------
def fig1(out):
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    import numpy as np
    xs = np.linspace(-3, 3, 9); ys = np.linspace(-3, 3, 9)
    X, Y = np.meshgrid(xs, ys)
    U = -0.75 * X          # 语言方向：强收敛
    V = -0.06 * Y          # 事实方向：几乎无拉力
    ax.quiver(X, Y, U, V, color=BLUE_L, scale=30, width=0.0035, alpha=0.85)
    ax.axhline(0, color=GREY, lw=1, ls=":")
    ax.axvline(0, color=GREY, lw=1, ls=":")
    # 语言方向（顶部右侧：强收敛）
    ax.annotate("语言连贯方向 L：Γ^L >> 0（强收敛）", xy=(1.55, 2.72),
                fontsize=11, color=BLUE, weight="bold", ha="center")
    ax.add_patch(FancyArrowPatch((0.25, 2.48), (2.85, 2.48), arrowstyle="-|>",
                 mutation_scale=16, color=BLUE, lw=2.2))
    # 事实方向（左侧：错误自由漂移、不收敛）
    ax.add_patch(FancyArrowPatch((-1.9, -2.3), (-1.9, 2.3), arrowstyle="-|>",
                 mutation_scale=16, color=RED, lw=2.0, linestyle="--"))
    ax.add_patch(Circle((-1.9, 1.5), 0.12, color=RED))
    ax.text(-1.5, 1.66, "一个事实错误", fontsize=9.5, color=RED,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"))
    ax.text(-3.05, -2.95, "事实校验方向 F：Γ^F ≈ 0（错误自由漂移、不收敛）",
            fontsize=10.4, color=RED, weight="bold", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.3, 3.3)
    ax.set_xlabel("→ 语言是否流畅"); ax.set_ylabel("→ 是否符合事实")
    ax.set_title("幻觉的谱图像：系统只把表达拉向“流畅”，不把错误拉回“事实”",
                 fontsize=12.5, weight="bold", color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)


# ---------- 图2：三端管线流程 ----------
def fig2(out):
    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    ax.set_xlim(0, 10.6); ax.set_ylim(0, 5.2); ax.axis("off")
    y, h, w = 3.0, 1.15, 1.62
    框(ax, 0.25, y, 1.4, h, "输入", sub="时序 / 传感器", ec=GREY)
    框(ax, 2.05, y, w, h, "core 小脑", sub="纯函数·零网络\nEDMD→γ/τ", ec=BLUE)
    框(ax, 4.15, y, w, h, "agent", sub="本地·可断网\n噪声地板/组prompt", ec=BLUE)
    框(ax, 6.25, y, w, h, "System-2", sub="大模型\n输出 Claim", ec=GOLD)
    框(ax, 8.35, y, w, h, "L2 裁决", sub="归一·比对\n放行/打回/人审", ec=GREEN)
    xs = [1.65, 3.67, 5.77, 7.87]
    for i, x in enumerate(xs):
        箭头(ax, (x, y + h / 2), (x + 0.38, y + h / 2))
    ax.text(3.0, y + h + 0.28, "AnchorEnvelope", ha="center", fontsize=8.6, color=BLUE)
    ax.text(7.2, y + h + 0.28, "ClaimForCheck", ha="center", fontsize=8.6, color=GOLD)
    # 闭环：L2 打回 → agent（下方回路）
    箭头(ax, (9.15, y), (4.95, 1.7), color=RED, lw=2.0, rad=0.25)
    箭头(ax, (4.95, 1.7), (4.95, y), color=RED, lw=2.0)
    ax.text(7.0, 1.15, "带偏离方向打回，真进下一次生成（闭环）", color=RED,
            fontsize=10, weight="bold", ha="center")
    ax.text(2.86, 0.55, "确定性计算", color=BLUE, fontsize=9, ha="center")
    ax.text(7.05, 0.55, "概率采样", color=GOLD, fontsize=9, ha="center")
    ax.set_title("工程三端：确定性小脑与概率模型物理隔离，纠错必须闭环",
                 fontsize=12.5, weight="bold", color=INK)
    fig.tight_layout(); fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)


# ---------- 图3：M0 闭环 vs 开环 ----------
def fig3(out):
    k = [0, 1, 2, 3]
    A = [1.00, 1.00, 0.75, 0.50]
    B = [1.00, 1.00, 1.00, 1.00]
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    ax.plot(k, A, "-o", color=BLUE, lw=2.3, ms=8, label="A 闭环（逐轮回灌反馈）")
    ax.plot(k, B, "--s", color=GREY, lw=2.0, ms=7, label="B 开环（同一锚、不回灌）")
    for x, y in zip(k, A):
        ax.text(x, y + 0.05, "%.2f" % y, ha="center", color=BLUE, fontsize=10)
    ax.text(3, 0.93, "四轮恒 1.00", ha="center", color="#6a7480", fontsize=10)
    ax.annotate("配对差 k3 = +0.50", xy=(3, 0.5), xytext=(1.7, 0.18),
                arrowprops=dict(arrowstyle="->", color=INK), fontsize=10)
    ax.text(0.98, 0.12, "τfb ≈ 5.8 轮（R²=0.99；真收敛 n=2，仅数量级）",
            transform=ax.transAxes, fontsize=9.6, color=GOLD, ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fbf3e2", ec=GOLD, lw=0.9))
    ax.set_ylim(0, 1.12); ax.set_xlim(-0.2, 3.4)
    ax.set_xticks(k); ax.set_xlabel("反馈轮 k"); ax.set_ylabel("错误率")
    ax.set_title("M0：闭环错误随轮下降，开环原地不动（Qwen2.5-7B / CPU）",
                 fontsize=12.3, weight="bold", color=INK)
    ax.legend(loc="upper right", frameon=False, fontsize=9.6)
    ax.grid(axis="y", ls=":", color="#d8dee6")
    fig.tight_layout(); fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)


# ---------- 图4：M1 四模型末轮错误率 + τfb ----------
def fig4(out):
    models = ["Qwen3-14B", "Qwen3-VL-8B", "Qwen2.5-7B", "gemma-3-12B"]
    A末 = [0.0, 0.0, 0.25, 0.0]
    B = [0.417, 0.50, 0.50, 0.458]
    tau = ["2.60", "4.37", "8.28", "17.93"]
    import numpy as np
    x = np.arange(len(models)); w = 0.36
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    b1 = ax.bar(x - w / 2, A末, w, label="A 闭环末轮错误率", color=BLUE)
    b2 = ax.bar(x + w / 2, B, w, label="B 开环错误率（四轮恒定）", color=GREY)
    for r, v in zip(b1, A末):
        ax.text(r.get_x() + w / 2, v + 0.015, "%.2f" % v, ha="center", fontsize=9.6, color=BLUE)
    for r, v in zip(b2, B):
        ax.text(r.get_x() + w / 2, v + 0.015, "%.2f" % v, ha="center", fontsize=9.6, color="#5a6470")
    for xi, t in zip(x, tau):
        ax.text(xi, -0.105, "τfb≈%s 轮" % t, ha="center", fontsize=9.2, color=GOLD,
                transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 0.62); ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("错误率")
    ax.set_title("M1：四模型闭环末轮（3/4 归零）vs 开环不降；τfb 仅数量级、CI 偏宽",
                 fontsize=12.0, weight="bold", color=INK)
    ax.legend(loc="upper left", frameon=False, fontsize=9.6)
    ax.grid(axis="y", ls=":", color="#d8dee6")
    ax.text(0.99, 0.95, "τfb 95%CI 下界均>0；跨口径不作精确排名",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.8, color="#6a7480")
    fig.tight_layout(); fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)


# ---------- 图5：六失效条件 ----------
def fig5(out):
    cards = [
        ("①", "锚点本身错", "错锚注入错误约束、反增错；锚须为 core 可复算量", RED),
        ("②", "方向正交", "小脑维度与出错维度无关，投影为零、增益为零", GOLD),
        ("③", "回路不闭合", "只并列展示、不回灌下一次生成（=开环，无效）", GREY),
        ("④", "超出锚定义域", "价值/意图/纯创造无物理锚，应放行不硬验", BLUE),
        ("⑤", "证据低于分辨阈", "不是锚错、是锚太弱，被噪声地板淹没", GREEN),
        ("⑥", "前端系统性读错", "事实在、但提取算子读错/读反，应拒绝准入", "#7a4fa0"),
    ]
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    ax.set_xlim(0, 10.8); ax.set_ylim(0, 5.4); ax.axis("off")
    w, h = 3.35, 2.05
    pos = [(0.25, 2.85), (3.72, 2.85), (7.2, 2.85),
           (0.25, 0.45), (3.72, 0.45), (7.2, 0.45)]
    for (num, title, desc, col), (x, y) in zip(cards, pos):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
                           fc="#f7f9fc", ec=col, lw=1.6)
        ax.add_patch(b)
        ax.text(x + 0.32, y + h - 0.42, num, fontsize=20, color=col, weight="bold")
        ax.text(x + 0.95, y + h - 0.45, title, fontsize=12.5, color=INK, weight="bold")
        ax.text(x + 0.22, y + 0.62, desc, fontsize=9.8, color="#4a5563",
                wrap=True, va="center")
    ax.set_title("六个失效条件：任一触发即无增益甚至反增错（实验前写死）",
                 fontsize=12.6, weight="bold", color=INK)
    fig.tight_layout(); fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    figs = [(fig1, "fig1_耗散方向各向异性.png"),
            (fig2, "fig2_三端管线流程.png"),
            (fig3, "fig3_M0闭环vs开环.png"),
            (fig4, "fig4_M1四模型末轮错误率.png"),
            (fig5, "fig5_六失效条件.png")]
    for fn, name in figs:
        p = os.path.join(here, name)
        fn(p)
        print("已生成 %s" % p)


if __name__ == "__main__":
    main()
