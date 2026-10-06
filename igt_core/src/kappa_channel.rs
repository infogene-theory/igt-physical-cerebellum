// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! κ 通道域 — 集体耦合 / κ_ij 测量 **纯计算**（T-71 段① 落地, 图边真实化唯一合法来源）
//!
//! **T-67/T-71 口径（理论侧续41/续42·T-72 消歧）**:
//!   - 状态空间 L 非对角元 = 构造合成 L 时的 off-diagonal 元素 → **仅合成靶**（零a 演示）;
//!   - 模态空间 κ_ij = w_i† δL v_j（左右矢 + 非对角耦合投影, D5+D6/P34）→ **真实测量唯一合法来源**;
//!   - ⛔ 严禁用单通道 HRV 验证 κ: 单通道物理上 κ≡0,
//!     全 0 是平凡成立、换错实现也全 0 ⇒ 零信息;
//!   - 段①只许用「合成含非对角复 L」（已知 κ 真值）对账。
//!
//! 理论锚点: D5/D6（κ=耦合）/ P34（模态投影）/ P12 §3.4（|κ|τ<10 安全区, ≥10 失锁）/
//!   D3（虚部=耗散耦合 Γ 非对角）/ P35（主权边界 κ_ij=κ_ji† 权责对等）。
//! core 五类约束（ARCH-1 v2.1 §5）: 无 fs/net/时间/随机, 纯计算。

#![allow(non_snake_case)]

use nalgebra::Complex as NComplex;
use nalgebra::DMatrix;

use crate::nonhermitian_spectrum::非厄米谱求解器;

/// 耦合项 κ_ij（复值, 实部=能量耦合 Re / 虚部=耗散耦合 Im, D3）
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct 耦合项 {
    /// 源索引 i
    pub i: usize,
    /// 目标索引 j
    pub j: usize,
    /// Re(κ_ij)——能量耦合（H 侧）
    pub 实部: f64,
    /// Im(κ_ij)——耗散耦合（Γ 非对角, D3 虚部语义）
    pub 虚部: f64,
}

impl 耦合项 {
    /// κ 模长 |κ|
    pub fn 模(&self) -> f64 {
        (self.实部 * self.实部 + self.虚部 * self.虚部).sqrt()
    }
}

/// 耦合矩阵（实虚分离存储, 行主序 n×n）
#[derive(Debug, Clone, PartialEq)]
pub struct 耦合矩阵 {
    /// 阶数 n
    pub 阶: usize,
    /// 实部（n×n 行主序）
    pub 实部: Vec<f64>,
    /// 虚部（n×n 行主序）
    pub 虚部: Vec<f64>,
}

impl 耦合矩阵 {
    /// 取 (i,j) 复值
    pub fn 取(&self, i: usize, j: usize) -> NComplex<f64> {
        NComplex::new(self.实部[i * self.阶 + j], self.虚部[i * self.阶 + j])
    }
}

/// ★ EP（异常点）条件数告警门限: 最大 Petermann 因子 K_max≥此值 ⇒ 本征矢病态、
/// κ 边数值不可信，须**标注、不静默给边**（理论侧续44 第四点③）。K=1 为正规阵。
pub const EP条件数门限: f64 = 10.0;

/// 模态空间 κ 测量结果（T-71: κ_ij = w_i† δL v_j, 完整复）
#[derive(Debug, Clone, PartialEq)]
pub struct κ测量结果 {
    /// 阶数 r
    pub 阶: usize,
    /// κ 实部（r×r 行主序; 对角=自耦合可非零, 0 表示闭环受限）
    pub κ实部: Vec<f64>,
    /// κ 虚部（r×r 行主序）
    pub κ虚部: Vec<f64>,
    /// 每模 ω（复平面坐标, 供对账索引映射）
    pub ω: Vec<f64>,
    /// 每模 γ＝**耗散率**（γ=−Im λ，稳定 > 0；γ符号统一令后口径；旧版存 Im λ 稳定负）
    pub γ: Vec<f64>,
    /// 每模 τ（=1/γ；数值不变; 供 P12 §3.4 |κ|τ 判据）
    pub τ: Vec<f64>,
    /// ★ 全模最大 Petermann 因子 K_max（EP 条件数; 1=良好, 趋大=近异常点本征矢病态）。
    /// 段②上图前用 `病态告警()` 判定：true 时标注不可信、不静默给边。
    pub 最大条件数: f64,
}

impl κ测量结果 {
    /// EP 邻域病态告警（T-71 段②准入③）: K_max ≥ `EP条件数门限` 即 true。
    /// 与"不得静默落绿"同纪律——数值不稳只可标注、不得当正常边画出。
    pub fn 病态告警(&self) -> bool {
        self.最大条件数 >= EP条件数门限
    }
}

// ============================================================================
// T-72 κ 相位基准规范（理论侧裁决 v1.0，Damon 拍板「分对象裁决·类 III→乙」）
//
// 逐模自由相位 v_j→e^{iθ_j}v_j、w_i→e^{−iθ_i}w_i 下  κ_ij→e^{i(θ_j−θ_i)}κ_ij：
//   · 对角 κ_jj 相位自抵消（e^{i(θ_j−θ_j)}=1）⇒ κ_jj 本就规范不变（Re 频移/Im 线宽，物理，给节点）；
//   · 非对角 i≠j 带模间自由相位 ⇒ 单边 Re/Im 随任意 θ 变、**不是物理量**。
// 故**类 III 封闭内禀图只呈现下列不变量**；下面类型**故意不含非对角 Re/Im 字段**——
// 类 III 取数路径在类型层就拿不到单边实/虚部（"禁保守/耗散边着色、禁首分量派生量"）。
// 复值 κ 算子本身保留（类 II 单模对角相对物理外扰 φ_d 的同相/正交仍需 Re/Im，走另一条接口）。
// ============================================================================

/// 方向比判据门限 R_c（规范 §二.2）：R_ij=|κ_ij|/|κ_ji|，|R|≥R_c 记强单向（消费侧判据用）。
pub const 方向比门限: f64 = 3.0;

/// 类 III 图边（i≠j）允许暴露的**规范不变量**（无 Re/Im 字段——单边量类型层不可达）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct 不变量边 {
    pub i: usize,
    pub j: usize,
    /// |κ_ij|（规范不变）
    pub κ模: f64,
    /// |κ_ij|·τ_j（P12 §3.4 安全量；τ_j 非有限时为 NaN，不进判据）
    pub κ模τ_j: f64,
    /// 有向性 R_ij = |κ_ij|/|κ_ji|（配 `方向比门限` R_c=3）
    pub 方向比: f64,
    /// 两模环模 |κ_ij κ_ji|（幅角也不变，见 `两模环幅角`；此处给模）
    pub 两模环模: f64,
    /// 两模环幅角 arg(κ_ij)+arg(κ_ji)（自由相位下 (θ_j−θ_i)+(θ_i−θ_j)=0，不变）
    pub 两模环幅角: f64,
}

/// 有向三环（r≥3）：环 κ_ab κ_bc κ_ca；自由相位望远镜和为 0，模与幅角皆规范不变。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct 有向三环 {
    pub a: usize,
    pub b: usize,
    pub c: usize,
    pub 环模: f64,
    pub 环幅角: f64,
}

/// 节点（对角）不变量：κ_jj 相位自抵消，复值整体规范不变（实=频移/虚=线宽，物理）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct 节点不变量 {
    pub j: usize,
    /// 对角 κ_jj（复数，规范不变）
    pub κjj实: f64,
    pub κjj虚: f64,
    pub κjj模: f64,
    pub τ: f64,
}

/// 类 III 图的**唯一合法呈现载体**（T-72 乙）：只含规范不变量 + 两道闸标志。
///
/// - **规范闸**：本结构存在即过（构造时就只放不变量，调用方拿不到非对角裸 Re/Im）；
/// - **病态闸**：`可信=false`（K_max>`EP条件数门限`，近 EP）⇒ 连不变量也**灰显挂起、不参与判据 A**。
#[derive(Debug, Clone, PartialEq)]
pub struct κ不变量结果 {
    pub 阶: usize,
    pub 边: Vec<不变量边>,
    pub 有向三环: Vec<有向三环>,
    pub 节点: Vec<节点不变量>,
    pub 最大条件数: f64,
    /// 两道闸合取：规范闸（恒真）∧ 病态闸（K_max≤门限）。false＝病态挂起，边灰显不入判据。
    pub 可信: bool,
}

/// ★ T-72 纯函数：复 κ（行主序 m×m，实/虚分离）＋τ＋最大 Petermann K → 类 III 规范不变量。
///
/// 无状态/无 IO（core 纯函数库定位）。**不读、不返回任何非对角 Re/Im**；对角 κ_jj 整体保留
/// （相位自抵消、物理）。`κ实部/κ虚部` 长度须 = m² 且与 τ 等长，否则返回 None（不静默造）。
pub fn κ不变量(κ实部: &[f64], κ虚部: &[f64], τ: &[f64], 最大条件数: f64) -> Option<κ不变量结果> {
    let n2 = κ实部.len();
    if n2 != κ虚部.len() || n2 == 0 {
        return None;
    }
    let m = (n2 as f64).sqrt() as usize;
    if m * m != n2 || τ.len() != m {
        return None;
    }
    let at = |i: usize, j: usize| -> NComplex<f64> { NComplex::new(κ实部[i * m + j], κ虚部[i * m + j]) };
    let 模 = |z: NComplex<f64>| z.norm();

    // 边（全部 i≠j 有向边；只放不变量）
    let mut 边 = Vec::with_capacity(m * m.saturating_sub(1));
    for i in 0..m {
        for j in 0..m {
            if i == j {
                continue;
            }
            let kij = at(i, j);
            let kji = at(j, i);
            let km = 模(kij);
            let 环 = kij * kji; // κ_ij κ_ji（两模环）
            边.push(不变量边 {
                i,
                j,
                κ模: km,
                κ模τ_j: if τ[j].is_finite() && τ[j] > 0.0 { km * τ[j] } else { f64::NAN },
                方向比: km / 模(kji).max(1e-300),
                两模环模: 模(环),
                两模环幅角: 环.arg(),
            });
        }
    }

    // 有向三环（a<b<c，正向 a→b→c→a；r≥3）
    let mut 有向三环 = Vec::new();
    for a in 0..m {
        for b in (a + 1)..m {
            for c in (b + 1)..m {
                let 环 = at(a, b) * at(b, c) * at(c, a);
                有向三环.push(有向三环 { a, b, c, 环模: 模(环), 环幅角: 环.arg() });
            }
        }
    }

    // 节点对角（κ_jj 规范不变，复值物理量）
    let 节点 = (0..m)
        .map(|j| {
            let z = at(j, j);
            节点不变量 { j, κjj实: z.re, κjj虚: z.im, κjj模: 模(z), τ: τ[j] }
        })
        .collect();

    Some(κ不变量结果 {
        阶: m,
        边,
        有向三环,
        节点,
        最大条件数,
        可信: 最大条件数 <= EP条件数门限,
    })
}

/// 集体耦合纯计算 trait（T-71 落地实现; 全部关联函数/纯/无 IO）
pub trait CollectiveCoupling {
    /// 耦合项列表 → 非对角耦合矩阵（对角置零）
    fn 构造非对角(项: &[耦合项], 阶: usize) -> Option<耦合矩阵>;
    /// κτ 安全判据（P12 §3.4: |κ|τ<10 安全区, ≥10 失锁）
    fn 安全判据(κ模: f64, τ: f64) -> bool;
    /// 厄米/权责对等校验（P35: κ_ij = κ_ji†）
    fn 厄米校验(项: &[耦合项]) -> bool;
    /// 模态空间 κ 测量（D5+D6/P34）: κ_ij = w_i† δL v_j
    /// l_flat = 复 L 实虚交错 [2·r·r]; δL = L - diag(L); w/v 取分解完整复左右矢
    fn κ测量(l_flat: &[f64], 阶: usize) -> Option<κ测量结果>;
}

/// 默认实现（T-71）——纯计算、确定性。
pub struct 集体耦合实现;

impl CollectiveCoupling for 集体耦合实现 {
    fn 构造非对角(项: &[耦合项], 阶: usize) -> Option<耦合矩阵> {
        if 阶 == 0 {
            return None;
        }
        let n = 阶;
        let mut 实部 = vec![0.0f64; n * n];
        let mut 虚部 = vec![0.0f64; n * n];
        for 项 in 项 {
            if 项.i >= n || 项.j >= n {
                return None;
            }
            实部[项.i * n + 项.j] = 项.实部;
            虚部[项.i * n + 项.j] = 项.虚部;
            // P35 对等: 共轭回填（覆盖显式/隐式方向, 保证权责对等）
            实部[项.j * n + 项.i] = 项.实部;
            虚部[项.j * n + 项.i] = -项.虚部;
        }
        Some(耦合矩阵 { 阶: n, 实部, 虚部 })
    }

    fn 安全判据(κ模: f64, τ: f64) -> bool {
        if κ模 < 0.0 || τ <= 0.0 || !κ模.is_finite() || !τ.is_finite() {
            return false;
        }
        κ模 * τ < 10.0 // P12 §3.4: <10 安全区; ≥10 失锁相变
    }

    fn 厄米校验(项: &[耦合项]) -> bool {
        for a in 项 {
            for b in 项 {
                // κ_ij == conj(κ_ji)
                if a.i == b.j && a.j == b.i {
                    let conj_ok = (a.实部 - b.实部).abs() < 1e-12
                        && (a.虚部 + b.虚部).abs() < 1e-12;
                    if !conj_ok {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn κ测量(l_flat: &[f64], 阶: usize) -> Option<κ测量结果> {
        if 阶 < 2 || l_flat.len() < 2 * 阶 * 阶 {
            return None;
        }
        let r = 阶;
        let mut l = DMatrix::<NComplex<f64>>::zeros(r, r);
        for i in 0..r {
            for j in 0..r {
                let k = 2 * (i * r + j);
                l[(i, j)] = NComplex::new(l_flat[k], l_flat[k + 1]);
            }
        }
        let 分解 = 非厄米谱求解器::default().分解(&l).ok()?;
        let m = 分解.模.len();
        if m == 0 {
            return None;
        }
        // W[列=左矢模], V[列=右矢模]: 完整复
        let 复 = |实: &Vec<f64>, 虚: &Vec<f64>, i: usize| -> NComplex<f64> {
            NComplex::new(实[i], *虚.get(i).unwrap_or(&0.0))
        };
        let mut W = vec![NComplex::new(0.0, 0.0); m * m];
        let mut V = vec![NComplex::new(0.0, 0.0); m * m];
        for k in 0..m {
            for a in 0..m {
                W[a * m + k] = 复(&分解.模[k].w, &分解.模[k].w虚, a);
                V[a * m + k] = 复(&分解.模[k].v, &分解.模[k].v虚, a);
            }
        }
        // δL = L - diag
        let mut dL = vec![NComplex::new(0.0, 0.0); m * m];
        for i in 0..m {
            for j in 0..m {
                dL[i * m + j] = if i == j {
                    NComplex::new(0.0, 0.0)
                } else {
                    l[(i, j)]
                };
            }
        }
        // κ = W† δL V（列矢约定: Wmat/Vmat 第 k 列 = 第 k 模的 w_k/v_k）。
        //   κ_ij = Σ_ab conj(w_i[a]) · δL[a,b] · v_j[b]
        //   [W†]_{ia}=conj(Wmat[a,i])=W[a·m+i].conj();  Vmat[b,j]=V[b·m+j]。
        //   （早期误写 W[i·m+a]=w_a[i] 属左右矢索引转置, 已修）
        let mut κ = vec![NComplex::new(0.0, 0.0); m * m];
        for i in 0..m {
            for j in 0..m {
                let mut acc = NComplex::new(0.0, 0.0);
                for a in 0..m {
                    for b in 0..m {
                        acc += W[a * m + i].conj() * dL[a * m + b] * V[b * m + j];
                    }
                }
                κ[i * m + j] = acc;
            }
        }
        let mut 实部 = vec![0.0f64; m * m];
        let mut 虚部 = vec![0.0f64; m * m];
        for i in 0..m {
            for j in 0..m {
                实部[i * m + j] = κ[i * m + j].re;
                虚部[i * m + j] = κ[i * m + j].im;
            }
        }
        let 最大条件数 = 分解
            .模
            .iter()
            .map(|x| x.petermann)
            .fold(1.0f64, f64::max);
        Some(κ测量结果 {
            阶: m,
            κ实部: 实部,
            κ虚部: 虚部,
            ω: 分解.模.iter().map(|x| x.ω).collect(),
            // ★ γ符号统一令 §4.1：κ测量结果的 γ 改为**耗散率**（γ=−Im λ，稳定 > 0；旧＝Im λ 稳定负）
            //   τ 数值不变（1/γ ≡ −1/Im λ）。FFI 消费方（engine l4, 冻结）对接时按此口径换算。
            γ: 分解.模.iter().map(|x| x.γ).collect(),
            τ: 分解.模.iter().map(|x| x.τ).collect(),
            最大条件数,
        })
    }
}

// ============================================================================
// 单测（T-71 段①: 只允许合成含非对角复 L 对账, ⛔严禁单通道 HRV）
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// T-71: 构造非对角 + P35 共轭回填
    #[test]
    fn 测试_构造非对角_对等() {
        let 项 = vec![
            耦合项 { i: 0, j: 1, 实部: 0.5, 虚部: -0.2 },
            耦合项 { i: 2, j: 1, 实部: -0.3, 虚部: 0.1 },
        ];
        let m = 集体耦合实现::构造非对角(&项, 3).expect("应构造");
        assert_eq!(m.阶, 3);
        assert!((m.取(0, 1).re - 0.5).abs() < 1e-12, "κ01 实部");
        assert!((m.取(1, 0).im - 0.2).abs() < 1e-12, "κ10 虚部=+0.2（共轭回填）");
        assert!((m.取(0, 0).re).abs() < 1e-12, "对角置零");
        assert!(集体耦合实现::厄米校验(&项), "κ_ij=conj(κ_ji) 成立");
    }

    /// T-71: P12 §3.4 |κ|τ 安全判据
    #[test]
    fn 测试_安全判据() {
        assert!(集体耦合实现::安全判据(0.5, 10.0), "|κ|τ=5<10 安全");
        assert!(!集体耦合实现::安全判据(2.0, 10.0), "|κ|τ=20≥10 失锁");
        assert!(!集体耦合实现::安全判据(-1.0, 5.0), "κ 模负 → false");
        assert!(!集体耦合实现::安全判据(1.0, 0.0), "τ≤0 → false");
    }

    /// 合成「含非对角复 L」（三恒等式/精度回归夹具）: 对角 ω−i/τ 充分分离（H 与 Γ 都强对角占优）
    /// ＋一般复非对角（非 Hermitian, 含显著虚部）。⛔ 非单通道 HRV。
    fn 合成含非对角复L(r: usize, ε: f64) -> Vec<f64> {
        // λ_im 等距强分离（避免长 τ 端挤一起致 Γ 不占优）; 全稳定 λ_im<0（γ=−λ_im>0）
        let ωs = [0.20, 0.60, 1.00, 1.40];
        let λ_ims = [-0.80, -0.50, -0.30, -0.15];
        let mut flat = vec![0.0f64; 2 * r * r];
        for i in 0..r {
            flat[2 * (i * r + i)] = ωs[i];
            flat[2 * (i * r + i) + 1] = λ_ims[i];
        }
        for i in 0..r {
            for j in (i + 1)..r {
                let cre = 0.05 + 0.02 * (i as f64 + j as f64);
                let cim = -0.03 + 0.01 * (i as f64 - j as f64);
                flat[2 * (i * r + j)] = ε * cre;
                flat[2 * (i * r + j) + 1] = ε * cim;
                flat[2 * (j * r + i)] = ε * (cre + 0.01 * (i as f64 + 1.0)); // 一般非对称
                flat[2 * (j * r + i) + 1] = -ε * (cim - 0.01 * j as f64);
            }
        }
        flat
    }

    /// ★★ T-71 段① 主验收对账: **精确金标准**（非近对角近似）。
    /// 造 L = Vr·Λ·Vr⁻¹（右矢 Vr、谱 Λ 全部已知）, 独立矩阵乘算 κ_ref = Wr†δL Vr
    /// （Wr=Vr(Vr†Vr)⁻¹, 与 core 同列约定）; core 通用 κ测量 从 flat L 自行分解再投影,
    /// 在**强混合（ε=.30, 非近对角）**下仍须对上:
    ///   · 对角 κ_ii —— 复值相位不变量, 直接对实虚部;
    ///   · 非对角 κ_ij —— |κ_ij| 为相位不变量（特征向量各模有自由相位 e^{iθ},
    ///     κ_ij→e^{i(θ_j−θ_i)}κ_ij 旋转实虚部但模长不变; |κ| 正是 P12 §3.4 |κ|τ 输入）;
    ///     实/虚部绝对相位基准属 T-72 落地口径, 段①以模长 + 三恒等式严格验算子。
    fn 合成旋转L金标准() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let r = 3;
        let ωs = [0.20, 0.70, 1.30];
        let λ_ims = [-0.80, -0.45, -0.20]; // τ = 1/γ = −1/λ_im = 1.25 / 2.222 / 5.0（互异可匹配）
        let mut λ = DMatrix::<NComplex<f64>>::zeros(r, r);
        for n in 0..r {
            λ[(n, n)] = NComplex::new(ωs[n], λ_ims[n]);
        }
        // 已知非对角耦合 B（一般复、非 Hermitian）× ε=0.30（强混合, 故意不近对角）
        let ε = 0.30;
        let b: [[(f64, f64); 3]; 3] = [
            [(0.0, 0.0), (0.20, -0.10), (-0.15, 0.25)],
            [(0.20, 0.10), (0.0, 0.0), (0.30, 0.05)],
            [(-0.15, -0.25), (0.30, -0.05), (0.0, 0.0)],
        ];
        let mut vr = DMatrix::<NComplex<f64>>::identity(r, r);
        for i in 0..r {
            for j in 0..r {
                let (re, im) = b[i][j];
                vr[(i, j)] += NComplex::new(ε * re, ε * im);
            }
        }
        for j in 0..r {
            let n = vr.column(j).norm();
            let col = vr.column(j).clone() / NComplex::new(n, 0.0);
            vr.set_column(j, &col);
        }
        let l = &vr * λ * vr.clone().try_inverse().expect("Vr 可逆");
        // 金标准左右矢 + δL + κ_ref（独立路径, 不经 core 分解）
        let wr = &vr * (vr.adjoint() * &vr).try_inverse().expect("V†V 可逆");
        let mut dl = l.clone();
        for i in 0..r {
            dl[(i, i)] = NComplex::new(0.0, 0.0);
        }
        let κref = wr.adjoint() * &dl * &vr;
        let mut flat = vec![0.0f64; 2 * r * r];
        let mut κflat = vec![0.0f64; 2 * r * r];
        for i in 0..r {
            for j in 0..r {
                flat[2 * (i * r + j)] = l[(i, j)].re;
                flat[2 * (i * r + j) + 1] = l[(i, j)].im;
                κflat[2 * (i * r + j)] = κref[(i, j)].re;
                κflat[2 * (i * r + j) + 1] = κref[(i, j)].im;
            }
        }
        let τ: Vec<f64> = λ_ims.iter().map(|g| -1.0 / g).collect(); // τ=1/γ=−1/λ_im（数值不变）
        (flat, κflat, τ)
    }

    #[test]
    fn 测试_段一对账_旋转L精确κ金标准() {
        let r = 3;
        let (flat, κref_flat, τ真) = 合成旋转L金标准();
        let res = 集体耦合实现::κ测量(&flat, r).expect("κ 测量应成功");
        assert_eq!(res.阶, r);

        // 分解按 τ 降序 ⇒ 建 res→orig(按 τ 最近) → 反转 orig→res。
        let mut map = vec![0usize; r];
        let mut used = vec![false; r];
        for k in 0..r {
            let (best, _) = (0..r)
                .filter(|&j| !used[j])
                .map(|j| (j, (res.τ[k] - τ真[j]).abs()))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .expect("匹配");
            used[best] = true;
            map[k] = best;
        }
        let mut inv = vec![0usize; r];
        for k in 0..r {
            inv[map[k]] = k;
        }

        let tol = 1e-7_f64; // 两边均机器精度分解; 留 ~100× 余量
        let mut max_diag_err = 0.0f64;
        let mut max_mod_err = 0.0f64;
        for i in 0..r {
            for j in 0..r {
                let 参 = NComplex::new(κref_flat[2 * (i * r + j)], κref_flat[2 * (i * r + j) + 1]);
                let (ki, kj) = (inv[i], inv[j]);
                let 测 = NComplex::new(res.κ实部[ki * r + kj], res.κ虚部[ki * r + kj]);
                if i == j {
                    let e = (测 - 参).norm();
                    max_diag_err = max_diag_err.max(e);
                    assert!(e < tol, "对角 κ_{}{} 复值 {} vs 金标准 {}（相位不变量）", i, j, 测, 参);
                } else {
                    // 非对角: |κ_ij| 相位不变量
                    let e = (测.norm() - 参.norm()).abs();
                    max_mod_err = max_mod_err.max(e);
                    assert!(
                        e < tol,
                        "非对角 |κ_{}{}|={:.8} vs 金标准 {:.8}（模长/相位不变量）",
                        i, j, 测.norm(), 参.norm()
                    );
                }
            }
        }
        // 非平凡守卫: 金标准必须确有非零非对角 κ（否则又是单通道式平凡 0）
        let mut 非对角能量 = 0.0f64;
        for i in 0..r {
            for j in 0..r {
                if i != j {
                    非对角能量 += κref_flat[2 * (i * r + j)].powi(2)
                        + κref_flat[2 * (i * r + j) + 1].powi(2);
                }
            }
        }
        非对角能量 = 非对角能量.sqrt();
        assert!(非对角能量 > 0.05, "金标准须含非平凡非对角 κ, 实际 |κ_off|_F={:.4}", 非对角能量);
        eprintln!(
            "[段一精确对账] max 对角复值误差={:.2e}  max 非对角模长误差={:.2e}  |κ_off|_F={:.4}",
            max_diag_err, max_mod_err, 非对角能量
        );
    }

    /// ★★ T-71 段②准入①·类 III 语义测试（理论侧续44 第四点, P39 §14.2 类 III）:
    /// 段①金标准只证"实现一致性"，这里证**语义因果**——κ 模态度量的是**通道间相互作用 B**、
    /// 不是对角失谐 D。两条：
    ///  (语义1) **B=0 ⇒ V=W=I ⇒ κ_off≡0，无论对角 D 失谐多强**（若 κ 只测 D 投影此刻应非零）；
    ///  (语义2) **固定 D、把 B 由 0 同向扫强 ⇒ ‖κ_off‖_F 从 0 单调（非减）增长**。
    /// 注：‖κ_off‖_F 对模序置换不变（Frobenius 范数），无需对齐 τ 排序。
    fn κ非对角能量(res: &κ测量结果) -> f64 {
        let m = res.阶;
        let mut s = 0.0f64;
        for i in 0..m {
            for j in 0..m {
                if i != j {
                    s += res.κ实部[i * m + j].powi(2) + res.κ虚部[i * m + j].powi(2);
                }
            }
        }
        s.sqrt()
    }

    /// 2×2 Hermitian L（对角 D 强失谐 + 单一耦合 g，强度由 ε 缩放）→ flat
    fn 二模含耦合L(ε: f64, g_re: f64, g_im: f64) -> Vec<f64> {
        let r = 2;
        // 对角强分离: Δω=0.8, Δγ=0.5 → |Δλ|≈0.943（ε·g 全程 <0.5·|Δλ|, 不过 EP）
        let diag = [(0.20_f64, -0.80_f64), (1.00_f64, -0.30_f64)];
        let mut flat = vec![0.0f64; 2 * r * r];
        for (i, &(w, g)) in diag.iter().enumerate() {
            flat[2 * (i * r + i)] = w;
            flat[2 * (i * r + i) + 1] = g;
        }
        // Hermitian 耦合: L01=ε(g_re+g_im i), L10=conj
        flat[2 * (0 * r + 1)] = ε * g_re;
        flat[2 * (0 * r + 1) + 1] = ε * g_im;
        flat[2 * (1 * r + 0)] = ε * g_re;
        flat[2 * (1 * r + 0) + 1] = -ε * g_im;
        flat
    }

    /// 语义1: B=0（纯对角、强失谐 D）→ κ_off≡0；3 模、Δω/Δγ 都拉开
    #[test]
    fn 测试_类III语义1_B零则κoff恒零_任意D失谐() {
        let r = 3;
        let diag = [(0.2, -0.9), (0.9, -0.5), (1.6, -0.15)]; // ω、γ 均强分离
        let mut flat = vec![0.0f64; 2 * r * r];
        for (i, &(w, g)) in diag.iter().enumerate() {
            flat[2 * (i * r + i)] = w;
            flat[2 * (i * r + i) + 1] = g;
        } // 非对角全 0
        let res = 集体耦合实现::κ测量(&flat, r).expect("可测");
        let off = κ非对角能量(&res);
        assert!(off < 1e-9, "B=0 时无论 D 多强 κ_off 必须≡0（反证因果在 B）: {}", off);
    }

    /// 语义2: 固定 D、B 由 0 同向扫强 → ‖κ_off‖_F 从 0 单调非减增长
    #[test]
    fn 测试_类III语义2_B扫强则κoff单调增() {
        let eps = [0.0_f64, 0.03, 0.08, 0.15, 0.25, 0.40];
        let mut prev = -1.0f64;
        let mut vals = Vec::new();
        for &e in eps.iter() {
            let flat = 二模含耦合L(e, 0.5, -0.3);
            let res = 集体耦合实现::κ测量(&flat, 2).expect("可测");
            let off = κ非对角能量(&res);
            vals.push(off);
            assert!(off >= prev - 1e-9, "‖κ_off‖ 须随 B 单调非减: {:?}", vals);
            prev = off;
        }
        assert!(vals[0] < 1e-9, "ε=0 起点须≈0: {}", vals[0]);
        // 末端显著非零且远大于起点（确有强相互作用被捕获）
        assert!(vals[vals.len() - 1] > 0.10, "强耦合末态 ‖κ_off‖ 应显著: {:?}", vals);
        eprintln!("[类III语义2] ‖κ_off‖ 随 ε 单调: {:?}", vals);
    }

    /// ★ T-71 段②准入③·EP（异常点）条件数告警（理论侧续44 第四点③, P12/P39）:
    /// Petermann 因子 K=(w†w)(v†v)——正规/良好矩阵≈1, 本征矢随非厄米非正交度增大而趋病态。
    /// 构造实非对称 2×2（ab 固定保特征值不简并, 仅令本征矢非正交）:
    ///   · 温和非对称 a/b=4（a=2,b=0.5）→ K≈1.55 < 门限 10 → **不告警、可给边**;
    ///   · 强非对称 a/b=100（a=10,b=0.1）→ K≈23 ≥ 门限 → **告警（标注、不静默给边）**。
    /// （Hermitian 对称耦合本征矢恒正交 K≈1; 非对称耗散/耦合才产生 EP 敏感性。）
    fn 二模非对称L(a: f64, b: f64) -> Vec<f64> {
        let mut l = DMatrix::<NComplex<f64>>::zeros(2, 2);
        l[(0, 0)] = NComplex::new(0.2, -0.8);
        l[(1, 1)] = NComplex::new(1.0, -0.3);
        l[(0, 1)] = NComplex::new(a, 0.0);
        l[(1, 0)] = NComplex::new(b, 0.0);
        let mut f = vec![0.0; 8];
        for i in 0..2 {
            for j in 0..2 {
                f[2 * (i * 2 + j)] = l[(i, j)].re;
                f[2 * (i * 2 + j) + 1] = l[(i, j)].im;
            }
        }
        f
    }

    #[test]
    fn 测试_段二准入_EP条件数告警() {
        // 温和非对称 → K≈1.55, 不告警
        let f_ok = 二模非对称L(2.0, 0.5);
        let r_ok = 集体耦合实现::κ测量(&f_ok, 2).expect("可测");
        assert!(!r_ok.病态告警(), "良好条件（K≈1.55<10）不应告警: Kmax={}", r_ok.最大条件数);
        assert!(r_ok.最大条件数 > 1.0 && r_ok.最大条件数 < EP条件数门限,
            "非厄米 K 略>1 但低于门限: {}", r_ok.最大条件数);

        // 强非对称（近 EP 本征矢病态）→ K≈23, 触发告警
        let f_ep = 二模非对称L(10.0, 0.1);
        let r_ep = 集体耦合实现::κ测量(&f_ep, 2).expect("可测");
        assert!(r_ep.病态告警(), "K≈23≥10 必须告警（不静默给边）: Kmax={}", r_ep.最大条件数);
        assert!(r_ep.最大条件数 > 10.0, "Kmax 应显著超门限: {}", r_ep.最大条件数);

        // 正规（对称）Hermitian → K≈1, 不告警（用既有强对称耦合夹具）
        let f_norm = 二模含耦合L(0.25, 0.5, -0.3);
        let r_n = 集体耦合实现::κ测量(&f_norm, 2).expect("可测");
        assert!((r_n.最大条件数 - 1.0).abs() < 0.2,
            "Hermitian 正规 K≈1: {}", r_n.最大条件数);
        eprintln!(
            "[EP告警] 良好 Kmax={:.3}（不告警）/ 近EP Kmax={:.3}（告警）/ 正规 Kmax={:.3}",
            r_ok.最大条件数, r_ep.最大条件数, r_n.最大条件数
        );
    }

    // ── T-72 κ 相位基准规范（乙·只画不变量）硬测试 ──────────────────────

    /// 构造一个含复非对角 + 复对角的 3×3 κ（行主序实/虚分离）与有限 τ
    fn 三模复κ() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // 任意复数（含非对称、复耦合、复对角）
        let k = [
            NComplex::new(0.10, -0.04), NComplex::new(0.20, 0.05), NComplex::new(-0.05, 0.02),
            NComplex::new(0.08, -0.03), NComplex::new(-0.02, 0.06), NComplex::new(0.15, -0.07),
            NComplex::new(0.03, 0.01), NComplex::new(0.10, 0.00), NComplex::new(0.05, 0.02),
        ];
        let re = k.iter().map(|z| z.re).collect();
        let im = k.iter().map(|z| z.im).collect();
        (re, im, vec![26.3, 19.9, 12.4])
    }

    /// 逐模自由相位变换：κ'_ij = e^{i(θ_j−θ_i)} κ_ij（v_j→e^{iθ_j}v_j、w_i→e^{−iθ_i}w_i 的诱导）
    fn 规范变换(
        re: &[f64], im: &[f64], theta: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let m = theta.len();
        let mut nr = vec![0.0; m * m];
        let mut ni = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..m {
                let z = NComplex::new(re[i * m + j], im[i * m + j])
                    * NComplex::new(0.0, theta[j] - theta[i]).exp();
                nr[i * m + j] = z.re;
                ni[i * m + j] = z.im;
            }
        }
        (nr, ni)
    }

    /// ★ 规范硬测试：随机逐模相位旋转后，类 III 图呈现（全部不变量）逐位不变、对角 κ_jj 不变。
    #[test]
    fn 测试_T72_逐模相位旋转_图呈现逐位不变() {
        let (re, im, tau) = 三模复κ();
        let inv0 = κ不变量(&re, &im, &tau, 1.55).expect("应构造成功");
        assert!(inv0.可信, "K=1.55<10 应可信");
        assert_eq!(inv0.边.len(), 6, "3 模有向非对角边 = 3·2 = 6");
        assert_eq!(inv0.有向三环.len(), 1, "3 模有向三环 a<b<c = 1");

        // 多组随机相位（含 0 与大角）
        for seed in 0..64u64 {
            // 简单确定性 LCG 取角，避免测试依赖外部 rand
            let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let mut theta = Vec::new();
            for _ in 0..3 {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                theta.push(((s >> 33) as f64 / u32::MAX as f64) * std::f64::consts::TAU);
            }
            let (nr, ni) = 规范变换(&re, &im, &theta);
            // 确认变换确实动了非对角单边实/虚部（否则测试无意义）
            let moved = (0..9).any(|k| (nr[k] - re[k]).abs() > 1e-9 || (ni[k] - im[k]).abs() > 1e-9);
            assert!(moved, "相位变换应改变至少一个 κ 元素的单边实/虚部");

            let inv1 = κ不变量(&nr, &ni, &tau, 1.55).expect("变换后应仍可构造");
            // 边不变量逐位相等
            assert_eq!(inv1.边.len(), inv0.边.len());
            for (a, b) in inv0.边.iter().zip(inv1.边.iter()) {
                assert_eq!(a.i, b.i); assert_eq!(a.j, b.j);
                assert!((a.κ模 - b.κ模).abs() < 1e-9, "|κ| 应不变 {:.3e}", (a.κ模-b.κ模).abs());
                assert!((a.κ模τ_j - b.κ模τ_j).abs() < 1e-7, "|κ|τ 应不变");
                assert!((a.方向比 - b.方向比).abs() < 1e-9, "R_ij 应不变");
                assert!((a.两模环模 - b.两模环模).abs() < 1e-9, "两模环模应不变");
                // 环幅角卷绕比较：角差归一化到 [-π,π]
                let d = (a.两模环幅角 - b.两模环幅角 + std::f64::consts::PI)
                    .rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
                assert!(d.abs() < 1e-9, "两模环幅角应不变");
            }
            // 三环模/幅角不变
            for (a, b) in inv0.有向三环.iter().zip(inv1.有向三环.iter()) {
                assert!((a.环模 - b.环模).abs() < 1e-9);
                let d = (a.环幅角 - b.环幅角 + std::f64::consts::PI)
                    .rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
                assert!(d.abs() < 1e-9, "三环幅角应不变");
            }
            // 节点对角 κ_jj 整体（实/虚/模）逐位不变
            for (a, b) in inv0.节点.iter().zip(inv1.节点.iter()) {
                assert!((a.κjj实 - b.κjj实).abs() < 1e-12, "κ_jj 实部(频移)应不变");
                assert!((a.κjj虚 - b.κjj虚).abs() < 1e-12, "κ_jj 虚部(线宽)应不变");
                assert!((a.κjj模 - b.κjj模).abs() < 1e-12);
            }
        }
    }

    /// 类型层挡死：`不变量边` 编译期就没有 Re/Im 字段（类 III 路径拿不到单边量）。
    /// 此测试以"只能取不变量字段"的方式把该约束钉在用法上（若有人加 re/im 字段，
    /// code review 与本模块规范注释即拦截；运行时不存在被禁量）。
    #[test]
    fn 测试_T72_类III路径只暴露不变量() {
        let (re, im, tau) = 三模复κ();
        let inv = κ不变量(&re, &im, &tau, 1.0).unwrap();
        let e = &inv.边[0];
        // 允许字段均可读且为有限实数标量（模/方向比/环），无任何单边相位量
        let _ = (e.i, e.j, e.κ模, e.κ模τ_j, e.方向比, e.两模环模, e.两模环幅角);
        // 节点对角 κ_jj 是允许的物理量（频移/线宽）
        let n0 = &inv.节点[0];
        let _ = (n0.κjj实, n0.κjj虚, n0.κjj模, n0.τ);
    }

    /// 病态闸：K>10 即便在规范闸后也 `可信=false`（近 EP 连不变量也挂起、不参与判据 A）。
    #[test]
    fn 测试_T72_病态闸_连不变量也挂起() {
        let (re, im, tau) = 三模复κ();
        assert!(κ不变量(&re, &im, &tau, 1.55).unwrap().可信, "K=1.55 可信");
        let 挂起 = κ不变量(&re, &im, &tau, 23.012).unwrap();
        assert!(!挂起.可信, "EP 第三态 K=23.012>10 必须病态挂起");
        assert_eq!(挂起.边.len(), 6, "挂起时不变量仍可算（供灰显），只是不参与判据");
        // 非法形状 → None（不静默造）
        assert!(κ不变量(&[1.0, 2.0], &[0.0, 0.0], &[1.0], 1.0).is_none());
    }

    /// 段①纪律: 单通道/纯对角 L → κ 非对角 ≡ 0（平凡成立, 仅演示算子行为）
    #[test]
    fn 测试_纯对角L_κ非对角为零() {
        let r = 3;
        let mut flat = vec![0.0f64; 2 * r * r];
        for i in 0..r {
            flat[2 * (i * r + i)] = 0.1 + 0.1 * i as f64;
            flat[2 * (i * r + i) + 1] = -0.2 / (i as f64 + 1.0);
        }
        let res = 集体耦合实现::κ测量(&flat, r).expect("应成功");
        for i in 0..r {
            for j in 0..r {
                if i != j {
                    assert!(
                        res.κ实部[i * r + j].abs() < 1e-9 && res.κ虚部[i * r + j].abs() < 1e-9,
                        "纯对角 L → κ_ij({},{}) ≡ 0（单通道同理: 零信息, 不作为验收）",
                        i, j
                    );
                }
            }
        }
    }

    /// ★ T-71 段① 精度前提回归: κ=w†δLv 依赖精确左右矢。校验分解三恒等式——
    /// 右矢 Lv=λv（残差须机器精度）、双正交 w†v=δ、谱表示 W†LV=Λ（非对角≈0）。
    /// （逆迭代修复前右矢残差曾达 1e-2, 与弱耦合 κ 同量级 ⇒ 必须守住此门。）
    #[test]
    fn 测试_左右矢精度_三恒等式() {
        let r = 4;
        let ε = 0.10;
        let flat = 合成含非对角复L(r, ε);
        let mut l = DMatrix::<NComplex<f64>>::zeros(r, r);
        for i in 0..r {
            for j in 0..r {
                l[(i, j)] = NComplex::new(flat[2 * (i * r + j)], flat[2 * (i * r + j) + 1]);
            }
        }
        let 分 = 非厄米谱求解器::default().分解(&l).expect("分解");
        let m = 分.模.len();
        let mut vmat = DMatrix::<NComplex<f64>>::zeros(r, m);
        let mut wmat = DMatrix::<NComplex<f64>>::zeros(r, m);
        for k in 0..m {
            for a in 0..r {
                vmat[(a, k)] = NComplex::new(分.模[k].v[a], 分.模[k].v虚[a]);
                wmat[(a, k)] = NComplex::new(分.模[k].w[a], 分.模[k].w虚[a]);
            }
        }
        // 1) 右矢残差 ‖Lv-λv‖ 须机器精度
        let mut max_残 = 0.0f64;
        for k in 0..m {
            let vk = vmat.column(k);
            let λ = NComplex::new(分.模[k].ω, 分.模[k].λ_im); // 本征值＝(ω, λ_im)；γ=−λ_im 不入矩阵（统一令 §3）
            let 残 = (&l * vk - vk * λ).norm();
            max_残 = max_残.max(残);
        }
        // 2)+3) 双正交 与 谱表示 W†LV≈Λ
        let whv = wmat.adjoint() * &vmat;
        let whlv = wmat.adjoint() * &l * &vmat;
        let mut max_正交 = 0.0f64;
        let mut max_非对角 = 0.0f64;
        for i in 0..m {
            for j in 0..m {
                let 期望 = if i == j { 1.0 } else { 0.0 };
                max_正交 = max_正交.max((whv[(i, j)] - 期望).norm());
                if i != j {
                    max_非对角 = max_非对角.max(whlv[(i, j)].norm());
                }
            }
        }
        assert!(max_残 < 1e-9, "右矢残差 {:.2e} 须 <1e-9（κ 精度前提）", max_残);
        assert!(max_正交 < 1e-9, "双正交偏差 {:.2e} 须 <1e-9", max_正交);
        assert!(max_非对角 < 1e-9, "W†LV 非对角 {:.2e} 须 <1e-9", max_非对角);
    }
}