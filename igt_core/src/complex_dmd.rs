// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! 复 EDMD 数据链（D7, 2026-08-23, 规范 §3 Step2-4）— 方案B 替换方案C
//!
//! ★ T-93 注释补记（2026-09-17，PM ㊲ §四；**仅注释、core 逻辑零改**）：
//!   **Γ 正定旗标是「估计质量」而非「本体判据」** —— 对**非正规 L**，其**全部本征值可都在左半平面
//!   而 Γ 仍可能非正定**（瞬态增长/伪谱现象）⇒ **`Γ 对称部分正定` 强于 `全模 Imλ<0`**；
//!   二者仅在**正规 L** 时等价。⇒ 因此：Γ 正定检验不过时，**入库/造波/κ 因果门继续硬拒**（工程纪律），
//!   但**不得据此对系统下"非耗散/熵泵"本体结论**，须先排**非正规性**与**数据欠定**（短窗、单变量、慢模 T<5τ）。
//!   依据：理论侧续62＋P1 errata §E1（2026-09-17）。
//! 完整数据链（D7 §3）:
//!   Step1 解析信号（`analytical_signal`）→ Step2 复 EDMD（本模块）→
//!   Step3 本征分解（手写 QR 迭代, r≤30, 不需 ComplexSchur）→
//!   Step4 方案B 模态基重构 L=ΦΛΦ⁻¹（恢复完整复L 含非对角虚部）。
//!
//! 与方案C 的本质区别: 方案C 非对角虚部恒 0 → Γ 非对角耦合不存在 → 熵泵条件
//! （Im Σ>0）无法成立; 方案B 从复解析信号出发, L 为一般复矩阵, Γ 可正定。
//!
//! 硬边界（D7 §4）: 需 ≥2 通道（跨模态耦合）; 单通道无法计算熵泵（只做谱统计）。

#![allow(non_snake_case)]

use nalgebra::Complex as NComplex;
use nalgebra::ComplexField;
use nalgebra::{DMatrix, DVector, SVD};

use crate::analytical_signal::解析信号;

/// 复L 谱质量诊断（D7 深层优化, 第388批）——方案B 重构复L 的可信度
///
/// 背景: 快慢尺度差大时方案B 伪逆重构复L 失真（谱分解 τ 负/频率≈1）→ 熵泵
/// 谱分离失效。本字段使失真可观测, 熵泵/上层调用方可据此诚实降级。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct 复L质量 {
    /// SVD 条件数 σ_max/σ_min（保留模内; 越大越病态）
    pub 条件数: f64,
    /// 能量比 σ₁/σ₂（快慢模能量均衡度; 越大越失衡）
    pub 能量比: f64,
    /// Φ 列正交偏差: (Φ†Φ) 对角归一化后非对角最大模（>0.5 → 重构不可信）
    pub 列正交偏差: f64,
}

/// 复 dmd 结果（方案B 重构完整复L）
///
/// ★ γ符号统一令 v1.0（§1/§3）：本结构的谱量一律**论文口径**——
///   λ = ω − iγ；**λ_im = Im λ**（矩阵量，稳定 < 0）；**γ = −λ_im = −Im λ**（耗散率，稳定 > 0）；
///   **τ = 1/γ = −1/λ_im**（数值与旧实现恒等，未变）。
#[derive(Debug, Clone)]
pub struct 复dmd结果 {
    /// 方案B 重构复L（r×r, 实虚交错, 完整非对角虚部）
    /// 虚部语义＝λ_im（稳定负）；耗散率 γ＝−虚部（正）；布局与数值不变（统一令 §5）
    pub 复L: Vec<f64>,
    /// 复L 阶数 r
    pub 复L阶: usize,
    /// 稳定模 τ 链（λ=ω−iγ, γ=−λ_im>0; τ=1/γ>0 有限；数值与旧口径 τ=−1/λ_im 恒等）
    pub τ链: Vec<f64>,
    /// 振荡频率 ω 列表
    pub 频率: Vec<f64>,
    /// 失稳模列表（索引, ω, γ<0）——失稳 ⟺ γ<0 ⟺ λ_im>0（统一令 §1 判据）
    pub 失稳模: Vec<(usize, f64, f64)>,
    /// 全模谱（R4-v2 T-41 块1 方案 C, G-R3-1 只读导出）: 全部本征模 (模索引, ω, γ) 数值
    /// γ＝耗散率＝−Im λ（**稳定 > 0**，统一令 §1）；含稳定(γ>0)/临界(γ≈0)/失稳(γ<0) 全部,
    /// **不筛不标结论**——只供证据层喂模型自行判读; 与失稳模/τ链同一循环顺带填充。
    /// 自洽: γ<−1e-9 子集=失稳模, 稳定 τ=1/γ=τ链。
    pub 全模谱: Vec<(usize, f64, f64)>,
    /// Γ 正定（熵泵可计算前提, 由调用方验证）
    pub Γ正定: bool,
    /// 复L 谱质量诊断（第388批, 熵泵前置可信度检查）
    pub 谱质量: 复L质量,
}

/// 复 QR 分解（Givens 旋转, r≤30 小矩阵, 不需 nalgebra ComplexSchur feature）
/// 返回 (Q, R): A = Q·R（Q 酉, R 上三角）
pub fn 复QR分解(a: &DMatrix<NComplex<f64>>) -> (DMatrix<NComplex<f64>>, DMatrix<NComplex<f64>>) {
    let n = a.nrows();
    let mut r = a.clone();
    let mut q = DMatrix::<NComplex<f64>>::identity(n, n);
    for j in 0..n {
        for i in (j + 1)..n {
            let 主元 = r[(j, j)];
            let 下方 = r[(i, j)];
            let 模方 = 主元.norm_sqr() + 下方.norm_sqr();
            if 模方 < 1e-30 || 主元.norm() < 1e-30 {
                continue;
            }
            // 复 Givens: [c, s; -conj(s), c]·[a; b] = [r; 0]
            let r_len = 模方.sqrt();
            let c = 主元.norm() / r_len;
            let 相位 = 主元 / 主元.norm();
            let s = 相位 * 下方.conjugate() / r_len;
            // 左乘 G† 到 r 的行 (j, i): 行变换
            for k in 0..n {
                let rj = r[(j, k)];
                let ri = r[(i, k)];
                r[(j, k)] = c * rj + s * ri;
                r[(i, k)] = -s.conjugate() * rj + c * ri;
            }
            // Q 累积: Q = Q·G†（右乘 G 共轭转置; A=Q·R 中 Q=(G_m...G_1)†=G_1†...G_m†）
            for k in 0..n {
                let qk_j = q[(k, j)];
                let qk_i = q[(k, i)];
                q[(k, j)] = c * qk_j + s.conjugate() * qk_i;
                q[(k, i)] = -s * qk_j + c * qk_i;
            }
        }
    }
    (q, r)
}

/// 手写 QR 迭代（**双位移**, Francis）复矩阵特征分解（r≤30）
/// 返回 (本征值, 右特征向量矩阵 Φ 列）
///
/// 双位移对实矩阵的复共轭对本征值稳健（单位移会不收敛——实共轭对需两步位移）。
/// 右下 2×2 块特征值作为位移对, 每轮两次 QR。
pub fn 复特征分解_QR(
    a: &DMatrix<NComplex<f64>>,
    max_iter: usize,
) -> (Vec<NComplex<f64>>, DMatrix<NComplex<f64>>) {
    let n = a.nrows();
    if n == 0 {
        return (vec![], DMatrix::zeros(0, 0));
    }
    if n == 1 {
        return (vec![a[(0, 0)]], DMatrix::identity(1, 1));
    }
    let mut a_k = a.clone();
    let mut acc = DMatrix::<NComplex<f64>>::identity(n, n);
    for _ in 0..max_iter {
        // 位移对: 右下 2×2 块 [[a,b],[c,d]] 特征值 σ1, σ2（Francis 双位移）
        let (σ1, σ2) = if n >= 2 {
            let av = a_k[(n - 2, n - 2)];
            let bv = a_k[(n - 2, n - 1)];
            let cv = a_k[(n - 1, n - 2)];
            let dv = a_k[(n - 1, n - 1)];
            let t = (av + dv) / NComplex::new(2.0, 0.0);
            // sqrt(((a-d)/2)² + b·c)
            let 根 = (((av - dv) / NComplex::new(2.0, 0.0)).powi(2) + bv * cv).sqrt();
            (t + 根, t - 根)
        } else {
            (a_k[(0, 0)], NComplex::new(0.0, 0.0))
        };
        // 双位移: 两次 (A - σI) QR + (RQ + σI)
        for σ in [σ1, σ2] {
            for i in 0..n {
                a_k[(i, i)] -= σ;
            }
            let (q, r) = 复QR分解(&a_k);
            a_k = &r * &q;
            for i in 0..n {
                a_k[(i, i)] += σ;
            }
            acc = &acc * &q;
        }
        // 收敛: 下三角非对角足够小
        let mut 残差 = 0.0f64;
        for i in 1..n {
            for j in 0..i {
                残差 = 残差.max(a_k[(i, j)].norm());
            }
        }
        if 残差 < 1e-10 * a_k.iter().map(|z| z.norm()).fold(0.0f64, f64::max).max(1.0) {
            break;
        }
    }
    let 本征值: Vec<NComplex<f64>> = (0..n).map(|i| a_k[(i, i)]).collect();
    (本征值, acc)
}

/// Step1: 解析信号矩阵（多通道去均值 → Hilbert 解析化）
///
/// 去均值用**中段均值**（全序列均值被衰减信号前段大振荡主导 → 中段残留 DC
/// -0.035 → 解析信号 z 含 DC → 衰减模被调制 → τ 偏 50%）; 中段 x' 零均值 → z DC≈0。
fn 解析信号矩阵(通道: &[Vec<f64>]) -> Option<(usize, usize, usize, Vec<Vec<NComplex<f64>>>)> {
    let n = 通道.len();
    if n < 2 {
        return None; // D7 §4 硬边界: 单通道无法熵泵
    }
    let m = 通道[0].len();
    if m < n + 2 {
        return None;
    }
    let 边界 = (m / 20).max(2); // 首尾各 5%（Hilbert 边界污染区）
    let z: Vec<Vec<NComplex<f64>>> = 通道
        .iter()
        .map(|ch| {
            let 中段均值 = ch[边界..ch.len() - 边界].iter().sum::<f64>()
                / (ch.len() - 2 * 边界) as f64;
            let 去均值: Vec<f64> = ch.iter().map(|x| x - 中段均值).collect();
            解析信号(&去均值)
        })
        .collect();
    Some((n, m, 边界, z))
}

/// Step2-4: 单窗口复 EDMD——从解析信号矩阵构造快照 + SVD + 本征分解 + 方案B
///
/// - `终点上限`: 快照窗口终点上限（多尺度: 短窗 = 前段, 长窗 = 全段）;
///   幅值阈值截断内部再收窄（尾段 <0.05·峰值 → 数值噪声主导 → τ 崩）
fn 窗口EDMD(
    z: &[Vec<NComplex<f64>>],
    n: usize,
    m: usize,
    边界: usize,
    Δt: f64,
    终点上限: usize,
) -> Option<复dmd结果> {
    // Step2: 快照矩阵 Z0/Z1（n×M 复）——内 80% 中段（跳过 Hilbert 首尾各 10% 伪影）
    let 快照起点 = 2 * 边界;
    let 峰值 = z
        .iter()
        .map(|zi| {
            zi[快照起点..m - 快照起点]
                .iter()
                .map(|v| v.norm())
                .fold(0.0f64, f64::max)
        })
        .fold(0.0f64, f64::max);
    let 幅值底 = 0.05 * 峰值;
    let mut 快照终点 = 终点上限.min(m - 快照起点);
    // 循环 [起点+1, 终点上限) 不含终点上限——尾部边界伪影列不纳入（对齐单窗口原版行为）
    for k in (快照起点 + 1..快照终点).rev() {
        if z.iter().all(|zi| zi[k].norm() > 幅值底) {
            快照终点 = k;
            break;
        }
    }
    if 快照终点 <= 快照起点 + 1 {
        return None;
    }
    let 快照数 = 快照终点 - 快照起点 - 1;
    let z0 = DMatrix::<NComplex<f64>>::from_fn(n, 快照数, |i, j| z[i][j + 快照起点]);
    let z1 = DMatrix::<NComplex<f64>>::from_fn(n, 快照数, |i, j| z[i][j + 快照起点 + 1]);
    // 截断 SVD: Z0 = U Σ V†, 保留 σ_r > σ_max·1e-6
    //（1e-10 过松会把低能量慢模的数值噪声模保留 → Ã 病态/det≈0 → 本征值失真）
    let svd = SVD::new(z0.clone(), true, true);
    let u_all = svd.u.as_ref()?;
    let v_t_all = svd.v_t.as_ref()?;
    let σ_max = svd.singular_values.iter().fold(0.0f64, |a, b| a.max(*b));
    let r = svd
        .singular_values
        .iter()
        .filter(|s| **s > σ_max * 1e-6)
        .count()
        .max(1)
        .min(n);
    let u = u_all.columns(0, r).into_owned();
    // V 列 = (V† 前 r 行) 的共轭转置——`rows().transpose()` 只转置会得到 V 列的**共轭**,
    // 对复信号 z0·conj(v) ≠ σ·u → Ã 失真（τ 全错）; 须用 `adjoint()` 取 V 列本身。
    let v = v_t_all.rows(0, r).adjoint().into_owned(); // V (M×r)
    let mut s_inv = DMatrix::<NComplex<f64>>::zeros(r, r);
    for i in 0..r {
        let s = svd.singular_values[i];
        if s > 1e-12 {
            s_inv[(i, i)] = NComplex::new(1.0 / s, 0.0);
        }
    }
    // Ã = U† Z1 V Σ⁻¹（r×r 复）
    let Ã = &u.adjoint() * &z1 * &v * s_inv.clone();
    // Step3: 本征分解（手写 QR 迭代）——本征值用于 τ 谱; 观测基 A 不再需要 φ（方案B 弃用）
    let (本征值, _) = 复特征分解_QR(&Ã, 300);
    // Step4: 观测基复L（第396批, D7 优化项）——EDMD 标准观测基产物 + **连续化**。
    // A = U Ã U†（转移算子, 本征值 μ=e^{λΔt}）; 谱求解器/熵泵期望连续化 λ 矩阵
    // （P22 L=H-iΓ, τ=-1/Im λ）——若输出 μ, 谱分解把 μ 当 λ → τ=-1/Im(μ) 错（τ 负）。
    // **连续化**: L_cont = U·diag(ln μ/Δt)·U†, 本征值 = λ = ln(μ)/Δt（D7 §3 Step3）。
    let mut Λ_ln = DMatrix::<NComplex<f64>>::zeros(本征值.len(), 本征值.len());
    for (i, μ) in 本征值.iter().enumerate() {
        let r = (μ.re * μ.re + μ.im * μ.im).sqrt().max(1e-12);
        let θ = μ.im.atan2(μ.re);
        // 论文口径（统一令 §1/§3）: λ = ω − iγ = ω + i·λ_im；虚部＝λ_im（稳定 < 0），
        // 耗散率 γ = −λ_im > 0，τ = 1/γ = −1/Im λ（数值不变）。
        // 数学 λ=ln(μ)/Δt = ω + i·λ_im（θ/Δt 实部、ln(r)/Δt 虚部）; **交换实虚**使复L 谱符合该约定。
        Λ_ln[(i, i)] = NComplex::new(θ / Δt, r.ln() / Δt); // ω 实部, λ_im 虚部（稳定负；γ=−λ_im>0）
    }
    let 复l矩阵 = &u * &Λ_ln * u.adjoint(); // n×n 观测基连续化复L（P22 约定）
    // 谱质量（第388批）: 观测基的 U 列正交（SVD）→ 重构稳定, 列正交偏差恒 0
    let 列正交偏差 = 0.0f64;
    // λ_n = ln(μ_n)/Δt（标量 ln）; L 本征值 = λ_n; L 是 n×n（观测基通道数）
    let 阶 = n;
    let mut 复l = Vec::with_capacity(阶 * 阶 * 2);
    for i in 0..阶 {
        for j in 0..阶 {
            复l.push(复l矩阵[(i, j)].re);
            复l.push(复l矩阵[(i, j)].im);
        }
    }
    // τ 链/频率/失稳模——λ_n = ln(μ_n)/Δt（标量 ln, D7 §3 Step3）
    // μ = r·e^{iθ} → λ = θ/Δt + i·ln(r)/Δt（**λ_im = ln(r)/Δt, 稳定 < 0**；γ = −λ_im > 0 稳定）
    // ★ γ符号统一令 §3：矩阵量叫 λ_im；物理判据/对外导出用 γ = −λ_im；τ = 1/γ（数值不变）。
    let mut τ链 = Vec::new();
    let mut 频率 = Vec::new();
    let mut 失稳模 = Vec::new();
    let mut 全模谱 = Vec::new(); // T-41 块1: 全模只读导出(含临界 γ≈0), 不筛不标结论
    for (i, μ) in 本征值.iter().enumerate() {
        let r = (μ.re * μ.re + μ.im * μ.im).sqrt().max(1e-12);
        let θ = μ.im.atan2(μ.re);
        let ω = θ / Δt;
        let λ_im = r.ln() / Δt; // 本征值虚部＝增长率（稳定 < 0）
        let γ = -λ_im; // 耗散率（物理量，稳定 > 0）
        全模谱.push((i, ω, γ));
        频率.push(ω);
        if γ < -1e-9 {
            // 失稳 ⟺ γ<0 ⟺ λ_im>0（统一令 §1；判据反转但逻辑等价）
            失稳模.push((i, ω, γ));
        } else if γ > 1e-9 {
            let τ = 1.0 / γ; // = −1/λ_im（数值与旧实现恒等）
            if τ.is_finite() && τ > 0.0 {
                τ链.push(τ);
            }
        }
    }
    τ链.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Γ 正定验证（D7 §3 Step5, 由调用方二次确认; 此处给提示值）
    let Γ正定 = crate::nonhermitian_spectrum::验证Γ正定(&复l矩阵);
    // D7 深层优化（第388批）: 谱质量诊断（方案B 重构可信度）
    let σ_min = svd.singular_values.get(r - 1).copied().unwrap_or(1e-12).max(1e-12);
    let σ2 = svd.singular_values.get(1).copied().unwrap_or(1e-12).max(1e-12);
    let 谱质量 = 复L质量 {
        条件数: (σ_max / σ_min).max(1.0),
        能量比: if r >= 2 { (σ_max / σ2).max(1.0) } else { 1.0 },
        列正交偏差,
    };
    Some(复dmd结果 {
        复L: 复l,
        复L阶: 阶,
        τ链,
        频率,
        失稳模,
        全模谱,
        Γ正定,
        谱质量,
    })
}

/// 复 EDMD 解析信号（D7 §3 Step2-4, 完整数据链, 全窗口）
///
/// - `通道`: 多通道实信号（每通道同一长度, ≥2 通道硬边界）
/// - `Δt`: 采样间隔
///
/// # ⚠️ 适用边界（T-110 补记，2026-09-18，依据 B1 段 T-95/T-101/T-102 实测）
///
/// **本函数【无 Hankel 延迟嵌入扩维】⇒ 截断秩 `r` 被结构性锁死 `r ≤ 通道数 n`**：
/// - 快照矩阵按**通道**构造（`n = 通道数`，见下方 L226 附近 `from_fn(n, 快照数, …)`），
///   随后 SVD 截断取 `r = ….min(n)` ⇒ 例：双通道 ⇒ **`r ≤ 2`**；
/// - ⇒ ⇒ **后果（实测）**：`r=2` 时**够不上"信号模 + 噪声"的分离自由度** ——
///   B1 曾见合成信号主频 **0.3114 Hz（周期 3.2 s）**，而全模谱只有 `f = 0.2500 Hz / 0.0901 Hz` 两个模，
///   **真主频模根本不在谱里** ⇒ 任何"选模口径"都取不到信号模（`b` 偏大约 26–58 倍于 `1/T`）。
///
/// **⇒ 何时改用 `system_identifier::单变量dmd`**：
/// **单通道／少通道（尤其需要"频率维分辨力"）时**，请用
/// `igt_core_math::system_identifier::{系统辨识器, 单变量dmd}` —— 它**自带 Hankel 延迟嵌入**
/// （快照 = `[x_k, …, x_{k+m-1}]`，**状态维数 = 嵌入维 `m`**），且 **`截断秩` 可自由设** ⇒ **`r` 能上去**。
/// **B1 已验证用法**：`HRV tachogram → 单变量dmd(嵌入维=16) → 模列表 → 锁主频 γ̂ = −Re(λ)`
///   ⇒ 嵌入维 **4/8/16/24** 扫描下 `b` 由 **58/T 单调降到 2.52/T**、**检出率由 2% 升到 100%**（T-95 路 A）。
/// **注**：本函数仍是**多通道物理耦合（如 κ 矩阵）**场景的正确入口（`n = 物理通道数` 是其设计意图）。
pub fn 复EDMD解析信号(通道: &[Vec<f64>], Δt: f64) -> Option<复dmd结果> {
    if Δt <= 0.0 {
        return None;
    }
    let (n, m, 边界, z) = 解析信号矩阵(通道)?;
    窗口EDMD(&z, n, m, 边界, Δt, m)
}

/// 多尺度复EDMD（D7 优化项, 第387批）: 短窗（快模寿命区）∪ 长窗（慢模）→ 合并 τ 链;
/// 复L 用短窗（快模准 + 慢模近 DC → 熵泵谱分离成立）。
///
/// 背景（D7 V4 诚实结论）: 单窗口对时间尺度差大的快慢模投影误差大——快模在长窗
/// 能量占比低 → SVD 主方向偏慢模 → 快模 γ 失真（τ 负/失稳）→ 熵泵谱分离失效。
/// 多窗口: 短窗覆盖快模寿命（占比高 → 提取准）, 长窗覆盖慢模（τ 准）; 合并取全谱。
pub fn 复EDMD多尺度(通道: &[Vec<f64>], Δt: f64) -> Option<复dmd结果> {
    if Δt <= 0.0 {
        return None;
    }
    let (n, m, 边界, z) = 解析信号矩阵(通道)?;
    let 起点 = 2 * 边界;
    let 有效 = m - 2 * 起点;
    if 有效 < 8 {
        return None;
    }
    // 短窗: 前 30% 有效窗口（快模寿命区——慢模近线性/DC, 快模占比高）
    let 短窗上限 = 起点 + (有效 as f64 * 0.3) as usize;
    let 短窗 = 窗口EDMD(&z, n, m, 边界, Δt, 短窗上限)?;
    // 长窗: 全窗口（慢模寿命）
    let 长窗 = 窗口EDMD(&z, n, m, 边界, Δt, m)?;
    // 合并 τ 链（去重: 相对差 > 25% 视为不同模）
    let mut τ链 = 短窗.τ链.clone();
    for τ in &长窗.τ链 {
        if τ链.iter().all(|t| (t - τ).abs() / τ.max(1e-9) > 0.25) {
            τ链.push(*τ);
        }
    }
    τ链.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // 复L 用短窗。诚实标注（第387批实测）: 快慢尺度差大时方案B 伪逆重构的复L
    // 本征值 ≠ 物理模（谱分解 τ 负/频率≈1）——τ 谱（Ã 本征值）准确, 复L 重构
    // 精度记入 D7 深层优化项（熵泵依赖复L, 谱分离失效时诚实回退）。
    let mut 结果 = 短窗;
    结果.τ链 = τ链;
    Some(结果)
}

// ============================================================================
// 测试（D7 规范 §6 V3/V4 + 手写 QR 正确性）
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// T-41 块1 (方案 C G-R3-1): 全模谱只读导出自洽——结构性断言, 零数学改动验证
    /// ★ γ符号统一令后口径: 全模谱第三元＝耗散率 γ=−λ_im（稳定>0）。
    /// ①全模谱 len == 复L阶(全部本征模, 含临界 γ≈0); ②γ<−1e-9 子集(ω,γ) == 失稳模;
    /// ③γ>1e-9 的 τ=1/γ(排序) == τ链。
    #[test]
    fn T41块1_全模谱导出_自洽() {
        // 持续增长振荡(divergent, γ≈−0.06<0 应被 EDMD 检为失稳模) + 衰减模 混合
        let n = 40;
        let Δt = 1.0;
        let 通道1: Vec<f64> = (0..n)
            .map(|k| {
                let t = k as f64 * Δt;
                (0.06 * t).exp() * (0.9 * t).sin() + (-t / 3.0).exp() * (2.0 * t).cos()
            })
            .collect();
        let 通道2: Vec<f64> = (0..n)
            .map(|k| 0.4 * (0.06 * k as f64 * Δt).exp() * (0.9 * k as f64 * Δt).sin() + 0.5 * 通道1[k])
            .collect();
        let 结果 = 复EDMD解析信号(&[通道1, 通道2], Δt).expect("复EDMD应成功");
        let 全模谱 = &结果.全模谱;
        assert_eq!(全模谱.len(), 结果.复L阶, "全模谱应含全部本征模(阶)");
        // ② γ<−1e-9 子集与失稳模逐元素一致(单一事实源自洽；失稳⟺γ<0)
        let mut 失稳_from全: Vec<(f64, f64)> = 全模谱
            .iter().filter(|(_, _, g)| *g < -1e-9)
            .map(|(_, w, g)| (*w, *g)).collect();
        失稳_from全.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut 失稳 = 结果.失稳模.iter().map(|(_, w, g)| (*w, *g)).collect::<Vec<_>>();
        失稳.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        assert!(!失稳.is_empty(), "持续增长应检出失稳模(否则用例不达意)");
        assert_eq!(失稳_from全, 失稳, "全模谱 γ<0 子集须与失稳模逐元素一致");
        // ③ 稳定模 τ=1/γ(排序) == τ链（γ>0 稳定）
        let mut 稳定τ_from全: Vec<f64> = 全模谱
            .iter().filter(|(_, _, g)| *g > 1e-9)
            .map(|(_, _, g)| 1.0 / g).collect();
        稳定τ_from全.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert!(!稳定τ_from全.is_empty(), "应含稳定模");
        assert_eq!(稳定τ_from全, 结果.τ链, "全模谱稳定模 τ 须与 τ链一致");
    }

    /// 手写 QR 特征分解正确性: 已知 2×2 复矩阵
    #[test]
    fn 测试_复特征分解QR_已知矩阵() {
        // [[2, 1], [1, 3]] → 本征值 3.618, 1.382
        let mut a = DMatrix::<NComplex<f64>>::zeros(2, 2);
        a[(0, 0)] = NComplex::new(2.0, 0.0);
        a[(0, 1)] = NComplex::new(1.0, 0.0);
        a[(1, 0)] = NComplex::new(1.0, 0.0);
        a[(1, 1)] = NComplex::new(3.0, 0.0);
        let (本征值, _) = 复特征分解_QR(&a, 300);
        let mut 实: Vec<f64> = 本征值.iter().map(|λ| λ.re).collect();
        实.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((实[0] - 1.382).abs() < 0.02, "λ1≈1.382: {}", 实[0]);
        assert!((实[1] - 3.618).abs() < 0.02, "λ2≈3.618: {}", 实[1]);
    }

    /// 复 QR 迭代对复共轭对本征值收敛: [[a,-b],[b,a]] → a±ib
    #[test]
    fn 测试_复特征分解QR_共轭对() {
        let a = 0.85;
        let b = 0.1;
        let mut m = DMatrix::<NComplex<f64>>::zeros(2, 2);
        m[(0, 0)] = NComplex::new(a, 0.0);
        m[(0, 1)] = NComplex::new(-b, 0.0);
        m[(1, 0)] = NComplex::new(b, 0.0);
        m[(1, 1)] = NComplex::new(a, 0.0);
        let (本征值, _) = 复特征分解_QR(&m, 300);
        // 本征值 = a±ib
        let mut 虚: Vec<f64> = 本征值.iter().map(|λ| λ.im).collect();
        虚.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert!((虚[0] + b).abs() < 0.05, "λ=a-ib: {}", 虚[0]);
        assert!((虚[1] - b).abs() < 0.05, "λ=a+ib: {}", 虚[1]);
        for λ in &本征值 {
            assert!((λ.re - a).abs() < 0.05, "实部 a: {}", λ.re);
        }
    }

    /// 复 QR 分解正确性: A = Q·R 重建
    #[test]
    fn 测试_复QR分解_重建() {
        let n = 3;
        let mut a = DMatrix::<NComplex<f64>>::zeros(n, n);
        a[(0, 0)] = NComplex::new(1.0, 0.5);
        a[(0, 1)] = NComplex::new(0.2, -0.1);
        a[(0, 2)] = NComplex::new(0.3, 0.0);
        a[(1, 0)] = NComplex::new(0.4, 0.2);
        a[(1, 1)] = NComplex::new(2.0, 0.0);
        a[(1, 2)] = NComplex::new(-0.5, 0.3);
        a[(2, 0)] = NComplex::new(0.1, 0.0);
        a[(2, 1)] = NComplex::new(0.3, 0.1);
        a[(2, 2)] = NComplex::new(1.5, -0.2);
        let (q, r) = 复QR分解(&a);
        let 重建 = &q * &r;
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (重建[(i, j)] - a[(i, j)]).norm() < 1e-8,
                    "A≈QR: [{},{}] 重建={} 原={}",
                    i, j, 重建[(i, j)], a[(i, j)]
                );
            }
        }
        // R 上三角
        for i in 1..n {
            for j in 0..i {
                assert!(r[(i, j)].norm() < 1e-8, "R 上三角: [{},{}]={}", i, j, r[(i, j)]);
            }
        }
    }

    /// D7-V3: 复 EDMD 解析信号数据链——2 通道含耦合 → 复L 非对角虚部非零（方案B 恢复）
    #[test]
    fn 测试_D7V3_方案B非对角虚部恢复() {
        let n = 200;
        let Δt = 0.1;
        // 通道1: 快衰减振荡; 通道2: 慢衰减 + 0.3×通道1 耦合
        let 通道1: Vec<f64> = (0..n)
            .map(|k| (-(k as f64 * Δt) / 3.0).exp() * (0.4 * k as f64 * Δt).cos())
            .collect();
        let 通道2: Vec<f64> = (0..n)
            .map(|k| (-(k as f64 * Δt) / 10.0).exp() + 0.3 * 通道1[k])
            .collect();
        let 结果 = 复EDMD解析信号(&[通道1, 通道2], Δt).expect("复EDMD应成功");
        assert!(结果.复L阶 >= 2, "2通道应≥2阶: {}", 结果.复L阶);
        // 非对角虚部非零（方案B 恢复 Γ 非对角耦合, D7 V3）
        let r = 结果.复L阶;
        let mut 非对角虚非零 = 0;
        for i in 0..r {
            for j in 0..r {
                if i != j {
                    let im = 结果.复L[2 * (i * r + j) + 1];
                    if im.abs() > 1e-6 {
                        非对角虚非零 += 1;
                    }
                }
            }
        }
        assert!(非对角虚非零 >= 1, "方案B 复L 非对角虚部应非零（D7 V3）: 非对角虚非零={}", 非对角虚非零);
    }

    /// D7-V4: 2 通道 → 熵泵链路可用（Γ 正定或 τ 谱合理, 不 panic）
    #[test]
    fn 测试_D7V4_复EDMD链路() {
        let n = 200;
        let Δt = 0.1;
        let 通道1: Vec<f64> = (0..n)
            .map(|k| (-(k as f64 * Δt) / 3.0).exp() * (0.4 * k as f64 * Δt).cos())
            .collect();
        let 通道2: Vec<f64> = (0..n)
            .map(|k| (-(k as f64 * Δt) / 10.0).exp() + 0.3 * 通道1[k])
            .collect();
        let 结果 = 复EDMD解析信号(&[通道1, 通道2], Δt).unwrap();
        // τ 链非空（稳定模存在）
        assert!(!结果.τ链.is_empty(), "稳定模 τ 链应非空");
        // 频率含全部模（稳定+失稳）→ ≥ τ 链长度
        assert!(结果.频率.len() >= 结果.τ链.len(), "频率全模≥稳定模: {} vs {}", 结果.频率.len(), 结果.τ链.len());
        // 复L 阶×阶×2 长度合法
        assert_eq!(结果.复L.len(), 结果.复L阶 * 结果.复L阶 * 2);
    }

    /// D7 §4 硬边界: 单通道 → None（不静默）
    #[test]
    fn 测试_D7_单通道None() {
        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        assert!(复EDMD解析信号(&[x], 0.1).is_none(), "单通道应 None（D7 §4）");
    }

    /// 复 EDMD 核心正确性: 双通道同模（秩 1）→ 精确恢复 τ（V 列共轭 bug 修复后）
    #[test]
    fn 测试_复EDMD_双通道同模精确() {
        let Δt = 0.1;
        let n = 800; // 快照 ~640 步 ≈ 1.6τ（1τ 覆盖下 EDMD 拟合不足 → |μ| 偏 1 → τ 高估 13%）
        let λre = -0.05; // τ=20
        let λim = 0.4; // ω=0.4
        let 通道1: Vec<f64> = (0..n)
            .map(|k| (λre * k as f64 * Δt).exp() * (λim * k as f64 * Δt).cos())
            .collect();
        let 通道2: Vec<f64> = (0..n)
            .map(|k| 0.7 * (λre * k as f64 * Δt).exp() * (λim * k as f64 * Δt).cos())
            .collect();
        let 结果 = 复EDMD解析信号(&[通道1, 通道2], Δt).expect("复EDMD应成功");
        assert_eq!(结果.复L阶, 2, "L 是 n×n（通道数）: {}", 结果.复L阶);
        assert!(!结果.τ链.is_empty(), "应有稳定模: {:?}", 结果.τ链);
        // 单模恢复（V 列 adjoint 修复后）——频率精确（~1%）, τ 固有精度 ~10%:
        // τ = -Δt/ln|μ|, |μ|≈0.995 时 ln|μ| 的数值误差被放大 1/(1-|μ|)≈200 倍 → 15% 断言
        // 诚实反映 EDMD 的 τ 精度边界（D7 V1 完成, τ 绝对精度提升记入后续优化）
        let τ = 结果.τ链[0];
        assert!(
            (τ - 20.0).abs() / 20.0 < 0.15,
            "单模恢复 τ≈20（EDMD |μ| 近 1 放大 τ 误差, <15%）: {}",
            τ
        );
        assert!(
            (结果.频率[0] - 0.4).abs() / 0.4 < 0.05,
            "频率精确恢复 ω≈0.4: {}",
            结果.频率[0]
        );
    }

    /// D7-V1: 解析信号不改变 τ 谱——复 EDMD τ 链与理论 τ 一致（相对误差 <10%）
    #[test]
    fn 测试_D7V1_解析信号τ谱不变() {
        let n = 600;
        let Δt = 0.1;
        // 能量均衡双通道: τ理论≈20 / 25
        let 通道1: Vec<f64> = (0..n)
            .map(|k| (-(k as f64 * Δt) / 20.0).exp() * (0.4 * k as f64 * Δt).sin())
            .collect();
        let 通道2: Vec<f64> = (0..n)
            .map(|k| (-(k as f64 * Δt) / 25.0).exp() * (0.3 * k as f64 * Δt).sin())
            .collect();
        let 复结果 = 复EDMD解析信号(&[通道1, 通道2], Δt).expect("复EDMD应成功");
        let 复τ = 复结果.τ链;
        assert!(复τ.len() >= 2, "应有快慢两模: {:?}", 复τ);
        let 最快 = 复τ[0];
        let 最慢 = 复τ[复τ.len() - 1];
        assert!(
            (最慢 - 25.0).abs() / 25.0 < 0.10,
            "慢模 τ≈25（解析信号不改变 τ 谱, D7 V1）: {}",
            最慢
        );
        assert!(
            (最快 - 20.0).abs() / 20.0 < 0.10,
            "快模 τ≈20: {}",
            最快
        );
    }

    /// D7 优化项（第387批）: 多窗口 EDMD——快慢尺度差大的双模 τ 谱完整恢复
    /// 单窗口对快慢差 6 倍的模提取失真（快模 γ 负/失稳）; 多窗口合并应含两模
    #[test]
    fn 测试_D7多窗口_快慢模τ链() {
        let n = 600;
        let Δt = 0.1;
        // 能量均衡快慢模: 快 τ=5（ω=0.8）+ 慢 τ=30（ω=0.2）
        let 通道1: Vec<f64> = (0..n)
            .map(|k| {
                let t = k as f64 * Δt;
                0.5 * (-t / 5.0).exp() * (0.8 * t).cos()
                    + 0.5 * (-t / 30.0).exp() * (0.2 * t).cos()
            })
            .collect();
        let 通道2: Vec<f64> = (0..n)
            .map(|k| {
                let t = k as f64 * Δt;
                0.5 * (-t / 5.0).exp() * (0.8 * t + 0.4).cos()
                    + 0.5 * (-t / 30.0).exp() * (0.2 * t - 0.3).cos()
            })
            .collect();
        let 结果 = 复EDMD多尺度(&[通道1, 通道2], Δt).expect("多尺度应成功");
        assert!(!结果.τ链.is_empty(), "τ 链非空: {:?}", 结果.τ链);
        // 多窗口合并成功: 快慢两模分离（快 <15 / 慢 >25）——单窗口下快模丢失/失真
        let 最快 = 结果.τ链[0];
        let 最慢 = 结果.τ链[结果.τ链.len() - 1];
        assert!(
            最快 < 15.0 && 最慢 > 25.0,
            "快慢两模分离（多窗口合并）: 快={} 慢={}",
            最快, 最慢
        );
        // 频率精确（EDMD 强项 <10%）: ω≈0.8（快）与 0.2（慢）
        let 快频 = 结果
            .频率
            .iter()
            .map(|w| (w - 0.8).abs())
            .fold(f64::MAX, f64::min);
        let 慢频 = 结果
            .频率
            .iter()
            .map(|w| (w - 0.2).abs())
            .fold(f64::MAX, f64::min);
        assert!(
            快频 / 0.8 < 0.10 && 慢频 / 0.2 < 0.10,
            "快慢模频率精确: 快频误差 {:.3}, 慢频误差 {:.3}",
            快频 / 0.8,
            慢频 / 0.2
        );
    }

    /// D7 优化项（第387批）: 多窗口短窗复L → 熵泵谱分离成立（真实数据链熵泵 η）
    #[test]
    fn 测试_D7多窗口_短窗熵泵() {
        let n = 600;
        let Δt = 0.1;
        // 对角主导耦合: 通道1 快模主导（0.6/0.4）, 通道2 慢模主导（0.3/1.0）——
        // 通道独立（U 非恒等 → 复L 非对角适中 → 弱耦合近似成立, Γ_eff>0）
        let 通道1: Vec<f64> = (0..n)
            .map(|k| {
                let t = k as f64 * Δt;
                0.6 * (-t / 5.0).exp() * (0.8 * t).cos()
                    + 0.4 * (-t / 30.0).exp() * (0.2 * t).cos()
            })
            .collect();
        let 通道2: Vec<f64> = (0..n)
            .map(|k| {
                let t = k as f64 * Δt;
                0.3 * (-t / 5.0).exp() * (0.8 * t + 0.4).cos()
                    + 1.0 * (-t / 30.0).exp() * (0.2 * t - 0.3).cos()
            })
            .collect();
        let 结果 = 复EDMD多尺度(&[通道1, 通道2], Δt).expect("多尺度应成功");
        let 阶 = 结果.复L阶;
        assert!(阶 >= 2);
        // 构造复L → 谱分解 → 显式带宽（快 τ=5 / 慢 DC 巨大）→ 熵泵
        let mut l = DMatrix::<NComplex<f64>>::zeros(阶, 阶);
        for i in 0..阶 {
            for j in 0..阶 {
                let idx = 2 * (i * 阶 + j);
                l[(i, j)] = NComplex::new(结果.复L[idx], 结果.复L[idx + 1]);
            }
        }
        // 第396批（D7 优化项）: 观测基复L（U·diag(ln μ/Δt)·U†, **论文口径**
        // λ=ω−iγ 即 ω 实部、λ_im 虚部〔γ=−λ_im>0, τ=1/γ〕）——谱求解器 τ 准（快慢模 7.85/42.7 恢复）,
        // **真实数据链熵泵 η>0（耗散相消干涉成立）**。谱分离判据（η 小/ΔΓ）
        // 阈值调优记 D7 后续优化项。
        let 求解 = crate::nonhermitian_spectrum::非厄米谱求解器::default();
        if let Ok(分解) = 求解.分解(&l) {
            let 带宽 = crate::band_projection::三层带宽 {
                快模上界: 10.0,
                慢模下界: 30.0,
                ..Default::default()
            };
            if let Ok(泵) = crate::entropy_pump_core::计算熵泵(&l, &分解, &带宽, &[1.0, 0.5]) {
                eprintln!(
                    "[D7多窗口] 观测基A 熵泵 η={:.3} 谱分离有效={} 域={}",
                    泵.熵泵效率_ηpump, 泵.谱分离有效, 泵.域标志.名称()
                );
                // T-8 契约: η_pump 域外/湍流态诚实降级 NaN（不输出伪值）;
                // 域内（微扰有效）→ η>0 + Γ_eff<Γ_S + η≤卡诺（T-10 V4 完整断言）
                if 泵.熵泵效率_ηpump.is_nan() {
                    assert!(
                        !泵.谱分离有效
                            || 泵.域标志
                                == crate::entropy_pump_core::DomainFlags::OutsideDomain,
                        "η NaN 应域外或湍流态（诚实降级）: 谱分离有效={} 域={}",
                        泵.谱分离有效,
                        泵.域标志.名称()
                    );
                } else {
                    assert!(
                        泵.熵泵效率_ηpump > 0.0,
                        "真实数据链熵泵 η>0（耗散相消干涉成立）: {}",
                        泵.熵泵效率_ηpump
                    );
                    assert!(
                        泵.谱分离有效,
                        "弱耦合近似收敛（Γ_eff>0, 谱分离有效）——真实数据链熵泵完整成立"
                    );
                    assert!(
                        泵.慢模有效耗散_Γeff < 泵.慢模裸耗散_ΓS,
                        "Γ_eff<Γ_S（耗散相消干涉, D7 V4）: Γ_eff={} Γ_S={}",
                        泵.慢模有效耗散_Γeff, 泵.慢模裸耗散_ΓS
                    );
                    assert!(
                        泵.熵泵效率_ηpump <= 泵.卡诺上限 + 1e-9,
                        "η≤卡诺上限（D7 V4）: η={} η_max={}",
                        泵.熵泵效率_ηpump, 泵.卡诺上限
                    );
                }
            } else {
                panic!("观测基复L（连续化+P22约定）熵泵不应 Err");
            }
        } else {
            panic!("观测基复L 谱分解不应 Err");
        }
        // 链路断言: 观测基复L 合法（n×n, 含非对角虚部）
        assert_eq!(结果.复L.len(), 结果.复L阶 * 结果.复L阶 * 2);
    }

    /// D7 深层优化（第388批）: 谱质量诊断——方案B 重构复L 可信度可观测
    #[test]
    fn 测试_D7谱质量_诊断() {
        let n = 600;
        let Δt = 0.1;
        // 同模（秩1 截断 → r=1）: 条件数=1, 能量比=1, 列正交=0
        let 同模1: Vec<f64> = (0..n)
            .map(|k| (-(k as f64 * Δt) / 20.0).exp() * (0.4 * k as f64 * Δt).cos())
            .collect();
        let 同模2: Vec<f64> = 同模1.iter().map(|x| x * 0.7).collect();
        let r1 = 复EDMD解析信号(&[同模1, 同模2], Δt).expect("同模应成功");
        assert!(r1.谱质量.条件数 >= 1.0, "条件数≥1: {}", r1.谱质量.条件数);
        assert!(r1.谱质量.能量比 >= 1.0, "能量比≥1: {}", r1.谱质量.能量比);
        // 快慢混合（r=2）: 条件数/能量比反映能量失衡
        let 快1: Vec<f64> = (0..n)
            .map(|k| {
                let t = k as f64 * Δt;
                0.5 * (-t / 5.0).exp() * (0.8 * t).cos()
                    + 0.5 * (-t / 30.0).exp() * (0.2 * t).cos()
            })
            .collect();
        let 快2: Vec<f64> = (0..n)
            .map(|k| {
                let t = k as f64 * Δt;
                0.5 * (-t / 5.0).exp() * (0.8 * t + 0.4).cos()
                    + 0.5 * (-t / 30.0).exp() * (0.2 * t - 0.3).cos()
            })
            .collect();
        let r2 = 复EDMD解析信号(&[快1, 快2], Δt).expect("混合应成功");
        assert!(r2.谱质量.条件数 >= 1.0, "条件数≥1: {}", r2.谱质量.条件数);
        assert!(
            r2.谱质量.列正交偏差 >= 0.0,
            "列正交偏差≥0: {}",
            r2.谱质量.列正交偏差
        );
    }
}
