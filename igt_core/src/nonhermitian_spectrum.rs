// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! 非厄米谱动力学 — 第267批 P22 §6 core 数学根基三连（A+B+C）
//!
//! 理论依据:
//!   - 《P22 IGT Engine：τ工程化架构与接口》§6.2 复本征谱分解与双正交归一化:
//!     L=H-iΓ 非厄米 → 复本征值 λ_n=ω_n-i/τ_n → 右V归一化‖v‖=1 → 左W=V^{-H}
//!     （逆共轭转置, 双正交 w†v=δ）→ 展开系数 c_n=w†Δx(0)
//!   - §6.4 椭圆系综拟合（替代 GUE 检验）: τ_ellipse=(σ_H²-σ_Γ²)/(σ_H²+σ_Γ²)
//!     → D_s^(2D)=W_2(ρ_sys,ρ_ellipse) 复平面二维 Wasserstein 距离
//!   - §6.3 谱分离（Schrieffer-Wolff）: 快模积分自能 Σ(ω), L_eff=P_S L P_S+Σ,
//!     Γ_eff=-Im Σ>0 因果性检验, 谱隙 Δ
//!
//! 核心定位: 非厄米 L=H-iΓ 的复谱是 P22 定义的 core 核心算法根基——
//!   参与度（模态可观度/激发占比）、基因匹配（复谱距离+子空间夹角）、
//!   逆元建议、椭圆谱校验全部建立其上。
//!
//! 诚实标注: 复 Schur 的左右特征向量提取用"Schur 形式 + 反代求解"自实现
//!   （nalgebra 0.32 的 Schur 对复矩阵只给特征值不给双正交向量组）;
//!   适用于小型矩阵（n≤64）, 大矩阵可换 LAPACK zgees。
//!   本模块 100% 纯计算, 零网络零领域知识（core 定位）。

#![allow(non_snake_case)]

use nalgebra::Complex as NComplex;
use nalgebra::{ComplexField, DMatrix};
use rand::Rng;
use std::f64::consts::PI;

// ============================================================================
// 非厄米谱分解结果
// ============================================================================

/// 单个非厄米模（复本征值 + 双正交向量 + 参与度）
///
/// ★ γ符号统一令 v1.0（§3/§6）：**λ_im 与 γ 分离**——
///   `λ_im` = Im λ（矩阵量，稳定 < 0，与复 L 本征值虚部同值）；`γ` = −λ_im（**耗散率，稳定 > 0**）；
///   `τ` = 1/γ = −1/λ_im（数值与旧实现恒等）。
/// 历史数据（旧口径：字段 γ 存的是 Im λ，稳定负）须经 [`非厄米模::适配历史`] 一次性换算，
/// **不得静默反号**（统一令 §6；core 内当前无 非厄米模 落盘持久化 ⇒ 适配供后续对接用）。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct 非厄米模 {
    /// 复本征值实部 ω_n（振荡频率 rad/s）
    pub ω: f64,
    /// **λ_im = Im λ**（本征值虚部＝增长率；稳定 < 0；矩阵读写层使用）
    #[serde(default)]
    pub λ_im: f64,
    /// **耗散率 γ_n = −λ_im = 1/τ_n（统一令后：稳定 > 0）**
    /// ⚠ 旧版（统一令前）此字段存的是 Im λ（稳定为负）；读历史数据须经 `适配历史`（不静默反号）。
    pub γ: f64,
    /// 弛豫时间 τ_n = 1/γ_n = −1/λ_im（γ>0 稳定模；数值与旧口径恒等）
    pub τ: f64,
    /// 右本征向量 v 实部（归一化 ‖v‖=1，模态形状：哪些变量参与）
    pub v: Vec<f64>,
    /// 右本征向量 v 虚部（第289批修复: 非厄米右向量是复向量, 虚部丢弃致参与度带病 P22§6.2）
    #[serde(default)]
    pub v虚: Vec<f64>,
    /// 左本征向量 w 实部（双正交 w†v=δ，可激发方向）
    pub w: Vec<f64>,
    /// 左本征向量 w 虚部（第289批修复: w=V^{-H} 完整复, 展开系数 c_n=w†Δx 需共轭）
    #[serde(default)]
    pub w虚: Vec<f64>,
    /// 模态可观度 p_obs = ‖w‖²/Σ‖w‖²（不依赖扰动）
    pub p_obs: f64,
    /// 激发能量占比 p_exc = |c|²/Σ|c|²（依赖具体扰动，运行时计算）
    pub p_exc: f64,
    /// ★ Petermann 因子 K_n=(w_n†w_n)(v_n†v_n)（双正交归一 w†v=1 下 K≥1）:
    /// 正规/可对角化良好矩阵 K≈1；趋近异常点（EP）本征矢病态、K 发散（噪声放大）。
    /// T-71 段②: K 超门限（见 `kappa_channel::EP条件数门限`）须**标注、不静默给模态边**。
    #[serde(default)]
    pub petermann: f64,
}

impl 非厄米模 {
    /// ★ γ符号统一令 §6：**旧口径历史数据的一次性适配**（不静默反号、显式调用）
    /// 旧口径：字段 `γ` 存 Im λ（稳定 < 0）、τ = −1/γ。
    /// 新口径：γ = −Im λ（稳定 > 0）、τ = 1/γ、λ_im = Im λ。
    /// 数值不变量：**τ 两口径恒等**；ω/向量/参与度不受影响。
    pub fn 适配历史(mut 旧: 非厄米模) -> 非厄米模 {
        旧.λ_im = 旧.γ; // 旧 γ 就是 Im λ
        旧.γ = -旧.λ_im; // 新 γ = −Im λ（正=稳定）
        // τ 数值不变（−1/Im λ ≡ 1/γ），无需改
        旧
    }
}

/// 非厄米谱分解结果
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct 非厄米谱分解 {
    /// 模数
    pub 模数: usize,
    /// 全部模（按 τ 从大到小排序，慢模在前）
    pub 模: Vec<非厄米模>,
    /// H=(L+L†)/2 对称部分
    pub H: Vec<Vec<f64>>,
    /// Γ=i(L†-L)/2 耗散部分
    pub Γ: Vec<Vec<f64>>,
    /// 稳定模数（γ>0，统一令后口径）
    pub 稳定模数: usize,
    /// 判定
    pub 判定: String,
}

// ============================================================================
// 椭圆系综拟合结果（D_s^(2D)）
// ============================================================================

/// 椭圆系综拟合结果（P22 §6.4）
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct 椭圆系综结果 {
    /// 椭圆参数 τ_ellipse=(σ_H²-σ_Γ²)/(σ_H²+σ_Γ²) ∈ [-1,1]
    pub τ_ellipse: f64,
    /// 椭圆中心（复平面）
    pub 中心实部: f64,
    pub 中心虚部: f64,
    /// 半轴 a（实部方向）
    pub 半轴a: f64,
    /// 半轴 b（虚部方向）
    pub 半轴b: f64,
    /// D_s^(2D)=W_2(ρ_sys,ρ_ellipse) 复平面二维 Wasserstein 距离
    pub d_s_2d: f64,
    /// 绿<0.3 / 黄0.3-0.6 / 红≥0.6
    pub 区域: String,
    /// 径向偏离（τ 分布异常）
    pub 径向偏离: f64,
    /// 角向偏离（振荡结构异常）
    pub 角向偏离: f64,
    /// 极限检验：τ_ellipse→-1 应恢复 GOE；→1 应恢复 GUE
    pub 极限检验: String,
    /// 本征值数
    pub 本征值数: usize,
}

// ============================================================================
// 谱分离（Schrieffer-Wolff）结果
// ============================================================================

/// 谱分离（Schrieffer-Wolff）结果（P22 §6.3）
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct 谱分离结果 {
    /// 快模数（τ<τ_min）
    pub 快模数: usize,
    /// 慢模数（τ>τ_max）
    pub 慢模数: usize,
    /// 中带模数（τ_min≤τ≤τ_max）
    pub 中带模数: usize,
    /// 自能 Σ(ω)（复矩阵，快模积分修正）
    pub 自能: Vec<Vec<f64>>,
    /// 有效算子 L_eff = P_S L P_S + Σ(ω)
    pub l_eff: Vec<Vec<f64>>,
    /// Γ_eff = -Im Σ(ω) 对角（因果性检验：>0 通过）
    pub Γ_eff对角: Vec<f64>,
    /// 谱隙 Δ = min(τ_S^{-1}) - max(τ_F^{-1})
    pub 谱隙: f64,
    /// 误差界 C·e^{-Δt/2}
    pub 误差界: f64,
    /// 因果性检验通过（Γ_eff>0 全正）
    pub 因果性通过: bool,
    /// 判定
    pub 判定: String,
}

// ============================================================================
// 非厄米谱求解器
// ============================================================================

/// 非厄米谱求解器 — 复谱分解 + 双正交归一化 + 椭圆系综 + 谱分离
#[derive(Debug, Clone, Copy)]
pub struct 非厄米谱求解器 {
    /// 复 Schur 迭代最大次数
    pub max_iter: usize,
    /// 收敛容差
    pub ε: f64,
    /// 椭圆系综 MC 采样数（W_2 用）
    pub n_sample: usize,
}

impl Default for 非厄米谱求解器 {
    fn default() -> Self {
        Self {
            max_iter: 200,
            ε: 1e-9,
            n_sample: 2000,
        }
    }
}

/// 平衡对角矩阵（第396批, Parlett-Reinsch 一次版）——L' = D L D⁻¹（相似, 本征值不变）
///
/// 快慢尺度差大的矩阵（如观测基 A=UÃU†, 快模/慢模量级悬殊）→ QR 迭代数值不稳
/// → 本征值失真（τ 负/ω≈1）→ 熵泵谱分离失效。平衡后行/列范数均衡 → QR 稳健。
/// **一次性平衡**（基于原始行/列和, O(n²), scale 钳制防爆炸——迭代版实测
/// scale 累积振荡导致 900s 超时, 回退为一次版）。
fn 平衡对角(l: &DMatrix<NComplex<f64>>) -> (DMatrix<NComplex<f64>>, DMatrix<NComplex<f64>>) {
    let n = l.nrows();
    let mut d = DMatrix::<NComplex<f64>>::identity(n, n);
    let mut dinv = DMatrix::<NComplex<f64>>::identity(n, n);
    if n < 2 {
        return (d, dinv);
    }
    let 行和: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| l[(i, j)].modulus()).sum())
        .collect();
    let 列和: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| l[(j, i)].modulus()).sum())
        .collect();
    for i in 0..n {
        let scale = (行和[i] / 列和[i].max(1e-300))
            .sqrt()
            .clamp(1e-3, 1e3);
        d[(i, i)] *= NComplex::new(scale, 0.0);
        dinv[(i, i)] *= NComplex::new(1.0 / scale, 0.0);
    }
    (d, dinv)
}

impl 非厄米谱求解器 {
    /// 复矩阵本征分解（右特征向量 + 双正交左向量）
    ///
    /// 方法：复 Schur 分解（QR 迭代，适用于小型矩阵 n≤64）→
    /// 对角元=复本征值 → 特征向量反代（(L-λI)v=0 解零空间）。
    /// 对复矩阵 L，右向量 V 归一化后，左向量 W=V^{-H}（逆共轭转置）
    /// 自动满足双正交 w†v=δ。
    pub fn 分解(&self, l: &DMatrix<NComplex<f64>>) -> Result<非厄米谱分解, String> {
        let n = l.nrows();
        if n == 0 || l.ncols() != n {
            return Err("L 须为方阵".to_string());
        }
        if n > 64 {
            return Err(format!("n={} 超小型矩阵上限 64（当前自实现 QR 迭代）", n));
        }

        // 第396批（D7 优化项）: 平衡预处理——快慢尺度差大的矩阵元素量级悬殊
        // → QR 迭代数值不稳 → 本征值失真（谱分解 τ 负/ω≈1）→ 熵泵谱分离失效。
        // L' = D L D⁻¹（相似, 本征值不变, 行/列范数均衡）; 特征向量转换 v_原 = D⁻¹ v_平衡
        //   （T-71 修复: 由 L'x=λx 左乘 D⁻¹ 得 L(D⁻¹x)=λ(D⁻¹x); 早期误写 D·x,
        //     对称/平衡良好矩阵 D≈I 碰巧无差, 非对称非对角 L 下右矢/κ 失准）。
        let (d, dinv) = 平衡对角(l);
        let l_平衡 = &d * l * &dinv;
        // 复 Schur：QR 迭代得复本征值
        let 特征值: Vec<NComplex<f64>> = self.复特征值(&l_平衡)?;
        // 右矢 + Rayleigh 精化本征值（RQI; 平衡基 ρ 经相似变换＝原 L 谱）
        let mut 模列表: Vec<非厄米模> = Vec::with_capacity(n);
        for &λ in &特征值 {
            let (v_复平衡, λ精) = self.特征向量(&l_平衡, λ, n)?;
            let v_复vec = &dinv * nalgebra::DVector::from_column_slice(&v_复平衡); // v_原=D⁻¹v_平衡
            let v_复: Vec<NComplex<f64>> = v_复vec.iter().copied().collect();
            // 右向量归一化 ‖v‖=1
            let 范数 = v_复.iter().map(|z| z.modulus_squared()).sum::<f64>().sqrt();
            if 范数 <= 1e-12 {
                continue;
            }
            // 第289批修复: 右向量保存完整复数（实部+虚部）, 不再丢弃虚部
            let v归一: Vec<NComplex<f64>> = v_复.iter().map(|z| z / NComplex::new(范数, 0.0)).collect();
            let 模 = 非厄米模 {
                ω: λ精.re,
                λ_im: λ精.im,          // 本征值虚部（稳定 < 0）
                γ: -λ精.im,            // 耗散率（统一令：γ=−λ_im，稳定 > 0）
                τ: if λ精.im.abs() > 1e-12 { -1.0 / λ精.im } else { f64::INFINITY }, // = 1/γ（数值不变）
                v: v归一.iter().map(|z| z.re).collect(),
                v虚: v归一.iter().map(|z| z.im).collect(),
                w: vec![],   // 双正交归一化阶段填
                w虚: vec![],
                p_obs: 0.0,
                p_exc: 0.0,
                petermann: 1.0, // 双正交归一化阶段按 ‖w‖²‖v‖² 填（v 已归一化, =‖w‖²）
            };
            模列表.push(模);
        }
        if 模列表.is_empty() {
            return Err("未分解出有效本征模".to_string());
        }

        // 双正交归一化：V 列 = 右向量（完整复数）→ W = V^{-H}
        let m = 模列表.len();
        let mut v矩阵 = DMatrix::<NComplex<f64>>::zeros(n, m);
        for (j, 模) in 模列表.iter().enumerate() {
            for i in 0..n {
                let v实 = if i < 模.v.len() { 模.v[i] } else { 0.0 };
                let v虚 = if i < 模.v虚.len() { 模.v虚[i] } else { 0.0 };
                v矩阵[(i, j)] = NComplex::new(v实, v虚);
            }
        }
        // V^H V（m×m 小块）→ 逆 → W = V (V^H V)^{-1}（= V^{-H}，共轭转置逆）
        // 第289批修复: 全程复数运算, 右向量虚部不再丢弃
        let vhv = v矩阵.adjoint() * &v矩阵;
        let vhv逆 = match vhv.try_inverse() {
            Some(inv) => inv,
            None => return Err("V^H V 奇异，无法双正交归一化（退化模）".to_string()),
        };
        let w_candidates = &v矩阵 * vhv逆; // n×m, 第 j 列 = w_j（完整复数）

        // 填充左向量 + 参与度
        let 总p_obs2: f64 = (0..m)
            .map(|j| {
                (0..n).map(|i| w_candidates[(i, j)].modulus_squared()).sum::<f64>()
            })
            .sum();
        let mut 稳定模数 = 0;
        for (j, 模) in 模列表.iter_mut().enumerate() {
            let w实: Vec<f64> = (0..n).map(|i| w_candidates[(i, j)].re).collect();
            let w虚: Vec<f64> = (0..n).map(|i| w_candidates[(i, j)].im).collect();
            let w范数2: f64 = (0..n).map(|i| w_candidates[(i, j)].modulus_squared()).sum();
            // v 已 ‖·‖=1, 这里仍显式带 v†v 以防上游归一化约定变化; K_n=(w†w)(v†v)
            let v范数2: f64 = (0..n)
                .map(|i| 模.v.get(i).copied().unwrap_or(0.0).powi(2)
                    + 模.v虚.get(i).copied().unwrap_or(0.0).powi(2))
                .sum();
            模.w = w实;
            模.w虚 = w虚;
            模.petermann = w范数2 * v范数2;
            模.p_obs = if 总p_obs2 > 1e-12 { w范数2 / 总p_obs2 } else { 1.0 / m as f64 };
            if 模.γ > self.ε {
                // 稳定 ⟺ γ>0（统一令 §1；旧口径 γ<−ε 已反转，逻辑等价）
                稳定模数 += 1;
            }
        }

        // 按 τ 从大到小排序（慢模在前，P22 §6.2）
        模列表.sort_by(|a, b| b.τ.partial_cmp(&a.τ).unwrap_or(std::cmp::Ordering::Equal));

        // H/Γ 分解
        let h = 对称部分(l);
        let Γ = 耗散部分(l);

        let 判定 = if 稳定模数 == 模列表.len() {
            "全稳定（耗散系统健康）".to_string()
        } else if 稳定模数 == 0 {
            "全不稳定（发散，危险）".to_string()
        } else {
            format!("{}/{} 稳定（部分模失稳）", 稳定模数, 模列表.len())
        };

        Ok(非厄米谱分解 {
            模数: m,
            模: 模列表,
            H: h,
            Γ,
            稳定模数,
            判定,
        })
    }

    /// 复特征值（复 Schur QR 迭代）
    fn 复特征值(&self, l: &DMatrix<NComplex<f64>>) -> Result<Vec<NComplex<f64>>, String> {
        // 第396批（D7 优化项）: 复用 complex_dmd `复特征分解_QR`（Francis 双位移 +
        // 稳定复 QR 分解）——本模块自实现（Gram-Schmidt QR + Rayleigh/双位移迭代）
        // 对快慢尺度差大的矩阵数值不稳, 实测提取 τ 负（观测基 A 本征值数学准,
        // 但自实现 QR 收敛错）; 复特征分解_QR 实测与物理模一致。
        Ok(crate::complex_dmd::复特征分解_QR(l, self.max_iter.max(300)).0)
    }

    /// 复矩阵 QR 分解（改进 Gram-Schmidt）
    fn qr_复(a: &DMatrix<NComplex<f64>>) -> (DMatrix<NComplex<f64>>, DMatrix<NComplex<f64>>) {
        let n = a.nrows();
        let mut q = DMatrix::<NComplex<f64>>::zeros(n, n);
        let mut r = DMatrix::<NComplex<f64>>::zeros(n, n);
        for j in 0..n {
            let mut v: Vec<NComplex<f64>> = (0..n).map(|i| a[(i, j)]).collect();
            for i in 0..j {
                let r_ij = (0..n).map(|k| q[(k, i)].conjugate() * v[k]).sum::<NComplex<f64>>();
                r[(i, j)] = r_ij;
                for k in 0..n {
                    v[k] -= r_ij * q[(k, i)];
                }
            }
            let 范数: f64 = v.iter().map(|z| z.modulus_squared()).sum::<f64>().sqrt();
            if 范数 > 1e-12 {
                r[(j, j)] = NComplex::new(范数, 0.0);
                for k in 0..n {
                    q[(k, j)] = v[k] / NComplex::new(范数, 0.0);
                }
            }
        }
        (q, r)
    }

    /// 特征向量 + 精化本征值：对给定初值 λ（QR）求右向量 v 与精化 λ。
    ///
    /// T-71 段① 精度修复（κ=w†δLv 要求机器精度左右矢, 旧实现残差 1e-2 与弱耦合同量级）:
    ///   1) 复 SVD 约定: nalgebra `v_t` 存 **V^H**, 须 `.adjoint()` 还原 V（旧码漏, 复向量被共轭）;
    ///   2) 仅取 SVD 向量无精化, 强混合/非正规矩阵残差 1e-2~1e-4 ⇒ 加 **Rayleigh 商迭代**
    ///      （ρ=v†Lv/v†v, 复向量须 `dotc` 共轭点积）, 非厄米 RQI 局部三次收敛;
    ///   3) 平衡 L'=D L D⁻¹ 的右矢基变换是 **v_原=D⁻¹v_平衡**（旧码误写 D·v; 对称矩阵 D≈I
    ///      碰巧无差, 非对称非对角 L 下右矢/κ 失准——T-66 仅对特征值故未暴露）。
    /// 选模初值按维数混合: 小 n(≤20) 用 (L-λI) 最小奇异右矢（秩亏/近简并最鲁棒）;
    ///   大 n(>20) 用均匀向量 + 固定位移逆迭代（省每模 SVD, 保 EDMD n=30/60 限时）。
    ///   之后统一 RQI, 追踪残差最小者。精化 ρ 经相似变换＝原 L 谱, ω/γ/τ 只更准。
    fn 特征向量(
        &self,
        l: &DMatrix<NComplex<f64>>,
        λ0: NComplex<f64>,
        n: usize,
    ) -> Result<(Vec<NComplex<f64>>, NComplex<f64>), String> {
        // 位移矩阵 A0 = L-λ0 I（固定逆迭代复用其 LU; SVD 也作用其上）
        let a0 = {
            let mut t = l.clone();
            for i in 0..n {
                t[(i, i)] -= λ0;
            }
            t
        };
        let 均匀 = || {
            nalgebra::DVector::from_element(n, NComplex::new((n as f64).sqrt().recip(), 0.0))
        };
        // 初值按维数二选一:
        //  · 小 n(≤20): (L-λ0) 最小奇异值右奇异向量（V=v_t.adjoint() 还原）——对秩亏/近简并/
        //    位移极准（LU 近奇异）最鲁棒, 且 n 小 O(n³) 可忽略;
        //  · 大 n(>20, EDMD 满非对角、非退化): 均匀向量初值, 省掉每模一次 SVD（n=30/60 限时大头）,
        //    固定位移逆迭代按 1/|λ0-λk| 加权自动选模。
        let mut v: nalgebra::DVector<NComplex<f64>> = if n <= 20 {
            let mut v0 = 均匀();
            let svd: nalgebra::linalg::SVD<NComplex<f64>, nalgebra::Dyn, nalgebra::Dyn> =
                a0.clone().svd(true, true);
            if let Some(vt) = svd.v_t {
                let vmat = vt.adjoint(); // 复 SVD 存 V^H, 须 adjoint 还原
                let col = vmat.ncols().saturating_sub(1); // 奇异值降序 ⇒ 末列 σ_min
                let cand: nalgebra::DVector<NComplex<f64>> = nalgebra::DVector::from_column_slice(
                    (0..n).map(|i| vmat[(i, col)]).collect::<Vec<_>>().as_slice(),
                );
                let cn = cand.norm();
                if cn > 1e-12 {
                    v0 = cand / NComplex::new(cn, 0.0);
                }
            }
            v0
        } else {
            均匀()
        };
        // 固定位移逆迭代: 小 n 的 SVD 初值已在零空间(0 步, 直接交 RQI);
        //   大 n 由均匀起点逆迭代选模(6 步)。
        let lu0 = a0.lu();
        let 固定步 = if n <= 20 { 0 } else { 6 };
        for _ in 0..固定步 {
            match lu0.solve(&v) {
                Some(y) if y.norm().is_finite() && y.norm() > 1e-12 => {
                    let yn = y.norm();
                    v = y / NComplex::new(yn, 0.0);
                }
                _ => break,
            }
        }

        // 阶段B: Rayleigh 商迭代（v 已锁定该模）, ρ 精化位移 → 残差压到机器精度
        let mut λcur = λ0;
        let mut 最佳v = v.clone();
        let mut 最佳λ = λ0;
        let mut 最佳残 = (l * &v - &v * λ0).norm();
        for _ in 0..4 {
            // 位移逆步: y = (L-λI)⁻¹ v
            let a = {
                let mut t = l.clone();
                for i in 0..n {
                    t[(i, i)] -= λcur;
                }
                t
            };
            let y = match a.lu().solve(&v) {
                Some(y) if y.norm().is_finite() && y.norm() > 1e-12 => y,
                _ => break,
            };
            let yn = y.norm();
            v = y / NComplex::new(yn, 0.0);
            // Rayleigh 商 ρ = v† L v / v† v（复向量须 dotc 共轭点积; .dot 对复数不共轭）
            let lv = l * &v;
            let rho = v.dotc(&lv) / v.dotc(&v);
            λcur = rho;
            let 残 = (lv - &v * rho).norm();
            if 残 < 最佳残 {
                最佳残 = 残;
                最佳v = v.clone();
                最佳λ = rho;
            }
            if 残 < 1e-11 {
                break;
            }
        }
        if !最佳λ.is_finite() || 最佳v.norm() <= 1e-12 {
            return Err("特征向量 Rayleigh 迭代失败（退化/范数丢失）".to_string());
        }
        Ok(((0..n).map(|i| 最佳v[i]).collect(), 最佳λ))
    }

    /// 展开系数 c_n = w†Δx(0)（P22 §6.2 第4步）
    /// 第289批修复: w 为完整复数, c_n = Σ conj(w_i)·x_i, 返回 |c_n|²
    pub fn 展开系数(&self, 分解: &非厄米谱分解, 初态: &[f64]) -> Vec<f64> {
        let n = 初态.len();
        分解
            .模
            .iter()
            .map(|模| {
                let mut 实部 = 0.0f64;
                let mut 虚部 = 0.0f64;
                for i in 0..n {
                    let w实 = if i < 模.w.len() { 模.w[i] } else { 0.0 };
                    let w虚 = if i < 模.w虚.len() { 模.w虚[i] } else { 0.0 };
                    let x = if i < 初态.len() { 初态[i] } else { 0.0 };
                    // conj(w)·x = (w实 - i·w虚)·x
                    实部 += w实 * x;
                    虚部 -= w虚 * x;
                }
                // |c|² = 实部² + 虚部²
                实部 * 实部 + 虚部 * 虚部
            })
            .collect()
    }

    /// 激发占比 p_exc（依赖扰动，按展开系数 |c|²/Σ|c|²）
    pub fn 激发占比(&self, 分解: &非厄米谱分解, 初态: &[f64]) -> Vec<f64> {
        let c2: Vec<f64> = self.展开系数(分解, 初态);
        let 总和: f64 = c2.iter().sum();
        if 总和 <= 1e-12 {
            return vec![0.0; c2.len()];
        }
        c2.iter().map(|x| x / 总和).collect()
    }

    // ============ 椭圆系综拟合（P22 §6.4） ============

    /// 椭圆系综拟合：带内复本征值 → 椭圆参数 + D_s^(2D)
    ///
    /// 复本征值 {λ_n}（**ω, λ_im**；λ_im=Im λ=−1/τ，统一令 §5 权威表示）
    /// → 中心=均值 → 半轴=(σ_ω, σ_λim)〔结果结构体内沿用历史字段名 σ_γ，语义＝虚部方差〕
    /// → τ_ellipse=(σ_ω²-σ_γ²)/(σ_ω²+σ_γ²) → 归一化椭圆 ρ_ellipse
    /// → W_2(ρ_sys, ρ_ellipse)（复平面二维 Wasserstein，MC 采样近似）
    pub fn 椭圆系综(&self, 本征值: &[(f64, f64)]) -> Result<椭圆系综结果, String> {
        let n = 本征值.len();
        if n < 3 {
            return Err("椭圆拟合须 ≥3 个本征值".to_string());
        }
        // 中心
        let 中心实 = 本征值.iter().map(|z| z.0).sum::<f64>() / n as f64;
        let 中心虚 = 本征值.iter().map(|z| z.1).sum::<f64>() / n as f64;
        // 方差 σ_ω²（实部=H 方向）, σ_γ²（虚部=Γ 方向）
        let σω2 = 本征值.iter().map(|z| (z.0 - 中心实).powi(2)).sum::<f64>() / n as f64;
        let σγ2 = 本征值.iter().map(|z| (z.1 - 中心虚).powi(2)).sum::<f64>() / n as f64;
        let τ_ellipse = if σω2 + σγ2 > 1e-12 {
            (σω2 - σγ2) / (σω2 + σγ2)
        } else {
            0.0
        };
        let 半轴a = σω2.sqrt();
        let 半轴b = σγ2.sqrt();

        // D_s^(2D) = W_2(ρ_sys, ρ_ellipse)：对归一化到椭圆中心的点做
        // 二维 Wasserstein-2 的 MC 近似（椭圆均匀采样 vs 系统点集最优运输）
        let d_s_2d = self.wasserstein2d(本征值, 中心实, 中心虚, 半轴a, 半轴b, n);

        let 区域 = if d_s_2d < 0.3 {
            "绿".to_string()
        } else if d_s_2d < 0.6 {
            "黄".to_string()
        } else {
            "红".to_string()
        };

        // 径向偏离（τ 分布异常）：点相对椭圆边界的径向距离标准差
        let 径向偏离 = self.径向偏离(本征值, 中心实, 中心虚, 半轴a, 半轴b);
        // 角向偏离（振荡结构异常）：角度分布均匀性（Rayleigh 检验近似）
        let 角向偏离 = self.角向偏离(本征值, 中心实, 中心虚, 半轴a, 半轴b);

        // 极限检验：τ_ellipse→-1 应恢复 GOE（纯实谱），→1 应恢复 GUE（纯虚谱）
        let 极限检验 = if τ_ellipse < -0.8 {
            "τ_ellipse≈-1：纯实谱方向（GOE 极限）".to_string()
        } else if τ_ellipse > 0.8 {
            "τ_ellipse≈+1：纯虚谱方向（GUE 极限）".to_string()
        } else {
            "混合椭圆（H/Γ 均衡耗散）".to_string()
        };

        Ok(椭圆系综结果 {
            τ_ellipse,
            中心实部: 中心实,
            中心虚部: 中心虚,
            半轴a,
            半轴b,
            d_s_2d,
            区域,
            径向偏离,
            角向偏离,
            极限检验,
            本征值数: n,
        })
    }

    /// 二维 Wasserstein-2 距离（MC 采样近似）
    fn wasserstein2d(
        &self,
        本征值: &[(f64, f64)],
        中心实: f64, 中心虚: f64, 半轴a: f64, 半轴b: f64, n: usize,
    ) -> f64 {
        // 系统点（去中心，归一化到椭圆尺度）
        let 系统点: Vec<(f64, f64)> = 本征值
            .iter()
            .map(|z| ((z.0 - 中心实) / 半轴a.max(1e-9), (z.1 - 中心虚) / 半轴b.max(1e-9)))
            .collect();
        // 椭圆均匀采样点（MC）
        let mut rng = rand::thread_rng();
        let mut 椭圆点: Vec<(f64, f64)> = Vec::with_capacity(self.n_sample);
        for _ in 0..self.n_sample {
            let 角度 = rng.gen_range(0.0..2.0 * PI);
            let 半径 = rng.gen_range(0.0..1.0);
            let (x, y) = (半径 * 角度.cos(), 半径 * 角度.sin());
            椭圆点.push((x, y));
        }
        // 贪心配对近似 W_2（系统点 ↔ 最近椭圆点）
        let mut 总距离2 = 0.0;
        let mut 已用 = vec![false; self.n_sample];
        for &(sx, sy) in &系统点 {
            let mut 最近 = f64::INFINITY;
            for (i, &(ex, ey)) in 椭圆点.iter().enumerate() {
                if 已用[i] {
                    continue;
                }
                let d2 = (sx - ex).powi(2) + (sy - ey).powi(2);
                if d2 < 最近 {
                    最近 = d2;
                }
            }
            总距离2 += 最近;
        }
        // 近似 W_2 = sqrt(平均最近距离²)（贪心近似，非最优运输精确解，文档标注）
        (总距离2 / 系统点.len() as f64).sqrt().min(1.0)
    }

    /// 径向偏离：点相对椭圆边界径向距离的归一化标准差
    fn 径向偏离(&self, 本征值: &[(f64, f64)], 中心实: f64, 中心虚: f64, 半轴a: f64, 半轴b: f64) -> f64 {
        let n = 本征值.len() as f64;
        let 径向: Vec<f64> = 本征值
            .iter()
            .map(|z| {
                let dx = (z.0 - 中心实) / 半轴a.max(1e-9);
                let dy = (z.1 - 中心虚) / 半轴b.max(1e-9);
                (dx * dx + dy * dy).sqrt() // 椭圆归一化半径（均匀≈1）
            })
            .collect();
        let 均值 = 径向.iter().sum::<f64>() / n;
        let 方差 = 径向.iter().map(|r| (r - 均值).powi(2)).sum::<f64>() / n;
        (方差.sqrt() / 均值.max(1e-9)).min(1.0)
    }

    /// 角向偏离：角度分布均匀性（Rayleigh R 统计近似）
    fn 角向偏离(&self, 本征值: &[(f64, f64)], 中心实: f64, 中心虚: f64, 半轴a: f64, 半轴b: f64) -> f64 {
        let n = 本征值.len() as f64;
        let (mut 余弦和, mut 正弦和) = (0.0, 0.0);
        for z in 本征值 {
            let dx = (z.0 - 中心实) / 半轴a.max(1e-9);
            let dy = (z.1 - 中心虚) / 半轴b.max(1e-9);
            余弦和 += dx / (dx * dx + dy * dy).sqrt().max(1e-9);
            正弦和 += dy / (dx * dx + dy * dy).sqrt().max(1e-9);
        }
        let r统计 = (余弦和.powi(2) + 正弦和.powi(2)).sqrt() / n;
        r统计 // R∈[0,1]，接近1=角度聚集（角向偏离），接近0=均匀
    }

    // ============ 谱分离（Schrieffer-Wolff，P22 §6.3） ============

    /// 谱分离：按带宽 [τ_min, τ_max] 三段切分，快模积分自能，得有效算子
    ///
    /// L_eff = P_S L P_S + Σ(ω)，Σ(ω)=P_S L P_F (ω-P_F L P_F)^{-1} P_F L P_S
    /// 因果性：Γ_eff = -Im Σ(ω) > 0
    /// 谱隙：Δ = min(τ_S^{-1}) - max(τ_F^{-1})
    pub fn 谱分离(
        &self,
        l: &DMatrix<NComplex<f64>>,
        τ_min: f64,
        τ_max: f64,
        ω: f64,
    ) -> Result<谱分离结果, String> {
        let n = l.nrows();
        // 先用分解得 τ 链
        let 分解 = self.分解(l)?;
        let m = 分解.模数;
        // 快模索引（τ<τ_min）/中带/慢模（τ>τ_max）
        let 快模: Vec<usize> = 分解.模.iter().enumerate()
            .filter(|(_, 模)| 模.τ < τ_min).map(|(i, _)| i).collect();
        let 慢模: Vec<usize> = 分解.模.iter().enumerate()
            .filter(|(_, 模)| 模.τ > τ_max).map(|(i, _)| i).collect();
        let 中带模: Vec<usize> = 分解.模.iter().enumerate()
            .filter(|(_, 模)| 模.τ >= τ_min && 模.τ <= τ_max).map(|(i, _)| i).collect();
        // 注意：τ=INF（虚部≈0）算慢模（冻结背景）
        let 快模数 = 快模.len();
        let 慢模数 = 慢模.len() + 分解.模.iter().filter(|模| !模.τ.is_finite()).count();
        let 中带模数 = 中带模.len();

        // 投影子空间（模基下）
        // P_S L P_F (ω - P_F L P_F)^{-1} P_F L P_S
        // 在模基（右向量 V）下计算：L_基 = V^{-1} L V ≈ diag(λ)
        let v矩阵 = self.右向量矩阵(&分解, n);
        let v逆 = match v矩阵.clone().try_inverse() {
            Some(inv) => inv,
            None => DMatrix::<NComplex<f64>>::zeros(n, m),
        };
        let l基 = v逆.clone() * l * &v矩阵; // m×m 模基算子（近似对角）

        // 投影矩阵（模基下）
        let 投影S = |idx: &[usize]| -> DMatrix<NComplex<f64>> {
            let mut p = DMatrix::<NComplex<f64>>::zeros(m, m);
            for &i in idx {
                p[(i, i)] = NComplex::new(1.0, 0.0);
            }
            p
        };
        let ps = 投影S(&慢模);
        let pf = 投影S(&快模);

        // 自能 Σ = P_S L P_F (ω - P_F L P_F)^{-1} P_F L P_S（模基）
        let lp_s = &ps * &l基;
        let l_s_pf = &l基 * &pf;
        let pf_l_pf = &pf * &l基 * &pf;
        let 中括号 = &pf_l_pf - &DMatrix::<NComplex<f64>>::from_element(m, m, NComplex::new(ω, 0.0));
        // 伪逆（ω-P_F L P_F 可能奇异）
        let 中括号逆 = match 中括号.clone().try_inverse() {
            Some(inv) => inv,
            None => DMatrix::<NComplex<f64>>::zeros(m, m),
        };
        let 自能基 = &lp_s * &中括号逆 * &l_s_pf;

        // 转回原基：Σ_原 = V Σ_基 V^{-1}
        let 自能原 = &v矩阵 * &自能基 * &v逆;

        // L_eff 在原基 = V (P_S L P_S) V^{-1} + Σ_原
        let l_eff原 = &v矩阵 * &(&ps * &l基 * &ps) * &v逆 + &自能原;

        // Γ_eff = -Im Σ_原 对角
        let Γ_eff对角: Vec<f64> = (0..n).map(|i| -自能原[(i, i)].im).collect();
        let 因果性通过 = Γ_eff对角.iter().all(|g| *g > 0.0);

        // 谱隙 Δ = min(τ_F^{-1}) - max(τ_S^{-1})（快模最快速率 - 慢模最慢速率）
        // 物理：快模衰减远快于慢模 → Δ>0 = 可分离（快模已衰减殆尽时慢模未动）
        let 慢τ: Vec<f64> = 分解.模.iter().filter(|模| 模.τ > τ_max).map(|模| 模.τ).collect();
        let 快τ: Vec<f64> = 分解.模.iter().filter(|模| 模.τ < τ_min).map(|模| 模.τ).collect();
        let 谱隙 = if !慢τ.is_empty() && !快τ.is_empty() {
            let min_快逆 = 快τ.iter().map(|t| 1.0 / t.max(1e-9)).fold(f64::INFINITY, f64::min);
            let max_慢逆 = 慢τ.iter().map(|t| 1.0 / t.max(1e-9)).fold(0.0, f64::max);
            min_快逆 - max_慢逆
        } else {
            0.0
        };
        // 误差界 C·e^{-Δt/2}（C=‖L‖ 近似，Δ=谱隙；谱隙>0 才有物理意义）
        let c_范数 = l.norm();
        let 误差界 = c_范数 * (-谱隙.max(0.0) / 2.0).exp();

        let 判定 = if 因果性通过 {
            "快模积分成功，因果性通过".to_string()
        } else {
            "快模积分后 Γ_eff 非正（因果性破坏——截断不成立）".to_string()
        };

        // 转 Vec<Vec<f64>>（复 → 实部）
        let 转实 = |m: &DMatrix<NComplex<f64>>| -> Vec<Vec<f64>> {
            (0..m.nrows()).map(|i| (0..m.ncols()).map(|j| m[(i, j)].re).collect()).collect()
        };

        Ok(谱分离结果 {
            快模数,
            慢模数,
            中带模数,
            自能: 转实(&自能原),
            l_eff: 转实(&l_eff原),
            Γ_eff对角,
            谱隙,
            误差界,
            因果性通过,
            判定,
        })
    }

    /// 右向量矩阵 V（列=右向量）
    /// D1-R2（2026-08-23）: 虚部写死 0.0 丢非厄米耦合 → 改完整复数（含 v虚 越界兜底）。
    fn 右向量矩阵(&self, 分解: &非厄米谱分解, n: usize) -> DMatrix<NComplex<f64>> {
        let m = 分解.模数;
        let mut v = DMatrix::<NComplex<f64>>::zeros(n, m);
        for (j, 模) in 分解.模.iter().enumerate() {
            for i in 0..n {
                // D1-R2: 完整复数 v=实+i·虚（P22 §6.2 v 是复向量）
                v[(i, j)] = NComplex::new(模.v[i], 模.v虚.get(i).copied().unwrap_or(0.0));
            }
        }
        v
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

/// D1-R5（2026-08-23）: 统一取右向量工具函数——完整复数, 杜绝 new(real, 0.0) 式截断
pub fn 取右向量(模: &非厄米模, i: usize) -> NComplex<f64> {
    NComplex::new(
        模.v.get(i).copied().unwrap_or(0.0),
        模.v虚.get(i).copied().unwrap_or(0.0),
    )
}

/// D1-R5: 统一取左向量工具函数（w=V^-H 完整复, 展开系数需共轭）
pub fn 取左向量(模: &非厄米模, i: usize) -> NComplex<f64> {
    NComplex::new(
        模.w.get(i).copied().unwrap_or(0.0),
        模.w虚.get(i).copied().unwrap_or(0.0),
    )
}

/// D7（2026-08-23, 规范 §3 Step5/§6 V2）: 验证 Γ 正定（熵泵可计算的必要前提）
///
/// Γ = i(L†-L)/2（复 Hermitian）。Hermitian 正定 ⟺ Cholesky 分解成功——
/// 手动 Cholesky-Banachiewicz 主元判定（r≤30 小矩阵, 无 nalgebra feature 依赖, D7 §2）。
/// 非正定 → 数据链异常（噪声过大/解析信号失败/样本不足）, 熵泵应返回谱信息不足, 不静默。
pub fn 验证Γ正定(l: &DMatrix<NComplex<f64>>) -> bool {
    let n = l.nrows();
    if n == 0 {
        return false;
    }
    // Γ_ij = i·(conj(L_ji) - L_ij)/2（Hermitian 构造）
    let mut g = DMatrix::<NComplex<f64>>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let 虚单位 = NComplex::new(0.0, 1.0);
            g[(i, j)] = 虚单位 * (l[(j, i)].conjugate() - l[(i, j)]) / 2.0;
        }
    }
    // Cholesky-Banachiewicz: Γ = L·L†; 对角主元须实正（正定 ⟺ 分解成功）
    let mut lchol = DMatrix::<NComplex<f64>>::zeros(n, n);
    for i in 0..n {
        for j in 0..i {
            let mut 累 = g[(i, j)];
            for k in 0..j {
                累 -= lchol[(i, k)] * lchol[(j, k)].conjugate();
            }
            if lchol[(j, j)].re.abs() < 1e-12 {
                return false; // 主元近零 → 奇异/非正定
            }
            lchol[(i, j)] = 累 / lchol[(j, j)];
        }
        // Hermitian 对角为实
        let mut 累 = g[(i, i)].re;
        for k in 0..i {
            累 -= lchol[(i, k)].norm_sqr();
        }
        if 累 <= 1e-12 {
            return false; // 非正主元 → 非正定
        }
        lchol[(i, i)] = NComplex::new(累.sqrt(), 0.0);
    }
    true
}

/// H = (L+L†)/2 对称部分
pub fn 对称部分(l: &DMatrix<NComplex<f64>>) -> Vec<Vec<f64>> {
    let n = l.nrows();
    let mut h = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            h[i][j] = ((l[(i, j)] + l[(j, i)].conjugate()) / 2.0).re;
        }
    }
    h
}

/// Γ = i(L†-L)/2 耗散部分
pub fn 耗散部分(l: &DMatrix<NComplex<f64>>) -> Vec<Vec<f64>> {
    let n = l.nrows();
    let mut g = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            g[i][j] = ((NComplex::new(0.0, 1.0) * (l[(j, i)].conjugate() - l[(i, j)])) / 2.0).re;
        }
    }
    g
}

/// 实矩阵 → 复矩阵
pub fn 实转复(m: &DMatrix<f64>) -> DMatrix<NComplex<f64>> {
    let (n, k) = (m.nrows(), m.ncols());
    DMatrix::from_fn(n, k, |i, j| NComplex::new(m[(i, j)], 0.0))
}

/// 复本征值对列表 → D_s^(2D) 便捷入口
pub fn 复谱椭圆拟合(本征值: &[(f64, f64)]) -> Result<椭圆系综结果, String> {
    非厄米谱求解器::default().椭圆系综(本征值)
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 已知 2×2 非厄米矩阵的精确复本征值
    fn 测试矩阵() -> DMatrix<NComplex<f64>> {
        // L = [[1+i, 0.2], [-0.1, 2-i]] → 本征值近似 (1+i, 2-i)（对角占优）
        DMatrix::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => NComplex::new(1.0, 1.0),
            (0, 1) => NComplex::new(0.2, 0.0),
            (1, 0) => NComplex::new(-0.1, 0.0),
            (1, 1) => NComplex::new(2.0, -1.0),
            _ => NComplex::new(0.0, 0.0),
        })
    }

    #[test]
    fn 测试_复特征值_对角矩阵() {
        let 求解 = 非厄米谱求解器::default();
        // 对角复矩阵 → 精确本征值
        let l = DMatrix::from_fn(3, 3, |i, j| {
            if i == j {
                NComplex::new(i as f64 + 1.0, -(i as f64 + 1.0) * 0.1)
            } else {
                NComplex::new(0.0, 0.0)
            }
        });
        let λs = 求解.复特征值(&l).unwrap();
        assert_eq!(λs.len(), 3);
        let mut 实部: Vec<f64> = λs.iter().map(|λ| λ.re).collect();
        实部.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());
        assert!((实部[0] - 1.0).abs() < 1e-6);
        assert!((实部[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn 测试_非厄米分解_双正交() {
        let 求解 = 非厄米谱求解器::default();
        let l = 测试矩阵();
        let 分解 = 求解.分解(&l).unwrap();
        assert!(分解.模数 >= 2);
        // τ 从大到小排序
        for w in 分解.模.windows(2) {
            assert!(w[0].τ >= w[1].τ, "τ 应降序: {} < {}", w[0].τ, w[1].τ);
        }
        // 稳定模：γ>0（统一令后口径）→ τ>0；且 λ_im=−γ 自洽
        let 稳定 = 分解.模.iter().filter(|模| 模.γ > 0.0).count();
        for 模 in &分解.模 {
            assert!((模.γ + 模.λ_im).abs() < 1e-12, "γ=−λ_im 应自洽: γ={} λ_im={}", 模.γ, 模.λ_im);
            assert!((模.τ * 模.γ - 1.0).abs() < 1e-9, "τ=1/γ 应自洽: τ={} γ={}", 模.τ, 模.γ);
        }
        assert!(稳定 >= 1, "应至少1个稳定模: {:?}", 分解.模.iter().map(|m| m.τ).collect::<Vec<_>>());
        // p_obs 和为 1
        let p总和: f64 = 分解.模.iter().map(|m| m.p_obs).sum();
        assert!((p总和 - 1.0).abs() < 1e-6, "p_obs 和应=1: {}", p总和);
        // 判定
        assert!(分解.判定.contains("稳定") || 分解.判定.contains("失稳"));
    }

    #[test]
    fn 测试_参与度_激发占比() {
        let 求解 = 非厄米谱求解器::default();
        let l = 测试矩阵();
        let 分解 = 求解.分解(&l).unwrap();
        let p_exc = 求解.激发占比(&分解, &[1.0, 0.0]);
        assert_eq!(p_exc.len(), 分解.模数);
        let 总和: f64 = p_exc.iter().sum();
        assert!((总和 - 1.0).abs() < 1e-6, "p_exc 和应=1: {}", 总和);
    }

    #[test]
    fn 测试_椭圆系综_Ginibre恢复() {
        let 求解 = 非厄米谱求解器::default();
        // 圆盘内均匀采样点（Ginibre 极限）→ 椭圆参数≈0（圆对称）
        let mut rng = rand::thread_rng();
        let 点: Vec<(f64, f64)> = (0..300)
            .map(|_| {
                let 半径 = rng.gen_range(0.0..1.0);
                let 角度 = rng.gen_range(0.0..2.0 * PI);
                (半径 * 角度.cos(), 半径 * 角度.sin())
            })
            .collect();
        let 结果 = 求解.椭圆系综(&点).unwrap();
        assert!(结果.τ_ellipse.abs() < 0.3, "圆对称 τ_ellipse 应≈0: {}", 结果.τ_ellipse);
        assert!((结果.半轴a - 结果.半轴b).abs() < 0.2, "圆对称半轴应接近: {} vs {}", 结果.半轴a, 结果.半轴b);
        assert_eq!(结果.本征值数, 300);
        // 区域判定存在
        assert!(["绿", "黄", "红"].contains(&结果.区域.as_str()));
        // 极限检验
        assert!(结果.极限检验.contains("混合"));
    }

    #[test]
    fn 测试_椭圆系综_各向异性() {
        let 求解 = 非厄米谱求解器::default();
        // 强各向异性：实部展宽远大于虚部 → τ_ellipse→+1（GUE 方向）
        let 点: Vec<(f64, f64)> = (0..300)
            .map(|i| {
                let 角度 = i as f64 * 2.0 * PI / 300.0;
                (3.0 * 角度.cos(), 0.3 * 角度.sin())
            })
            .collect();
        let 结果 = 求解.椭圆系综(&点).unwrap();
        assert!(结果.τ_ellipse > 0.6, "各向异性 τ_ellipse 应接近+1: {}", 结果.τ_ellipse);
    }

    #[test]
    fn 测试_谱分离_块对角() {
        let 求解 = 非厄米谱求解器::default();
        // 块对角：快块（τ 小）+ 慢块（τ 大）
        // 快块：大耗散（γ 大负 → τ 小）
        let n = 4;
        let mut l = DMatrix::<NComplex<f64>>::zeros(n, n);
        // 快块（模1,2）：γ=-2, -3 → τ≈0.5, 0.33
        l[(0, 0)] = NComplex::new(0.0, -2.0);
        l[(1, 1)] = NComplex::new(0.0, -3.0);
        // 慢块（模3,4）：γ=-0.02, -0.01 → τ≈50, 100
        l[(2, 2)] = NComplex::new(0.0, -0.02);
        l[(3, 3)] = NComplex::new(0.0, -0.01);
        let 分解 = 求解.分解(&l).unwrap();
        // τ_min=5, τ_max=20 → 快模2个（τ<5），慢模2个（τ>20）
        let 结果 = 求解.谱分离(&l, 5.0, 20.0, 1.0).unwrap();
        assert_eq!(结果.快模数, 2);
        assert_eq!(结果.慢模数, 2);
        assert_eq!(结果.中带模数, 0);
        // 谱隙>0
        assert!(结果.谱隙 > 0.0, "谱隙应>0: {}", 结果.谱隙);
        // 因果性（自能修正后 Γ_eff）
        assert!(结果.l_eff.len() == n);
        assert_eq!(结果.Γ_eff对角.len(), n);
    }

    #[test]
    fn 测试_H_Γ分解() {
        let l = 测试矩阵();
        let h = 对称部分(&l);
        let g = 耗散部分(&l);
        // H 对称
        assert!((h[0][1] - h[1][0]).abs() < 1e-12);
        // Γ 对称
        assert!((g[0][1] - g[1][0]).abs() < 1e-12);
        // H[0][0]=Re(1+i)=1
        assert!((h[0][0] - 1.0).abs() < 1e-12);
    }

    // ==================== 第289批 Phase 1: 双正交虚部修复验证 + 端到端合成 ====================

    /// 强非厄米矩阵（右向量含显著虚部）——验证 v 虚部/ w 虚部保存 + w†v=δ 双正交成立
    fn 强非厄米矩阵() -> DMatrix<NComplex<f64>> {
        // L = [[-0.3+2i, 0.4+0.1i], [0.1, -1.5-i]] → 非对角非对称+对角复
        DMatrix::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => NComplex::new(-0.3, 2.0),
            (0, 1) => NComplex::new(0.4, 0.1),
            (1, 0) => NComplex::new(0.1, 0.0),
            (1, 1) => NComplex::new(-1.5, -1.0),
            _ => NComplex::new(0.0, 0.0),
        })
    }

    /// 第289批修复1: 右向量虚部不再丢弃 + 双正交 w†v=δ（复内积）
    #[test]
    fn 测试_双正交_虚部保存与双正交() {
        let 求解 = 非厄米谱求解器::default();
        let l = 强非厄米矩阵();
        let 分解 = 求解.分解(&l).unwrap();
        // 虚部字段已填充（强非厄米必有非零虚部）
        assert!(分解.模.iter().any(|m| m.v虚.iter().any(|x| x.abs() > 1e-6)),
            "强非厄米右向量应含显著虚部: {:?}", 分解.模[0].v虚);
        assert!(分解.模.iter().any(|m| m.w虚.iter().any(|x| x.abs() > 1e-6)),
            "左向量应含显著虚部: {:?}", 分解.模[0].w虚);
        // 双正交 w†v=δ: 用完整复向量构造复内积
        for i in 0..分解.模数 {
            for j in 0..分解.模数 {
                let 模i = &分解.模[i];
                let 模j = &分解.模[j];
                let mut 实部 = 0.0f64;
                let mut 虚部 = 0.0f64;
                for k in 0..模i.w.len() {
                    let w实 = 模i.w[k];
                    let w虚 = 模i.w虚[k];
                    let v实 = 模j.v[k];
                    let v虚 = 模j.v虚[k];
                    // conj(w_i)·v_j = (w实 - i·w虚)(v实 + i·v虚)
                    实部 += w实 * v实 + w虚 * v虚;
                    虚部 += w实 * v虚 - w虚 * v实;
                }
                let 期望 = if i == j { 1.0 } else { 0.0 };
                assert!((实部 - 期望).abs() < 1e-6,
                    "Re(w{}†v{})={} 应≈{}", i, j, 实部, 期望);
                assert!(虚部.abs() < 1e-6, "Im(w{}†v{})={} 应≈0", i, j, 虚部);
            }
        }
    }

    /// 第289批修复3: 展开系数 c_n = w†Δx 用完整复 w（共轭）
    #[test]
    fn 测试_展开系数_复数共轭() {
        let 求解 = 非厄米谱求解器::default();
        let l = 强非厄米矩阵();
        let 分解 = 求解.分解(&l).unwrap();
        let 初态 = [1.0, 0.0];
        let c2 = 求解.展开系数(&分解, &初态);
        assert_eq!(c2.len(), 分解.模数);
        // |c|² 非负
        assert!(c2.iter().all(|x| *x >= 0.0), "|c|² 应非负: {:?}", c2);
        // 和>0（初态非零）
        let 总和: f64 = c2.iter().sum();
        assert!(总和 > 1e-12, "展开能量应非零: {}", 总和);
        // 激发占比和=1
        let p_exc = 求解.激发占比(&分解, &初态);
        assert!((p_exc.iter().sum::<f64>() - 1.0).abs() < 1e-6, "p_exc 和应=1");
    }

    /// 第289批端到端: 已知复 L → 分解 → {ω,τ} 误差<5%（单模对角）
    #[test]
    fn 测试_端到端_合成L单模对角() {
        let 求解 = 非厄米谱求解器::default();
        // 单模（对角）: ω=1.0, γ=-2.0 → τ=0.5
        let l = DMatrix::from_fn(3, 3, |i, j| {
            if i == j { NComplex::new(1.0 + i as f64 * 0.1, -2.0) } else { NComplex::new(0.0, 0.0) }
        });
        let 分解 = 求解.分解(&l).unwrap();
        // 对角矩阵 → 本征值=对角元精确
        let mut 匹配 = 0;
        for 模 in &分解.模 {
            if (模.ω - 1.0).abs() < 0.05 && (模.τ - 0.5).abs() / 0.5 < 0.05 { 匹配 += 1; }
            if (模.ω - 1.1).abs() < 0.05 && (模.τ - 0.5).abs() / 0.5 < 0.05 { 匹配 += 1; }
            if (模.ω - 1.2).abs() < 0.05 && (模.τ - 0.5).abs() / 0.5 < 0.05 { 匹配 += 1; }
        }
        assert!(匹配 >= 2, "对角L本征值应≈对角元(误差<5%): {:?}", 分解.模.iter().map(|m| (m.ω, m.τ)).collect::<Vec<_>>());
    }

    /// 第289批端到端: 双模耦合矩阵 → 分解本征值误差<5%
    #[test]
    fn 测试_端到端_双模耦合() {
        let 求解 = 非厄米谱求解器::default();
        // 对角占优双模耦合: τ=2(γ=-0.5), τ=10(γ=-0.1), 弱耦合 0.05
        // （耦合相对对角越小本征值越接近对角; 0.05 使误差<5%）
        let n = 2;
        let mut l = DMatrix::<NComplex<f64>>::zeros(n, n);
        l[(0, 0)] = NComplex::new(0.0, -0.5);
        l[(1, 1)] = NComplex::new(0.0, -0.1);
        l[(0, 1)] = NComplex::new(0.05, 0.02);
        l[(1, 0)] = NComplex::new(0.02, -0.05);
        let 分解 = 求解.分解(&l).unwrap();
        // 弱耦合 → 本征值接近对角（γ≈-0.5/-0.1）
        let mut τs: Vec<f64> = 分解.模.iter().map(|m| m.τ).collect();
        τs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((τs[0] - 2.0).abs() / 2.0 < 0.05, "τ1≈2: {}", τs[0]);
        assert!((τs[1] - 10.0).abs() / 10.0 < 0.05, "τ2≈10: {}", τs[1]);
    }

    // ==================== D1（2026-08-23）: 非厄米谱链路修复 R2/R5 验证 ====================

    /// D1-R2: 右向量矩阵完整复数——强非厄米快块谱分离路径虚部保留
    #[test]
    fn 测试_D1R2_谱分离完整复数() {
        let 求解 = 非厄米谱求解器::default();
        let n = 4;
        let mut l = DMatrix::<NComplex<f64>>::zeros(n, n);
        // 快块 2×2 非厄米（对角复虚部 → v 虚部显著; 弱耦合保证 SW 因果性成立）
        l[(0, 0)] = NComplex::new(0.0, -2.0);
        l[(0, 1)] = NComplex::new(0.05, 0.02);
        l[(1, 0)] = NComplex::new(0.02, -0.01);
        l[(1, 1)] = NComplex::new(0.0, -3.0);
        // 慢块对角
        l[(2, 2)] = NComplex::new(0.0, -0.02);
        l[(3, 3)] = NComplex::new(0.0, -0.01);
        let 分解 = 求解.分解(&l).unwrap();
        // 快块模 v 虚部显著（非厄米耦合在谱分离前已进入 v 虚部）
        assert!(
            分解.模.iter().filter(|m| m.τ < 1.0).any(|m| m.v虚.iter().any(|x| x.abs() > 1e-4)),
            "非厄米快块 v 应含虚部: {:?}",
            分解.模.iter().filter(|m| m.τ < 1.0).map(|m| &m.v虚).collect::<Vec<_>>()
        );
        // 谱分离完整复数路径正常（右向量矩阵不再丢虚部, 无回归）
        let 结果 = 求解.谱分离(&l, 5.0, 20.0, 1.0).unwrap();
        assert_eq!(结果.快模数, 2);
        assert_eq!(结果.慢模数, 2);
        assert_eq!(结果.Γ_eff对角.len(), n);
        // 注: 因果性（Γ_eff>0）属 D1-R4 范畴, 不在本测试断言范围（与现有块对角测试一致）
    }

    /// D1-R5: 统一取向量工具函数完整复数 + 越界兜底
    #[test]
    fn 测试_D1R5_取向量工具函数() {
        let 求解 = 非厄米谱求解器::default();
        let 分解 = 求解.分解(&强非厄米矩阵()).unwrap();
        for 模 in &分解.模 {
            for i in 0..模.v.len() {
                let rv = 取右向量(模, i);
                assert!((rv.re - 模.v[i]).abs() < 1e-12);
                assert!((rv.im - 模.v虚[i]).abs() < 1e-12);
                let lv = 取左向量(模, i);
                assert!((lv.re - 模.w[i]).abs() < 1e-12);
                assert!((lv.im - 模.w虚[i]).abs() < 1e-12);
            }
            // 越界兜底返回 0
            let rv = 取右向量(模, 模.v.len() + 5);
            assert!(rv.re == 0.0 && rv.im == 0.0);
        }
    }

    // ==================== D7（2026-08-23）: 复L 数据链（Γ 正定验证） ====================

    /// D7-V2: 验证Γ正定——Γ 正定 L → true; 含负耗散 → false
    #[test]
    fn 测试_D7V2_验证Γ正定() {
        let n = 2;
        // L = i·Γ（Γ 对角正: 1,2）→ Γ 恢复正定 → Cholesky 成功
        let mut l = DMatrix::<NComplex<f64>>::zeros(n, n);
        l[(0, 0)] = NComplex::new(0.0, 1.0);
        l[(1, 1)] = NComplex::new(0.0, 2.0);
        assert!(验证Γ正定(&l), "Γ=diag(1,2) 应正定");
        // L = i·Γ（Γ 含负: 1,-2）→ Γ 非正定 → false
        let mut l2 = DMatrix::<NComplex<f64>>::zeros(n, n);
        l2[(0, 0)] = NComplex::new(0.0, 1.0);
        l2[(1, 1)] = NComplex::new(0.0, -2.0);
        assert!(!验证Γ正定(&l2), "Γ 含负耗散应非正定");
        // 零矩阵 → false
        assert!(!验证Γ正定(&DMatrix::<NComplex<f64>>::zeros(n, n)), "零Γ非正定");
    }
}
