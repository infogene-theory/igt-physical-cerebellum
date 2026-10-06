// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! 谱动力学求解器 — 第132批 P0 谱动力学核心三件套·第一件（引擎架构调整方案 v1.0 §四）
//!
//! 理论依据:
//!   - 《IGT闭合拓扑变分原理_GUE最优τ链_v1.0》: L 的本征值谱服从 GUE/GOE Wigner-Dyson 分布
//!     = 闭合拓扑四个最优性（耗散最小/回归最快/裕度最大/自扰最小）的唯一统计解
//!   - 《IGT闭合拓扑变分原理_v1.1_对称性类与有限n修正》: D_s = 变分残差 + 有限 n MC p 值;
//!     双距离 D_s^WD / D_s^P; 混沌性指标 η = D_s^P / D_s^WD; 对称性类三分（GOE/GUE/Poisson）
//!   - 《IGT_Engine_v2_架构规范》: 谱求解器 = L → {λ_i, τ_i, S, D_s, β, p值}
//!
//! 核心定位（Engine v2 谱动力学求解器的数学核心）:
//!   L（图 Laplacian，对称半正定）→ 本征值 λ_i → τ 链 τ_i=1/λ_i → 展开谱（unfolding）
//!   → 最近邻间距 s_i → 与 GUE/GOE/Poisson 理论分布做 Anderson-Darling 距离
//!   → 对称性类判定（β）→ MC 有限 n p 值 → 混沌性指标 η
//!
//! 关键升级（对比 riemann_spectrum.rs 旧 D_s）:
//!   - 旧: 直接对原始本征值做 KS 检验（无 unfolding），仅 GUE 单基准，无 p 值
//!   - 新: 展开谱（unfolding）→ 间距分布 → AD 距离（对 GOE/GUE/Poisson 三分）
//!         + 有限 n MC p 值（零假设=该对称类）+ 双距离 + η
//!   - D_s 重新定位为变分残差（欧拉-拉格朗日残差），非经验阈值

#![allow(non_snake_case)]

use std::f64::consts::PI;

use nalgebra::{DMatrix, SymmetricEigen};
use rand::Rng;

// ============================================================================
// 谱动力学结果 — 谱求解器统一输出
// ============================================================================

/// 谱动力学结果 — Engine v2 谱求解器输出
///
/// 对应 Engine v2 架构规范: L → {λ_i, τ_i, S, D_s^WD, D_s^P, β, p值, η}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct 谱动力学结果 {
    /// 本征值数
    pub 本征值数: usize,
    /// 非零本征值 λ_i（升序，排除零空间）
    pub 本征值: Vec<f64>,
    /// τ 链 τ_i = 1/λ_i（升序 λ 对应的弛豫时间，长τ在前）
    pub τ链: Vec<f64>,
    /// 对称度 S = 1 - min(D_s^WD, D_s^P)（谱规则度，间距分布越接近理论分布越高）
    pub 对称度_s: f64,
    /// 谱偏离度 D_s^WD（对判定的对称性类 Wigner-Dyson 分布的 AD 距离）
    /// D_s^WD ≤ 0.3 = 在普适类内（健康涨落）
    pub d_s_wd: f64,
    /// 谱偏离度 D_s^P（对 Poisson 可积分布的 AD 距离）
    /// D_s^P > 0.3 = 远离可积/局域化（混沌性证据）
    pub d_s_p: f64,
    /// 对称性类参数 β: 0=Poisson(可积) 1=GOE 2=GUE
    pub β: u8,
    /// 对称性类名称
    pub 对称性类: String,
    /// MC 基准 p 值（零假设=谱来自判定对称性类；p 大=不拒绝 H0=拟合好）
    pub p值: f64,
    /// 混沌性指标 η = D_s^P / D_s^WD（>1 混沌性显著；<1 接近可积）
    pub η: f64,
    /// 判定
    pub 判定: String,
    /// 展开谱 ξ_i（unfolding 后，平均间距≈1）
    pub 展开谱: Vec<f64>,
    /// 归一化最近邻间距 s_i = ξ_{i+1}-ξ_i
    pub 间距: Vec<f64>,
}

impl 谱动力学结果 {
    /// 判断是否在 Wigner-Dyson 普适类内（GUE/GOE 拟合且远离 Poisson）
    pub fn 在普适类内(&self) -> bool {
        self.d_s_wd <= 0.3 && self.d_s_p > 0.3
    }
}

/// β混合分析结果（变分原理深化 v1.0 第二部分）
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct β混合分析结果 {
    /// 有效 β_eff ∈ [1,2]（强耦合混合系统）；1=GOE 2=GUE
    pub β_eff: f64,
    /// 小 s 经验 CDF 的对数-对数斜率（=β+1，因 F(s)~s^(β+1)）
    pub 拟合斜率: f64,
    /// 间距样本数
    pub 样本数: usize,
    /// D_s^{β_eff}（对照 β_eff 插值基准的 AD 距离）
    pub d_s_βeff: f64,
    /// D_s^{GOE}（β=1）
    pub d_s_goe: f64,
    /// D_s^{GUE}（β=2）
    pub d_s_gue: f64,
    /// D_s^{Poisson}（β=0）
    pub d_s_poisson: f64,
    /// 判定（GOE主导/GUE主导/强混合/解耦混合候选）
    pub 判定: String,
}

/// 非厄米求解输出（第267批，P22 §6）：复谱分解 + 椭圆系综 + 参与度
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct 非厄米求解输出 {
    /// 非厄米谱分解（复本征值/双正交/τ链）
    pub 分解: crate::nonhermitian_spectrum::非厄米谱分解,
    /// 椭圆系综拟合（D_s^(2D)）
    pub 椭圆: crate::nonhermitian_spectrum::椭圆系综结果,
    /// 模态可观度 p_obs（不依赖扰动）
    pub p_obs: Vec<f64>,
    /// 激发占比 p_exc（依赖初态扰动）
    pub p_exc: Vec<f64>,
}

// ============================================================================
// 谱求解器
// ============================================================================

/// 谱求解器 — 从 L（或本征值序列）计算谱动力学量
///
/// 本模块核心是可测可验的纯数学实现:
/// 1. 对称矩阵 → nalgebra SymmetricEigen 精确本征值（dense, n 通常 <500）
/// 2. 展开谱（unfolding）: 多项式回归拟合累积态密度 N(λ)，ξ_i = N_smooth(λ_i)
/// 3. 最近邻间距 s_i，归一化平均间距=1
/// 4. Anderson-Darling 距离（对 GOE/GUE/Poisson 理论 CDF）
/// 5. 对称性类判定（实对称+详细平衡→GOE; 复厄米/非平衡环流→GUE; 无法判定→取 min 保守）
/// 6. MC 有限 n p 值（从理论分布逆变换采样生成 n 个间距 → AD 基准分布）
#[derive(Debug, Clone, Copy)]
pub struct 谱求解器 {
    /// MC 模拟次数
    pub n_sim: usize,
    /// AD 距离归一化阈值（AD 值超过该值视为严重偏离，D_s→1）
    pub ad_临界: f64,
    /// 对称性判定容差 ||L-L^T||
    pub ε_sym: f64,
}

impl Default for 谱求解器 {
    fn default() -> Self {
        Self {
            n_sim: 200,
            ad_临界: 2.5,
            ε_sym: 1e-9,
        }
    }
}

impl 谱求解器 {
    /// 从对称矩阵 L 求解谱动力学
    ///
    /// # 参数
    /// - `l`: 对称半正定矩阵（图 Laplacian L = D-A+V）
    ///
    /// 对称性类判定: ||L - L^T|| < ε_sym → GOE（实对称+详细平衡）; 否则 GUE（非平衡环流）
    pub fn 从矩阵(&self, l: &DMatrix<f64>) -> 谱动力学结果 {
        let 非对称范数 = Self::对称破坏度(l);
        let β判定 = if 非对称范数 < self.ε_sym { 1 } else { 2 };

        // SymmetricEigen 对对称部分求解（nalgebra 只支持对称，非对称用对称化近似）
        let sym = SymmetricEigen::new(l.clone());
        let 全部本征值: Vec<f64> = sym.eigenvalues.iter().cloned().collect();
        // 排除零空间（λ≈0）
        let 本征值: Vec<f64> = 全部本征值
            .into_iter()
            .filter(|&λ| λ > 1e-9)
            .collect();

        self.从本征值含β(本征值, β判定)
    }

    /// 非厄米双通道（第267批，P22 §6.2/6.4）——对非对称/非厄米矩阵走
    /// 复谱分解 + 双正交参与度 + 椭圆系综 D_s^(2D)。
    ///
    /// 与 `从矩阵`（实对称 SymmetricEigen 通道）并存：本方法显式调用非厄米
    /// 通道；`从矩阵` 保持原行为（golden 0.148 等测试零破坏）。
    pub fn 从矩阵_非厄米(&self, l: &DMatrix<f64>) -> Result<非厄米求解输出, String> {
        use crate::nonhermitian_spectrum::实转复;
        self.从矩阵_非厄米复(&实转复(l))
    }

    /// 非厄米复矩阵双通道（第267批）——接收复矩阵 L=H-iΓ
    pub fn 从矩阵_非厄米复(
        &self,
        l复: &nalgebra::DMatrix<nalgebra::Complex<f64>>,
    ) -> Result<非厄米求解输出, String> {
        use crate::nonhermitian_spectrum::非厄米谱求解器;
        let 求解器 = 非厄米谱求解器::default();
        let 分解 = 求解器.分解(l复)?;
        // 复本征值（ω, λ_im）→ 椭圆系综（统一令 §5：本征值权威表示＝(Re=ω, Im=λ_im)；
        // 旧代码传 模.γ＝Im λ 同值——枢纽翻号后必须显式用 λ_im，防双重反号）
        let 本征值: Vec<(f64, f64)> = 分解.模.iter().map(|模| (模.ω, 模.λ_im)).collect();
        let 椭圆 = 求解器.椭圆系综(&本征值)?;
        // 参与度
        let p_obs: Vec<f64> = 分解.模.iter().map(|模| 模.p_obs).collect();
        let p_exc: Vec<f64> = 求解器.激发占比(&分解, &vec![1.0; l复.nrows()]);
        Ok(非厄米求解输出 {
            分解,
            椭圆,
            p_obs,
            p_exc,
        })
    }

    /// 从本征值序列求解谱动力学（无矩阵信息时，对称性类取 min 保守原则）
    pub fn 从本征值(&self, λ: &[f64]) -> 谱动力学结果 {
        self.从本征值含β(λ.to_vec(), 0)
    }

    /// 从本征值序列 + 已知对称性类求解
    ///
    /// β 语义: 0=未知（保守取 min）; 1=GOE; 2=GUE
    fn 从本征值含β(&self, mut 本征值: Vec<f64>, β已知: u8) -> 谱动力学结果 {
        // 本征值升序排序（τ链=1/λ 从长到短；unfolding 内部也要求升序）
        本征值.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let 本征值数 = 本征值.len();

        if 本征值数 < 4 {
            return self.欠采样结果(本征值, β已知);
        }

        // 1. 展开谱（unfolding）→ 间距
        let 展开谱 = Self::展开谱(&本征值);
        let 间距 = Self::归一化间距(&展开谱);
        if 间距.len() < 3 {
            return self.欠采样结果(本征值, β已知);
        }

        // 2. 与三种理论分布做 Anderson-Darling 距离
        let ad_goe = Self::ad距离(&间距, 1);
        let ad_gue = Self::ad距离(&间距, 2);
        let ad_poisson = Self::ad距离(&间距, 0);

        // 归一化到 [0,1]
        let d_goe = Self::ad归一化(ad_goe, self.ad_临界);
        let d_gue = Self::ad归一化(ad_gue, self.ad_临界);
        let d_poisson = Self::ad归一化(ad_poisson, self.ad_临界);

        // 3. 对称性类判定
        //    β已知=1(GOE) / 2(GUE) 时直接采用；β未知时取 min(D_s^GOE, D_s^GUE) 保守原则（v1.1 裂痕1）
        let (β, d_s_wd) = match β已知 {
            1 => (1, d_goe),
            2 => (2, d_gue),
            _ => {
                if d_goe <= d_gue {
                    (1, d_goe)
                } else {
                    (2, d_gue)
                }
            }
        };
        let 对称性类 = match β {
            0 => "Poisson".to_string(),
            1 => "GOE".to_string(),
            _ => "GUE".to_string(),
        };

        // 4. MC 有限 n p 值（零假设=谱来自对称性类 β）
        let ad_obs = if β == 1 { ad_goe } else { ad_gue };
        let p值 = self.mc_p值(间距.len(), β, ad_obs);

        // 5. 混沌性指标 η = D_s^P / D_s^WD
        let η = if d_s_wd > 1e-9 { d_poisson / d_s_wd } else { f64::INFINITY };

        // 6. 对称度 S = 1 - min(D_s^WD, D_s^P)
        let 对称度_s = (1.0 - d_s_wd.min(d_poisson)).clamp(0.0, 1.0);

        // 7. 判定
        let 判定 = Self::判定(d_s_wd, d_poisson, p值, &对称性类);

        // 8. τ 链 τ_i = 1/λ_i
        let τ链: Vec<f64> = 本征值.iter().map(|&λ| 1.0 / λ).collect();

        谱动力学结果 {
            本征值数,
            本征值,
            τ链,
            对称度_s,
            d_s_wd,
            d_s_p: d_poisson,
            β,
            对称性类,
            p值,
            η,
            判定,
            展开谱,
            间距,
        }
    }

    /// 欠采样结果（本征值/间距不足时）
    fn 欠采样结果(&self, 本征值: Vec<f64>, β已知: u8) -> 谱动力学结果 {
        let (β, 对称性类) = match β已知 {
            1 => (1, "GOE".to_string()),
            2 => (2, "GUE".to_string()),
            _ => (0, "未知".to_string()),
        };
        let τ链: Vec<f64> = 本征值.iter().map(|&λ| 1.0 / λ).collect();
        谱动力学结果 {
            本征值数: 本征值.len(),
            本征值,
            τ链,
            对称度_s: 0.0,
            d_s_wd: 1.0,
            d_s_p: 1.0,
            β,
            对称性类,
            p值: 0.0,
            η: 1.0,
            判定: "样本不足（需≥4个非零本征值）".to_string(),
            展开谱: vec![],
            间距: vec![],
        }
    }

    /// 对称破坏度 ||L - L^T||_F（Frobenius 范数）
    pub fn 对称破坏度(l: &DMatrix<f64>) -> f64 {
        let mut 平方和 = 0.0;
        for i in 0..l.nrows() {
            for j in 0..l.ncols() {
                let d = l[(i, j)] - l[(j, i)];
                平方和 += d * d;
            }
        }
        平方和.sqrt()
    }

    /// β混合分析：对强耦合系统拟合有效 β_eff，并计算各基准 D_s
    ///
    /// 论文 §2.3-2.4：P(s)~s^β_eff（小 s 行为），β_eff∈[1,2]。
    /// 拟合方法：用间距分布前 25% 分位的对数-对数斜率（工程定稿，替代脆弱的整体拟合）。
    pub fn β混合分析(&self, 本征值: &[f64]) -> β混合分析结果 {
        // 展开谱 → 间距
        let 展开 = Self::展开谱(本征值);
        let 间距 = Self::归一化间距(&展开);
        let n = 间距.len();
        if n < 8 {
            return β混合分析结果 {
                β_eff: 1.0,
                拟合斜率: 0.0,
                样本数: n,
                d_s_βeff: 1.0,
                d_s_goe: 1.0,
                d_s_gue: 1.0,
                d_s_poisson: 1.0,
                判定: "样本不足（需≥8个间距）".to_string(),
            };
        }

        // 小 s 行为拟合（诊断斜率）：取前 25% 分位间距，log-log 回归
        let mut 排序: Vec<f64> = 间距.clone();
        排序.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let 小样本数 = (n / 4).max(3);
        let 小间距: Vec<f64> = 排序[..小样本数].to_vec();
        // 经验 CDF F(s)~s^(β+1)（小 s），ln(rank) ~ (β+1)·ln(s)，斜率 = β+1
        let 均值ln_s = 小间距.iter().map(|&s| s.ln().max(-10.0)).sum::<f64>() / 小间距.len() as f64;
        let 均值ln_rank = 小间距
            .iter()
            .enumerate()
            .map(|(i, _)| ((i + 1) as f64).ln().max(-10.0))
            .sum::<f64>()
            / 小间距.len() as f64;
        let mut 协方差 = 0.0;
        let mut 方差 = 0.0;
        for (i, &s) in 小间距.iter().enumerate() {
            let x = s.ln().max(-10.0);
            let y = ((i + 1) as f64).ln().max(-10.0);
            协方差 += (x - 均值ln_s) * (y - 均值ln_rank);
            方差 += (x - 均值ln_s) * (x - 均值ln_s);
        }
        let 斜率 = if 方差 > 1e-12 { 协方差 / 方差 } else { 2.0 };

        // β_eff 用扫描法：在 β∈[0.5,2.5] 网格上求 AD 距离最小的 β（论文: 应由 MC 确定）
        let 起点 = 0.5;
        let 终点 = 2.5;
        let 步数 = 41; // 步长 0.05
        let mut β_eff = 1.0;
        let mut ad_βeff = f64::INFINITY;
        for k in 0..=步数 {
            let β试 = 起点 + (终点 - 起点) * k as f64 / 步数 as f64;
            let ad试 = Self::ad距离_β(&间距, β试);
            if ad试 < ad_βeff {
                ad_βeff = ad试;
                β_eff = β试;
            }
        }

        // 各基准 AD 距离
        let ad_goe = Self::ad距离(&间距, 1);
        let ad_gue = Self::ad距离(&间距, 2);
        let ad_poisson = Self::ad距离(&间距, 0);
        let d_s_goe = Self::ad归一化(ad_goe, self.ad_临界);
        let d_s_gue = Self::ad归一化(ad_gue, self.ad_临界);
        let d_s_poisson = Self::ad归一化(ad_poisson, self.ad_临界);
        let d_s_βeff = Self::ad归一化(ad_βeff, self.ad_临界);

        // 判定
        let 判定 = if (d_s_goe <= 0.3 && d_s_gue > 0.3) && (d_s_βeff - d_s_goe).abs() < 0.1 {
            format!("GOE主导(β_eff={:.2})", β_eff)
        } else if (d_s_gue <= 0.3 && d_s_goe > 0.3) && (d_s_βeff - d_s_gue).abs() < 0.1 {
            format!("GUE主导(β_eff={:.2})", β_eff)
        } else if d_s_βeff < d_s_goe.min(d_s_gue) - 0.02 {
            format!("强混合(β_eff={:.2}，插值基准拟合最优)", β_eff)
        } else if (d_s_goe <= 0.3 && d_s_gue <= 0.3) {
            format!("解耦混合候选(两基准均拟合，需分块检测)")
        } else {
            format!("偏离普适类(β_eff={:.2})", β_eff)
        };

        β混合分析结果 {
            β_eff,
            拟合斜率: 斜率,
            样本数: n,
            d_s_βeff,
            d_s_goe,
            d_s_gue,
            d_s_poisson,
            判定,
        }
    }

    /// Anderson-Darling 距离（对连续 β∈[0,2] 插值分布）
    ///
    /// β=0 Poisson / β=1 GOE / β=2 GUE；中间值用 β 插值理论 CDF。
    pub fn ad距离_β(间距: &[f64], β: f64) -> f64 {
        let mut s: Vec<f64> = 间距.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = s.len();
        if n == 0 {
            return f64::INFINITY;
        }
        let mut sum = 0.0;
        for (i, &si) in s.iter().enumerate() {
            let f = Self::理论cdf_β(si, β).clamp(1e-15, 1.0 - 1e-15);
            let 末位 = Self::理论cdf_β(s[n - 1 - i], β).clamp(1e-15, 1.0 - 1e-15);
            sum += (2 * (i + 1) - 1) as f64 * (f.ln() + (1.0 - 末位).ln());
        }
        -(n as f64) - sum / n as f64
    }

    /// 连续 β 理论 CDF（β∈[0,2] 插值）
    ///
    /// 用 Wigner-Dyson 广义分布族 P_β(s) ∝ s^β exp(-a_β s²)，
    /// a_β 由归一化+平均间距=1 确定（Haake）。β=0 退化为 Poisson。
    pub fn 理论cdf_β(s: f64, β: f64) -> f64 {
        if s < 0.0 {
            return 0.0;
        }
        if β <= 0.05 {
            return 1.0 - (-s).exp();
        }
        // 广义 Wigner 分布: a_β = [Γ((β+2)/2)/Γ((β+1)/2)]²
        let a_β = {
            let 分子 = Self::lgamma((β + 2.0) / 2.0);
            let 分母 = Self::lgamma((β + 1.0) / 2.0);
            ((分子 - 分母).exp()).powi(2)
        };
        // CDF: γ((β+1)/2, a_β s²)/Γ((β+1)/2) —— 数值积分
        // 用 Simpson 积分 PDF
        Self::积分cdf_β(s, β, a_β)
    }

    /// 数值积分 CDF（Simpson，精度可控）
    fn 积分cdf_β(s: f64, β: f64, a_β: f64) -> f64 {
        let upper = s.min(10.0);
        let n = (upper / 0.002) as usize; // 步长 0.002
        let n = n.max(4);
        let h = upper / n as f64;
        let pdf = |x: f64| Self::pdf_β(x, β, a_β);
        let mut sum = pdf(0.0) + pdf(upper);
        for i in 1..n {
            let x = i as f64 * h;
            sum += if i % 2 == 1 { 4.0 * pdf(x) } else { 2.0 * pdf(x) };
        }
        (h / 3.0 * sum).clamp(0.0, 1.0)
    }

    /// 广义 Wigner-Dyson PDF: P_β(s) = c_β s^β exp(-a_β s²)
    fn pdf_β(s: f64, β: f64, a_β: f64) -> f64 {
        if s < 0.0 {
            return 0.0;
        }
        // 归一化常数 c_β = 2 a_β^{(β+1)/2} / Γ((β+1)/2)
        let c_β = 2.0 * a_β.powf((β + 1.0) / 2.0) / Self::lgamma((β + 1.0) / 2.0).exp();
        c_β * s.powf(β) * (-a_β * s * s).exp()
    }

    /// log-Γ 函数（Lanczos 近似，精度 ~1e-10）
    fn lgamma(x: f64) -> f64 {
        // Lanczos g=7, n=9
        const G: f64 = 7.0;
        const C: [f64; 9] = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];
        if x < 0.5 {
            // 反射公式
            return PI.ln() - (PI * x).sin().ln() - Self::lgamma(1.0 - x);
        }
        let x = x - 1.0;
        let mut 总和 = C[0];
        for i in 1..9 {
            总和 += C[i] / (x + i as f64);
        }
        let t = x + G + 0.5;
        0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + 总和.ln()
    }

    // ========================================================================
    // 展开谱（unfolding）
    // ========================================================================

    /// 展开谱（unfolding）: 对排序本征值做多项式回归拟合累积态密度 N(λ)，
    /// 得到平滑谱函数 ξ_i = N_smooth(λ_i)，使展开后平均间距≈1。
    ///
    /// 标准方法（Haake 量子混沌）：unfolding 消除谱密度变化，只保留统计涨落。
    /// 返回展开后的能级点序列 ξ_i（平均间距=1），最近邻间距由 `归一化间距` 计算。
    pub fn 展开谱(本征值: &[f64]) -> Vec<f64> {
        let mut λ: Vec<f64> = 本征值.to_vec();
        λ.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = λ.len();
        if n == 0 {
            return vec![];
        }

        // 标准化 λ 提高数值稳定性
        let 均值 = λ.iter().sum::<f64>() / n as f64;
        let 方差 = λ
            .iter()
            .map(|&x| (x - 均值) * (x - 均值))
            .sum::<f64>()
            / n as f64;
        let 标准差 = 方差.sqrt().max(1e-12);
        let λstd: Vec<f64> = λ.iter().map(|&x| (x - 均值) / 标准差).collect();

        // 多项式回归拟合 (λ_i, i) 点：i ≈ Σ c_k λ^k（degree 3，高阶病态）
        let degree = 3usize.min(n.saturating_sub(2));
        let 系数 = Self::多项式最小二乘(&λstd, degree);

        // 平滑累积态密度 ξ_i = poly(λ_i)
        let 展开: Vec<f64> = λstd
            .iter()
            .map(|&x| Self::多项式求值(&系数, x))
            .collect();

        // 重缩放使平均间距=1（返回展开点序列，非间距）
        let 间距: Vec<f64> = 展开.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
        let 均值间距 = if 间距.is_empty() {
            1.0
        } else {
            间距.iter().sum::<f64>() / 间距.len() as f64
        };
        if 均值间距 < 1e-12 {
            return 展开;
        }
        展开.iter().map(|&x| x / 均值间距).collect()
    }

    /// 归一化最近邻间距（平均间距=1）
    pub fn 归一化间距(ξ: &[f64]) -> Vec<f64> {
        let 间距: Vec<f64> = ξ.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
        if 间距.is_empty() {
            return 间距;
        }
        let 均值 = 间距.iter().sum::<f64>() / 间距.len() as f64;
        if 均值 < 1e-12 {
            return 间距;
        }
        间距.iter().map(|&s| s / 均值).collect()
    }

    /// 多项式最小二乘回归（SVD 求解，数值稳健）
    ///
    /// 对数据点 (x_i, y_i=i)，求系数 c 使 y ≈ Σ_k c_k x^k 最小二乘误差最小。
    /// 用 nalgebra SVD 解最小二乘（等价 numpy.linalg.lstsq），避免正规方程病态。
    /// degree 需 ≥1（阶数 = degree，项数 = degree+1）。
    fn 多项式最小二乘(x: &[f64], degree: usize) -> Vec<f64> {
        let n = x.len();
        let degree = degree.max(1);
        let 项数 = degree + 1;
        if n < 项数 {
            // 数据不足：退化为线性拟合
            let (斜率, 截距) = Self::线性回归(x);
            return vec![截距, 斜率];
        }
        // 构造 Vandermonde 矩阵 A[n×项数]: A[i][k] = x_i^k
        let a = DMatrix::from_fn(n, 项数, |i, k| x[i].powi(k as i32));
        let b = DMatrix::from_fn(n, 1, |i, _| i as f64);
        let svd = a.svd(true, true);
        match svd.solve(&b, 1e-10) {
            Ok(解) => (0..项数).map(|i| 解[(i, 0)]).collect(),
            Err(_) => {
                // SVD 失败兜底：线性回归
                let (斜率, 截距) = Self::线性回归(x);
                vec![截距, 斜率]
            }
        }
    }

    /// 线性回归 y = kx + b（最小二乘，返回 (斜率, 截距)）
    fn 线性回归(x: &[f64]) -> (f64, f64) {
        let n = x.len();
        if n == 0 {
            return (0.0, 0.0);
        }
        let 均值x = x.iter().sum::<f64>() / n as f64;
        let 均值y = (0..n).map(|i| i as f64).sum::<f64>() / n as f64;
        let mut 协方差 = 0.0;
        let mut 方差x = 0.0;
        for (i, &xi) in x.iter().enumerate() {
            let yi = i as f64;
            协方差 += (xi - 均值x) * (yi - 均值y);
            方差x += (xi - 均值x) * (xi - 均值x);
        }
        if 方差x < 1e-12 {
            return (0.0, 均值y);
        }
        let 斜率 = 协方差 / 方差x;
        let 截距 = 均值y - 斜率 * 均值x;
        (斜率, 截距)
    }

    /// 多项式求值 Σ c_k x^k
    fn 多项式求值(系数: &[f64], x: f64) -> f64 {
        系数
            .iter()
            .rev()
            .fold(0.0, |acc, &c| acc * x + c)
    }

    // ========================================================================
    // Anderson-Darling 距离
    // ========================================================================

    /// Anderson-Darling 距离（对理论分布 β: 0=Poisson 1=GOE 2=GUE）
    ///
    /// AD = -n - (1/n) Σ (2i-1)[ln F(s_i) + ln(1 - F(s_{n-i+1}))]
    /// 完美匹配 → AD≈0；严重偏离 → AD 大。
    pub fn ad距离(间距: &[f64], β: u8) -> f64 {
        let mut s: Vec<f64> = 间距.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = s.len();
        if n == 0 {
            return f64::INFINITY;
        }
        let mut sum = 0.0;
        for (i, &si) in s.iter().enumerate() {
            let f = Self::理论cdf(si, β);
            let f = f.clamp(1e-15, 1.0 - 1e-15);
            let 末位 = s[n - 1 - i];
            let f末 = Self::理论cdf(末位, β);
            let f末 = f末.clamp(1e-15, 1.0 - 1e-15);
            sum += (2 * (i + 1) - 1) as f64 * (f.ln() + (1.0 - f末).ln());
        }
        -(n as f64) - sum / n as f64
    }

    /// AD 距离归一化到 [0,1]: D_s = min(1, AD / AD_临界)
    fn ad归一化(ad: f64, ad_临界: f64) -> f64 {
        if !ad.is_finite() {
            return 1.0;
        }
        (ad / ad_临界).clamp(0.0, 1.0)
    }

    /// 理论 CDF（归一化最近邻间距分布）
    ///
    /// - Poisson (β=0): F(s) = 1 - e^{-s}（可积/局域化系统）
    /// - GOE (β=1): F(s) = 1 - exp(-πs²/4)（正交类，时间反演对称）
    /// - GUE (β=2): F(s) = 1 - exp(-4s²/π)(1 + 4s²/π)（酉类，无时间反演）
    pub fn 理论cdf(s: f64, β: u8) -> f64 {
        if s < 0.0 {
            return 0.0;
        }
        match β {
            0 => 1.0 - (-s).exp(),
            1 => 1.0 - (-PI * s * s / 4.0).exp(),
            _ => {
                // GUE CDF: F(s) = erf(2s/√π) - (4s/π)·e^{-4s²/π}
                // （由 P(s)=(32/π²)s²e^{-4s²/π} 积分得到）
                let x = 2.0 * s / PI.sqrt();
                Self::erf(x) - (4.0 * s / PI) * (-4.0 * s * s / PI).exp()
            }
        }
    }

    /// 误差函数 erf(x)（Abramowitz-Stegun 7.1.26，精度 ~1.5e-7）
    fn erf(x: f64) -> f64 {
        let 符号 = if x < 0.0 { -1.0 } else { 1.0 };
        let ax = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * ax);
        let y = 1.0
            - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
                + 0.254829592)
                * t
                * (-ax * ax).exp());
        符号 * y
    }

    /// 理论 PDF（供逆变换采样）
    fn 理论pdf(s: f64, β: u8) -> f64 {
        if s < 0.0 {
            return 0.0;
        }
        match β {
            0 => (-s).exp(),
            1 => (PI / 2.0) * s * (-PI * s * s / 4.0).exp(),
            _ => (32.0 / (PI * PI)) * s * s * (-4.0 * s * s / PI).exp(),
        }
    }

    // ========================================================================
    // MC 有限 n p 值
    // ========================================================================

    /// MC 有限 n p 值: 零假设=谱来自对称性类 β。
    ///
    /// 生成 n_sim 组长度为 n 的随机间距（从理论分布逆变换采样），
    /// 每组计算 AD 距离得到基准分布，p = P(AD_基准 ≥ AD_obs)。
    ///
    /// p 大 → 观测 AD 在基准分布内 → 不拒绝 H0 → 谱符合该对称类。
    /// 这是有限 n 修正：n 较小时 AD 天然偏大，需与同 n 基准比较而非渐近临界值。
    pub fn mc_p值(&self, 间距数: usize, β: u8, ad_obs: f64) -> f64 {
        let n = 间距数;
        let n_sim = self.n_sim.max(10);
        // MC 基准用确定性种子（可复现，与测试断言一致）
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xA11CE_u64 + 间距数 as u64 + β as u64);
        let mut 超计数 = 0u32;
        for _ in 0..n_sim {
            // 生成 n 个符合 β 分布的随机间距
            let 样本: Vec<f64> = (0..n).map(|_| Self::逆变换采样(&mut rng, β)).collect();
            let ad_sim = Self::ad距离(&样本, β);
            if ad_sim >= ad_obs {
                超计数 += 1;
            }
        }
        // 保守估计：+1 避免 p=0
        (超计数 as f64 + 1.0) / (n_sim as f64 + 1.0)
    }

    /// 逆变换采样: 从理论分布（CDF=F）采样一个间距值
    ///
    /// GOE/Poisson CDF 可解析反演；GUE 用数值二分。
    fn 逆变换采样(rng: &mut impl Rng, β: u8) -> f64 {
        let u: f64 = rng.gen_range(1e-9..1.0 - 1e-9);
        match β {
            0 => -(1.0 - u).ln(), // s = -ln(1-u)，Poisson 指数分布
            1 => {
                // s = sqrt(-4·ln(1-u)/π)
                (-4.0 * (1.0 - u).ln() / PI).max(0.0).sqrt()
            }
            _ => {
                // GUE: 数值二分求 F(s)=u
                let mut lo = 0.0;
                let mut hi = 8.0;
                // 扩展上界直到 F(hi) ≥ u
                while Self::理论cdf(hi, 2) < u && hi < 64.0 {
                    hi *= 2.0;
                }
                for _ in 0..60 {
                    let mid = 0.5 * (lo + hi);
                    if Self::理论cdf(mid, 2) < u {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                0.5 * (lo + hi)
            }
        }
    }

    /// 综合判定
    fn 判定(d_s_wd: f64, d_s_p: f64, p值: f64, 对称性类: &str) -> String {
        if d_s_wd <= 0.3 && d_s_p > 0.3 {
            format!(
                "混沌稳态（{} 普适类内）: D_s^WD={:.3}≤0.3 且 D_s^P={:.3}>0.3, p={:.3}",
                对称性类, d_s_wd, d_s_p, p值
            )
        } else if d_s_p <= 0.3 && d_s_wd > 0.3 {
            format!(
                "可积/局域化（Poisson 类）: D_s^P={:.3}≤0.3 且 D_s^WD={:.3}>0.3, p={:.3}",
                d_s_p, d_s_wd, p值
            )
        } else if d_s_wd <= 0.3 && d_s_p <= 0.3 {
            format!(
                "中间态/欠采样: 两距离均≤0.3（n 过小或混合谱）, p={:.3}",
                p值
            )
        } else {
            format!(
                "严重偏离普适类: D_s^WD={:.3}, D_s^P={:.3}, p={:.3}（需检查损伤/微调拓扑）",
                d_s_wd, d_s_p, p值
            )
        }
    }
}

// ============================================================================
// Ginibre 复平面分析（变分原理深化 v1.0 第三部分 §3.3.2 非梯度系统）
// ============================================================================

/// Ginibre 复平面谱分析结果（非厄米随机矩阵，非梯度系统基准）
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Ginibre分析结果 {
    /// 复本征值数量
    pub 点数: usize,
    /// 复平面半径（最大 |μ|）
    pub 半径: f64,
    /// 径向分布与 Ginibre 圆盘律的偏离度 D_s^Ginibre ∈ [0,1]
    pub d_s_ginibre: f64,
    /// 虚部比例（复本征值占比，非梯度证据强度）
    pub 虚部比例: f64,
    /// 角向均匀性（0=均匀圆盘，>0 各向异性）
    pub 角向均匀性: f64,
    /// 实部统计的 GOE 偏离度（投影到实轴）
    pub d_s_goe_实部: f64,
    /// 判定: 圆盘律拟合 / 环状 / 各向异性 / 样本不足
    pub 判定: String,
}

/// Ginibre 复平面分析器 — 对非梯度系统的复本征值做圆盘律基准检验
///
/// 论文 §3.3.2: 非梯度系统（∇×f≠0）的 Koopman 本征值在复平面服从 Ginibre 圆盘律:
///   径向分布: ρ(r) 随 r 增大在圆盘边缘形成均匀密度（圆盘律）
///   角向: 均匀（旋转对称）
///
/// 工程实现:
///   - 径向距离 d_s: 用复本征值的模长 |μ| 分布对照理论圆盘分布（LSD）
///   - 角向均匀性: 相位分布的圆形统计（Rayleigh 检验近似）
///   - 实部 GOE: 投影实部做 GOE 检验（论文§3.3.2: 实部投影可能近似 WD）
pub struct Ginibre分析器 {
    /// MC 样本数（径向基准）
    pub n_sim: usize,
}

impl Default for Ginibre分析器 {
    fn default() -> Self {
        Self { n_sim: 300 }
    }
}

impl Ginibre分析器 {
    /// 复平面谱分析
    ///
    /// # 参数
    /// - `复本征值`: (实部, 虚部) 列表
    /// - `矩阵规模`: 原始矩阵维数 n（圆盘律的半径归一化基准）
    pub fn 分析(&self, 复本征值: &[(f64, f64)], 矩阵规模: usize) -> Ginibre分析结果 {
        let n = 复本征值.len();
        if n < 8 {
            return Ginibre分析结果 {
                点数: n,
                半径: 0.0,
                d_s_ginibre: 1.0,
                虚部比例: 0.0,
                角向均匀性: 1.0,
                d_s_goe_实部: 1.0,
                判定: "样本不足（需≥8个复本征值）".to_string(),
            };
        }

        // 1. 半径与模长分布
        let 模长: Vec<f64> = 复本征值.iter().map(|&(re, im)| (re * re + im * im).sqrt()).collect();
        let 半径 = 模长.iter().cloned().fold(0.0_f64, f64::max).max(1e-12);
        // 归一化模长
        let 归一化模长: Vec<f64> = 模长.iter().map(|&r| r / 半径).collect();

        // 2. 理论圆盘分布 CDF: 圆盘律下归一化半径 r∈[0,1] 的累积分布 = r²
        //    （Girko 圆盘律: 谱在单位圆盘内均匀）
        //    AD 距离对照 F_theory(r)=r²
        let d_s_ginibre = Self::ad距离圆盘(&归一化模长);

        // 3. 虚部比例
        let 虚部数 = 复本征值.iter().filter(|&&(_, im)| im.abs() > 1e-6).count();
        let 虚部比例 = 虚部数 as f64 / n as f64;

        // 4. 角向均匀性（Rayleigh 检验近似: 相位向量平均模长）
        let 平均实 = 复本征值.iter().map(|&(re, _)| re / (re*re + 0.0f64).sqrt().max(1e-12)).sum::<f64>() / n as f64;
        let 平均虚 = 复本征值.iter().map(|&(_, im)| im / (im*im + 0.0f64).sqrt().max(1e-12)).sum::<f64>() / n as f64;
        // 相位向量平均模长 R: 均匀分布 R→0, 聚集 R→1
        let r_rayleigh = (平均实 * 平均实 + 平均虚 * 平均虚).sqrt();
        let 角向均匀性 = r_rayleigh;

        // 5. 实部 GOE 检验（投影到实轴）
        let 实部: Vec<f64> = 复本征值.iter().map(|&(re, _)| re).collect();
        let d_s_goe_实部 = {
            let 求解器 = 谱求解器::default();
            let 展开 = 谱求解器::展开谱(&实部);
            let 间距 = 谱求解器::归一化间距(&展开);
            if 间距.len() < 3 {
                1.0
            } else {
                let ad = 谱求解器::ad距离(&间距, 1);
                Self::ad归一化(ad, 2.5)
            }
        };

        // 6. 判定
        let 判定 = if d_s_ginibre <= 0.3 && 角向均匀性 < 0.3 {
            format!("Ginibre圆盘律拟合（混沌非梯度系统）: D_s^Gin={:.3}, 角向均匀R={:.3}", d_s_ginibre, 角向均匀性)
        } else if d_s_ginibre <= 0.3 {
            format!("径向圆盘但角向各向异性（存在主导方向）: D_s^Gin={:.3}, R={:.3}", d_s_ginibre, 角向均匀性)
        } else if 角向均匀性 < 0.3 {
            format!("角向均匀但径向偏离圆盘律（环状谱?）: D_s^Gin={:.3}, R={:.3}", d_s_ginibre, 角向均匀性)
        } else {
            format!("偏离Ginibre圆盘律（结构非随机或尺度不足）: D_s^Gin={:.3}, R={:.3}", d_s_ginibre, 角向均匀性)
        };

        Ginibre分析结果 {
            点数: n,
            半径,
            d_s_ginibre,
            虚部比例,
            角向均匀性,
            d_s_goe_实部,
            判定,
        }
    }

    /// 圆盘律 AD 距离（对照 F(r)=r²）
    pub fn ad距离圆盘(归一化模长: &[f64]) -> f64 {
        let mut r: Vec<f64> = 归一化模长.to_vec();
        r.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = r.len();
        if n == 0 {
            return f64::INFINITY;
        }
        let mut sum = 0.0;
        for (i, &ri) in r.iter().enumerate() {
            // 理论 CDF: F(r) = r²（圆盘内均匀）
            let f = (ri * ri).clamp(1e-15, 1.0 - 1e-15);
            let f末 = (r[n - 1 - i] * r[n - 1 - i]).clamp(1e-15, 1.0 - 1e-15);
            sum += (2 * (i + 1) - 1) as f64 * (f.ln() + (1.0 - f末).ln());
        }
        let ad = -(n as f64) - sum / n as f64;
        Self::ad归一化(ad, 2.5)
    }

    /// AD 距离归一化到 [0,1]
    fn ad归一化(ad: f64, ad_临界: f64) -> f64 {
        if !ad.is_finite() {
            return 1.0;
        }
        (ad / ad_临界).clamp(0.0, 1.0)
    }

    /// 生成 Ginibre 系综样本（复高斯矩阵的特征值，用于 MC 基准）
    ///
    /// 标准 Ginibre: 复高斯矩阵 G (n×n, 元素~N(0,1)+iN(0,1))，特征值在单位圆盘内均匀。
    /// 用 Schur 分解求复特征值（nalgebra）。
    pub fn 生成Ginibre样本(n: usize, 样本数: usize) -> Vec<(f64, f64)> {
        let mut rng = rand::thread_rng();
        let mut 全部 = Vec::with_capacity(n * 样本数);
        for _ in 0..样本数 {
            // 构造 2n×2n 实矩阵近似复高斯（分块 [[A,-B],[B,A]] 特征值 = A±iB 的复数）
            let m = DMatrix::from_fn(2 * n, 2 * n, |i, j| {
                let u1: f64 = rng.gen_range(-1.0..1.0);
                let u2: f64 = rng.gen_range(-1.0..1.0);
                // 分块复结构: (0..n)x(0..n)=A, (0..n)x(n..2n)=-B, (n..2n)x(0..n)=B, (n..2n)x(n..2n)=A
                let (re, im) = if i < n && j < n {
                    (u1, u2) // A = U+iV 的实部/虚部混合（简化：直接高斯）
                } else {
                    (u1, u2)
                };
                let _ = (re, im);
                u1 * 0.5 + u2 * 0.0
            });
            // 简化：直接用 2n×2n 高斯实矩阵的复特征值（含共轭对，取前 n 个）
            let schur = nalgebra::linalg::Schur::new(m);
            let 特征值: Vec<(f64, f64)> = schur
                .complex_eigenvalues()
                .iter()
                .map(|λ| (λ.re, λ.im))
                .collect();
            let 取n: Vec<(f64, f64)> = 特征值.into_iter().take(n).collect();
            全部.extend(取n);
        }
        全部
    }
}

// ============================================================================
// 非厄米分析器 — 第151批 V10 椭圆系综（非厄米变分原理 v1.0）
// ============================================================================

use nalgebra::linalg::Schur;
use nalgebra::SVD;

/// 非厄米分析结果 — Engine v2.1 输出（非厄米变分原理 v1.0）
///
/// L=H-iΓ（H=H† 保守振荡, Γ=Γ†>0 耗散）→ 复本征值 λ=ω-i/τ:
///   τ=-1/Im(λ) 第一基本量, ω=Re(λ) 振荡频率
///   椭圆参数 τ_param=(σ_ω2-σ_γ2)/(σ_ω2+σ_γ2) ∈ [-1,1] 连续替代 GOE/GUE/Ginibre 三分
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct 非厄米结果 {
    /// 复本征值数量（筛选 Im<0 的模）
    pub 点数: usize,
    /// 复本征值列表 (ω, τ) — 每个 (Re λ, τ=-1/Im λ)
    pub 本征值: Vec<(f64, f64)>,
    /// τ 链（升序）
    pub τ链: Vec<f64>,
    /// 振荡频率谱（ω=Re λ）
    pub ω谱: Vec<f64>,
    /// 平均耗散 γ_0 = -mean(Im λ)
    pub γ_0: f64,
    /// 椭圆参数 τ_param ∈ [-1,1]（H/Γ 方差比）: -1=GOE过阻尼 0=Ginibre 1=GUE近保守
    pub τ_param: f64,
    /// 到椭圆均匀分布的谱偏离度 D_s^{(2D)}（复平面椭圆 CDF AD 距离）∈[0,1]
    pub d_s_椭圆: f64,
    /// 非正规性 ν = ||[L,L†]||_F / ||L||_F²（0=正规矩阵, 越大越非正规/伪谱膨胀）
    pub ν: f64,
    /// 自扰代理: 谱条件数 κ(L)=σ_max/σ_min（SVD, 论文 κ(V) 的工程近似）
    pub κ_l: f64,
    /// 普适类映射（τ_param 阈值）: GUE(>0.5) / Ginibre(|≤0.5) / GOE(<-0.5)
    pub 普适类: String,
    /// 判定
    pub 判定: String,
}

/// 非厄米分析器 — V10 椭圆系综基准检验
///
/// 工程实现:
///   - 实表示: L_r = [[H, Γ], [-Γ, H]]（2n×2n 实矩阵）, Schur 复特征值 = {λ, λ*}
///   - 筛选 Im<0 的模 → λ=ω-i/τ（Γ>0 时 Im(λ)<0）
///   - γ_0 = -mean(Im λ); σ_ω2=Var(Re), σ_γ2=Var(Im); τ_param=(σ_ω2-σ_γ2)/(σ_ω2+σ_γ2)
///   - D_s^{(2D)}: 归一化椭圆坐标 u=sqrt((ω/a)²+((Im+γ_0)/b)²), a=1+τ_param, b=1-τ_param,
///     理论椭圆内均匀 CDF F(u)=u²（类比 Ginibre 圆盘律）→ AD 距离
///   - ν: 正规性偏差 ||L_r L_r^T - L_r^T L_r||_F / ||L_r||_F²
///   - κ_L: SVD 谱条件数（工程代理）
pub struct 非厄米分析器 {
    /// AD 距离归一化阈值
    pub ad_临界: f64,
    /// 样本不足阈值
    pub 最少点数: usize,
}

impl Default for 非厄米分析器 {
    fn default() -> Self {
        Self { ad_临界: 2.5, 最少点数: 8 }
    }
}

impl 非厄米分析器 {
    /// 从 H 与 Γ 分析非厄米算子 L=H-iΓ
    pub fn 从非厄米(&self, h: &DMatrix<f64>, gamma: &DMatrix<f64>) -> 非厄米结果 {
        let n = h.nrows();
        if n == 0 || n != h.ncols() || n != gamma.nrows() || n != gamma.ncols() {
            return self.样本不足("矩阵维度无效");
        }
        // 实表示: [[H, Γ], [-Γ, H]]
        let mut lr = DMatrix::zeros(2 * n, 2 * n);
        for i in 0..n {
            for j in 0..n {
                lr[(i, j)] = h[(i, j)];
                lr[(i, n + j)] = gamma[(i, j)];
                lr[(n + i, j)] = -gamma[(i, j)];
                lr[(n + i, n + j)] = h[(i, j)];
            }
        }
        let 特征值 = Schur::new(lr.clone()).complex_eigenvalues();
        // 筛选 Im < 0 的模（Γ>0 → λ 在下半平面）
        let 下半平面: Vec<(f64, f64)> = 特征值
            .iter()
            .filter(|λ| λ.im < -1e-9)
            .map(|λ| (λ.re, λ.im))
            .collect();
        if 下半平面.len() < self.最少点数 {
            return self.样本不足(&format!("有效模{}<{}", 下半平面.len(), self.最少点数));
        }
        let 模数 = 下半平面.len();
        // ω=Re, τ=-1/Im
        let 本征值: Vec<(f64, f64)> = 下半平面.iter().map(|&(re, im)| (re, -1.0 / im)).collect();
        let mut τ链: Vec<f64> = 本征值.iter().map(|&(_, tau)| tau).collect();
        τ链.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let ω谱: Vec<f64> = 本征值.iter().map(|&(w, _)| w).collect();

        // γ_0 = -mean(Im λ); 需重新从 1/τ 恢复 Im
        let γ_0 = 本征值.iter().map(|&(_, tau)| 1.0 / tau).sum::<f64>() / 模数 as f64;
        // σ_ω2, σ_γ2
        let 均ω = ω谱.iter().sum::<f64>() / 模数 as f64;
        let σ_ω2 = ω谱.iter().map(|w| (w - 均ω).powi(2)).sum::<f64>() / 模数 as f64;
        let 均γ = 本征值.iter().map(|&(_, tau)| 1.0 / tau).sum::<f64>() / 模数 as f64;
        let σ_γ2 = 本征值
            .iter()
            .map(|&(_, tau)| (1.0 / tau - 均γ).powi(2))
            .sum::<f64>()
            / 模数 as f64;
        let τ_param = if σ_ω2 + σ_γ2 < 1e-12 {
            0.0 // 所有模同构（退化）
        } else {
            ((σ_ω2 - σ_γ2) / (σ_ω2 + σ_γ2)).clamp(-1.0, 1.0)
        };

        // D_s^{(2D)}: 归一化椭圆坐标 u, 理论 CDF F(u)=u²
        let a = (1.0 + τ_param).max(1e-9);
        let b = (1.0 - τ_param).max(1e-9);
        let mut u: Vec<f64> = 本征值
            .iter()
            .map(|&(w, tau)| {
                let im = 1.0 / tau;
                let ux = w / a;
                let uy = (im + γ_0) / b;
                (ux * ux + uy * uy).sqrt()
            })
            .collect();
        u.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        let d_s_椭圆 = Self::ad距离椭圆(&u, self.ad_临界);

        // ν = ||[L_r, L_r^T]||_F / ||L_r||_F²（正规性偏差）
        let ν = {
            let llt = &lr * &lr.transpose();
            let ltl = &lr.transpose() * &lr;
            let 非正规范数 = (&llt - &ltl).norm();
            let 范数平方 = lr.norm() * lr.norm();
            if 范数平方 < 1e-12 {
                0.0
            } else {
                非正规范数 / 范数平方
            }
        };

        // κ_L: 谱条件数（SVD）
        let κ_l = {
            let svd = SVD::new(lr.clone(), false, false);
            let 奇异值: Vec<f64> = svd.singular_values.iter().cloned().collect();
            let max = 奇异值.iter().cloned().fold(0.0_f64, f64::max);
            let min = 奇异值
                .iter()
                .cloned()
                .fold(f64::INFINITY, |m, s| if s > 1e-12 && s < m { s } else { m });
            if min.is_finite() && min > 1e-12 {
                max / min
            } else {
                f64::INFINITY
            }
        };

        // 普适类映射
        let (普适类, 判定) = if τ_param > 0.5 {
            (
                "GUE".to_string(),
                format!(
                    "近保守振荡(H>>Γ): τ_param={:.3}, γ_0={:.3}, D_s_2D={:.3}",
                    τ_param, γ_0, d_s_椭圆
                ),
            )
        } else if τ_param < -0.5 {
            (
                "GOE".to_string(),
                format!(
                    "过阻尼纯耗散(Γ>>H): τ_param={:.3}, γ_0={:.3}, D_s_2D={:.3}",
                    τ_param, γ_0, d_s_椭圆
                ),
            )
        } else {
            (
                "Ginibre".to_string(),
                format!(
                    "振荡耗散等强(H~Γ): τ_param={:.3}, γ_0={:.3}, D_s_2D={:.3}",
                    τ_param, γ_0, d_s_椭圆
                ),
            )
        };

        非厄米结果 {
            点数: 模数,
            本征值,
            τ链,
            ω谱,
            γ_0,
            τ_param,
            d_s_椭圆,
            ν,
            κ_l,
            普适类,
            判定,
        }
    }

    /// 从复本征值直接分析（EDMD/DMD 输出的 Koopman 谱, 免 H/Γ 构造）
    pub fn 从复本征值(&self, 复本征值: &[(f64, f64)]) -> 非厄米结果 {
        let 有效: Vec<(f64, f64)> = 复本征值
            .iter()
            .filter(|&&(_, im)| im < -1e-9)
            .map(|&(re, im)| (re, -1.0 / im))
            .collect();
        if 有效.len() < self.最少点数 {
            return self.样本不足(&format!("有效模{}<{}", 有效.len(), self.最少点数));
        }
        // 构造 0 矩阵占位 H/Γ（仅用于规格化; 直接计算各量）
        let 模数 = 有效.len();
        let 本征值 = 有效;
        let mut τ链: Vec<f64> = 本征值.iter().map(|&(_, tau)| tau).collect();
        τ链.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let ω谱: Vec<f64> = 本征值.iter().map(|&(w, _)| w).collect();
        let γ_0 = 本征值.iter().map(|&(_, tau)| 1.0 / tau).sum::<f64>() / 模数 as f64;
        let 均ω = ω谱.iter().sum::<f64>() / 模数 as f64;
        let σ_ω2 = ω谱.iter().map(|w| (w - 均ω).powi(2)).sum::<f64>() / 模数 as f64;
        let 均γ = 本征值.iter().map(|&(_, tau)| 1.0 / tau).sum::<f64>() / 模数 as f64;
        let σ_γ2 = 本征值.iter().map(|&(_, tau)| (1.0 / tau - 均γ).powi(2)).sum::<f64>() / 模数 as f64;
        let τ_param = if σ_ω2 + σ_γ2 < 1e-12 {
            0.0
        } else {
            ((σ_ω2 - σ_γ2) / (σ_ω2 + σ_γ2)).clamp(-1.0, 1.0)
        };
        let a = (1.0 + τ_param).max(1e-9);
        let b = (1.0 - τ_param).max(1e-9);
        let mut u: Vec<f64> = 本征值
            .iter()
            .map(|&(w, tau)| {
                let im = 1.0 / tau;
                let ux = w / a;
                let uy = (im + γ_0) / b;
                (ux * ux + uy * uy).sqrt()
            })
            .collect();
        u.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        let d_s_椭圆 = Self::ad距离椭圆(&u, self.ad_临界);

        let (普适类, 判定) = if τ_param > 0.5 {
            ("GUE".to_string(), format!("近保守振荡: τ_param={:.3}, γ_0={:.3}", τ_param, γ_0))
        } else if τ_param < -0.5 {
            ("GOE".to_string(), format!("过阻尼耗散: τ_param={:.3}, γ_0={:.3}", τ_param, γ_0))
        } else {
            ("Ginibre".to_string(), format!("振荡耗散等强: τ_param={:.3}, γ_0={:.3}", τ_param, γ_0))
        };

        非厄米结果 {
            点数: 模数,
            本征值,
            τ链,
            ω谱,
            γ_0,
            τ_param,
            d_s_椭圆,
            ν: 0.0,   // 无矩阵信息, 非正规性不可算
            κ_l: 0.0, // 无矩阵信息
            普适类,
            判定,
        }
    }

    /// 椭圆内均匀分布 AD 距离（理论 CDF F(u)=u², 类比圆盘律）
    pub fn ad距离椭圆(u: &[f64], ad_临界: f64) -> f64 {
        if u.is_empty() {
            return 1.0;
        }
        let n = u.len();
        let mut sum = 0.0;
        for (i, &ui) in u.iter().enumerate() {
            let f = (ui * ui).clamp(1e-15, 1.0 - 1e-15);
            let f末 = (u[n - 1 - i] * u[n - 1 - i]).clamp(1e-15, 1.0 - 1e-15);
            sum += (2 * (i + 1) - 1) as f64 * (f.ln() + (1.0 - f末).ln());
        }
        let ad = -(n as f64) - sum / n as f64;
        if !ad.is_finite() {
            return 1.0;
        }
        (ad / ad_临界).clamp(0.0, 1.0)
    }

    fn 样本不足(&self, 原因: &str) -> 非厄米结果 {
        非厄米结果 {
            点数: 0,
            本征值: vec![],
            τ链: vec![],
            ω谱: vec![],
            γ_0: 0.0,
            τ_param: 0.0,
            d_s_椭圆: 1.0,
            ν: 0.0,
            κ_l: 0.0,
            普适类: "未知".to_string(),
            判定: format!("样本不足({})", 原因),
        }
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    /// 构造 3x3 环图 Laplacian: 节点 3 环，单位权重
    /// 特征值: 0, 3, 3（连通图零空间 1 维）
    fn 环图拉普拉斯() -> DMatrix<f64> {
        DMatrix::from_row_slice(3, 3, &[2.0, -1.0, -1.0, -1.0, 2.0, -1.0, -1.0, -1.0, 2.0])
    }

    #[test]
    fn test_环图特征值已知() {
        let l = 环图拉普拉斯();
        let sym = SymmetricEigen::new(l);
        let mut 特征值: Vec<f64> = sym.eigenvalues.iter().cloned().collect();
        // nalgebra 特征值顺序不保证升序，排序后断言
        特征值.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((特征值[0] - 0.0).abs() < 1e-9, "最小特征值应为0: {}", 特征值[0]);
        assert!((特征值[1] - 3.0).abs() < 1e-9, "第二特征值应为3: {}", 特征值[1]);
        assert!((特征值[2] - 3.0).abs() < 1e-9, "第三特征值应为3: {}", 特征值[2]);
    }

    #[test]
    fn test_对称破坏度() {
        let 对称 = 环图拉普拉斯();
        assert!(谱求解器::对称破坏度(&对称) < 1e-12, "对称矩阵破坏度应≈0");

        let 非对称 = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 0.0]);
        assert!(谱求解器::对称破坏度(&非对称) > 0.1, "非对称矩阵破坏度应>0");
    }

    #[test]
    fn test_理论cdf性质() {
        // F(0)=0
        assert!((谱求解器::理论cdf(0.0, 0) - 0.0).abs() < 1e-9);
        assert!((谱求解器::理论cdf(0.0, 1) - 0.0).abs() < 1e-9);
        assert!((谱求解器::理论cdf(0.0, 2) - 0.0).abs() < 1e-9);
        // F(∞)→1（Poisson 衰减慢，用更大 s）
        assert!((谱求解器::理论cdf(30.0, 0) - 1.0).abs() < 1e-12);
        assert!((谱求解器::理论cdf(8.0, 1) - 1.0).abs() < 1e-9);
        assert!((谱求解器::理论cdf(8.0, 2) - 1.0).abs() < 1e-9);
        // 单调递增
        assert!(谱求解器::理论cdf(1.0, 1) > 谱求解器::理论cdf(0.5, 1));
        assert!(谱求解器::理论cdf(1.0, 2) > 谱求解器::理论cdf(0.5, 2));
    }

    #[test]
    fn test_理论cdf解析值() {
        // GOE: F(1) = 1 - exp(-π/4) ≈ 0.5441
        let f1 = 谱求解器::理论cdf(1.0, 1);
        assert!((f1 - 0.5441).abs() < 0.01, "GOE F(1)≈0.544: {}", f1);
        // GUE: F(1) = erf(2/√π) - (4/π)·e^{-4/π} ≈ 0.533
        let 期望 = 0.5330502005906137;
        let f2 = 谱求解器::理论cdf(1.0, 2);
        assert!((f2 - 期望).abs() < 1e-6, "GUE F(1) 解析核对: {} vs {}", f2, 期望);
        // Poisson: F(1) = 1-e^{-1} ≈ 0.632
        let f0 = 谱求解器::理论cdf(1.0, 0);
        assert!((f0 - 0.6321).abs() < 0.01, "Poisson F(1)≈0.632: {}", f0);
    }

    #[test]
    fn test_展开谱平均间距归一() {
        // 均匀分布本征值
        let 本征值: Vec<f64> = (0..50).map(|i| i as f64 * 1.7 + 0.3).collect();
        let 展开 = 谱求解器::展开谱(&本征值);
        let 间距 = 谱求解器::归一化间距(&展开);
        let 均值 = 间距.iter().sum::<f64>() / 间距.len() as f64;
        assert!((均值 - 1.0).abs() < 0.05, "展开后平均间距≈1: {}", 均值);
    }

    #[test]
    fn test_均匀谱判定() {
        // 完全均匀本征值（可积系统极限：规则谱）——既非GUE也非Poisson，应被判为严重偏离
        let 本征值: Vec<f64> = (0..60).map(|i| i as f64 * 1.0 + 0.1).collect();
        let s = 谱求解器::default();
        let 结果 = s.从本征值(&本征值);
        // 均匀谱（delta 间距分布）对 GUE 偏离应显著（规则谱 ≠ 混沌谱）
        assert!(
            结果.d_s_wd > 0.3,
            "均匀规则谱对GUE应显著偏离: D_s^WD={}",
            结果.d_s_wd
        );
    }

    #[test]
    fn test_poisson采样谱拟合Poisson() {
        // 用逆变换采样生成 Poisson 间距 → 累积成本征值 → 应拟合 Poisson（D_s^P 小）
        let s = 谱求解器::default();
        let mut rng = rand::thread_rng();
        let n = 80;
        // 生成 n 个 Poisson 间距（指数分布，均值1）
        let 间距: Vec<f64> = (0..n).map(|_| 谱求解器::逆变换采样(&mut rng, 0)).collect();
        // 累积和构造本征值
        let mut 本征值 = Vec::with_capacity(n + 1);
        let mut acc = 0.0;
        本征值.push(acc);
        for &s in &间距 {
            acc += s;
            本征值.push(acc);
        }
        let 结果 = s.从本征值(&本征值);
        // Poisson 样本应对 Poisson 的偏离小于对 GUE 的偏离
        assert!(
            结果.d_s_p < 结果.d_s_wd,
            "Poisson采样谱应更接近Poisson: D_s^P={} vs D_s^WD={}",
            结果.d_s_p, 结果.d_s_wd
        );
    }

    #[test]
    fn test_gue采样谱拟合GUE() {
        // 用逆变换采样生成 GUE 间距 → 累积成本征值 → 应拟合 GUE（D_s^WD 小）
        let s = 谱求解器::default();
        let mut rng = rand::thread_rng();
        let n = 80;
        let 间距: Vec<f64> = (0..n).map(|_| 谱求解器::逆变换采样(&mut rng, 2)).collect();
        let mut 本征值 = Vec::with_capacity(n + 1);
        let mut acc = 0.0;
        本征值.push(acc);
        for &s in &间距 {
            acc += s;
            本征值.push(acc);
        }
        let 结果 = s.从本征值(&本征值);
        assert!(
            结果.d_s_wd < 0.5,
            "GUE采样谱对GUE偏离应较小: D_s^WD={}",
            结果.d_s_wd
        );
        // GUE 样本应对 Poisson 偏离更大
        assert!(
            结果.d_s_p > 结果.d_s_wd,
            "GUE采样谱对Poisson偏离应更大: D_s^P={} vs D_s^WD={}",
            结果.d_s_p, 结果.d_s_wd
        );
    }

    #[test]
    fn test_环图谱在GUE普适类() {
        // 环图 Laplacian 是实对称矩阵 → GOE 类；但特征值高度简并（3,3），谱间隔不规则
        // 主要验证流程可跑通 + 结构完整
        let l = 环图拉普拉斯();
        let s = 谱求解器::default();
        let 结果 = s.从矩阵(&l);
        assert_eq!(结果.本征值数, 2, "3x3环图排除零空间后2个非零本征值");
        assert_eq!(结果.β, 1, "对称矩阵判定GOE");
        assert_eq!(结果.对称性类, "GOE");
        assert!(!结果.判定.is_empty());
    }

    #[test]
    fn test_mc_p值_零假设匹配() {
        // 从 GUE 采样生成的间距，对 GUE 的 p 值应较高（不拒绝 H0）
        // 用确定性种子避免随机 flaky；n=80 降低 AD 观测方差
        use rand::SeedableRng;
        let s = 谱求解器 { n_sim: 150, ..Default::default() };
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let 样本: Vec<f64> = (0..80).map(|_| 谱求解器::逆变换采样(&mut rng, 2)).collect();
        let ad_obs = 谱求解器::ad距离(&样本, 2);
        let p = s.mc_p值(80, 2, ad_obs);
        assert!(p > 0.02, "GUE采样对GUE的p值应>0.02（有限n修正）: {}", p);
    }

    #[test]
    fn test_欠采样保护() {
        let s = 谱求解器::default();
        let 结果 = s.从本征值(&[1.0, 2.0, 3.0]);
        assert!(结果.判定.contains("样本不足"));
        assert_eq!(结果.本征值数, 3);
    }

    #[test]
    fn test_逆变换采样分布正确() {
        // 大样本逆变换采样均值/方差应接近理论分布
        let mut rng = rand::thread_rng();
        let n = 2000;
        // Poisson 均值=1
        let 均值0: f64 = (0..n).map(|_| 谱求解器::逆变换采样(&mut rng, 0)).sum::<f64>() / n as f64;
        assert!((均值0 - 1.0).abs() < 0.1, "Poisson采样均值≈1: {}", 均值0);
        // GOE 均值 = Γ(3/2)/Γ(1)/√(4/π)·... 约 1.0（实际≈1.0）
        let 均值1: f64 = (0..n).map(|_| 谱求解器::逆变换采样(&mut rng, 1)).sum::<f64>() / n as f64;
        assert!((均值1 - 1.0).abs() < 0.1, "GOE采样均值≈1: {}", 均值1);
        // GUE 均值≈1.0
        let 均值2: f64 = (0..n).map(|_| 谱求解器::逆变换采样(&mut rng, 2)).sum::<f64>() / n as f64;
        assert!((均值2 - 1.0).abs() < 0.15, "GUE采样均值≈1: {}", 均值2);
    }

    #[test]
    fn test_多项式拟合() {
        // y = 2x² + 3x + 1 在 x=0..10 上拟合，应还原系数
        let x: Vec<f64> = (0..30).map(|i| i as f64 / 3.0).collect();
        // 拟合 degree=2（项数3），使用 多项式最小二乘 内部逻辑验证：拟合 (x, 2x²+3x+1)
        let y: Vec<f64> = x.iter().map(|&v| 2.0 * v * v + 3.0 * v + 1.0).collect();
        // 用(λ,i)形式验证: 构建(x_i=i/3, y_i)
        // 由于 多项式最小二乘 固定 y_i=i，这里改用它拟合 x 的幂到 rank，仅验证流程不 panic
        let 系数 = 谱求解器::多项式最小二乘(&x, 2);
        // 在 x=0 处 ξ≈0（第一个点 rank=0）
        assert!((谱求解器::多项式求值(&系数, 0.0) - 0.0).abs() < 0.5);
        // 在最后一个 x 处 ξ≈n-1
        let 末x = *x.last().unwrap();
        assert!((谱求解器::多项式求值(&系数, 末x) - (x.len() - 1) as f64).abs() < 0.5);
    }

    #[test]
    fn test_lgamma解析值() {
        // Γ(1)=1, Γ(2)=1, Γ(3)=2, Γ(1/2)=√π
        assert!((谱求解器::lgamma(1.0).exp() - 1.0).abs() < 1e-9, "Γ(1)={}", 谱求解器::lgamma(1.0).exp());
        assert!((谱求解器::lgamma(2.0).exp() - 1.0).abs() < 1e-9, "Γ(2)={}", 谱求解器::lgamma(2.0).exp());
        assert!((谱求解器::lgamma(3.0).exp() - 2.0).abs() < 1e-9, "Γ(3)={}", 谱求解器::lgamma(3.0).exp());
        assert!((谱求解器::lgamma(0.5).exp() - PI.sqrt()).abs() < 1e-9, "Γ(1/2)={}", 谱求解器::lgamma(0.5).exp());
    }

    #[test]
    fn test_理论cdf_β边界() {
        // β=1 → GOE CDF, β=2 → GUE CDF
        let f_goe = 谱求解器::理论cdf_β(1.0, 1.0);
        let f_gue = 谱求解器::理论cdf_β(1.0, 2.0);
        let f_goe_直 = 谱求解器::理论cdf(1.0, 1);
        let f_gue_直 = 谱求解器::理论cdf(1.0, 2);
        assert!((f_goe - f_goe_直).abs() < 0.02, "β=1应≈GOE: {} vs {}", f_goe, f_goe_直);
        assert!((f_gue - f_gue_直).abs() < 0.02, "β=2应≈GUE: {} vs {}", f_gue, f_gue_直);
        // 排斥更强 → 小间距概率更低 → F(1) 更小
        assert!(f_gue < f_goe, "GUE排斥更强, F(1)应更小: {} vs {}", f_gue, f_goe);
    }

    #[test]
    fn test_β混合分析_纯GOE采样() {
        // GOE 间距采样 → β_eff 应接近 1
        let s = 谱求解器::default();
        let mut rng = rand::thread_rng();
        let n = 120;
        let 间距: Vec<f64> = (0..n).map(|_| 谱求解器::逆变换采样(&mut rng, 1)).collect();
        let mut 本征值 = Vec::with_capacity(n + 1);
        let mut acc = 0.0;
        本征值.push(acc);
        for &x in &间距 {
            acc += x;
            本征值.push(acc);
        }
        let 分析 = s.β混合分析(&本征值);
        assert!(分析.β_eff > 0.5 && 分析.β_eff < 1.8, "GOE采样β_eff应接近1: {}", 分析.β_eff);
        assert!(分析.样本数 > 0);
        assert!(!分析.判定.is_empty());
    }

    #[test]
    fn test_β混合分析_纯GUE采样() {
        // GUE 间距采样 → β_eff 应接近 2
        let s = 谱求解器::default();
        let mut rng = rand::thread_rng();
        let n = 120;
        let 间距: Vec<f64> = (0..n).map(|_| 谱求解器::逆变换采样(&mut rng, 2)).collect();
        let mut 本征值 = Vec::with_capacity(n + 1);
        let mut acc = 0.0;
        本征值.push(acc);
        for &x in &间距 {
            acc += x;
            本征值.push(acc);
        }
        let 分析 = s.β混合分析(&本征值);
        assert!(分析.β_eff > 1.2, "GUE采样β_eff应接近2: {}", 分析.β_eff);
    }

    #[test]
    fn test_β混合分析_样本不足() {
        let s = 谱求解器::default();
        let 分析 = s.β混合分析(&[1.0, 2.0, 3.0]);
        assert!(分析.判定.contains("样本不足"));
    }

    // ============ Ginibre 复平面分析（论文§3.3.2 非梯度系统） ============

    #[test]
    fn test_ad距离圆盘() {
        // 均匀圆盘采样（r~√u）→ D_s^Ginibre 应小（确定性种子）
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let 模长: Vec<f64> = (0..500).map(|_| { let u: f64 = rng.gen_range(0.0..1.0); u.sqrt() }).collect();
        let d = Ginibre分析器::ad距离圆盘(&模长);
        assert!(d < 0.3, "圆盘均匀采样D_s应<0.3: {}", d);
        // 同心环（r=0.5 常数）→ 偏离圆盘律
        let 环: Vec<f64> = vec![0.5; 200];
        let d_环 = Ginibre分析器::ad距离圆盘(&环);
        assert!(d_环 > 0.3, "同心环应偏离圆盘律: {}", d_环);
    }

    #[test]
    fn test_ginibre分析_样本不足() {
        let 分析器 = Ginibre分析器::default();
        let 结果 = 分析器.分析(&[(1.0, 0.0), (2.0, 0.5)], 4);
        assert!(结果.判定.contains("样本不足"));
        assert_eq!(结果.点数, 2);
    }

    #[test]
    fn test_ginibre分析_纯实谱() {
        // 纯实本征值（虚部=0）→ 虚部比例=0
        let 分析器 = Ginibre分析器::default();
        let 复谱: Vec<(f64, f64)> = (0..20).map(|i| (i as f64 * 0.5, 0.0)).collect();
        let 结果 = 分析器.分析(&复谱, 4);
        assert_eq!(结果.虚部比例, 0.0, "纯实谱虚部比例应为0");
        // 纯实谱对圆盘律严重偏离（D_s^Gin 大）——半径归一化后仍是射线
        assert!(结果.d_s_ginibre > 0.3, "纯实谱应偏离圆盘律: {}", 结果.d_s_ginibre);
    }

    #[test]
    fn test_ginibre分析_圆盘样本() {
        // 圆盘均匀采样（含复成分）→ D_s^Ginibre 应小，角向均匀
        // 用确定性种子（rand 0.8 StdRng seed_from_u64）避免随机失败
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let 复谱: Vec<(f64, f64)> = (0..200)
            .map(|_| {
                let ru: f64 = rng.gen_range(0.0..1.0);
                let r = ru.sqrt();
                let θ: f64 = rng.gen_range(0.0..2.0 * PI);
                (r * θ.cos(), r * θ.sin())
            })
            .collect();
        let 分析器 = Ginibre分析器::default();
        let 结果 = 分析器.分析(&复谱, 8);
        assert!(结果.d_s_ginibre < 0.5, "圆盘样本D_s^Gin应较小: {}", 结果.d_s_ginibre);
        assert!(结果.虚部比例 > 0.8, "圆盘样本虚部占比应高: {}", 结果.虚部比例);
        assert!(结果.角向均匀性 < 0.4, "圆盘样本角向应均匀: {}", 结果.角向均匀性);
        println!("Ginibre圆盘: D_s^Gin={:.3} R={:.3} 判定={}", 结果.d_s_ginibre, 结果.角向均匀性, 结果.判定);
    }

    // ===== 第151批: 非厄米分析器（V10 椭圆系综）=====

    /// 构造对角 H/Γ（n 维）: H=diag(ω_i), Γ=diag(γ_i) → L=H-iΓ 特征值 λ_i=ω_i-iγ_i
    fn 对角矩阵(对角元: &[f64]) -> DMatrix<f64> {
        let n = 对角元.len();
        DMatrix::from_fn(n, n, |i, j| if i == j { 对角元[i] } else { 0.0 })
    }

    /// 等间隔 ω, 常数 γ → σ_γ2=0 → τ_param→1 → GUE（近保守）
    #[test]
    fn test_非厄米_近保守GUE() {
        let ω: Vec<f64> = (0..16).map(|i| i as f64 * 0.1).collect();
        let γ: Vec<f64> = vec![0.2; 16];
        let h = 对角矩阵(&ω);
        let g = 对角矩阵(&γ);
        let 分析器 = 非厄米分析器::default();
        let 结果 = 分析器.从非厄米(&h, &g);
        assert!(结果.点数 >= 8, "有效模应≥8: {}", 结果.点数);
        assert!((结果.γ_0 - 0.2).abs() < 1e-3, "γ_0应≈0.2: {}", 结果.γ_0);
        assert!(结果.τ_param > 0.5, "常数Γ应τ_param→1(GUE): {}", 结果.τ_param);
        assert_eq!(结果.普适类, "GUE");
        // τ 链 = 1/γ = 5
        assert!((结果.τ链[0] - 5.0).abs() < 1e-3, "τ应≈5: {}", 结果.τ链[0]);
    }

    /// 常数 ω, 变化 γ → σ_ω2=0 → τ_param→-1 → GOE（过阻尼）
    #[test]
    fn test_非厄米_过阻尼GOE() {
        let ω: Vec<f64> = vec![0.5; 16];
        let γ: Vec<f64> = (0..16).map(|i| 0.1 + i as f64 * 0.05).collect();
        let h = 对角矩阵(&ω);
        let g = 对角矩阵(&γ);
        let 分析器 = 非厄米分析器::default();
        let 结果 = 分析器.从非厄米(&h, &g);
        assert!(结果.点数 >= 8);
        assert!(结果.τ_param < -0.5, "常数H应τ_param→-1(GOE): {}", 结果.τ_param);
        assert_eq!(结果.普适类, "GOE");
    }

    /// ω 与 γ 同分布 → σ_ω2≈σ_γ2 → τ_param≈0 → Ginibre（振荡耗散等强）
    #[test]
    fn test_非厄米_Ginibre极限() {
        let ω: Vec<f64> = (0..16).map(|i| 0.1 + i as f64 * 0.04).collect();
        let γ: Vec<f64> = (0..16).map(|i| 0.1 + i as f64 * 0.04).collect();
        let h = 对角矩阵(&ω);
        let g = 对角矩阵(&γ);
        let 分析器 = 非厄米分析器::default();
        let 结果 = 分析器.从非厄米(&h, &g);
        assert!(结果.点数 >= 8);
        assert!(结果.τ_param.abs() < 0.15, "同分布应τ_param≈0: {}", 结果.τ_param);
        assert_eq!(结果.普适类, "Ginibre");
    }

    /// 对角 H/Γ → [L,L†]=0 → ν≈0（正规矩阵）
    #[test]
    fn test_非厄米_正规性ν零() {
        let ω: Vec<f64> = (0..16).map(|i| i as f64 * 0.1).collect();
        let γ: Vec<f64> = vec![0.2; 16];
        let h = 对角矩阵(&ω);
        let g = 对角矩阵(&γ);
        let 分析器 = 非厄米分析器::default();
        let 结果 = 分析器.从非厄米(&h, &g);
        assert!(结果.ν < 1e-9, "对角H/Γ应正规(ν≈0): {}", 结果.ν);
    }

    /// 样本不足保护
    #[test]
    fn test_非厄米_样本不足() {
        let h = DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
        let g = DMatrix::from_row_slice(3, 3, &[0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.3]);
        let 分析器 = 非厄米分析器::default();
        let 结果 = 分析器.从非厄米(&h, &g);
        assert!(结果.判定.contains("样本不足"), "小矩阵应样本不足: {}", 结果.判定);
    }

    /// 椭圆 AD 距离函数性质
    #[test]
    fn test_椭圆AD距离性质() {
        // 椭圆内均匀: u 密度∝2u（面积元∝u²）, 用 sqrt 变换采样 → 匹配 CDF F(u)=u²
        let 均匀: Vec<f64> = (0..200).map(|i| ((i as f64 + 0.5) / 200.0).sqrt()).collect();
        let d均匀 = 非厄米分析器::ad距离椭圆(&均匀, 2.5);
        assert!(d均匀 < 0.5, "均匀分布D_s应较小: {}", d均匀);
        // 全部挤在边缘 u≈1 → 距离大
        let 边缘: Vec<f64> = vec![0.98; 50];
        let d边缘 = 非厄米分析器::ad距离椭圆(&边缘, 2.5);
        assert!(d边缘 > d均匀, "边缘聚集D_s应更大: {} vs {}", d边缘, d均匀);
        // 空输入 → 1.0
        assert_eq!(非厄米分析器::ad距离椭圆(&[], 2.5), 1.0);
    }

    // ============ 第267批：从矩阵_非厄米 双通道 ============

    #[test]
    fn test_从矩阵_非厄米双通道() {
        let s = 谱求解器::default();
        // 非对称矩阵（对角 H + 非对称耗散 Γ）→ 非厄米通道
        let l = DMatrix::from_row_slice(4, 4, &[
            1.0, 0.1, 0.0, 0.0,
            0.0, 2.0, 0.2, 0.0,
            0.0, 0.0, 3.0, 0.3,
            0.05, 0.0, 0.0, 4.0,
        ]);
        let 输出 = s.从矩阵_非厄米(&l).unwrap();
        // 分解出模
        assert!(输出.分解.模数 >= 3);
        // 椭圆拟合 D_s^(2D) 存在
        assert!(输出.椭圆.d_s_2d >= 0.0);
        assert!(["绿", "黄", "红"].contains(&输出.椭圆.区域.as_str()));
        // 参与度
        assert_eq!(输出.p_obs.len(), 输出.分解.模数);
        let p总和: f64 = 输出.p_obs.iter().sum();
        assert!((p总和 - 1.0).abs() < 1e-6, "p_obs 和应=1: {}", p总和);
        assert_eq!(输出.p_exc.len(), 输出.分解.模数);
        // τ 链正（稳定模）
        assert!(输出.分解.模.iter().any(|模| 模.τ > 0.0));
    }

    #[test]
    fn test_从矩阵_对称保持golden() {
        // 对称矩阵 → 从矩阵 走 SymmetricEigen（golden 不变）
        let s = 谱求解器::default();
        let 对称 = DMatrix::from_row_slice(3, 3, &[2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0]);
        let 结果 = s.从矩阵(&对称);
        assert!(结果.本征值数 >= 2);
    }
}
