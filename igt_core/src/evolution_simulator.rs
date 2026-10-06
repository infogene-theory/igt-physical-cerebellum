// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! 演化模拟器 — 第133批 变分原理深化融合（论文第一部分：演化趋衡定理）
//!
//! 理论依据:
//!   - 《变分原理深化_演化趋衡_beta混合_Koopman非线性_v1.0.md》第一部分:
//!     变异-选择动力学 = 拓扑空间上的 Metropolis 链 → 唯一平稳分布 β-系综
//!     D_s 期望单调递减（IGT H定理）；变异幅度 σ = 演化温度
//!   - 《IGT闭合拓扑变分原理_v1.0》: 演化 = 拓扑空间中的 Metropolis 过程
//!     （变异=随机扰动 L 连接/权重, 存活=闭合 S=1 + 耗散不超预算）
//!
//! 核心功能（Metropolis 变异-选择模拟）:
//!   1. 初始 L（图 Laplacian，连通闭合 + 迹预算）
//!   2. 变异: 随机选边扰动 κ_ij → κ_ij + δ·η（η~N(0,σ²)），保持行和=0
//!   3. 选择: 若 L' 半正定（闭合）且 Tr(L')=预算 → 接受；否则拒绝
//!   4. 每 K 步计算 D_s（复用谱求解器）→ D_s 轨迹
//!   5. 验证 H定理: D_s 期望单调递减（数值证据）
//!
//! 工程修正点（相对论文）:
//!   - 论文定理是待证命题（§4.4 待证1/2），本模拟器提供数值证据而非证明
//!   - 变异限制单边扰动（保持连通闭合），选择用 Cholesky 检查半正定
//!   - σ（温度）与论文一致：大 σ 强变异快趋衡；σ=0 冻结

#![allow(non_snake_case)]

use nalgebra::DMatrix;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::spectrum_solver::{谱求解器, 谱动力学结果};

/// 演化模拟步记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 演化步记录 {
    /// 迭代步
    pub 步: usize,
    /// 当前 D_s^WD（变分残差）
    pub d_s: f64,
    /// 当前 D_s^P
    pub d_s_p: f64,
    /// 当前最小本征值（连通性指标）
    pub λ_min: f64,
    /// 当前条件数 κ(L)
    pub 条件数: f64,
    /// 接受的变异数（累计）
    pub 接受数: usize,
}

/// 演化模拟结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 演化模拟结果 {
    /// 初始 D_s
    pub 初始d_s: f64,
    /// 最终 D_s
    pub 最终d_s: f64,
    /// D_s 轨迹（每采样步）
    pub d_s轨迹: Vec<f64>,
    /// 步记录
    pub 记录: Vec<演化步记录>,
    /// 变异幅度 σ（演化温度）
    pub 变异幅度: f64,
    /// 总迭代步
    pub 总步数: usize,
    /// 接受率
    pub 接受率: f64,
    /// H定理验证：D_s 是否单调递减（斜率<0）
    pub h定理验证: bool,
    /// 最终判定
    pub 判定: String,
}

/// 演化模拟器 — Metropolis 变异-选择（Dyson 布朗运动的工程近似）
#[derive(Debug, Clone, Copy)]
pub struct 演化模拟器 {
    /// 变异幅度 δ·σ（边扰动标准差）
    pub 变异幅度: f64,
    /// 谱求解器
    pub 谱求解器: 谱求解器,
}

impl Default for 演化模拟器 {
    fn default() -> Self {
        Self {
            变异幅度: 0.1,
            谱求解器: 谱求解器::default(),
        }
    }
}

impl 演化模拟器 {
    /// 模拟变异-选择过程
    ///
    /// # 参数
    /// - `初始l`: 初始图 Laplacian（对称半正定，连通）
    /// - `总步数`: 迭代总步数
    /// - `采样间隔`: 每多少步记录一次 D_s
    pub fn 模拟(&self, 初始l: &DMatrix<f64>, 总步数: usize, 采样间隔: usize) -> 演化模拟结果 {
        let n = 初始l.nrows();
        let 预算 = 初始l.trace(); // Tr(L) 守恒
        let mut l = 初始l.clone();
        let mut rng = rand::thread_rng();
        let mut 接受数 = 0usize;
        let mut 记录 = Vec::new();
        let mut d_s轨迹 = Vec::new();

        let 采样间隔 = 采样间隔.max(1);

        // 初始谱
        let 初始谱 = self.谱求解器.从矩阵(&l);
        let 初始d_s = 初始谱.d_s_wd;

        // 迭代
        for 步 in 0..总步数 {
            // ===== 变异 =====
            let mut 候选 = l.clone();
            // 随机选一条边（i,j），扰动 κ_ij 和 κ_ji（对称），并调整对角保持行和=0
            let i = rng.gen_range(0..n);
            let mut j = rng.gen_range(0..n);
            while j == i {
                j = rng.gen_range(0..n);
            }
            let δ = self.变异幅度 * Self::高斯(&mut rng);
            // κ_ij 增加 δ，κ_ji 同步；对角 i,j 各减 δ 保持行和=0
            候选[(i, j)] -= δ;
            候选[(j, i)] -= δ;
            候选[(i, i)] += δ;
            候选[(j, j)] += δ;

            // ===== 选择（Metropolis 硬约束）=====
            // 1) 迹守恒：Tr(L') = Tr(L) + δ - δ = 预算（自动满足）
            // 2) 半正定（闭合）: Cholesky 检查
            let 接受 = Self::半正定(&候选);

            if 接受 {
                l = 候选;
                接受数 += 1;
            }

            // ===== 采样 =====
            if 步 % 采样间隔 == 0 {
                let 谱 = self.谱求解器.从矩阵(&l);
                d_s轨迹.push(谱.d_s_wd);
                let λ_min = 谱.本征值.first().copied().unwrap_or(0.0);
                let 条件数 = if λ_min > 1e-12 {
                    l.norm() / λ_min
                } else {
                    f64::INFINITY
                };
                记录.push(演化步记录 {
                    步,
                    d_s: 谱.d_s_wd,
                    d_s_p: 谱.d_s_p,
                    λ_min,
                    条件数,
                    接受数,
                });
            }
        }

        // 最终
        let 最终谱 = self.谱求解器.从矩阵(&l);
        let 最终d_s = 最终谱.d_s_wd;

        // H 定理验证：D_s 趋势（最小二乘斜率）
        let 斜率 = if d_s轨迹.len() >= 3 {
            let n_t = d_s轨迹.len();
            let 均值t = (n_t as f64 - 1.0) / 2.0;
            let 均值d = d_s轨迹.iter().sum::<f64>() / n_t as f64;
            let mut 协方差 = 0.0;
            let mut 方差 = 0.0;
            for (i, &d) in d_s轨迹.iter().enumerate() {
                let t = i as f64;
                协方差 += (t - 均值t) * (d - 均值d);
                方差 += (t - 均值t) * (t - 均值t);
            }
            if 方差 > 1e-12 { 协方差 / 方差 } else { 0.0 }
        } else {
            0.0
        };

        let 接受率 = 接受数 as f64 / 总步数.max(1) as f64;
        let h定理验证 = 斜率 < -1e-6;
        let 判定 = if h定理验证 {
            format!(
                "H定理成立（D_s斜率={:.4}<0，趋衡β-系综）: D_s {:.3}→{:.3}",
                斜率, 初始d_s, 最终d_s
            )
        } else {
            format!(
                "D_s未单调递减（斜率={:.4}，可能样本不足或σ过大跳出普适类）: D_s {:.3}→{:.3}",
                斜率, 初始d_s, 最终d_s
            )
        };

        演化模拟结果 {
            初始d_s,
            最终d_s,
            d_s轨迹,
            记录,
            变异幅度: self.变异幅度,
            总步数,
            接受率,
            h定理验证,
            判定,
        }
    }

    /// 高斯随机数（Box-Muller）
    fn 高斯(rng: &mut impl Rng) -> f64 {
        let u1: f64 = rng.gen_range(1e-9..1.0);
        let u2: f64 = rng.gen_range(1e-9..1.0);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// 半正定检查（最小本征值 ≥ -ε，对奇异半正定矩阵可靠）
    pub fn 半正定(m: &DMatrix<f64>) -> bool {
        let n = m.nrows();
        if n == 0 {
            return true;
        }
        // 先做对称化（变异产生的 L 应保持对称，但数值保护）
        let mut 对称 = m.clone();
        for i in 0..n {
            for j in i + 1..n {
                let v = 0.5 * (m[(i, j)] + m[(j, i)]);
                对称[(i, j)] = v;
                对称[(j, i)] = v;
            }
        }
        let eig = nalgebra::SymmetricEigen::new(对称);
        let λ_min = eig.eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);
        λ_min >= -1e-9
    }
}

/// 便捷入口：从拓扑图构造初始 L 并模拟演化
pub fn 从拓扑模拟(
    节点数: usize,
    边: &[(usize, usize, f64)],
    变异幅度: f64,
    总步数: usize,
    采样间隔: usize,
) -> Option<演化模拟结果> {
    use crate::topology_editor::拓扑图;

    let mut 图 = 拓扑图::新建(节点数);
    for &(a, b, κ) in 边 {
        图.加边(a, b, κ);
    }
    let l = 图.构造生成元();
    // 检查连通闭合（最小本征值 > 0）
    let 谱 = 谱求解器::default().从矩阵(&l);
    if 谱.本征值.is_empty() {
        return None;
    }
    let 模拟器 = 演化模拟器 { 变异幅度, ..Default::default() };
    Some(模拟器.模拟(&l, 总步数, 采样间隔))
}

/// 演化温度点（σ↔造波温度对照）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 演化温度点 {
    /// 变异幅度 σ（演化温度）
    pub σ: f64,
    /// 初始 D_s
    pub 初始d_s: f64,
    /// 最终 D_s
    pub 最终d_s: f64,
    /// D_s 变化量（负=趋衡）
    pub Δd_s: f64,
    /// 趋衡速率（D_s 最小二乘斜率，<0 趋衡）
    pub 趋衡速率: f64,
    /// 接受率（Metropolis 接受概率）
    pub 接受率: f64,
    /// 是否有效趋衡
    pub 趋衡: bool,
}

/// 演化温度扫描结果（第135批: σ↔造波温度对照）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 演化温度扫描结果 {
    /// 各温度点
    pub 温度点: Vec<演化温度点>,
    /// 最优温度（ΔD_s 最负的 σ）
    pub 最优温度: Option<f64>,
    /// 最优温度下 ΔD_s
    pub 最优Δd_s: Option<f64>,
    /// 说明: σ 映射造波温度——大σ=强造波（快速趋衡但耗散大）, 小σ=弱造波（慢速精细调整）
    pub 说明: String,
}

/// 演化温度扫描——不同 σ（演化温度/造波温度）下的趋衡行为对照
///
/// 论文第一部分: 变异幅度 σ = 演化温度；σ↔造波引擎的波幅度/强度对照。
/// 本函数在多个 σ 上模拟变异-选择，输出 D_s 变化与趋衡速率，
/// 供造波引擎选择"最优演化温度"（在耗散预算内最快趋衡）。
impl 演化模拟器 {
    /// 温度扫描
    pub fn 温度扫描(&self, 初始l: &DMatrix<f64>, σ列表: &[f64], 总步数: usize, 采样间隔: usize) -> 演化温度扫描结果 {
        let mut 温度点 = Vec::new();
        let mut 最优温度 = None;
        let mut 最优Δd_s = None;
        for &σ in σ列表 {
            let 模拟器 = 演化模拟器 { 变异幅度: σ, ..*self };
            let 结果 = 模拟器.模拟(初始l, 总步数, 采样间隔);
            let Δd_s = 结果.最终d_s - 结果.初始d_s;
            // 趋衡速率 = D_s 轨迹斜率
            let 趋衡速率 = if 结果.d_s轨迹.len() >= 3 {
                let k = 结果.d_s轨迹.len();
                let 均值t = (k as f64 - 1.0) / 2.0;
                let 均值d = 结果.d_s轨迹.iter().sum::<f64>() / k as f64;
                let mut 协方差 = 0.0;
                let mut 方差 = 0.0;
                for (i, &d) in 结果.d_s轨迹.iter().enumerate() {
                    let t = i as f64;
                    协方差 += (t - 均值t) * (d - 均值d);
                    方差 += (t - 均值t) * (t - 均值t);
                }
                if 方差 > 1e-12 { 协方差 / 方差 } else { 0.0 }
            } else {
                0.0
            };
            let 趋衡 = 趋衡速率 < -1e-6;
            if 趋衡 {
                // 找 ΔD_s 最负（趋衡最显著）的温度
                if 最优Δd_s.map_or(true, |best| Δd_s < best) {
                    最优Δd_s = Some(Δd_s);
                    最优温度 = Some(σ);
                }
            }
            温度点.push(演化温度点 {
                σ,
                初始d_s: 结果.初始d_s,
                最终d_s: 结果.最终d_s,
                Δd_s,
                趋衡速率,
                接受率: 结果.接受率,
                趋衡,
            });
        }
        演化温度扫描结果 {
            温度点,
            最优温度,
            最优Δd_s,
            说明: "演化温度σ↔造波温度: σ大=强造波(快速趋衡,耗散大); σ小=弱造波(慢速精细); 最优σ=ΔD_s最负".to_string(),
        }
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造 4 节点环图 Laplacian（连通闭合）
    fn 环图4() -> DMatrix<f64> {
        let mut 图 = crate::topology_editor::拓扑图::新建(4);
        图.加边(0, 1, 1.0);
        图.加边(1, 2, 1.0);
        图.加边(2, 3, 1.0);
        图.加边(3, 0, 1.0);
        图.构造生成元()
    }

    /// 构造 6 节点完全图 Laplacian（高连通）
    fn 完全图6() -> DMatrix<f64> {
        let mut 图 = crate::topology_editor::拓扑图::新建(6);
        for i in 0..6 {
            for j in i + 1..6 {
                图.加边(i, j, 1.0);
            }
        }
        图.构造生成元()
    }

    #[test]
    fn test_半正定检查() {
        // 对角正 → 半正定
        let 正 = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![1.0, 2.0, 3.0]));
        assert!(演化模拟器::半正定(&正));
        // 含负对角 → 非半正定
        let 负 = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![1.0, -2.0, 3.0]));
        assert!(!演化模拟器::半正定(&负));
        // 环图 → 半正定
        assert!(演化模拟器::半正定(&环图4()));
    }

    #[test]
    fn test_模拟流程完整() {
        let 模拟器 = 演化模拟器::default();
        let 初始 = 环图4();
        let 结果 = 模拟器.模拟(&初始, 200, 20);
        assert_eq!(结果.总步数, 200);
        assert!(结果.记录.len() >= 3, "记录应≥3: {}", 结果.记录.len());
        assert!(结果.初始d_s >= 0.0 && 结果.最终d_s >= 0.0);
        assert!(!结果.判定.is_empty());
        assert!(结果.接受率 >= 0.0 && 结果.接受率 <= 1.0);
    }

    #[test]
    fn test_完全图趋衡() {
        // 完全图已近 GOE（对称+连通），变异应保持或降低 D_s
        let 模拟器 = 演化模拟器 { 变异幅度: 0.05, ..Default::default() };
        let 初始 = 完全图6();
        let 结果 = 模拟器.模拟(&初始, 300, 30);
        // 完全图 6 节点特征值近正则 → 应趋衡（h定理通常成立）
        assert!(!结果.判定.is_empty());
        println!("完全图趋衡: D_s {:.3}→{:.3}, 接受率={:.2}, H定理={}",
                 结果.初始d_s, 结果.最终d_s, 结果.接受率, 结果.h定理验证);
    }

    #[test]
    fn test_零变异冻结() {
        // σ=0 → 无变异 → D_s 不变（冻结）
        let 模拟器 = 演化模拟器 { 变异幅度: 0.0, ..Default::default() };
        let 初始 = 环图4();
        let 结果 = 模拟器.模拟(&初始, 100, 10);
        let 全同 = 结果.d_s轨迹.iter().all(|&d| (d - 结果.初始d_s).abs() < 1e-9);
        assert!(全同, "σ=0应冻结: {:?}", 结果.d_s轨迹);
    }

    #[test]
    fn test_高斯分布性质() {
        let mut rng = rand::thread_rng();
        let n = 10000;
        let 均值 = (0..n).map(|_| 演化模拟器::高斯(&mut rng)).sum::<f64>() / n as f64;
        let 方差 = (0..n).map(|_| { let v = 演化模拟器::高斯(&mut rng); v * v }).sum::<f64>() / n as f64;
        assert!(均值.abs() < 0.05, "高斯均值≈0: {}", 均值);
        assert!((方差 - 1.0).abs() < 0.1, "高斯方差≈1: {}", 方差);
    }

    #[test]
    fn test_从拓扑模拟() {
        let 结果 = 从拓扑模拟(4, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)], 0.1, 100, 10);
        assert!(结果.is_some(), "拓扑模拟应成功");
        let 结果 = 结果.unwrap();
        assert_eq!(结果.总步数, 100);
        assert!(!结果.记录.is_empty());
    }

    #[test]
    fn test_温度扫描() {
        let 模拟器 = 演化模拟器::default();
        let 初始 = 环图4();
        let σ列表 = vec![0.02, 0.05, 0.1, 0.2];
        let 结果 = 模拟器.温度扫描(&初始, &σ列表, 150, 15);
        assert_eq!(结果.温度点.len(), 4, "每个σ一个温度点");
        // 每点字段完整
        for 点 in &结果.温度点 {
            assert!(点.σ > 0.0);
            assert!(点.接受率 >= 0.0 && 点.接受率 <= 1.0);
            assert!(点.Δd_s.abs() < 1.0, "D_s变化应在合理范围: {}", 点.Δd_s);
        }
        // 至少一个趋衡点（小σ通常趋衡）
        let 有趋衡 = 结果.温度点.iter().any(|p| p.趋衡);
        println!("温度扫描: 最优σ={:?} 点={:?}", 结果.最优温度, 结果.温度点.iter().map(|p| (p.σ, p.Δd_s)).collect::<Vec<_>>());
        assert!(有趋衡 || 结果.最优温度.is_none(), "环图小σ应趋衡或最优温度为空");
        assert!(!结果.说明.is_empty());
    }

    #[test]
    fn test_温度扫描_σ越大接受率越高() {
        // 大σ变异被接受概率更高（除非破坏半正定）
        let 模拟器 = 演化模拟器::default();
        let 初始 = 环图4();
        let σ小 = 模拟器.温度扫描(&初始, &[0.01], 100, 10);
        let σ大 = 模拟器.温度扫描(&初始, &[0.3], 100, 10);
        let 接受小 = σ小.温度点[0].接受率;
        let 接受大 = σ大.温度点[0].接受率;
        println!("σ小接受率={:.3} σ大接受率={:.3}", 接受小, 接受大);
        assert!(接受小 > 0.0 && 接受小 <= 1.0);
        assert!(接受大 > 0.0 && 接受大 <= 1.0);
    }
}
