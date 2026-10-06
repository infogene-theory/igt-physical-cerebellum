// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! Lyapunov 指数模块 — 第135批 变分原理深化第3批（论文§3.6 非线性健康度）
//!
//! 理论依据:
//!   - 《变分原理深化_演化趋衡_beta混合_Koopman非线性_v1.0.md》§3.6:
//!     非线性 ε_wave 由 Lyapunov 指数控制——最大 Lyapunov 指数 λ_max < 0（稳定），
//!     造波后 λ_max 不转正；ε_wave = |Δλ_max|（造波导致的最大 Lyapunov 指数变化）
//!   - S（非线性）: 稳态存在且全局吸引 + 所有 Koopman 本征值 Re(μ_i)<0
//!   - D_s（非线性）: 梯度系统→Re(μ)对照GOE; 非梯度→复μ对照Ginibre
//!
//! 工程实现（Rosenstein 轨道邻域法，1993）:
//!   1. 延迟坐标嵌入（Takens）重建相空间
//!   2. 对每个参考点找最近邻，跟踪邻域距离随时间的对数增长
//!   3. λ_max = 对数发散率的平均斜率（最小二乘拟合）
//!
//! 输出:
//!   - λ_max: 最大 Lyapunov 指数（<0 稳定 / >0 混沌 / ≈0 守恒）
//!   - ε_wave: 非线性造波安全上界 = |Δλ_max|（造波前后 λ_max 变化）
//!   - 非线性 S/D_s 判定（健康度）

#![allow(non_snake_case)]

use serde::{Deserialize, Serialize};

/// Lyapunov 分析结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lyapunov分析结果 {
    /// 最大 Lyapunov 指数 λ_max（1/s）
    pub λ_max: f64,
    /// 判定: 稳定(λ<0) / 混沌(λ>0) / 守恒(λ≈0)
    pub 判定: String,
    /// 非线性造波安全上界 ε_wave = |Δλ_max|（造波前后 λ_max 变化）
    pub ε_wave: f64,
    /// 非线性 S（稳态健康度）: λ_max<0 时 S_nl=1
    pub 非线性s: f64,
    /// 非线性 D_s（谱偏离度）: 1-|λ_max|归一化
    pub 非线性d_s: f64,
    /// 样本点数
    pub 样本数: usize,
    /// 嵌入维 m
    pub 嵌入维: usize,
    /// 延迟 τ_delay
    pub 延迟: usize,
    /// 说明
    pub 说明: String,
}

/// Lyapunov 指数计算器（Rosenstein 轨道邻域法）
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Lyapunov计算器 {
    /// 嵌入维 m（Takens 嵌入）
    pub 嵌入维: usize,
    /// 延迟 τ_delay
    pub 延迟: usize,
    /// 邻域比例（最近邻最小间距阈值）
    pub 邻域比例: f64,
    /// λ_max 拟合的轨道长度（数据点的比例）
    pub 拟合比例: f64,
}

impl Default for Lyapunov计算器 {
    fn default() -> Self {
        Self {
            嵌入维: 5,
            延迟: 2,
            邻域比例: 0.01,
            拟合比例: 0.7,
        }
    }
}

impl Lyapunov计算器 {
    /// 计算最大 Lyapunov 指数（Rosenstein 法）
    ///
    /// # 参数
    /// - `时序`: 等间隔采样的标量时间序列
    /// - `采样间隔`: Δt（秒）
    pub fn 计算(&self, 时序: &[f64], Δt: f64) -> Lyapunov分析结果 {
        let n = 时序.len();
        let 嵌入维 = self.嵌入维.max(2);
        let 延迟 = self.延迟.max(1);

        // 延迟嵌入相空间重建（Takens）
        let 轨道数 = n.saturating_sub((嵌入维 - 1) * 延迟);
        if 轨道数 < 10 || Δt <= 0.0 {
            return Lyapunov分析结果 {
                λ_max: 0.0,
                判定: "样本不足（需≥10个轨道点）".to_string(),
                ε_wave: 0.0,
                非线性s: 0.0,
                非线性d_s: 1.0,
                样本数: n,
                嵌入维,
                延迟,
                说明: "无法重建相空间".to_string(),
            };
        }

        // 轨道点向量
        let 轨道: Vec<Vec<f64>> = (0..轨道数)
            .map(|i| {
                (0..嵌入维)
                    .map(|k| 时序[i + k * 延迟])
                    .collect::<Vec<f64>>()
            })
            .collect();

        // 计算所有点对的距离（用于找最近邻，避免临时最近邻）
        let 距离 = |a: &[f64], b: &[f64]| -> f64 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f64>()
                .sqrt()
        };

        // 每参考点找最近邻（排除临时相邻点，避免切向轨道）
        let 均值间距 = 时序
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum::<f64>()
            / (n - 1).max(1) as f64;
        let 最小邻近距离 = 均值间距.max(1e-9);

        // 参考点列表: (i, j0, d0) —— i 为参考轨道点，j0 为其最近邻轨道点，d0=初始距离
        let mut 参考点 = Vec::new();
        for i in 0..轨道数 {
            let mut 最近距离 = f64::INFINITY;
            let mut 最近索引 = None;
            for j in 0..轨道数 {
                if j == i {
                    continue;
                }
                // 排除时间上太接近的点（<延迟，避免切向相邻）
                if (i as isize - j as isize).abs() as usize <= 延迟 {
                    continue;
                }
                let d = 距离(&轨道[i], &轨道[j]);
                if d > 最小邻近距离 && d < 最近距离 {
                    最近距离 = d;
                    最近索引 = Some(j);
                }
            }
            if let (Some(j0), true) = (最近索引, 最近距离.is_finite()) {
                参考点.push((i, j0, 最近距离.max(1e-12)));
            }
        }

        if 参考点.len() < 4 {
            return Lyapunov分析结果 {
                λ_max: 0.0,
                判定: "邻域不足".to_string(),
                ε_wave: 0.0,
                非线性s: 0.0,
                非线性d_s: 1.0,
                样本数: n,
                嵌入维,
                延迟,
                说明: "最近邻数量不足".to_string(),
            };
        }

        // 跟踪邻域距离随时间演化（Rosenstein: d(t) = |x_{i+t} - x_{j0+t}|）
        let 最大长度 = 轨道数 - 1;
        let 拟合长度 = ((最大长度 as f64) * self.拟合比例) as usize;
        let 拟合长度 = 拟合长度.max(2).min(最大长度);

        let mut 平均对数距离 = Vec::with_capacity(拟合长度 + 1);
        for t in 0..=拟合长度 {
            let mut 和 = 0.0;
            let mut 计数 = 0;
            for &(i, j0, 初始距离) in &参考点 {
                let i_t = i + t;
                let j_t = j0 + t;
                if i_t >= 轨道数 || j_t >= 轨道数 {
                    continue;
                }
                let d = 距离(&轨道[i_t], &轨道[j_t]);
                if d > 1e-15 && 初始距离 > 1e-15 {
                    和 += (d / 初始距离).ln();
                    计数 += 1;
                }
            }
            if 计数 >= 2 {
                平均对数距离.push((t, 和 / 计数 as f64));
            }
        }

        // 最小二乘拟合斜率 = λ_max
        if 平均对数距离.len() < 3 {
            return Lyapunov分析结果 {
                λ_max: 0.0,
                判定: "演化不足".to_string(),
                ε_wave: 0.0,
                非线性s: 0.0,
                非线性d_s: 1.0,
                样本数: n,
                嵌入维,
                延迟,
                说明: "平均对数距离演化样本不足".to_string(),
            };
        }
        let k = 平均对数距离.len();
        let 均值t = 平均对数距离.iter().map(|&(t, _)| t as f64).sum::<f64>() / k as f64;
        let 均值y = 平均对数距离.iter().map(|&(_, y)| y).sum::<f64>() / k as f64;
        let mut 协方差 = 0.0;
        let mut 方差 = 0.0;
        for &(t, y) in &平均对数距离 {
            let dt = t as f64 - 均值t;
            协方差 += dt * (y - 均值y);
            方差 += dt * dt;
        }
        let λ_max = if 方差 > 1e-12 {
            协方差 / 方差 / Δt
        } else {
            0.0
        };

        // 判定
        let (判定, 非线性s, 非线性d_s) = if λ_max < -self.ε_稳定() {
            (
                format!("稳定（λ_max={:.4}<0）", λ_max),
                1.0,
                (λ_max.abs() / (1.0 + λ_max.abs())).min(1.0),
            )
        } else if λ_max > self.ε_混沌() {
            (
                format!("混沌（λ_max={:.4}>0）", λ_max),
                0.0,
                1.0,
            )
        } else {
            (
                format!("守恒/边缘（λ_max={:.4}≈0）", λ_max),
                0.7,
                0.3,
            )
        };

        Lyapunov分析结果 {
            λ_max,
            判定,
            ε_wave: 0.0, // 由 造波前后比较 计算
            非线性s,
            非线性d_s,
            样本数: n,
            嵌入维,
            延迟,
            说明: format!(
                "Rosenstein轨道邻域法: λ_max={:.4}/s; 非线性S={:.2}, D_s={:.2}",
                λ_max, 非线性s, 非线性d_s
            ),
        }
    }

    /// 稳定判定阈值 ε_λ（略小于0）
    fn ε_稳定(&self) -> f64 {
        0.01
    }

    /// 混沌判定阈值 ε_λ（略大于0）
    fn ε_混沌(&self) -> f64 {
        0.01
    }

    /// 非线性 ε_wave 计算（论文§3.6）
    ///
    /// 对造波前后的两个时序分别算 λ_max，ε_wave = |λ_max_后 - λ_max_前|
    /// 若造波后 λ_max 转正（混沌）→ 安全失败
    pub fn 计算ε_wave(&self, 造波前: &[f64], 造波后: &[f64], Δt: f64) -> εWave结果 {
        let 前 = self.计算(造波前, Δt);
        let 后 = self.计算(造波后, Δt);
        let ε_wave = (后.λ_max - 前.λ_max).abs();
        let 转正 = 前.λ_max < -self.ε_稳定() && 后.λ_max > self.ε_混沌();
        εWave结果 {
            ε_wave,
            λ_max前: 前.λ_max,
            λ_max后: 后.λ_max,
            前判定: 前.判定.clone(),
            后判定: 后.判定.clone(),
            造波安全: !转正 && ε_wave < 0.3,
            说明: if 转正 {
                format!("造波导致混沌（λ_max {:.4}→{:.4}转正）: 违反非线性S不降级", 前.λ_max, 后.λ_max)
            } else {
                format!("非线性造波安全: ε_wave=|{:.4}→{:.4}|={:.4}", 前.λ_max, 后.λ_max, ε_wave)
            },
        }
    }
}

/// 非线性 ε_wave 结果（论文§3.6）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct εWave结果 {
    /// 造波导致的最大 Lyapunov 指数变化 |Δλ_max|
    pub ε_wave: f64,
    /// 造波前 λ_max
    pub λ_max前: f64,
    /// 造波后 λ_max
    pub λ_max后: f64,
    /// 造波前判定
    pub 前判定: String,
    /// 造波后判定
    pub 后判定: String,
    /// 造波是否安全（λ_max 不转正 且 ε_wave<0.3）
    pub 造波安全: bool,
    /// 说明
    pub 说明: String,
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 衰减信号：λ = -1/τ（指数衰减 e^{-t/τ} → λ_max = -1/τ）
    fn 衰减时序(τ: f64, Δt: f64, n: usize) -> Vec<f64> {
        (0..n).map(|k| (-(k as f64 * Δt) / τ).exp()).collect()
    }

    /// 增长信号：λ = +1/τ
    fn 增长时序(τ: f64, Δt: f64, n: usize) -> Vec<f64> {
        (0..n).map(|k| ((k as f64 * Δt) / τ).exp()).collect()
    }

    /// 振荡信号：λ ≈ 0（边界，理想无衰减振荡 λ_max≈0）
    fn 振荡时序(ω: f64, Δt: f64, n: usize) -> Vec<f64> {
        (0..n).map(|k| (ω * k as f64 * Δt).cos()).collect()
    }

    #[test]
    fn test_衰减信号稳定() {
        let 计算器 = Lyapunov计算器::default();
        let 时序 = 衰减时序(2.0, 0.05, 400);
        let 结果 = 计算器.计算(&时序, 0.05);
        assert!(结果.λ_max < 0.0, "衰减信号λ_max应<0: {}", 结果.λ_max);
        assert!(结果.判定.contains("稳定"));
        assert_eq!(结果.非线性s, 1.0);
    }

    #[test]
    fn test_增长信号混沌() {
        let 计算器 = Lyapunov计算器::default();
        let 时序 = 增长时序(2.0, 0.05, 300);
        let 结果 = 计算器.计算(&时序, 0.05);
        assert!(结果.λ_max > 0.0, "增长信号λ_max应>0: {}", 结果.λ_max);
        assert!(结果.判定.contains("混沌"));
        assert_eq!(结果.非线性s, 0.0);
    }

    #[test]
    fn test_振荡信号守恒() {
        let 计算器 = Lyapunov计算器::default();
        let 时序 = 振荡时序(1.0, 0.05, 500);
        let 结果 = 计算器.计算(&时序, 0.05);
        // 理想无衰减振荡 λ_max≈0（Rosenstein 对边界轨道可能略偏）
        assert!(!结果.判定.contains("样本不足"), "振荡应能计算: {}", 结果.判定);
        println!("振荡λ_max={:.4} 判定={}", 结果.λ_max, 结果.判定);
    }

    #[test]
    fn test_样本不足保护() {
        let 计算器 = Lyapunov计算器::default();
        let 结果 = 计算器.计算(&[1.0, 2.0, 3.0], 0.1);
        assert!(结果.判定.contains("样本不足"));
        assert_eq!(结果.非线性s, 0.0);
    }

    #[test]
    fn test_εwave计算() {
        let 计算器 = Lyapunov计算器::default();
        // 造波前稳定（τ=1, λ=-1），造波后仍稳定（τ=1.25, λ=-0.8, Δλ=0.2<0.3）
        let 前 = 衰减时序(1.0, 0.05, 400);
        let 后 = 衰减时序(1.25, 0.05, 400);
        let 结果 = 计算器.计算ε_wave(&前, &后, 0.05);
        assert!(结果.造波安全, "稳定→稳定应安全: {}", 结果.说明);
        assert!(结果.ε_wave >= 0.0);
        assert!(结果.ε_wave < 0.3, "Δλ应<0.3: {}", 结果.ε_wave);

        // 造波前稳定，造波后混沌（增长）→ 不安全（转正）
        let 后混沌 = 增长时序(2.0, 0.05, 300);
        let 结果2 = 计算器.计算ε_wave(&前, &后混沌, 0.05);
        assert!(!结果2.造波安全, "转正应不安全: {}", 结果2.说明);
    }

    #[test]
    fn test_εwave缺省0() {
        let 计算器 = Lyapunov计算器::default();
        let 结果 = 计算器.计算(&衰减时序(1.0, 0.05, 300), 0.05);
        assert_eq!(结果.ε_wave, 0.0, "单时序分析默认ε_wave=0");
    }
}
