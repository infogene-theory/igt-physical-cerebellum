// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! 谱等价标量判定（MIGRATION-FINAL flutter 归一第四例）
//!
//! 从 flutter `lib/services/spectral_equivalence.dart`（第352批 P22 §10.5 前端可视化深化）迁移——
//! **移动不改算法**：当前蒸馏基因 vs 本地基因库 → ω差<0.15 ∧ γ差<0.1 ∧ p_obs≥0.5 → 谱等价（本次复现）。
//!
//! 注: 本模块是**标量三元判定**（前端可视化，对齐 engine gene_distillation/spectral_equivalence.rs）；
//! 生产级谱等价（复谱距离+子空间夹角+参与度余弦+椭圆）见 `complex_spectrum_match::谱等价判定`。

use serde::{Deserialize, Serialize};

/// 谱等价标量候选（基因的 ω/γ/参与度标量）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 谱等价候选 {
    /// 本征频率 ω
    pub omega: f64,
    /// 弛豫速率 γ
    pub gamma: f64,
    /// 观测参与度 p_obs
    pub p_obs: f64,
}

/// 单基因谱等价匹配结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 谱等价匹配结果 {
    pub omega_dist: f64,
    pub gamma_dist: f64,
    pub equivalent: bool,
}

/// 谱等价判定常量（P22 §10.5 三元）
pub const OMEGA_TOL: f64 = 0.15;
pub const GAMMA_TOL: f64 = 0.1;
pub const P_OBS_MIN: f64 = 0.5;

/// 单基因匹配库中最近基因（等价判据全过才 equivalent；移动不改算法）
pub fn 谱等价最近匹配(
    当前: &谱等价候选,
    库: &[谱等价候选],
) -> Option<(usize, 谱等价匹配结果)> {
    let mut best_idx: Option<usize> = None;
    let mut best_dist = f64::INFINITY;
    let mut best_match = None;
    for (i, rec) in 库.iter().enumerate() {
        let d_w = (rec.omega - 当前.omega).abs();
        let d_g = (rec.gamma - 当前.gamma).abs();
        let equivalent = d_w < OMEGA_TOL && d_g < GAMMA_TOL && 当前.p_obs >= P_OBS_MIN;
        let dist = d_w * d_w + d_g * d_g; // 欧氏加权距离（找最近）
        if dist < best_dist {
            best_dist = dist;
            best_idx = Some(i);
            best_match = Some(谱等价匹配结果 {
                omega_dist: d_w,
                gamma_dist: d_g,
                equivalent,
            });
        }
    }
    best_idx.map(|i| (i, best_match.unwrap()))
}

/// 批量判定 → 命中的库基因索引集合（本次观测复现了哪些库基因）
pub fn 谱等价命中集(当前列表: &[谱等价候选], 库: &[谱等价候选]) -> Vec<usize> {
    let mut result = Vec::new();
    for c in 当前列表 {
        if let Some((idx, m)) = 谱等价最近匹配(c, 库) {
            if m.equivalent && !result.contains(&idx) {
                result.push(idx);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn 基因(omega: f64, gamma: f64, p_obs: f64) -> 谱等价候选 {
        谱等价候选 { omega, gamma, p_obs }
    }

    #[test]
    fn test_最近匹配_谱等价() {
        let 当前 = 基因(1.0, 0.5, 0.7);
        let 库 = vec![基因(1.1, 0.45, 0.6), 基因(5.0, 5.0, 0.9)];
        let (idx, m) = 谱等价最近匹配(&当前, &库).unwrap();
        assert_eq!(idx, 0, "应命中最近库基因");
        assert!(m.equivalent, "ω差0.1<0.15 ∧ γ差0.05<0.1 ∧ p_obs0.7≥0.5 应等价");
        assert!((m.omega_dist - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_最近匹配_不等价() {
        let 当前 = 基因(1.0, 0.5, 0.2); // p_obs<0.5 → 不等价
        let 库 = vec![基因(1.05, 0.45, 0.9)];
        let (_, m) = 谱等价最近匹配(&当前, &库).unwrap();
        assert!(!m.equivalent, "p_obs 0.2<0.5 应不等价");
    }

    #[test]
    fn test_最近匹配_γ超容差() {
        // 对齐 flutter 原用例: 库(1.0,-0.2) vs 当前(1.02,-0.5) → γ差0.3>0.1
        let 当前 = 基因(1.02, -0.5, 0.8);
        let 库 = vec![基因(1.0, -0.2, 0.8)];
        let (_, m) = 谱等价最近匹配(&当前, &库).unwrap();
        assert!(!m.equivalent, "γ差 0.3>0.1 应不等价");
    }

    #[test]
    fn test_最近匹配_ω超容差() {
        // 对齐 flutter 原用例: 库(1.0,-0.2) vs 当前(1.3,-0.18) → ω差0.3>0.15
        let 当前 = 基因(1.3, -0.18, 0.8);
        let 库 = vec![基因(1.0, -0.2, 0.8)];
        let (_, m) = 谱等价最近匹配(&当前, &库).unwrap();
        assert!(!m.equivalent, "ω差 0.3>0.15 应不等价");
    }

    #[test]
    fn test_空库_None() {
        let 当前 = 基因(1.0, 0.5, 0.7);
        assert!(谱等价最近匹配(&当前, &[]).is_none());
    }

    #[test]
    fn test_命中集() {
        let 当前 = vec![基因(1.0, 0.5, 0.7), 基因(2.0, 1.0, 0.6)];
        let 库 = vec![基因(1.05, 0.45, 0.9), 基因(2.05, 1.05, 0.8), 基因(9.0, 9.0, 0.9)];
        let 命中 = 谱等价命中集(&当前, &库);
        assert_eq!(命中.len(), 2, "两个当前基因都应命中: {:?}", 命中);
        assert!(!命中.contains(&2), "远基因不应命中");
    }

    #[test]
    fn test_序列化往返() {
        let c = 基因(1.0, 0.5, 0.7);
        let json = serde_json::to_string(&c).unwrap();
        let back: 谱等价候选 = serde_json::from_str(&json).unwrap();
        assert!((back.omega - 1.0).abs() < 1e-9);
    }
}
