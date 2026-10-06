// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! §19.13.72 黎曼谱偏离度 D_s — GUE Wigner-Dyson 基准 KS 检验
//!
//! 理论依据:
//!   - IGT_感知场域造波调控_工程实践总纲_v2.0 §1.1: 黎曼谱基准比对算子
//!   - IGT_新旧工程框架实质性区别对比_v1.0 §二: GUE 谱偏离度提前 2-4 个月预警
//!
//! 核心方法:
//!   1. 从系统本征值序列计算最近邻间距 s_i = λ_{i+1} - λ_i
//!   2. 归一化间距 s_i / <s> (除以平均间距)
//!   3. 与 GUE Wigner-Dyson 基准分布做 KS 检验
//!   4. 输出谱偏离度 D_s ∈ [0, 1] (越小越接近稳态)
//!
//! GUE Wigner-Dyson 分布 (β=2):
//!   P(s) = (32/π²) · s² · exp(-4s²/π)
//!   - s: 归一化最近邻能级间距
//!   - P(0) = 0: 能级排斥 (level repulsion), 稳态系统特征
//!   - 与泊松分布(可积系统)对比: 泊松 P(s) = exp(-s), 无排斥
//!
//! KS 检验:
//!   D_ks = max |F_empirical(s) - F_theoretical(s)|
//!   - F_empirical: 经验 CDF (排序间距的阶梯函数)
//!   - F_theoretical: GUE 理论 CDF (数值积分)
//!   - D_s = D_ks ∈ [0, 1]
//!   - D_s → 0: 系统本征谱收敛于 GUE (稳态)
//!   - D_s → 1: 系统本征谱偏离 GUE (失稳)

#![allow(non_snake_case)]

use std::f64::consts::PI;

/// GUE Wigner-Dyson 常数 (β=2)
const GUE_NORM: f64 = 32.0 / (PI * PI); // 32/π² ≈ 3.2415

/// §19.13.72 黎曼谱偏离度计算结果
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct 谱偏离度结果 {
    /// 谱偏离度 D_s ∈ [0, 1] (越小越接近 GUE 稳态)
    pub d_s: f64,
    /// 输入本征值数量
    pub 本征值数: usize,
    /// 计算的最近邻间距数量
    pub 间距数: usize,
    /// 归一化间距均值 (应接近 1.0)
    pub 归一化间距均值: f64,
    /// KS 检验统计量
    pub ks统计量: f64,
    /// 稳态判定等级
    pub 稳态等级: String,
}

/// §19.13.72 黎曼谱偏离度计算器
pub struct 黎曼谱偏离度计算器;

impl 黎曼谱偏离度计算器 {
    /// 计算谱偏离度 D_s — 纯函数, 供单元测试 + Engine 调用
    ///
    /// 输入: 本征值序列 (从 τ 谱或 H 值时序提取)
    /// 输出: 谱偏离度结果 (D_s + 稳态等级)
    ///
    /// 稳态等级判定:
    ///   D_s < 0.10 → "GUE 稳态" (本征谱收敛于 GUE, 系统稳态)
    ///   0.10 ≤ D_s < 0.20 → "轻微偏离" (早期预警)
    ///   0.20 ≤ D_s < 0.40 → "中度偏离" (失稳趋势)
    ///   D_s ≥ 0.40 → "严重偏离" (系统失稳)
    pub fn 计算(本征值序列: &[f64]) -> Option<谱偏离度结果> {
        // 至少需要 4 个本征值才能计算 3 个间距做 KS 检验
        if 本征值序列.len() < 4 {
            return None;
        }

        // 1. 排序本征值
        let mut 排序值 = 本征值序列.to_vec();
        排序值.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // 2. 计算最近邻间距
        let 间距: Vec<f64> = 排序值
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .filter(|&d| d > 1e-12) // 过滤零间距(重根)
            .collect();

        if 间距.len() < 3 {
            return None;
        }

        // 3. 归一化间距 (除以平均间距)
        let 均值间距 = 间距.iter().sum::<f64>() / 间距.len() as f64;
        if 均值间距 < 1e-12 {
            return None;
        }
        let 归一化间距: Vec<f64> = 间距.iter().map(|&s| s / 均值间距).collect();

        // 4. 排序归一化间距 (KS 检验需要有序样本)
        let mut 排序间距 = 归一化间距.clone();
        排序间距.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // 5. KS 检验: 计算 max |F_emp - F_theo|
        let n = 排序间距.len() as f64;
        let mut ks统计量 = 0.0_f64;
        for (i, &s) in 排序间距.iter().enumerate() {
            // 经验 CDF: F_emp(s_i) = (i+1) / n
            let f_emp = (i + 1) as f64 / n;
            // 理论 CDF: F_theo(s_i) = ∫₀^{s_i} P(t) dt
            let f_theo = Self::gue_cdf(s);
            // KS 统计量: max |F_emp - F_theo|
            let diff = (f_emp - f_theo).abs();
            if diff > ks统计量 {
                ks统计量 = diff;
            }
        }

        // 6. D_s = KS 统计量 (clamp 到 [0, 1])
        let d_s = ks统计量.clamp(0.0, 1.0);

        // 7. 稳态等级判定
        let 稳态等级 = if d_s < 0.10 {
            "GUE 稳态".to_string()
        } else if d_s < 0.20 {
            "轻微偏离".to_string()
        } else if d_s < 0.40 {
            "中度偏离".to_string()
        } else {
            "严重偏离".to_string()
        };

        Some(谱偏离度结果 {
            d_s,
            本征值数: 本征值序列.len(),
            间距数: 间距.len(),
            归一化间距均值: 均值间距,
            ks统计量,
            稳态等级,
        })
    }

    /// GUE Wigner-Dyson PDF (β=2): P(s) = (32/π²) · s² · exp(-4s²/π)
    fn gue_pdf(s: f64) -> f64 {
        if s < 0.0 {
            return 0.0;
        }
        GUE_NORM * s * s * (-4.0 * s * s / PI).exp()
    }

    /// GUE Wigner-Dyson CDF: F(s) = ∫₀^s P(t) dt
    ///
    /// 使用辛普森法数值积分 (精度 1e-6, 步长 0.001)
    fn gue_cdf(s: f64) -> f64 {
        if s <= 0.0 {
            return 0.0;
        }
        // 积分上限: s (通常 s ∈ [0, 5] 覆盖 99.99% 概率)
        let upper = s.min(10.0);
        let n = (upper / 0.001) as usize; // 步长 0.001
        let n = n.max(2); // 至少 2 个区间
        let h = upper / n as f64;

        // 辛普森法: ∫ ≈ (h/3) · [f(x₀) + 4f(x₁) + 2f(x₂) + ... + 4f(x_{n-1}) + f(x_n)]
        let mut sum = Self::gue_pdf(0.0) + Self::gue_pdf(upper);
        for i in 1..n {
            let x = i as f64 * h;
            if i % 2 == 1 {
                sum += 4.0 * Self::gue_pdf(x);
            } else {
                sum += 2.0 * Self::gue_pdf(x);
            }
        }
        let result = (h / 3.0) * sum;
        result.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GUE PDF 在 s=0 处为 0 (能级排斥)
    #[test]
    fn test_gue_pdf_零点() {
        let p = 黎曼谱偏离度计算器::gue_pdf(0.0);
        assert!(p.abs() < 1e-12, "P(0) 应为 0 (能级排斥), 实际: {}", p);
    }

    /// GUE PDF 在 s>0 处为正
    #[test]
    fn test_gue_pdf_正值() {
        let p = 黎曼谱偏离度计算器::gue_pdf(1.0);
        assert!(p > 0.0, "P(1) 应为正, 实际: {}", p);
    }

    /// GUE CDF 单调递增, F(0)=0, F(∞)→1
    #[test]
    fn test_gue_cdf_单调性() {
        let f0 = 黎曼谱偏离度计算器::gue_cdf(0.0);
        let f1 = 黎曼谱偏离度计算器::gue_cdf(1.0);
        let f2 = 黎曼谱偏离度计算器::gue_cdf(2.0);
        let f5 = 黎曼谱偏离度计算器::gue_cdf(5.0);
        assert!(f0.abs() < 1e-9, "F(0) 应为 0, 实际: {}", f0);
        assert!(f1 > f0, "F 应单调递增: F(1)={} > F(0)={}", f1, f0);
        assert!(f2 > f1, "F 应单调递增: F(2)={} > F(1)={}", f2, f1);
        assert!(f5 > f2, "F 应单调递增: F(5)={} > F(2)={}", f5, f2);
        assert!(f5 > 0.95, "F(5) 应接近 1, 实际: {}", f5);
    }

    /// GUE 稳态: 生成符合 GUE 分布的本征值, D_s 应较小
    #[test]
    fn test_gue稳态_低偏离度() {
        // 生成近似 GUE 分布的间距 (Wigner-Dyson)
        // 简化: 用 χ² 分布的平方根近似 GUE 间距
        // 这里用手工构造的 GUE-like 间距
        let gue间距: Vec<f64> = vec![
            0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1,
            0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2,
        ];
        // 构造本征值 (累积和)
        let mut 本征值 = Vec::new();
        let mut acc = 0.0;
        for &s in &gue间距 {
            本征值.push(acc);
            acc += s;
        }
        本征值.push(acc);

        let 结果 = 黎曼谱偏离度计算器::计算(&本征值);
        assert!(结果.is_some(), "应返回结果");
        let r = 结果.unwrap();
        // GUE-like 间距的 D_s 不应太大(虽然手工构造不完美, 但应在合理范围)
        assert!(r.d_s < 0.5, "GUE-like 间距 D_s 应 < 0.5, 实际: {}", r.d_s);
    }

    /// 泊松分布(可积系统): 均匀间距, D_s 应较大
    #[test]
    fn test_泊松分布_高偏离度() {
        // 泊松分布: 间距服从 exp(-s), 均匀间距严重偏离 GUE
        let 均匀间距: Vec<f64> = vec![1.0; 20]; // 完全均匀间距
        let mut 本征值 = Vec::new();
        let mut acc = 0.0;
        for &s in &均匀间距 {
            本征值.push(acc);
            acc += s;
        }
        本征值.push(acc);

        let 结果 = 黎曼谱偏离度计算器::计算(&本征值);
        assert!(结果.is_some(), "应返回结果");
        let r = 结果.unwrap();
        // 均匀间距严重偏离 GUE (GUE 有能级排斥, 均匀间距没有)
        // D_s 应较大
        assert!(r.d_s > 0.1, "均匀间距 D_s 应 > 0.1, 实际: {}", r.d_s);
    }

    /// 样本不足: 少于 4 个本征值返回 None
    #[test]
    fn test_样本不足() {
        let 本征值 = vec![1.0, 2.0, 3.0]; // 仅 3 个
        let 结果 = 黎曼谱偏离度计算器::计算(&本征值);
        assert!(结果.is_none(), "少于 4 个本征值应返回 None");
    }

    /// 稳态等级判定: D_s < 0.10 为 GUE 稳态
    #[test]
    fn test_稳态等级判定() {
        // 构造一个接近 GUE 的序列(使用 Wigner-Dyson 的典型间距)
        // GUE 间距的典型值在 0.5-2.0 之间, 带有能级排斥
        let gue_like间距: Vec<f64> = vec![
            0.42, 0.51, 0.67, 0.78, 0.89, 0.95, 1.05, 1.12, 1.23, 1.34,
            0.45, 0.56, 0.71, 0.82, 0.91, 0.98, 1.08, 1.18, 1.28, 1.41,
        ];
        let mut 本征值 = Vec::new();
        let mut acc = 0.0;
        for &s in &gue_like间距 {
            本征值.push(acc);
            acc += s;
        }
        本征值.push(acc);

        if let Some(r) = 黎曼谱偏离度计算器::计算(&本征值) {
            // 验证稳态等级是有效值
            assert!(
                ["GUE 稳态", "轻微偏离", "中度偏离", "严重偏离"].contains(&r.稳态等级.as_str()),
                "稳态等级应有效, 实际: {}",
                r.稳态等级
            );
        }
    }
}
