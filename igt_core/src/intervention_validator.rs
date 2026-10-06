// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! IGT干预验证工具 — 基线测量 + 效果对比 + 统计分析
//!
//! **MIGRATION-FINAL Step0-3b-2**: engine `l1/intervention_validator.rs` **逐字搬迁**
//! （不改算法、不改命名，Step2 engine 只改 use 路径核对）；engine 原位 Step2 删除。
//! 依赖 `crate::shared_types::Sbi8Result`（Step0-3b-1 已下沉）+ serde——**无 IO**。
//!
//! 基于三层八器官统一心跳体系，对每次干预进行严格的科学验证:
//!   1. 干预前基线SBI-8测量
//!   2. 干预后SBI-8测量
//!   3. 效果量计算 (Cohen's d)
//!   4. 显著性评估
//!   5. 器官级效果分解
//!
//! 注: engine 原文件 `use crate::l1::organ_router::OrganId` 在实现中未实际使用
//! （仅测试可能引用），core 版本不引入该 import。
#![allow(non_snake_case)]

use crate::shared_types::Sbi8Result;
use serde::{Deserialize, Serialize};

// ============================================================================
// 干预验证结果
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionValidationReport {
    pub intervention_name: String,
    pub baseline_sbi8: f64,
    pub post_sbi8: f64,
    pub sbi8_delta: f64,
    pub effect_level: EffectLevel,
    pub cohens_d: f64,
    pub estimated_p_value: f64,
    pub organ_deltas: Vec<OrganDelta>,
    pub statistical_power: f64,
    pub recommendation: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffectLevel {
    LargePositive,
    MediumPositive,
    SmallPositive,
    Negligible,
    SmallNegative,
    MediumNegative,
    LargeNegative,
}

impl EffectLevel {
    pub fn from_cohens_d(d: f64) -> Self {
        if d >= 0.80 {
            Self::LargePositive
        } else if d >= 0.50 {
            Self::MediumPositive
        } else if d >= 0.20 {
            Self::SmallPositive
        } else if d > -0.20 {
            Self::Negligible
        } else if d > -0.50 {
            Self::SmallNegative
        } else if d > -0.80 {
            Self::MediumNegative
        } else {
            Self::LargeNegative
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::LargePositive => "大正向效果",
            Self::MediumPositive => "中正向效果",
            Self::SmallPositive => "小正向效果",
            Self::Negligible => "无显著效果",
            Self::SmallNegative => "小负向效果",
            Self::MediumNegative => "中负向效果",
            Self::LargeNegative => "大负向效果",
        }
    }

    pub fn is_positive(&self) -> bool {
        matches!(
            self,
            Self::LargePositive | Self::MediumPositive | Self::SmallPositive
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganDelta {
    pub organ_id: String,
    pub organ_label: String,
    pub baseline_health: f64,
    pub post_health: f64,
    pub health_delta: f64,
    pub sigma_delta: f64,
    pub k_delta: f64,
    pub tau_delta: f64,
    pub effect_level: EffectLevel,
}

// ============================================================================
// 干预验证器
// ============================================================================

pub struct InterventionValidator {
    pub min_effect_size: f64,
    pub significance_threshold: f64,
}

impl Default for InterventionValidator {
    fn default() -> Self {
        Self {
            min_effect_size: 0.20,
            significance_threshold: 0.05,
        }
    }
}

impl InterventionValidator {
    pub fn new(min_effect: f64, sig_threshold: f64) -> Self {
        Self {
            min_effect_size: min_effect,
            significance_threshold: sig_threshold,
        }
    }

    pub fn validate(
        &self,
        intervention_name: &str,
        baseline: &Sbi8Result,
        post: &Sbi8Result,
    ) -> InterventionValidationReport {
        let sbi8_delta = post.sbi8_score - baseline.sbi8_score;

        let organ_deltas = self.compute_organ_deltas(baseline, post);

        let cohens_d = self.compute_cohens_d(&organ_deltas);
        let effect_level = EffectLevel::from_cohens_d(cohens_d);

        let estimated_p_value = self.estimate_p_value(cohens_d, organ_deltas.len());
        let statistical_power = self.compute_statistical_power(cohens_d, organ_deltas.len());

        let recommendation = self.generate_recommendation(effect_level, sbi8_delta, &organ_deltas);

        InterventionValidationReport {
            intervention_name: intervention_name.to_string(),
            baseline_sbi8: baseline.sbi8_score,
            post_sbi8: post.sbi8_score,
            sbi8_delta,
            effect_level,
            cohens_d,
            estimated_p_value,
            organ_deltas,
            statistical_power,
            recommendation,
        }
    }

    fn compute_organ_deltas(&self, baseline: &Sbi8Result, post: &Sbi8Result) -> Vec<OrganDelta> {
        let mut deltas = Vec::new();

        for post_organ in &post.organ_details {
            let baseline_organ = baseline
                .organ_details
                .iter()
                .find(|b| b.organ_id == post_organ.organ_id);

            if let Some(base) = baseline_organ {
                let health_delta = post_organ.health_score - base.health_score;
                let sigma_delta = post_organ.sigma - base.sigma;
                let k_delta = post_organ.k - base.k;
                let tau_delta = post_organ.tau_minutes - base.tau_minutes;

                deltas.push(OrganDelta {
                    organ_id: post_organ.organ_id.clone(),
                    organ_label: post_organ.organ_label.clone(),
                    baseline_health: base.health_score,
                    post_health: post_organ.health_score,
                    health_delta,
                    sigma_delta,
                    k_delta,
                    tau_delta,
                    effect_level: EffectLevel::from_cohens_d(health_delta / 0.15),
                });
            }
        }

        deltas
    }

    fn compute_cohens_d(&self, organ_deltas: &[OrganDelta]) -> f64 {
        if organ_deltas.is_empty() {
            return 0.0;
        }

        let n = organ_deltas.len() as f64;
        let mean_delta: f64 = organ_deltas.iter().map(|d| d.health_delta).sum::<f64>() / n;

        if n < 2.0 {
            return if mean_delta > 0.0 {
                1.0
            } else if mean_delta < 0.0 {
                -1.0
            } else {
                0.0
            };
        }

        let variance: f64 = organ_deltas
            .iter()
            .map(|d| {
                let diff = d.health_delta - mean_delta;
                diff * diff
            })
            .sum::<f64>()
            / (n - 1.0);

        let pooled_sd = variance.sqrt().max(0.01);
        mean_delta / pooled_sd
    }

    fn estimate_p_value(&self, cohens_d: f64, n: usize) -> f64 {
        let n = n.max(2) as f64;
        let t_stat = cohens_d.abs() * n.sqrt();

        let df = n - 1.0;
        let x = df / (df + t_stat * t_stat);

        let p = if x <= 0.0 {
            1.0
        } else if x >= 1.0 {
            0.0
        } else {
            regularized_beta(x, df / 2.0, 0.5)
        };

        (p * 2.0).min(1.0)
    }

    fn compute_statistical_power(&self, cohens_d: f64, n: usize) -> f64 {
        let n = n.max(2) as f64;
        let ncp = cohens_d.abs() * n.sqrt();
        let z_alpha = 1.96;

        if ncp <= z_alpha {
            (ncp / z_alpha) * 0.5
        } else {
            let z = ncp - z_alpha;
            0.5 + 0.5 * (z / (1.0 + z.abs())).min(0.49)
        }
    }

    fn generate_recommendation(
        &self,
        effect_level: EffectLevel,
        sbi8_delta: f64,
        organ_deltas: &[OrganDelta],
    ) -> String {
        match effect_level {
            EffectLevel::LargePositive | EffectLevel::MediumPositive => {
                let improved: Vec<_> = organ_deltas
                    .iter()
                    .filter(|d| d.health_delta > 0.05)
                    .map(|d| d.organ_label.as_str())
                    .collect();
                if improved.is_empty() {
                    format!(
                        "干预效果显著(SBI-8 Δ={:.3})，建议纳入标准干预库",
                        sbi8_delta
                    )
                } else {
                    format!(
                        "干预效果显著(SBI-8 Δ={:.3})，{}等器官明显改善，建议纳入标准干预库",
                        sbi8_delta,
                        improved.join("、")
                    )
                }
            }
            EffectLevel::SmallPositive => {
                format!(
                    "干预有轻微正向效果(SBI-8 Δ={:.3})，可优化参数后重试",
                    sbi8_delta
                )
            }
            EffectLevel::Negligible => {
                format!(
                    "干预无显著效果(SBI-8 Δ={:.3})，建议调整干预策略",
                    sbi8_delta
                )
            }
            EffectLevel::SmallNegative
            | EffectLevel::MediumNegative
            | EffectLevel::LargeNegative => {
                let worsened: Vec<_> = organ_deltas
                    .iter()
                    .filter(|d| d.health_delta < -0.03)
                    .map(|d| d.organ_label.as_str())
                    .collect();
                if worsened.is_empty() {
                    format!(
                        "干预有负面趋势(SBI-8 Δ={:.3})，建议立即停止并复盘",
                        sbi8_delta
                    )
                } else {
                    format!(
                        "干预有负面效果(SBI-8 Δ={:.3})，{}等器官恶化，建议立即停止",
                        sbi8_delta,
                        worsened.join("、")
                    )
                }
            }
        }
    }
}

fn regularized_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);

    let mut sum = 0.0;
    let mut term = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp();
    let mut k = 0;

    while term.abs() > 1e-12 && k < 200 {
        sum += term;
        k += 1;
        let kf = k as f64;
        term *= (a + b + kf - 1.0) * x / (a + kf);
    }

    sum
}

fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let c = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let y = x;
    let tmp = x + 5.5;
    let tmp = tmp - (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015;
    for (i, &ci) in c.iter().enumerate() {
        ser += ci / (y + (i + 1) as f64);
    }
    -tmp + (2.5066282746310005 * ser / y).ln()
}

// ============================================================================
// 批量验证 (多次干预对比)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchValidationResult {
    pub validations: Vec<InterventionValidationReport>,
    pub best_intervention: Option<String>,
    pub worst_intervention: Option<String>,
    pub average_effect: f64,
    pub total_significant_count: usize,
}

impl InterventionValidator {
    pub fn batch_validate(
        &self,
        interventions: &[(String, Sbi8Result, Sbi8Result)],
    ) -> BatchValidationResult {
        let validations: Vec<_> = interventions
            .iter()
            .map(|(name, baseline, post)| self.validate(name, baseline, post))
            .collect();

        let best = validations
            .iter()
            .max_by(|a, b| {
                a.cohens_d
                    .partial_cmp(&b.cohens_d)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|v| v.intervention_name.clone());

        let worst = validations
            .iter()
            .min_by(|a, b| {
                a.cohens_d
                    .partial_cmp(&b.cohens_d)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|v| v.intervention_name.clone());

        let avg_effect = if validations.is_empty() {
            0.0
        } else {
            validations.iter().map(|v| v.cohens_d).sum::<f64>() / validations.len() as f64
        };

        let sig_count = validations
            .iter()
            .filter(|v| v.effect_level.is_positive())
            .count();

        BatchValidationResult {
            validations,
            best_intervention: best,
            worst_intervention: worst,
            average_effect: avg_effect,
            total_significant_count: sig_count,
        }
    }
}

// ============================================================================
// 测试（搬迁验证: 统计口径与 engine 原实现一致）
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared_types::OrganHealthResult;

    fn 构造器官(id: &str, label: &str, 健康: f64, sigma: f64) -> OrganHealthResult {
        OrganHealthResult {
            organ_id: id.to_string(),
            organ_label: label.to_string(),
            organ_icon: "◉".to_string(),
            thermodynamic_type: "耗散型".to_string(),
            sigma,
            k: 1.0,
            tau_minutes: 10.0,
            s_sigma: 0.5,
            s_k: 0.5,
            s_tau: 0.5,
            health_score: 健康,
            global_weight: 0.125,
            status: "健康".to_string(),
        }
    }

    fn 构造sbi8(分数: f64, 器官: Vec<OrganHealthResult>) -> Sbi8Result {
        Sbi8Result {
            sbi8_score: 分数,
            health_level: "B".to_string(),
            health_label_cn: "良好".to_string(),
            organ_details: 器官,
            total_weight: 1.0,
            weighted_sum: 分数,
            timestamp: 0,
        }
    }

    /// EffectLevel 分级（搬自 engine 原实现阈值: 0.8/0.5/0.2/-0.2/-0.5/-0.8）
    #[test]
    fn 测试_效果分级() {
        assert_eq!(EffectLevel::from_cohens_d(1.0), EffectLevel::LargePositive);
        assert_eq!(EffectLevel::from_cohens_d(0.6), EffectLevel::MediumPositive);
        assert_eq!(EffectLevel::from_cohens_d(0.3), EffectLevel::SmallPositive);
        assert_eq!(EffectLevel::from_cohens_d(0.0), EffectLevel::Negligible);
        assert_eq!(EffectLevel::from_cohens_d(-0.3), EffectLevel::SmallNegative);
        assert_eq!(EffectLevel::from_cohens_d(-0.6), EffectLevel::MediumNegative);
        assert_eq!(EffectLevel::from_cohens_d(-1.0), EffectLevel::LargeNegative);
        assert_eq!(EffectLevel::LargePositive.label(), "大正向效果");
        assert!(EffectLevel::SmallPositive.is_positive());
        assert!(!EffectLevel::Negligible.is_positive());
    }

    /// validate 全流程（器官分解 + Cohen's d + p 值 + 功效 + 建议）
    #[test]
    fn 测试_干预验证流程() {
        let 基线 = 构造sbi8(
            70.0,
            vec![
                构造器官("O1", "认知器官", 0.60, 0.5),
                构造器官("O3", "安全器官", 0.70, 0.4),
            ],
        );
        let 干预后 = 构造sbi8(
            82.0,
            vec![
                构造器官("O1", "认知器官", 0.80, 0.3),
                构造器官("O3", "安全器官", 0.85, 0.2),
            ],
        );
        let 验证器 = InterventionValidator::default();
        let 报告 = 验证器.validate("测试干预", &基线, &干预后);
        assert!((报告.sbi8_delta - 12.0).abs() < 1e-12, "Δ={}", 报告.sbi8_delta);
        assert_eq!(报告.organ_deltas.len(), 2, "器官分解");
        assert!(报告.organ_deltas.iter().all(|d| d.health_delta > 0.0));
        assert!(报告.cohens_d > 0.0, "Cohen's d 应正: {}", 报告.cohens_d);
        assert!(报告.effect_level.is_positive(), "应判正向: {:?}", 报告.effect_level);
        assert!(报告.estimated_p_value >= 0.0 && 报告.estimated_p_value <= 1.0);
        assert!(报告.statistical_power >= 0.0 && 报告.statistical_power <= 1.0);
        assert!(报告.recommendation.contains("显著"), "建议: {}", 报告.recommendation);
    }

    /// 空器官明细（退化: d=0 不 panic）
    #[test]
    fn 测试_空器官明细() {
        let 基线 = 构造sbi8(70.0, vec![]);
        let 干预后 = 构造sbi8(70.0, vec![]);
        let 报告 = InterventionValidator::default().validate("空", &基线, &干预后);
        assert_eq!(报告.cohens_d, 0.0);
        assert_eq!(报告.effect_level, EffectLevel::Negligible);
    }

    /// batch_validate（best/worst/平均/显著计数）
    #[test]
    fn 测试_批量验证() {
        let 基线 = 构造sbi8(70.0, vec![构造器官("O1", "认知器官", 0.60, 0.5)]);
        let 好 = 构造sbi8(85.0, vec![构造器官("O1", "认知器官", 0.90, 0.2)]);
        let 差 = 构造sbi8(65.0, vec![构造器官("O1", "认知器官", 0.50, 0.8)]);
        let 验证器 = InterventionValidator::default();
        let 批量 = 验证器.batch_validate(&[
            ("A".to_string(), 基线.clone(), 好),
            ("B".to_string(), 基线, 差),
        ]);
        assert_eq!(批量.validations.len(), 2);
        assert_eq!(批量.best_intervention.as_deref(), Some("A"));
        assert_eq!(批量.worst_intervention.as_deref(), Some("B"));
        assert_eq!(批量.total_significant_count, 1, "仅 A 正向");
    }

    /// regularized_beta 边界（纯统计函数: x≤0→0 / x≥1→1 / 区间内∈[0,1]）
    #[test]
    fn 测试_正则化贝塔边界() {
        assert_eq!(regularized_beta(0.0, 1.0, 0.5), 0.0);
        assert_eq!(regularized_beta(1.0, 1.0, 0.5), 1.0);
        let 中 = regularized_beta(0.5, 2.0, 0.5);
        assert!(中 >= 0.0 && 中 <= 1.0, "区间内: {}", 中);
        // ln_gamma 非正入参 → 0（诚实下限）
        assert_eq!(ln_gamma(0.0), 0.0);
        assert!(ln_gamma(5.0) > 0.0, "Γ(5)=ln(24)≈3.178");
    }
}
