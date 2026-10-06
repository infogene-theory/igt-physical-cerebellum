// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! P9 自指观测者涌现定理
//!
//! 绑定版本: 元衍拓扑学 V8.0 / drift_hash_root_v8.0
//!
//! 系统复杂度达到阈值时涌现自指观测者
#![allow(non_snake_case)]

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObserverState {
    Dormant,
    Awakening,
    Active,
    Transcendent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergenceCondition {
    pub complexity: f64,
    pub self_reference: f64,
    pub information_density: f64,
    pub feedback_loops: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergenceResult {
    pub state: ObserverState,
    pub emergence_probability: f64,
    pub readiness_score: f64,
    pub missing_conditions: Vec<String>,
}

pub const COMPLEXITY_THRESHOLD: f64 = 0.6;
pub const SELF_REF_THRESHOLD: f64 = 0.5;
pub const INFO_DENSITY_THRESHOLD: f64 = 0.4;
pub const MIN_FEEDBACK_LOOPS: u32 = 3;

pub struct EmergenceDetector {
    complexity_threshold: f64,
    self_ref_threshold: f64,
    info_density_threshold: f64,
    min_feedback_loops: u32,
}

impl EmergenceDetector {
    pub fn new() -> Self {
        Self {
            complexity_threshold: COMPLEXITY_THRESHOLD,
            self_ref_threshold: SELF_REF_THRESHOLD,
            info_density_threshold: INFO_DENSITY_THRESHOLD,
            min_feedback_loops: MIN_FEEDBACK_LOOPS,
        }
    }

    pub fn detect(&self, condition: &EmergenceCondition) -> EmergenceResult {
        let mut missing = Vec::new();
        let mut scores = Vec::new();

        if condition.complexity < self.complexity_threshold {
            missing.push(format!(
                "复杂度不足: {} < {}",
                condition.complexity, self.complexity_threshold
            ));
            scores.push(condition.complexity / self.complexity_threshold);
        } else {
            scores.push(1.0);
        }

        if condition.self_reference < self.self_ref_threshold {
            missing.push(format!(
                "自指度不足: {} < {}",
                condition.self_reference, self.self_ref_threshold
            ));
            scores.push(condition.self_reference / self.self_ref_threshold);
        } else {
            scores.push(1.0);
        }

        if condition.information_density < self.info_density_threshold {
            missing.push(format!(
                "信息密度不足: {} < {}",
                condition.information_density, self.info_density_threshold
            ));
            scores.push(condition.information_density / self.info_density_threshold);
        } else {
            scores.push(1.0);
        }

        if condition.feedback_loops < self.min_feedback_loops {
            missing.push(format!(
                "反馈环不足: {} < {}",
                condition.feedback_loops, self.min_feedback_loops
            ));
            scores.push(condition.feedback_loops as f64 / self.min_feedback_loops as f64);
        } else {
            scores.push(1.0);
        }

        let readiness_score = scores.iter().sum::<f64>() / scores.len() as f64;

        let emergence_probability = self.compute_emergence_probability(condition);

        let state = self.determine_state(readiness_score, emergence_probability);

        EmergenceResult {
            state,
            emergence_probability,
            readiness_score,
            missing_conditions: missing,
        }
    }

    fn compute_emergence_probability(&self, condition: &EmergenceCondition) -> f64 {
        let complexity_factor = (condition.complexity / self.complexity_threshold).min(1.0);
        let self_ref_factor = (condition.self_reference / self.self_ref_threshold).min(1.0);
        let info_factor = (condition.information_density / self.info_density_threshold).min(1.0);
        let loop_factor =
            (condition.feedback_loops as f64 / self.min_feedback_loops as f64).min(1.0);

        complexity_factor * self_ref_factor * info_factor * loop_factor
    }

    fn determine_state(&self, readiness: f64, probability: f64) -> ObserverState {
        let combined = readiness * probability;

        if combined >= 0.9 {
            ObserverState::Transcendent
        } else if combined >= 0.7 {
            ObserverState::Active
        } else if combined >= 0.4 {
            ObserverState::Awakening
        } else {
            ObserverState::Dormant
        }
    }

    pub fn compute_self_reference_degree(
        &self,
        internal_states: &[f64],
        observations: &[f64],
    ) -> f64 {
        if internal_states.is_empty() || observations.is_empty() {
            return 0.0;
        }

        let correlation = compute_correlation(internal_states, observations);
        correlation
    }
}

impl Default for EmergenceDetector {
    fn default() -> Self {
        Self::new()
    }
}

fn compute_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()) as f64;
    if n == 0.0 {
        return 0.0;
    }

    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sum_x2: f64 = x.iter().map(|a| a * a).sum();
    let sum_y2: f64 = y.iter().map(|a| a * a).sum();

    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

    if denominator == 0.0 {
        0.0
    } else {
        (numerator / denominator).abs()
    }
}

pub fn check_emergence(
    complexity: f64,
    self_ref: f64,
    info_density: f64,
    feedback_loops: u32,
) -> bool {
    complexity >= COMPLEXITY_THRESHOLD
        && self_ref >= SELF_REF_THRESHOLD
        && info_density >= INFO_DENSITY_THRESHOLD
        && feedback_loops >= MIN_FEEDBACK_LOOPS
}

pub fn get_observer_state(readiness: f64) -> ObserverState {
    if readiness >= 0.9 {
        ObserverState::Transcendent
    } else if readiness >= 0.7 {
        ObserverState::Active
    } else if readiness >= 0.4 {
        ObserverState::Awakening
    } else {
        ObserverState::Dormant
    }
}
