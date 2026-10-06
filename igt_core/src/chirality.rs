// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! P4a 手性对称性破缺定理
//!
//! 绑定版本: 元衍拓扑学 V8.0 / drift_hash_root_v8.0
//!
//! 系统演化方向由手性对称性破缺决定
#![allow(non_snake_case)]

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Chirality {
    Left,
    Right,
    Neutral,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChiralityState {
    pub left_weight: f64,
    pub right_weight: f64,
    pub symmetry_degree: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChiralityResult {
    pub dominant: Chirality,
    pub broken: bool,
    pub bias: f64,
    pub evolution_direction: String,
}

pub struct ChiralityAnalyzer {
    symmetry_threshold: f64,
    bias_threshold: f64,
}

impl ChiralityAnalyzer {
    pub fn new() -> Self {
        Self {
            symmetry_threshold: 0.1,
            bias_threshold: 0.05,
        }
    }

    pub fn analyze(&self, state: &ChiralityState) -> ChiralityResult {
        let bias = state.left_weight - state.right_weight;

        let dominant = if bias.abs() < self.bias_threshold {
            Chirality::Neutral
        } else if bias > 0.0 {
            Chirality::Left
        } else {
            Chirality::Right
        };

        let broken = state.symmetry_degree < self.symmetry_threshold;

        let evolution_direction = match dominant {
            Chirality::Left => "向左演化（收敛型）".to_string(),
            Chirality::Right => "向右演化（发散型）".to_string(),
            Chirality::Neutral => "对称态（临界点）".to_string(),
        };

        ChiralityResult {
            dominant,
            broken,
            bias,
            evolution_direction,
        }
    }

    pub fn compute_symmetry_breaking(&self, left: f64, right: f64) -> f64 {
        let total = left + right;
        if total <= 0.0 {
            return 0.0;
        }

        let ratio = (left - right).abs() / total;
        ratio
    }

    pub fn predict_evolution(&self, current_bias: f64, historical_bias: f64) -> f64 {
        let trend = current_bias - historical_bias;
        current_bias + trend
    }
}

impl Default for ChiralityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

pub fn detect_chirality(left: f64, right: f64) -> Chirality {
    let bias = left - right;
    if bias.abs() < 0.05 {
        Chirality::Neutral
    } else if bias > 0.0 {
        Chirality::Left
    } else {
        Chirality::Right
    }
}

pub fn is_symmetry_broken(symmetry_degree: f64) -> bool {
    symmetry_degree < 0.1
}
