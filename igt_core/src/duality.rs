// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! P8 健康-进化对偶定理
//!
//! 绑定版本: 元衍拓扑学 V8.0 / drift_hash_root_v8.0
//!
//! 健康度与进化水平存在对偶关系
#![allow(non_snake_case)]

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DualityPhase {
    StableHealthy,
    Evolving,
    Critical,
    Collapsed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualityState {
    pub health: f64,
    pub evolution_level: f64,
    pub duality_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualityResult {
    pub phase: DualityPhase,
    pub balance: f64,
    pub trade_off: f64,
    pub optimal_point: (f64, f64),
}

pub const HEALTH_CRITICAL: f64 = 0.3;
pub const HEALTH_OPTIMAL: f64 = 0.7;
pub const EVOLUTION_THRESHOLD: f64 = 0.5;

pub struct DualityAnalyzer {
    h_crit: f64,
    h_star: f64,
    el_threshold: f64,
}

impl DualityAnalyzer {
    pub fn new() -> Self {
        Self {
            h_crit: HEALTH_CRITICAL,
            h_star: HEALTH_OPTIMAL,
            el_threshold: EVOLUTION_THRESHOLD,
        }
    }

    pub fn analyze(&self, state: &DualityState) -> DualityResult {
        let phase = self.determine_phase(state.health, state.evolution_level);

        let balance = self.compute_balance(state.health, state.evolution_level);

        let trade_off = self.compute_trade_off(state.health, state.evolution_level);

        let optimal_point = self.find_optimal_point(state.duality_strength);

        DualityResult {
            phase,
            balance,
            trade_off,
            optimal_point,
        }
    }

    fn determine_phase(&self, h: f64, el: f64) -> DualityPhase {
        if h < self.h_crit {
            DualityPhase::Collapsed
        } else if h < self.h_star {
            if el > self.el_threshold {
                DualityPhase::Evolving
            } else {
                DualityPhase::Critical
            }
        } else {
            DualityPhase::StableHealthy
        }
    }

    fn compute_balance(&self, h: f64, el: f64) -> f64 {
        1.0 - (h - el).abs()
    }

    fn compute_trade_off(&self, h: f64, el: f64) -> f64 {
        if h < self.h_crit {
            return 0.0;
        }

        let max_evolution = 1.0 - h;
        if max_evolution <= 0.0 {
            return 0.0;
        }

        (el / max_evolution).min(1.0)
    }

    fn find_optimal_point(&self, duality_strength: f64) -> (f64, f64) {
        let optimal_h = self.h_star + (1.0 - self.h_star) * duality_strength * 0.5;
        let optimal_el = (1.0 - optimal_h) * 0.8;
        (optimal_h, optimal_el)
    }

    pub fn predict_evolution_path(
        &self,
        current_h: f64,
        current_el: f64,
        steps: usize,
    ) -> Vec<(f64, f64)> {
        let mut path = Vec::with_capacity(steps);
        let mut h = current_h;
        let mut el = current_el;

        for _ in 0..steps {
            let dh = (self.h_star - h) * 0.1;
            let del = if h > self.h_crit {
                (1.0 - h - el) * 0.05
            } else {
                -el * 0.1
            };

            h = (h + dh).clamp(0.0, 1.0);
            el = (el + del).clamp(0.0, 1.0);

            path.push((h, el));
        }

        path
    }
}

impl Default for DualityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

pub fn compute_duality_balance(h: f64, el: f64) -> f64 {
    1.0 - (h - el).abs()
}

pub fn get_duality_phase(h: f64, el: f64) -> DualityPhase {
    let analyzer = DualityAnalyzer::new();
    let state = DualityState {
        health: h,
        evolution_level: el,
        duality_strength: 0.5,
    };
    analyzer.analyze(&state).phase
}
