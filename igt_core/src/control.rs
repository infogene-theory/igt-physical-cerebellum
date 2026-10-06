// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! P7 调控能力定义定理
//!
//! 绑定版本: 元衍拓扑学 V8.0 / drift_hash_root_v8.0
//!
//! 系统调控能力由信息量和自由度决定
#![allow(non_snake_case)]

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlCapacity {
    pub information: f64,
    pub freedom: f64,
    pub energy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlResult {
    pub capacity: f64,
    pub efficiency: f64,
    pub is_sufficient: bool,
    pub deficit: f64,
}

pub const MIN_CONTROL_CAPACITY: f64 = 0.3;

pub struct ControlAnalyzer {
    min_capacity: f64,
    energy_weight: f64,
    info_weight: f64,
    freedom_weight: f64,
}

impl ControlAnalyzer {
    pub fn new() -> Self {
        Self {
            min_capacity: MIN_CONTROL_CAPACITY,
            energy_weight: 0.4,
            info_weight: 0.35,
            freedom_weight: 0.25,
        }
    }

    pub fn analyze(&self, capacity: &ControlCapacity) -> ControlResult {
        let total = capacity.information * self.info_weight
            + capacity.freedom * self.freedom_weight
            + capacity.energy * self.energy_weight;

        let efficiency = if capacity.energy > 0.0 {
            total / capacity.energy
        } else {
            0.0
        };

        let is_sufficient = total >= self.min_capacity;
        let deficit = if is_sufficient {
            0.0
        } else {
            self.min_capacity - total
        };

        ControlResult {
            capacity: total,
            efficiency: efficiency.clamp(0.0, 1.0),
            is_sufficient,
            deficit,
        }
    }

    pub fn compute_required_capacity(&self, target_health: f64, current_health: f64) -> f64 {
        let gap = (target_health - current_health).abs();
        gap * 2.0
    }

    pub fn optimize_allocation(&self, total_energy: f64) -> ControlCapacity {
        ControlCapacity {
            information: total_energy * self.info_weight / self.energy_weight,
            freedom: total_energy * self.freedom_weight / self.energy_weight,
            energy: total_energy,
        }
    }
}

impl Default for ControlAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

pub fn compute_control_capacity(info: f64, freedom: f64, energy: f64) -> f64 {
    let analyzer = ControlAnalyzer::new();
    let capacity = ControlCapacity {
        information: info,
        freedom,
        energy,
    };
    analyzer.analyze(&capacity).capacity
}

pub fn has_sufficient_control(info: f64, freedom: f64, energy: f64) -> bool {
    compute_control_capacity(info, freedom, energy) >= MIN_CONTROL_CAPACITY
}
