// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! P6 三维惯性守恒定理
//!
//! 绑定版本: 元衍拓扑学 V8.0 / drift_hash_root_v8.0
//!
//! 健康度、进化水平、协同度三维惯性守恒
#![allow(non_snake_case)]

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InertiaVector {
    pub health_inertia: f64,
    pub evolution_inertia: f64,
    pub synergy_inertia: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InertiaResult {
    pub is_conserved: bool,
    pub total_inertia: f64,
    pub conservation_error: f64,
    pub dominant_dimension: usize,
}

pub const INERTIA_TOLERANCE: f64 = 0.1;

pub struct InertiaConservator {
    tolerance: f64,
    reference_inertia: f64,
}

impl InertiaConservator {
    pub fn new() -> Self {
        Self {
            tolerance: INERTIA_TOLERANCE,
            reference_inertia: 1.0,
        }
    }

    pub fn with_reference(reference: f64) -> Self {
        Self {
            tolerance: INERTIA_TOLERANCE,
            reference_inertia: reference,
        }
    }

    pub fn check(&self, inertia: &InertiaVector) -> InertiaResult {
        // I_total = √(I_S² + I_ω² + I_C²) — 欧几里得范数
        let total = (inertia.health_inertia.powi(2)
            + inertia.evolution_inertia.powi(2)
            + inertia.synergy_inertia.powi(2))
        .sqrt();

        let conservation_error = (total - self.reference_inertia).abs();

        let is_conserved = conservation_error < self.tolerance;

        let dominant_dimension = if inertia.health_inertia >= inertia.evolution_inertia
            && inertia.health_inertia >= inertia.synergy_inertia
        {
            0
        } else if inertia.evolution_inertia >= inertia.synergy_inertia {
            1
        } else {
            2
        };

        InertiaResult {
            is_conserved,
            total_inertia: total,
            conservation_error,
            dominant_dimension,
        }
    }

    pub fn compute_inertia_transfer(
        &self,
        from_dim: usize,
        to_dim: usize,
        amount: f64,
        current: &InertiaVector,
    ) -> InertiaVector {
        let mut new_inertia = current.clone();

        let dims = [
            &mut new_inertia.health_inertia,
            &mut new_inertia.evolution_inertia,
            &mut new_inertia.synergy_inertia,
        ];

        if from_dim < 3 && to_dim < 3 {
            *dims[from_dim] = (*dims[from_dim] - amount).max(0.0);
            *dims[to_dim] = (*dims[to_dim] + amount).max(0.0);
        }

        new_inertia
    }

    pub fn normalize(&self, inertia: &InertiaVector) -> InertiaVector {
        let total = inertia.health_inertia + inertia.evolution_inertia + inertia.synergy_inertia;

        if total <= 0.0 {
            return InertiaVector {
                health_inertia: 1.0 / 3.0,
                evolution_inertia: 1.0 / 3.0,
                synergy_inertia: 1.0 / 3.0,
            };
        }

        InertiaVector {
            health_inertia: inertia.health_inertia / total,
            evolution_inertia: inertia.evolution_inertia / total,
            synergy_inertia: inertia.synergy_inertia / total,
        }
    }
}

impl Default for InertiaConservator {
    fn default() -> Self {
        Self::new()
    }
}

pub fn compute_total_inertia(h: f64, el: f64, k: f64) -> f64 {
    // I_total = √(I_S² + I_ω² + I_C²) — 欧几里得范数
    (h.powi(2) + el.powi(2) + k.powi(2)).sqrt()
}

pub fn is_inertia_conserved(h: f64, el: f64, k: f64, reference: f64) -> bool {
    let total = compute_total_inertia(h, el, k);
    (total - reference).abs() < INERTIA_TOLERANCE
}
