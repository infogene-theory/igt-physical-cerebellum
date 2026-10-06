// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! P3 三重锁定定理
//!
//! 绑定版本: 元衍拓扑学 V8.0 / drift_hash_root_v8.0
//!
//! 结构-功能-演化三重锁定保证系统稳定性
#![allow(non_snake_case)]

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LockType {
    Structure,
    Function,
    Evolution,
}

const UNLOCK_ORDER: [LockType; 3] = [LockType::Evolution, LockType::Function, LockType::Structure];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockState {
    pub structure_locked: bool,
    pub function_locked: bool,
    pub evolution_locked: bool,
    pub lock_strength: f64,
    pub next_unlock: LockType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleLockResult {
    pub all_locked: bool,
    pub lock_count: u8,
    pub stability_score: f64,
    pub warnings: Vec<String>,
}

pub struct TripleLockEngine {
    structure_threshold: f64,
    function_threshold: f64,
    evolution_threshold: f64,
}

impl TripleLockEngine {
    pub fn new() -> Self {
        Self {
            structure_threshold: 0.8,
            function_threshold: 0.7,
            evolution_threshold: 0.6,
        }
    }

    pub fn check(&self, structure: f64, function: f64, evolution: f64) -> TripleLockResult {
        let structure_locked = self.verify_lock(LockType::Structure, structure);
        let function_locked = self.verify_lock(LockType::Function, function);
        let evolution_locked = self.verify_lock(LockType::Evolution, evolution);

        let mut lock_count = 0u8;
        let mut warnings = Vec::new();

        if structure_locked {
            lock_count += 1;
        } else {
            warnings.push("结构锁定失效".to_string());
        }

        if function_locked {
            lock_count += 1;
        } else {
            warnings.push("功能锁定失效".to_string());
        }

        if evolution_locked {
            lock_count += 1;
        } else {
            warnings.push("演化锁定失效".to_string());
        }

        let lock_strength = self.compute_lock_strength(structure, function, evolution);
        let stability_score = (lock_count as f64 / 3.0) * lock_strength;

        TripleLockResult {
            all_locked: lock_count == 3,
            lock_count,
            stability_score,
            warnings,
        }
    }

    pub fn verify_lock(&self, lock_type: LockType, value: f64) -> bool {
        let threshold = match lock_type {
            LockType::Structure => self.structure_threshold,
            LockType::Function => self.function_threshold,
            LockType::Evolution => self.evolution_threshold,
        };

        value >= threshold
    }

    pub fn compute_lock_strength(&self, structure: f64, function: f64, evolution: f64) -> f64 {
        let s = if structure >= self.structure_threshold {
            1.0
        } else {
            structure / self.structure_threshold
        };
        let f = if function >= self.function_threshold {
            1.0
        } else {
            function / self.function_threshold
        };
        let e = if evolution >= self.evolution_threshold {
            1.0
        } else {
            evolution / self.evolution_threshold
        };

        (s + f + e) / 3.0
    }

    pub fn next_unlock_type(&self, state: &LockState) -> LockType {
        for &lock_type in &UNLOCK_ORDER {
            let is_locked = match lock_type {
                LockType::Structure => state.structure_locked,
                LockType::Function => state.function_locked,
                LockType::Evolution => state.evolution_locked,
            };
            if is_locked {
                return lock_type;
            }
        }
        LockType::Evolution
    }

    pub fn unlock(&self, state: &mut LockState, lock_type: LockType) -> Result<(), String> {
        let expected = self.next_unlock_type(state);
        if lock_type != expected {
            return Err(format!(
                "解锁顺序错误: 期望先解锁 {:?}, 实际尝试解锁 {:?}",
                expected, lock_type
            ));
        }

        match lock_type {
            LockType::Structure => {
                if !state.structure_locked {
                    return Err("结构锁已处于解锁状态".to_string());
                }
                state.structure_locked = false;
            }
            LockType::Function => {
                if !state.function_locked {
                    return Err("功能锁已处于解锁状态".to_string());
                }
                state.function_locked = false;
            }
            LockType::Evolution => {
                if !state.evolution_locked {
                    return Err("演化锁已处于解锁状态".to_string());
                }
                state.evolution_locked = false;
            }
        }

        state.next_unlock = self.next_unlock_type(state);
        Ok(())
    }
}

impl Default for TripleLockEngine {
    fn default() -> Self {
        Self::new()
    }
}

pub fn is_triple_locked(structure: f64, function: f64, evolution: f64) -> bool {
    let engine = TripleLockEngine::new();
    engine.check(structure, function, evolution).all_locked
}
