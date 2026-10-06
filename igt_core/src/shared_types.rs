// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! 共享纯类型层（MIGRATION-FINAL Step0-2a）
//!
//! engine 与 agent **共用**的纯数据结构下沉 core——消除跨端重复定义（总攻单 §1:
//! "engine 多主体链与 agent 共用同一类型"）。本模块**只有类型与纯函数**，无 IO、
//! 无异步、无全局状态（core 五类约束 ARCH-1 v2.1 §5）。
//!
//! ## 来源（逐字搬迁，不改结构/不改命名）
//! - `OrganId` ← engine `l1/organ_router.rs:31-90`（八器官标识，P34 工程拓扑）
//! - `FrequencyPayload` / `BeatAtom` / `ReferenceSystemType` / `RelaxationPayload`
//!   ← engine `l1/igt_message.rs:86-190`（BP-01 频率/弛豫载体）
//!
//! **命名冲突说明**: 本模块 `BeatAtom`（BP-01 拍频原子: atom_id/frequency_a/frequency_b/
//! beat_ratio…）与 `beat_kernel::BeatAtom`（beat_library 频率原子: token/freq_text…）是
//! **两个不同结构**，分模块隔离保留原名（Step2 engine 改引时按模块路径区分，改动面最小）。
//!
//! engine 原位文件由 Step2 删除（本批不动，中间态允许重复）。

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use serde::{Deserialize, Serialize};

// ============================================================================
// 八器官标识（下沉自 engine l1/organ_router.rs）
// ============================================================================

/// 八器官标识（P34 工程拓扑: O1 认知 / O2 经济 / O3 安全 / O4 生产 /
/// O5 支撑 / O6 观测 / O7 繁衍 / O8 调控）
///
/// 注: 代码编号按功能分组，论文编号按涌现顺序；跨模块引用使用命名枚举，不依赖数字编号。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum OrganId {
    #[default]
    O1_Cognitive,
    O2_Economic,
    O3_Safety,
    O4_Production,
    O5_Support,
    O6_Observation,
    O7_Reproduction,
    O8_Regulation,
}

impl OrganId {
    pub fn name(&self) -> &'static str {
        match self {
            Self::O1_Cognitive => "认知器官",
            Self::O2_Economic => "经济器官",
            Self::O3_Safety => "安全器官",
            Self::O4_Production => "生产器官",
            Self::O5_Support => "支撑器官",
            Self::O6_Observation => "观测器官",
            Self::O7_Reproduction => "繁衍器官",
            Self::O8_Regulation => "调控器官",
        }
    }

    pub fn id(&self) -> u8 {
        match self {
            Self::O1_Cognitive => 1,
            Self::O2_Economic => 2,
            Self::O3_Safety => 3,
            Self::O4_Production => 4,
            Self::O5_Support => 5,
            Self::O6_Observation => 6,
            Self::O7_Reproduction => 7,
            Self::O8_Regulation => 8,
        }
    }

    pub fn all() -> Vec<OrganId> {
        vec![
            Self::O1_Cognitive,
            Self::O2_Economic,
            Self::O3_Safety,
            Self::O4_Production,
            Self::O5_Support,
            Self::O6_Observation,
            Self::O7_Reproduction,
            Self::O8_Regulation,
        ]
    }
}

impl std::fmt::Display for OrganId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "O{}", self.id())
    }
}

// ============================================================================
// BP-01 频率 / 弛豫载体（下沉自 engine l1/igt_message.rs）
// ============================================================================

/// 参考系类型 (BP-01 §3.2 定理3.2)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReferenceSystemType {
    Light,
    Sound,
    Space,
    Time,
}

impl ReferenceSystemType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Light => "R_light 光频",
            Self::Sound => "R_sound 声频",
            Self::Space => "R_space 空间",
            Self::Time => "R_time 时间",
        }
    }

    pub fn global_dimension(&self) -> &'static str {
        match self {
            Self::Light => "D3 能量-自由能弛豫",
            Self::Sound => "D4 全域场耦合共振",
            Self::Space => "D1 空间拓扑弛豫",
            Self::Time => "D2 时间演化弛豫",
        }
    }

    pub fn all() -> Vec<ReferenceSystemType> {
        vec![Self::Light, Self::Sound, Self::Space, Self::Time]
    }
}

/// 拍频原子 (BP-01 §5.3, §S-01 L3 格式层)
///
/// 拍频是所有保拓扑参考系变换下的绝对不变量。信息跨参考系传递的物理基础。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeatAtom {
    pub atom_id: String,
    pub frequency_a: f64,
    pub frequency_b: f64,
    /// 拍频比 = |f_a - f_b| / √(f_a·f_b)
    pub beat_ratio: f64,
    pub amplitude: f64,
    pub phase: f64,
    /// +1 同手性耦合, -1 异手性耦合
    pub chirality: i8,
    pub reference_system: ReferenceSystemType,
}

impl BeatAtom {
    /// 拍频比 = |f_a − f_b| / √(f_a·f_b)（纯计算，与字段定义一致）
    pub fn compute_beat_ratio(frequency_a: f64, frequency_b: f64) -> f64 {
        let denom = (frequency_a * frequency_b).sqrt();
        if denom > 1e-12 {
            (frequency_a - frequency_b).abs() / denom
        } else {
            0.0
        }
    }
}

/// BP-01 频率载体: 四维频率矢量 + 拍频原子组合
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyPayload {
    /// 四维频率矢量 (R_light, R_sound, R_space, R_time)
    pub frequency_vector: [f64; 4],
    /// 拍频原子组合 (跨参考系守恒的信息载体, BP-01 §5)
    pub beat_atoms: Vec<BeatAtom>,
    /// 参考系锚定: 观测者基准谱系的哈希
    pub reference_frame_hash: String,
    /// 信噪比估计 (用于 L_noise 计算)
    pub snr: f64,
}

/// BP-01 §10 弛豫度量载体
///
/// 对接到 IGT-5-09 §6 RelaxationTimeProfile + 三重信息损失分解
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelaxationPayload {
    /// 弛豫时间谱系 (IGT-5-09 §6)
    pub tau_s: f64,
    pub tau_omega: f64,
    pub tau_c: f64,
    pub mean_tau: f64,
    pub variance_tau: f64,
    pub stability_score: f64,
    /// 节律异常标记 [0,1]
    pub rhythm_anomaly: f64,
    /// 与用户基准谱系的综合共振耦合指数 K_AB (BP-01 §11.2)
    pub user_observer_K_AB: f64,
    /// 与用户匹配频率 match_score (BP-01 §11.1)
    pub user_match_score: f64,
    /// 三重信息损失分解 (BP-01 §10 定理10.1)
    pub loss_freq: f64,
    pub loss_phase: f64,
    pub loss_noise: f64,
    /// 综合RCI k_total = √(1-(L_freq+L_phase+L_noise))
    pub rci_k_total: f64,
    /// 信息传递效率 η = k_total²
    pub transfer_efficiency: f64,
}

impl RelaxationPayload {
    /// 综合 RCI k_total = √(1 − (L_freq + L_phase + L_noise))（纯计算，与字段定义一致）
    pub fn compute_rci_k_total(loss_freq: f64, loss_phase: f64, loss_noise: f64) -> f64 {
        let 内 = 1.0 - (loss_freq + loss_phase + loss_noise);
        if 内 > 0.0 {
            内.sqrt()
        } else {
            0.0
        }
    }

    /// 信息传递效率 η = k_total²（纯计算，与字段定义一致）
    pub fn compute_transfer_efficiency(k_total: f64) -> f64 {
        k_total * k_total
    }
}

// ============================================================================
// SBI-8 健康度量结构（Step0-3b，下沉自 engine l1/sbi8.rs:150-184）
//
// PM 裁决（2026-08-29）: 先把 Sbi8Result 纯结构下沉 shared_types（保持无 IO），
// 再搬 intervention_validator。本组结构字段全为 String/f64/Vec——**无 organ_router
// 依赖**（sbi8.rs 顶部的 organ_router use 是给 Sbi8Calculator 计算器用的，结构本身不依赖）。
// Sbi8Calculator 计算引擎随业务归属处理（本批不下沉）。
// ============================================================================

/// 单器官健康输入
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganHealthInput {
    pub organ_id: String,
    pub sigma: f64,
    pub k: f64,
    pub tau_minutes: f64,
}

/// 单器官健康结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganHealthResult {
    pub organ_id: String,
    pub organ_label: String,
    pub organ_icon: String,
    pub thermodynamic_type: String,
    pub sigma: f64,
    pub k: f64,
    pub tau_minutes: f64,
    pub s_sigma: f64,
    pub s_k: f64,
    pub s_tau: f64,
    pub health_score: f64,
    pub global_weight: f64,
    pub status: String,
}

/// SBI-8 综合健康度量结果（八器官聚合）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sbi8Result {
    pub sbi8_score: f64,
    pub health_level: String,
    pub health_label_cn: String,
    pub organ_details: Vec<OrganHealthResult>,
    pub total_weight: f64,
    pub weighted_sum: f64,
    pub timestamp: u64,
}

// ============================================================================
// 测试（搬迁验证: 枚举/结构/纯计算与 engine 原实现一致）
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 测试_器官标识_编号与命名() {
        assert_eq!(OrganId::all().len(), 8, "八器官");
        assert_eq!(OrganId::O1_Cognitive.id(), 1);
        assert_eq!(OrganId::O8_Regulation.id(), 8);
        assert_eq!(OrganId::O2_Economic.name(), "经济器官");
        assert_eq!(format!("{}", OrganId::O3_Safety), "O3");
        assert_eq!(OrganId::default(), OrganId::O1_Cognitive);
    }

    #[test]
    fn 测试_参考系类型() {
        assert_eq!(ReferenceSystemType::all().len(), 4, "四参考系");
        assert_eq!(ReferenceSystemType::Light.name(), "R_light 光频");
        assert_eq!(ReferenceSystemType::Time.global_dimension(), "D2 时间演化弛豫");
    }

    #[test]
    fn 测试_拍频比计算() {
        // f_a=100, f_b=400 → |Δf|/√(f_a·f_b) = 300/200 = 1.5
        let r = BeatAtom::compute_beat_ratio(100.0, 400.0);
        assert!((r - 1.5).abs() < 1e-12, "拍频比={}", r);
        // 同频 → 0
        assert_eq!(BeatAtom::compute_beat_ratio(100.0, 100.0), 0.0);
        // 零频 → 0（防除零）
        assert_eq!(BeatAtom::compute_beat_ratio(0.0, 100.0), 0.0);
    }

    #[test]
    fn 测试_弛豫载体纯计算() {
        // L=0.1+0.1+0.1=0.3 → k_total=√0.7
        let k = RelaxationPayload::compute_rci_k_total(0.1, 0.1, 0.1);
        assert!((k - 0.7_f64.sqrt()).abs() < 1e-12, "k_total={}", k);
        assert!((RelaxationPayload::compute_transfer_efficiency(k) - 0.7).abs() < 1e-12);
        // 损失和 ≥1 → 0（诚实下限，不产生负值 NaN）
        assert_eq!(RelaxationPayload::compute_rci_k_total(0.5, 0.3, 0.4), 0.0);
    }

    /// Step0-3b: SBI-8 结构下沉验证（字段全 String/f64，序列化往返一致）
    #[test]
    fn 测试_sbi8结构下沉() {
        let 器官 = OrganHealthResult {
            organ_id: "O3".to_string(),
            organ_label: "安全器官".to_string(),
            organ_icon: "🛡".to_string(),
            thermodynamic_type: "耗散型".to_string(),
            sigma: 0.5,
            k: 2.0,
            tau_minutes: 10.0,
            s_sigma: 0.8,
            s_k: 0.9,
            s_tau: 0.7,
            health_score: 0.85,
            global_weight: 0.15,
            status: "健康".to_string(),
        };
        let 结果 = Sbi8Result {
            sbi8_score: 82.5,
            health_level: "B".to_string(),
            health_label_cn: "良好".to_string(),
            organ_details: vec![器官],
            total_weight: 1.0,
            weighted_sum: 82.5,
            timestamp: 1_700_000_000,
        };
        assert_eq!(结果.organ_details.len(), 1);
        assert_eq!(结果.organ_details[0].organ_id, "O3");
        // 序列化往返（纯数据结构，无 IO）
        let json = serde_json::to_string(&结果).unwrap();
        let 还原: Sbi8Result = serde_json::from_str(&json).unwrap();
        assert!((还原.sbi8_score - 82.5).abs() < 1e-12);
        assert_eq!(还原.health_level, "B");
        assert_eq!(还原.timestamp, 1_700_000_000);
        // 输入结构
        let 输入 = OrganHealthInput {
            organ_id: "O1".to_string(),
            sigma: 0.3,
            k: 3.0,
            tau_minutes: 5.0,
        };
        assert_eq!(输入.organ_id, "O1");
    }

    #[test]
    fn 测试_序列化roundtrip() {
        let 原子 = BeatAtom {
            atom_id: "a1".to_string(),
            frequency_a: 100.0,
            frequency_b: 400.0,
            beat_ratio: 1.5,
            amplitude: 0.8,
            phase: 0.25,
            chirality: 1,
            reference_system: ReferenceSystemType::Sound,
        };
        let json = serde_json::to_string(&原子).unwrap();
        let 还原: BeatAtom = serde_json::from_str(&json).unwrap();
        assert_eq!(还原.atom_id, "a1");
        assert_eq!(还原.reference_system, ReferenceSystemType::Sound);
        assert!((还原.beat_ratio - 1.5).abs() < 1e-12);
    }
}
