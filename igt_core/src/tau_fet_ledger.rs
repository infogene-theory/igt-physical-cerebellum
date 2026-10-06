// SPDX-License-Identifier: BSL-1.0
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! tau-FET 本地贡献账本规则（MIGRATION-FINAL flutter 归一第十三例）
//!
//! 从 flutter `lib/services/tau_fet_ledger.dart`（batch 393, D12 agent-frontend）
//! 迁移**纯规则部分**——**移动不改算法**。
//!
//! 对齐 engine `tau_fet.rs`（D12 §3/§4）: 贡献→拉取许可, 无交易/兑换, 本地账本 +
//! 接口预约（服务请求携带贡献凭证）。tau-FET 不是货币, 是跨尺度自由能贡献/消耗的
//! 统一度量凭证（P21 对称性: 贡献 ∝ τ 延长）。
//! 账本状态（credit/entries/entropyDebt）保留在 flutter 侧; 本模块只提供**判定规则**。
//!
//! T-182（例10·纠缠项，理论侧163拆分线 §4）**只标注不激活**：
//!   - 本模块 = **无状态规则镜像**（递减因子/贡献值/成本/权限重算/是否免费，纯判定，无成句文案）
//!     ⇒ 按拆分线"core 算量"保留；
//!   - engine `tau_fet.rs` = **有状态账本内核**（贡献条目/账本/凭证）⇒ "engine 持公共规则"，
//!     本轮 engine **保持冻结**：仅同批加一处分层标注（注释），零功能改动、不接线；
//!   - agent 私域呈现（个体账本措辞/明细）按需另立，不在本单。

use serde::{Deserialize, Serialize};

/// 贡献类型（D12 §3.1 三种）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum 贡献类型 {
    Gene,
    Compute,
    Data,
}

/// 消耗类型（D12 §3.2 两种）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum 消耗类型 {
    GenePull,
    EngineCompute,
}

/// 拉取许可（D12 §3.3）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum 拉取权限 {
    Basic,
    Standard,
    Advanced,
    Restricted,
}

/// D12 §4.3 V4 递减因子: 第1次 ×1.0, 第100次 ×0.05（下限 0.05）
pub fn 递减因子(实例序数: i64) -> f64 {
    if 实例序数 <= 1 {
        return 1.0;
    }
    (1.0 / 实例序数 as f64).clamp(0.05, 1.0)
}

/// 贡献基础值（D12 §3.1: 基因 10 / 算力 5 / 数据 8）
pub fn 贡献基础值(类型: 贡献类型) -> f64 {
    match 类型 {
        贡献类型::Gene => 10.0,
        贡献类型::Compute => 5.0,
        贡献类型::Data => 8.0,
    }
}

/// 贡献实际入账值 = 基础值 × 递减因子
pub fn 贡献价值(类型: 贡献类型, 实例序数: i64) -> f64 {
    贡献基础值(类型) * 递减因子(实例序数)
}

/// 消耗成本（D12 §3.2: 基因拉取 3 / 引擎算力 6）
pub fn 消耗成本(类型: 消耗类型) -> f64 {
    match 类型 {
        消耗类型::GenePull => 3.0,
        消耗类型::EngineCompute => 6.0,
    }
}

/// 权限重算（D12 §3.3; 移动不改算法: 已受限保持受限）:
///   credit ≥ 20 且无熵债 → advanced; credit ≥ 5 → standard; 否则 basic
pub fn 权限重算(credit: f64, 熵债: f64, 已受限: bool) -> 拉取权限 {
    if 已受限 {
        return 拉取权限::Restricted;
    }
    if credit >= 20.0 && 熵债 <= 0.0 {
        拉取权限::Advanced
    } else if credit >= 5.0 {
        拉取权限::Standard
    } else {
        拉取权限::Basic
    }
}

/// 跨尺度免费规则（V5, P22 §10.10）: local/family/G0public 免费, G1/G2 消耗
pub fn 是否免费(尺度: &str) -> bool {
    matches!(尺度, "local" | "family" | "G0public")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_递减因子_首次与百次() {
        // 对齐 flutter 原用例（V4）: 1st ×1.0, 100th ×0.05
        assert_eq!(递减因子(1), 1.0);
        assert_eq!(递减因子(0), 1.0, "≤1 → 1.0");
        assert!((递减因子(100) - 0.05).abs() < 1e-9, "100 次 → 0.05");
        assert!(递减因子(1000) >= 0.05, "下限 0.05");
        assert!((递减因子(2) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_贡献基础值() {
        assert_eq!(贡献基础值(贡献类型::Gene), 10.0);
        assert_eq!(贡献基础值(贡献类型::Compute), 5.0);
        assert_eq!(贡献基础值(贡献类型::Data), 8.0);
    }

    #[test]
    fn test_贡献价值_累减() {
        // 对齐 flutter V1 用例: 3 次基因贡献 → 10*(1+0.5+0.333)=18.33
        let total = (1..=3).map(|i| 贡献价值(贡献类型::Gene, i)).sum::<f64>();
        assert!(total > 15.0 && total < 20.0, "18.33: {total}");
        assert!((贡献价值(贡献类型::Gene, 1) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_消耗成本() {
        assert_eq!(消耗成本(消耗类型::GenePull), 3.0);
        assert_eq!(消耗成本(消耗类型::EngineCompute), 6.0);
    }

    #[test]
    fn test_权限阈值() {
        assert_eq!(权限重算(0.0, 0.0, false), 拉取权限::Basic);
        assert_eq!(权限重算(4.99, 0.0, false), 拉取权限::Basic);
        assert_eq!(权限重算(5.0, 0.0, false), 拉取权限::Standard, "≥5 → standard");
        assert_eq!(权限重算(19.9, 0.0, false), 拉取权限::Standard);
        assert_eq!(权限重算(20.0, 0.0, false), 拉取权限::Advanced, "≥20 且无债 → advanced");
    }

    #[test]
    fn test_权限_有熵债不进高级() {
        assert_eq!(权限重算(30.0, 1.0, false), 拉取权限::Standard, "有熵债 → 降 standard");
    }

    #[test]
    fn test_权限_受限保持() {
        assert_eq!(权限重算(100.0, 0.0, true), 拉取权限::Restricted, "已受限保持受限");
    }

    #[test]
    fn test_免费规则() {
        assert!(是否免费("local"));
        assert!(是否免费("family"));
        assert!(是否免费("G0public"));
        assert!(!是否免费("G1"));
        assert!(!是否免费("G2"));
        assert!(!是否免费(""), "未知尺度 → 不免费");
    }

    #[test]
    fn test_序列化往返() {
        let p = 拉取权限::Advanced;
        let json = serde_json::to_string(&p).unwrap();
        let back: 拉取权限 = serde_json::from_str(&json).unwrap();
        assert_eq!(back, 拉取权限::Advanced);
    }
}
