// SPDX-License-Identifier: BSL-1.0
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! 多模态感知节汇总（MIGRATION-FINAL flutter 归一第八例）
//!
//! 从 flutter `lib/services/perception_status.dart`（第354批 Phase 2/4·感知 8 通道前端汇总）
//! 迁移——**移动不改算法**。
//!
//! 五源感知通道状态 → 覆盖度 + 分级 code + 缺通道列表。数据 100% 本地（R3）。
//!
//! T-182（理论侧163拆分线 §2）：core 只出**中性 code + 数值/结构化字段**；覆盖度成句
//! 文案与补数据建议归 agent（`igt_agent::presentation`）；展示标识 `icon` 与摘要
//! `detail` 属 UI/agent 层，core 不再持有。

use serde::{Deserialize, Serialize};

/// 感知通道（T-182：`icon`/`detail` 属展示标识与成句摘要，归 agent/UI 层）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 感知通道 {
    pub name: String,
    /// 该通道是否有数据
    pub active: bool,
    pub count: i64,
}

/// 感知状态汇总输出（T-182：core 只出中性 code + 数值/结构化字段）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 感知汇总 {
    pub active_count: usize,
    /// 覆盖度 0-1
    pub coverage: f64,
    /// 覆盖度分级中性 code: `"all"` / `"major"` / `"partial"` / `"few"`
    pub coverage_grade: String,
    /// 未激活通道名列表（中性结构化字段；成句建议归 agent）
    pub 缺通道: Vec<String>,
}

/// 覆盖度 0-1（空通道列表 → 0）
pub fn 覆盖度(通道: &[感知通道]) -> f64 {
    if 通道.is_empty() {
        return 0.0;
    }
    let 激活 = 通道.iter().filter(|c| c.active).count();
    激活 as f64 / 通道.len() as f64
}

/// 覆盖度分级中性 code: ≥0.8 `"all"` / ≥0.5 `"major"` / ≥0.3 `"partial"` / 其余 `"few"`
pub fn 覆盖度分级(coverage: f64) -> String {
    let grade = if coverage >= 0.8 {
        "all"
    } else if coverage >= 0.5 {
        "major"
    } else if coverage >= 0.3 {
        "partial"
    } else {
        "few"
    };
    grade.to_string()
}

/// 未激活通道名列表（全激活 → 空；成句建议归 agent）
pub fn 缺通道列表(通道: &[感知通道]) -> Vec<String> {
    通道
        .iter()
        .filter(|c| !c.active)
        .map(|c| c.name.clone())
        .collect()
}

/// 汇总: 覆盖度 + 分级 code + 缺通道列表
pub fn 感知状态汇总(通道: &[感知通道]) -> 感知汇总 {
    let coverage = 覆盖度(通道);
    感知汇总 {
        active_count: 通道.iter().filter(|c| c.active).count(),
        coverage,
        coverage_grade: 覆盖度分级(coverage),
        缺通道: 缺通道列表(通道),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn 通道(name: &str, active: bool) -> 感知通道 {
        感知通道 {
            name: name.to_string(),
            active,
            count: 0,
        }
    }

    #[test]
    fn test_覆盖度计算() {
        let c = vec![通道("a", true), 通道("b", true), 通道("c", false)];
        let s = 感知状态汇总(&c);
        assert_eq!(s.active_count, 2);
        assert!((s.coverage - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_覆盖度分级_中性code() {
        // T-182: 分级为中性 code（中文文案归 agent presentation）
        // 5/5 = 1.0 → all
        let all: Vec<感知通道> = (0..5).map(|i| 通道(&format!("c{i}"), true)).collect();
        assert_eq!(感知状态汇总(&all).coverage_grade, "all");
        // 1/3 = 0.333 → partial
        let part = vec![通道("a", true), 通道("b", false), 通道("c", false)];
        assert_eq!(感知状态汇总(&part).coverage_grade, "partial");
        // 1/2 = 0.5 → major
        let main = vec![通道("a", true), 通道("b", false)];
        assert_eq!(感知状态汇总(&main).coverage_grade, "major");
        // 0/2 = 0 → few
        let few = vec![通道("a", false), 通道("b", false)];
        assert_eq!(感知状态汇总(&few).coverage_grade, "few");
    }

    #[test]
    fn test_分级边界() {
        assert_eq!(覆盖度分级(0.8), "all", "0.8 边界 → all");
        assert_eq!(覆盖度分级(0.79), "major");
        assert_eq!(覆盖度分级(0.5), "major", "0.5 边界 → major");
        assert_eq!(覆盖度分级(0.49), "partial");
        assert_eq!(覆盖度分级(0.3), "partial", "0.3 边界 → partial");
        assert_eq!(覆盖度分级(0.29), "few");
    }

    #[test]
    fn test_缺通道列表_列名() {
        let c = vec![通道("HRV", true), 通道("运动", false), 通道("睡眠", false)];
        let s = 感知状态汇总(&c);
        assert_eq!(s.缺通道, vec!["运动".to_string(), "睡眠".to_string()]);
        assert!(!s.缺通道.contains(&"HRV".to_string()), "已激活通道不应出现在待补列表");
    }

    #[test]
    fn test_全覆盖_缺通道为空() {
        let c = vec![通道("HRV", true)];
        assert!(感知状态汇总(&c).缺通道.is_empty());
    }

    #[test]
    fn test_空通道() {
        let s = 感知状态汇总(&[]);
        assert_eq!(s.coverage, 0.0);
        assert_eq!(s.active_count, 0);
        assert_eq!(s.coverage_grade, "few");
    }

    #[test]
    fn test_序列化往返() {
        let c = 通道("HRV", true);
        let json = serde_json::to_string(&c).unwrap();
        let back: 感知通道 = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "HRV");
        assert!(back.active);
    }
}
