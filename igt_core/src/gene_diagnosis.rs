// SPDX-License-Identifier: BSL-1.0
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! 基因驱动诊断（MIGRATION-FINAL flutter 归一第十例）
//!
//! 从 flutter `lib/services/gene_diagnosis.dart`（第307批 Phase 4, P22 §10.7）
//! 迁移——**移动不改算法**（含文案格式与容差默认值）。
//!
//! 有稳态基因的领域可做诊断，校验模只做谱级提示（P22 §10.7）:
//!   当前窗口蒸馏基因 ω 匹配基因库稳态基因（双校验+锚定）→ 语义诊断
//!   匹配校验模（无锚定语义）→ 谱级提示
//! 诊断边界随基因库成熟度动态扩展。
//! 输入为基因库记录的**诊断相关字段**（omega/doubleVerified/anchored/label）——maturity 计算一并归位。

use serde::{Deserialize, Serialize};

/// 当前窗口蒸馏基因候选（诊断仅需 ω）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 基因候选 {
    pub omega: f64,
}

/// 基因库记录输入（诊断相关字段）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 基因记录输入 {
    pub omega: f64,
    pub double_verified: bool,
    pub anchored: bool,
    /// 语义标签（开会/跑步/吵架/放松...）
    pub label: Option<String>,
}

/// 默认 ω 匹配容差（rad/s）
pub const OMEGA_TOLERANCE: f64 = 0.15;

/// 三级成熟度（P22 §10.4; 移动不改算法: 稳态基因 = 双校验 + 人工锚定 + 复现）
///
/// T-182（理论侧163拆分线 §2）：返回**中性 code**（`"stable"`/`"verified"`/`"candidate"`），
/// 面向用户的中文等级名归 agent（`igt_agent::presentation`）。
pub fn 成熟度(rec: &基因记录输入) -> &'static str {
    if rec.double_verified && rec.anchored {
        "stable"
    } else if rec.double_verified {
        "verified"
    } else {
        "candidate"
    }
}

/// 诊断条目（结构化中性输出；对应用户成句文案归 agent）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 诊断条目 {
    /// 中性类型 code: `"diagnosis"`（稳态基因激活）/ `"spectral_hint"`（谱级提示）
    pub kind: String,
    /// 匹配命中的当前基因 ω
    pub omega: f64,
    /// 稳态基因锚定语义标签（库中数据，非 core 措辞；无则 None，由 agent 给默认名）
    pub label: Option<String>,
}

/// 基因驱动诊断（P22 §10.7: 稳态基因→语义诊断, 校验模→谱级提示）
///
/// 对每个当前基因, 在库中找 ω 容差内的第一个稳态基因（否则第一个校验模）,
/// 产出**结构化诊断条目**（T-182：core 只出中性 code + 数值/结构化字段）;
/// 无匹配 → 不产出（诚实）。
pub fn 基因驱动诊断(当前: &[基因候选], 库: &[基因记录输入], 容差: f64) -> Vec<诊断条目> {
    let mut results = Vec::new();
    for gene in 当前 {
        let mut stable: Option<&基因记录输入> = None;
        let mut verified: Option<&基因记录输入> = None;
        for rec in 库 {
            if (rec.omega - gene.omega).abs() < 容差 {
                if 成熟度(rec) == "stable" && stable.is_none() {
                    stable = Some(rec);
                }
                if rec.double_verified && verified.is_none() {
                    verified = Some(rec);
                }
            }
        }
        if let Some(s) = stable {
            results.push(诊断条目 {
                kind: "diagnosis".to_string(),
                omega: gene.omega,
                label: s.label.clone(),
            });
        } else if verified.is_some() {
            results.push(诊断条目 {
                kind: "spectral_hint".to_string(),
                omega: gene.omega,
                label: None,
            });
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn 候选(omega: f64) -> 基因候选 {
        基因候选 { omega }
    }

    fn 记录(omega: f64, double_verified: bool, anchored: bool, label: Option<&str>) -> 基因记录输入 {
        基因记录输入 {
            omega,
            double_verified,
            anchored,
            label: label.map(|s| s.to_string()),
        }
    }

    #[test]
    fn test_成熟度三级_中性code() {
        // T-182: 成熟度为中性 code（中文等级名归 agent presentation）
        assert_eq!(成熟度(&记录(1.0, true, true, None)), "stable");
        assert_eq!(成熟度(&记录(1.0, true, false, None)), "verified");
        assert_eq!(成熟度(&记录(1.0, false, false, None)), "candidate");
        // 锚定但未双校验 → 仍候选（移动不改算法）
        assert_eq!(成熟度(&记录(1.0, false, true, None)), "candidate");
    }

    #[test]
    fn test_稳态基因匹配_语义诊断() {
        let 库 = vec![记录(0.5, true, true, Some("刚跑完步"))];
        let 当前 = vec![候选(0.52)];
        let r = 基因驱动诊断(&当前, &库, OMEGA_TOLERANCE);
        assert_eq!(r.len(), 1, "应产出 1 条: {:?}", r);
        assert_eq!(r[0].kind, "diagnosis", "稳态基因带锚定语义 → 语义诊断");
        assert_eq!(r[0].label.as_deref(), Some("刚跑完步"), "锚定标签随条目透传");
        assert!((r[0].omega - 0.52).abs() < 1e-12);
    }

    #[test]
    fn test_仅校验模_谱级提示() {
        let 库 = vec![记录(0.5, true, false, None)];
        let 当前 = vec![候选(0.52)];
        let r = 基因驱动诊断(&当前, &库, OMEGA_TOLERANCE);
        assert!(r.iter().any(|x| x.kind == "spectral_hint"));
        assert!(r.iter().all(|x| x.kind != "diagnosis"), "校验模不可做语义诊断");
    }

    #[test]
    fn test_候选模_无诊断() {
        let 库 = vec![记录(0.5, false, false, None)];
        let 当前 = vec![候选(0.52)];
        let r = 基因驱动诊断(&当前, &库, OMEGA_TOLERANCE);
        assert!(r.is_empty(), "候选模 → 诚实不产出");
    }

    #[test]
    fn test_ω不匹配_无诊断() {
        let 库 = vec![记录(0.5, true, true, Some("冥想"))];
        let 当前 = vec![候选(1.3)];
        assert!(基因驱动诊断(&当前, &库, OMEGA_TOLERANCE).is_empty());
    }

    #[test]
    fn test_多基因_各产一条() {
        let 库 = vec![记录(0.5, true, true, None), 记录(2.0, true, false, None)];
        let 当前 = vec![候选(0.51), 候选(2.02)];
        let r = 基因驱动诊断(&当前, &库, OMEGA_TOLERANCE);
        assert_eq!(r.len(), 2, "稳态诊断 + 谱级提示: {:?}", r);
    }

    #[test]
    fn test_序列化往返() {
        let c = 基因候选 { omega: 0.52 };
        let json = serde_json::to_string(&c).unwrap();
        let back: 基因候选 = serde_json::from_str(&json).unwrap();
        assert!((back.omega - 0.52).abs() < 1e-12);
    }
}
