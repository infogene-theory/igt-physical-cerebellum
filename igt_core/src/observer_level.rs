// SPDX-License-Identifier: BSL-1.0
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! 观察者等级评估（MIGRATION-FINAL flutter 归一第六例）
//!
//! 从 flutter `lib/services/observer_level.dart`（第355批 Phase 5·观察者等级联动; 第369批 D2 对齐 O5 预留）
//! 迁移——**移动不改算法**。
//!
//! O0-O5（P22/V10: 自指深度 r=τ_self/τ_env → O2≥2/O3≥5/O4≥10; O5 文明级理论预留）。
//! 本模块是**能力代理聚合评分**（感知覆盖/诊断/预测/基因成熟/反思深度）——
//! 诚实标注: 严格 τ_self/τ_env 在 core 端（band_projection 观测者层级带宽），此处为能力代理近似;
//! O5 为深层集体/文明级理论预留（对齐 engine observer_level.rs）——不从此路径产生, 最高评定 O4。

use serde::{Deserialize, Serialize};

/// 观察者能力集（数据由调用方注入）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 观察者能力 {
    /// 感知覆盖度 0-1
    pub coverage: f64,
    /// 诊断评估完成（ED/D_s/临界慢化 已算）
    pub has_diagnosis: bool,
    /// 预测能力（造波建议/ε扫描 可用）
    pub has_forecast: bool,
    /// 稳态基因数
    pub steady_genes: i64,
    /// 逆元对话轮数（反思深度）
    pub reflective_turns: i64,
    /// 当前 D_s（<0.3 稳态加分; 负值=未知）
    pub ds: f64,
}

/// 观察者等级结果
///
/// T-182（理论侧 163 拆分线 §2）：core 只出 `level/name/score` 中性 code+数值；
/// 面向用户的等级描述文案归 agent（`igt_agent::presentation`）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 观察者等级 {
    /// 0-4（O5 预留不从此路径产生）
    pub level: u8,
    /// O0-O4（中性等级 code）
    pub name: String,
    /// 0-100 能力评分
    pub score: f64,
}

/// O5 是否启用（对齐 engine: 多主体共享协议扩展后启用; 当前 false）
pub const O5_ENABLED: bool = false;

const NAMES: [&str; 6] = ["O0", "O1", "O2", "O3", "O4", "O5"];

/// 能力评分 0-100（感知 25 + 诊断 15 + 预测 15 + 基因成熟 30 + 反思深度 20 + 稳态 10）
pub fn 观察者等级评分(c: &观察者能力) -> f64 {
    let s = c.coverage.clamp(0.0, 1.0) * 25.0
        + if c.has_diagnosis { 15.0 } else { 0.0 }
        + if c.has_forecast { 15.0 } else { 0.0 }
        + c.steady_genes.clamp(0, 3) as f64 * 10.0
        + (c.reflective_turns.clamp(0, 8) as f64 / 2.0) * 5.0
        + if c.ds >= 0.0 && c.ds < 0.3 { 10.0 } else { 0.0 };
    s.clamp(0.0, 100.0)
}

/// 评分 → 等级（≥80 O4 / ≥60 O3 / ≥40 O2 / ≥20 O1 / 其余 O0）
pub fn 观察者等级评定(c: &观察者能力) -> 观察者等级 {
    let s = 观察者等级评分(c);
    let level: u8 = if s >= 80.0 {
        4
    } else if s >= 60.0 {
        3
    } else if s >= 40.0 {
        2
    } else if s >= 20.0 {
        1
    } else {
        0
    };
    观察者等级 {
        level,
        name: NAMES[level as usize].to_string(),
        score: s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn 能力(
        coverage: f64,
        has_diagnosis: bool,
        has_forecast: bool,
        steady_genes: i64,
        reflective_turns: i64,
        ds: f64,
    ) -> 观察者能力 {
        观察者能力 {
            coverage,
            has_diagnosis,
            has_forecast,
            steady_genes,
            reflective_turns,
            ds,
        }
    }

    #[test]
    fn test_零能力_O0() {
        let c = 能力(0.0, false, false, 0, 0, -1.0);
        let r = 观察者等级评定(&c);
        assert_eq!(r.level, 0);
        assert_eq!(r.name, "O0");
        assert_eq!(r.score, 0.0);
    }

    #[test]
    fn test_满分_O4() {
        let c = 能力(1.0, true, true, 3, 8, 0.1);
        // 25 + 15 + 15 + 30 + 20 + 10 = 115 → clamp 100
        let r = 观察者等级评定(&c);
        assert_eq!(r.level, 4);
        assert_eq!(r.name, "O4");
        assert_eq!(r.score, 100.0);
    }

    #[test]
    fn test_等级阈值() {
        // 感知 25 + 诊断 15 = 40 → O2（≥40）
        let c = 能力(1.0, true, false, 0, 0, -1.0);
        assert_eq!(观察者等级评定(&c).level, 2);
        // 感知 25 → O1（≥20）
        let c2 = 能力(1.0, false, false, 0, 0, -1.0);
        assert_eq!(观察者等级评定(&c2).level, 1);
        // 感知 25 + 诊断 15 + 预测 15 = 55 → O2（<60）
        let c3 = 能力(1.0, true, true, 0, 0, -1.0);
        assert_eq!(观察者等级评定(&c3).level, 2);
    }

    #[test]
    fn test_O5不从此路径产生() {
        assert!(!O5_ENABLED);
        let c = 能力(1.0, true, true, 3, 8, 0.1);
        assert!(观察者等级评定(&c).level < 5, "最高评定 O4");
    }

    #[test]
    fn test_序列化往返() {
        let c = 能力(0.5, true, false, 2, 4, 0.2);
        let json = serde_json::to_string(&c).unwrap();
        let back: 观察者能力 = serde_json::from_str(&json).unwrap();
        assert!((back.coverage - 0.5).abs() < 1e-9);
    }
}
