// SPDX-License-Identifier: BSL-1.0
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! 九类生活预警扫描（MIGRATION-FINAL flutter 归一第十一例）
//!
//! 从 flutter `lib/services/warning_scanner.dart`（第304批 Phase 3 九类预警数据通道）
//! 迁移——**移动不改算法**（含文案格式与阈值）。
//!
//! 对齐 agent 侧 warning.rs（第239-254批）阈值。九场景全返回（触发或未触发原因）。
//! 输入已参数化: 打卡/睡眠的**窗口过滤**由调用方（flutter LifeDataStore）完成, 谱级
//! 数据（τ历史/D_s）由调用方（DiagnosisStore）传入; 本模块仅做规则判定。
//! 注意: 场景二"不同日期数"以 UTC 整数天（time_ms/86400000）近似（平台对齐, 诚实标注）。

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// 打卡记录（时间窗口已由调用方过滤）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 打卡记录 {
    pub r#type: String,
    /// Unix 毫秒时间戳
    pub time_ms: i64,
}

/// 预警扫描输入（参数化; 时间过滤在调用方）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 预警输入 {
    /// 近 14 天打卡
    pub checkins_14: Vec<打卡记录>,
    /// 近 7 天打卡
    pub checkins_7: Vec<打卡记录>,
    /// 近 14 天睡眠记录条数
    pub sleeps_14_len: i64,
    /// 近 14 天睡眠债务（h）
    pub sleep_debt_14: f64,
    /// 运动累计打卡次数
    pub sport_count: i64,
    /// 最后运动距今天数（-1 = 无运动记录）
    pub days_since_sport: i64,
    /// 主导 τ 历史
    pub tau_history: Vec<f64>,
    /// D_s 历史
    pub ds_history: Vec<f64>,
    /// D_s 三区（绿/黄/红; None=未知）
    pub ds_zone: Option<String>,
    /// 不可逆失稳类型（'正常'=无）
    pub instability_type: Option<String>,
}

/// 预警判定事实（T-182：中性结构化字段，供 agent 组织面向用户文案；
/// 每场景用到其中一部分，其余保持默认值。文案本身归 agent `igt_agent::warning`）
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct 预警事实 {
    /// 命中标签（场景一/二 主导打卡类型；场景六 不可逆失稳类型）——调用方数据透传
    pub label: String,
    /// D_s 三区（场景六）——调用方数据透传（绿/黄/红）
    pub zone: String,
    /// 样本量（场景一 近14天打卡数；场景二 近7天打卡数；场景三/八 τ测量数；场景四 睡眠记录数）
    pub sample_len: i64,
    /// 场景一/二 占比 0-1
    pub ratio: f64,
    /// 场景二 不同日期数
    pub days: i64,
    /// 场景三 τ 变异系数
    pub cv: f64,
    /// 场景四 睡眠债务（h）
    pub debt: f64,
    /// 场景五 近7天咖啡杯数
    pub coffee: i64,
    /// 场景七 运动累计次数
    pub sport_count: i64,
    /// 场景七 最后运动距今天数（-1 = 无运动记录）
    pub days_since_sport: i64,
    /// 场景八 τ 趋势比（后1/3均值 / 前1/3均值）
    pub trend: f64,
    /// 未触发/边界原因中性 code:
    /// `""`=常规（触发或有数据未触发）/ `"insufficient"`=样本不足 /
    /// `"no_data"`=无记录 / `"idle"`=未检出（场景六）/ `"not_connected"`=未接入（场景九）
    pub reason: String,
}

/// 原因 code（`预警事实.reason`）
pub mod 原因 {
    /// 常规（触发，或有数据的未触发）
    pub const 常规: &str = "";
    /// 样本不足
    pub const 样本不足: &str = "insufficient";
    /// 无记录
    pub const 无记录: &str = "no_data";
    /// 未检出（场景六）
    pub const 未检出: &str = "idle";
    /// 未接入（场景九）
    pub const 未接入: &str = "not_connected";
}

/// 生活预警输出（T-182：core 只出中性 `scene/triggered/strength` + 结构化事实；
/// 场景中文名与成句 message 归 agent `igt_agent::warning`）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 生活预警 {
    pub scene: u8,
    pub triggered: bool,
    /// 0-1（未触发=0）
    pub strength: f64,
    /// 判定事实（供 agent 组织文案）
    pub facts: 预警事实,
}

fn 预警(scene: u8, triggered: bool, strength: f64, facts: 预警事实) -> 生活预警 {
    生活预警 {
        scene,
        triggered,
        strength,
        facts,
    }
}

/// 九类生活预警扫描（移动不改算法: 场景一/二/四/五/七 打卡数据驱动, 三/六/八 谱级, 九 未接入诚实标注）
pub fn 扫描预警(input: &预警输入) -> Vec<生活预警> {
    let mut warnings = Vec::new();

    // 场景一 顾此失彼（注意力分散）——近14天打卡单类型占比≥70% 且 ≥5次
    let c14 = &input.checkins_14;
    if c14.len() >= 5 {
        let mut counts: HashMap<&str, i32> = HashMap::new();
        for c in c14 {
            *counts.entry(c.r#type.as_str()).or_insert(0) += 1;
        }
        let (max_type, max_count) = counts
            .iter()
            .max_by_key(|(_, v)| **v)
            .map(|(k, v)| (*k, *v))
            .unwrap_or(("", 0));
        let ratio = max_count as f64 / c14.len() as f64;
        let facts = 预警事实 {
            label: max_type.to_string(),
            sample_len: c14.len() as i64,
            ratio,
            ..Default::default()
        };
        if ratio >= 0.7 && max_count >= 5 {
            warnings.push(预警(1, true, ratio, facts));
        } else {
            warnings.push(预警(1, false, 0.0, facts));
        }
    } else {
        warnings.push(预警(
            1,
            false,
            0.0,
            预警事实 {
                sample_len: c14.len() as i64,
                reason: 原因::样本不足.to_string(),
                ..Default::default()
            },
        ));
    }

    // 场景二 过度沉迷（带宽收窄）——近7天单一类型≥3个不同日期 且 占比>70%
    let c7 = &input.checkins_7;
    if c7.len() >= 3 {
        let mut by_type: HashMap<&str, Vec<i64>> = HashMap::new();
        for c in c7 {
            by_type
                .entry(c.r#type.as_str())
                .or_default()
                .push(c.time_ms / 86_400_000); // UTC 整数天（平台对齐近似）
        }
        let (max_type, days_vec) = by_type
            .iter()
            .max_by_key(|(_, v)| v.len())
            .map(|(k, v)| (*k, v.clone()))
            .unwrap_or(("", vec![]));
        let ratio = days_vec.len() as f64 / c7.len() as f64;
        let day_set: std::collections::HashSet<i64> = days_vec.iter().copied().collect();
        let days = day_set.len();
        let facts = 预警事实 {
            label: max_type.to_string(),
            sample_len: c7.len() as i64,
            ratio,
            days: days as i64,
            ..Default::default()
        };
        if ratio > 0.7 && days >= 3 {
            warnings.push(预警(2, true, ratio, facts));
        } else {
            warnings.push(预警(2, false, 0.0, facts));
        }
    } else {
        warnings.push(预警(
            2,
            false,
            0.0,
            预警事实 {
                sample_len: c7.len() as i64,
                reason: 原因::样本不足.to_string(),
                ..Default::default()
            },
        ));
    }

    // 场景三 方向迷失——谱级判定（主导τ漂移大 = 无稳定方向）
    let tau_hist = &input.tau_history;
    if tau_hist.len() >= 5 {
        let mean = tau_hist.iter().sum::<f64>() / tau_hist.len() as f64;
        let variance = tau_hist
            .iter()
            .map(|t| (t - mean) * (t - mean))
            .sum::<f64>()
            / tau_hist.len() as f64;
        let cv = variance.sqrt() / (if mean.abs() > 1e-9 { mean.abs() } else { 1e-9 });
        let facts = 预警事实 {
            cv,
            sample_len: tau_hist.len() as i64,
            ..Default::default()
        };
        if cv > 0.6 {
            warnings.push(预警(3, true, (cv / 2.0).clamp(0.0, 1.0), facts));
        } else {
            warnings.push(预警(3, false, 0.0, facts));
        }
    } else {
        warnings.push(预警(
            3,
            false,
            0.0,
            预警事实 {
                sample_len: tau_hist.len() as i64,
                reason: 原因::样本不足.to_string(),
                ..Default::default()
            },
        ));
    }

    // 场景四 慢性透支（睡眠债务）——近14天睡眠≥3天 且 债务≥10h
    let debt = input.sleep_debt_14;
    if input.sleeps_14_len >= 3 && debt >= 10.0 {
        warnings.push(预警(
            4,
            true,
            (debt / 40.0).clamp(0.0, 1.0),
            预警事实 {
                debt,
                sample_len: input.sleeps_14_len,
                ..Default::default()
            },
        ));
    } else if input.sleeps_14_len == 0 {
        warnings.push(预警(
            4,
            false,
            0.0,
            预警事实 {
                debt,
                sample_len: 0,
                reason: 原因::无记录.to_string(),
                ..Default::default()
            },
        ));
    } else {
        warnings.push(预警(
            4,
            false,
            0.0,
            预警事实 {
                debt,
                sample_len: input.sleeps_14_len,
                ..Default::default()
            },
        ));
    }

    // 场景五 虚假稳态（咖啡输血）——近7天咖啡打卡≥8次
    let coffee7 = c7.iter().filter(|c| c.r#type == "咖啡").count();
    let facts5 = 预警事实 {
        coffee: coffee7 as i64,
        ..Default::default()
    };
    if coffee7 >= 8 {
        warnings.push(预警(
            5,
            true,
            (coffee7 as f64 / 14.0).clamp(0.0, 1.0),
            facts5,
        ));
    } else {
        warnings.push(预警(5, false, 0.0, facts5));
    }

    // 场景六 应急模式固化——谱级判定（不可逆失稳持续 + D_s 黄/红 = 应急模式固化）
    let inst = input.instability_type.as_deref();
    let zone = input.ds_zone.as_deref();
    if inst.is_some() && inst != Some("正常") && (zone == Some("黄") || zone == Some("红")) {
        warnings.push(预警(
            6,
            true,
            if zone == Some("红") { 0.9 } else { 0.6 },
            预警事实 {
                label: inst.unwrap_or("").to_string(),
                zone: zone.unwrap_or("—").to_string(),
                ..Default::default()
            },
        ));
    } else if inst.is_none() || inst == Some("正常") {
        warnings.push(预警(
            6,
            false,
            0.0,
            预警事实 {
                reason: 原因::未检出.to_string(),
                ..Default::default()
            },
        ));
    } else {
        warnings.push(预警(
            6,
            false,
            0.0,
            预警事实 {
                label: inst.unwrap_or("").to_string(),
                zone: zone.unwrap_or("—").to_string(),
                ..Default::default()
            },
        ));
    }

    // 场景七 反馈延迟误判（运动中断）——运动累计≥3次 且 最后距今≥14天
    if input.sport_count >= 3 && input.days_since_sport >= 0 && input.days_since_sport >= 14 {
        warnings.push(预警(
            7,
            true,
            (input.days_since_sport as f64 / 42.0).clamp(0.0, 1.0),
            预警事实 {
                sport_count: input.sport_count,
                days_since_sport: input.days_since_sport,
                ..Default::default()
            },
        ));
    } else if input.days_since_sport < 0 {
        warnings.push(预警(
            7,
            false,
            0.0,
            预警事实 {
                sport_count: input.sport_count,
                days_since_sport: input.days_since_sport,
                reason: 原因::无记录.to_string(),
                ..Default::default()
            },
        ));
    } else {
        warnings.push(预警(
            7,
            false,
            0.0,
            预警事实 {
                sport_count: input.sport_count,
                days_since_sport: input.days_since_sport,
                ..Default::default()
            },
        ));
    }

    // 场景八 慢变量盲区——谱级判定（主导τ缓慢上升趋势 + D_s 绿被忽略）
    if tau_hist.len() >= 6 {
        let third = tau_hist.len() / 3;
        let front = tau_hist[..third].iter().sum::<f64>() / third as f64;
        let back = tau_hist[tau_hist.len() - third..].iter().sum::<f64>() / third as f64;
        let trend = if front > 1e-9 { back / front } else { 0.0 };
        let ds_ok = input.ds_history.is_empty()
            || zone == Some("绿")
            || input.ds_zone.is_none();
        if trend > 1.3 && ds_ok {
            warnings.push(预警(
                8,
                true,
                ((trend - 1.0) / 1.0).clamp(0.0, 1.0),
                预警事实 {
                    trend,
                    sample_len: tau_hist.len() as i64,
                    ..Default::default()
                },
            ));
        } else {
            warnings.push(预警(
                8,
                false,
                0.0,
                预警事实 {
                    trend,
                    sample_len: tau_hist.len() as i64,
                    ..Default::default()
                },
            ));
        }
    } else {
        warnings.push(预警(
            8,
            false,
            0.0,
            预警事实 {
                sample_len: tau_hist.len() as i64,
                reason: 原因::样本不足.to_string(),
                ..Default::default()
            },
        ));
    }

    // 场景九 多层共振——需多通道（未接入诚实标注）
    warnings.push(预警(
        9,
        false,
        0.0,
        预警事实 {
            reason: 原因::未接入.to_string(),
            ..Default::default()
        },
    ));

    warnings
}

#[cfg(test)]
mod tests {
    use super::*;

    fn 打卡(r#type: &str, time_ms: i64) -> 打卡记录 {
        打卡记录 {
            r#type: r#type.to_string(),
            time_ms,
        }
    }

    fn 输入(
        checkins_14: Vec<打卡记录>,
        checkins_7: Vec<打卡记录>,
    ) -> 预警输入 {
        预警输入 {
            checkins_14,
            checkins_7,
            sleeps_14_len: 0,
            sleep_debt_14: 0.0,
            sport_count: 0,
            days_since_sport: -1,
            tau_history: vec![],
            ds_history: vec![],
            ds_zone: None,
            instability_type: None,
        }
    }

    #[test]
    fn test_场景一_触发() {
        // 对齐 flutter 原用例: 8 会议 + 2 运动 → ratio 0.8
        let now = 1_750_000_000_000i64;
        let mut c14 = vec![];
        for i in 0..8 {
            c14.push(打卡("会议", now - i * 86_400_000));
        }
        for i in 0..2 {
            c14.push(打卡("运动", now - i * 86_400_000));
        }
        let w = 扫描预警(&输入(c14, vec![]));
        let w1 = w.iter().find(|x| x.scene == 1).unwrap();
        assert!(w1.triggered);
        assert!((w1.strength - 0.8).abs() < 1e-9, "8/10");
        assert_eq!(w1.facts.label, "会议", "T-182: 命中标签为中性事实字段");
        assert_eq!(w1.facts.sample_len, 10);
        assert!((w1.facts.ratio - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_场景一_数据不足() {
        let w = 扫描预警(&输入(vec![打卡("会议", 0)], vec![]));
        let w1 = w.iter().find(|x| x.scene == 1).unwrap();
        assert!(!w1.triggered);
        assert_eq!(w1.facts.reason, 原因::样本不足);
        assert_eq!(w1.facts.sample_len, 1);
    }

    #[test]
    fn test_场景四_触发() {
        // 对齐 flutter 原用例: 5 晚 5h → 债务 15h
        let mut input = 输入(vec![], vec![]);
        input.sleeps_14_len = 5;
        input.sleep_debt_14 = 15.0;
        let w = 扫描预警(&input);
        let w4 = w.iter().find(|x| x.scene == 4).unwrap();
        assert!(w4.triggered);
        assert!((w4.strength - 0.375).abs() < 0.01, "15/40");
    }

    #[test]
    fn test_场景五_触发() {
        // 对齐 flutter 原用例: 9 咖啡 → 9/14
        let c7: Vec<打卡记录> = (0..9).map(|i| 打卡("咖啡", 1_750_000_000_000 - i * 3_600_000)).collect();
        let w = 扫描预警(&输入(vec![], c7));
        let w5 = w.iter().find(|x| x.scene == 5).unwrap();
        assert!(w5.triggered);
        assert!((w5.strength - 9.0 / 14.0).abs() < 1e-9);
        assert_eq!(w5.facts.coffee, 9, "T-182: 咖啡杯数为中性事实字段");
    }

    #[test]
    fn test_场景七_触发() {
        // 对齐 flutter 原用例: 3 次运动 + 最后距今 16 天
        let mut input = 输入(vec![], vec![]);
        input.sport_count = 3;
        input.days_since_sport = 16;
        let w = 扫描预警(&input);
        let w7 = w.iter().find(|x| x.scene == 7).unwrap();
        assert!(w7.triggered);
        assert_eq!(w7.facts.days_since_sport, 16);
        assert_eq!(w7.facts.sport_count, 3);
    }

    #[test]
    fn test_场景九_诚实标注() {
        let w = 扫描预警(&输入(vec![], vec![]));
        let w9 = w.iter().find(|x| x.scene == 9).unwrap();
        assert!(!w9.triggered);
        assert_eq!(w9.facts.reason, 原因::未接入);
    }

    #[test]
    fn test_场景三_漂移大触发() {
        // 对齐 flutter 原用例: [3,10,25,5,20,30,8,22]
        let mut input = 输入(vec![], vec![]);
        input.tau_history = vec![3.0, 10.0, 25.0, 5.0, 20.0, 30.0, 8.0, 22.0];
        let w = 扫描预警(&input);
        let w3 = w.iter().find(|x| x.scene == 3).unwrap();
        assert!(w3.triggered);
        assert!(w3.facts.cv > 0.6, "变异系数 > 0.6: {}", w3.facts.cv);
        assert_eq!(w3.facts.sample_len, 8);
    }

    #[test]
    fn test_场景三_稳定未触发() {
        let mut input = 输入(vec![], vec![]);
        input.tau_history = vec![10.0, 10.5, 9.8, 10.2, 10.1, 9.9, 10.3];
        let w = 扫描预警(&input);
        let w3 = w.iter().find(|x| x.scene == 3).unwrap();
        assert!(!w3.triggered);
    }

    #[test]
    fn test_场景六_应急触发() {
        let mut input = 输入(vec![], vec![]);
        input.instability_type = Some("点逃逸".to_string());
        input.ds_zone = Some("红".to_string());
        let w = 扫描预警(&input);
        let w6 = w.iter().find(|x| x.scene == 6).unwrap();
        assert!(w6.triggered);
        assert_eq!(w6.facts.label, "点逃逸");
        assert_eq!(w6.facts.zone, "红");
        assert!((w6.strength - 0.9).abs() < 1e-9, "红 → 0.9");
    }

    #[test]
    fn test_场景六_无失稳未触发() {
        let mut input = 输入(vec![], vec![]);
        input.instability_type = Some("正常".to_string());
        input.ds_zone = Some("绿".to_string());
        let w = 扫描预警(&input);
        let w6 = w.iter().find(|x| x.scene == 6).unwrap();
        assert!(!w6.triggered);
        assert_eq!(w6.facts.reason, 原因::未检出);
    }

    #[test]
    fn test_场景八_趋势触发() {
        // 递增 τ: 前 1/3 均小, 后 1/3 均大 → trend > 1.3
        let mut input = 输入(vec![], vec![]);
        input.tau_history = vec![10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 40.0, 41.0, 42.0];
        input.ds_history = vec![0.1, 0.15]; // 绿
        input.ds_zone = Some("绿".to_string());
        let w = 扫描预警(&input);
        let w8 = w.iter().find(|x| x.scene == 8).unwrap();
        assert!(w8.triggered, "trend=42/10=4.2>1.3 且 D_s 绿: {:?}", w8.facts);
        assert!(w8.facts.trend > 1.3);
    }

    #[test]
    fn test_场景八_稳定未触发() {
        let mut input = 输入(vec![], vec![]);
        input.tau_history = vec![10.0, 10.2, 10.1, 10.3, 10.2, 10.4];
        let w = 扫描预警(&input);
        let w8 = w.iter().find(|x| x.scene == 8).unwrap();
        assert!(!w8.triggered);
    }

    #[test]
    fn test_九场景全返回() {
        let w = 扫描预警(&输入(vec![], vec![]));
        assert_eq!(w.len(), 9, "九场景全返回（触发或未触发原因）");
    }

    #[test]
    fn test_序列化往返() {
        let c = 打卡记录 { r#type: "会议".to_string(), time_ms: 1_750_000_000_000 };
        let json = serde_json::to_string(&c).unwrap();
        let back: 打卡记录 = serde_json::from_str(&json).unwrap();
        assert_eq!(back.r#type, "会议");
    }
}
