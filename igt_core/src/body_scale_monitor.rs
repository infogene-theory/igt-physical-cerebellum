// SPDX-License-Identifier: BSL-1.0
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! 身体尺度监测+预警（MIGRATION-FINAL flutter 归一第七例）
//!
//! 从 flutter `lib/services/body_scale_monitor.dart`（第299批 Phase 2 任务4, P22 §7.2/§7.5）
//! 迁移——**移动不改算法**（含边界行为: 无历史时 H 不 clamp, 与 Dart 一致）。
//!
//! ① D_s^(2D) 三区预警: <0.3 绿(稳态) / 0.3-0.6 黄(注意) / ≥0.6 红(预警)
//! ② 临界慢化信号（P22 §7.5）: 滑动窗口 RR 的 方差/滞后1自相关 趋势——
//!    方差上升>20% 或 自相关上升>0.1 → 临界慢化（τ 发散前兆）
//! ③ 健康度 H=1-ED/ε_c（P22 §7.1, ε_c 取历史 D_s 高分的自适应估计）——Dart 侧近似

use serde::{Deserialize, Serialize};

/// D_s 三区预警
///
/// T-182（理论侧 163 拆分线 §2）：core 只出**中性 code+数值**；面向用户的等级
/// 中文与成句文案归 agent（`igt_agent::presentation`）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DsZone {
    /// 中性等级 code: `"green"` / `"yellow"` / `"red"`
    pub level: String,
    pub value: f64,
}

/// 临界慢化信号
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlowingSignals {
    pub variance_prev: f64,
    pub variance_cur: f64,
    pub autocorr_prev: f64,
    pub autocorr_cur: f64,
    /// 方差上升>20%
    pub variance_rising: bool,
    /// 自相关上升>0.1
    pub autocorr_rising: bool,
    /// 任一信号
    pub triggered: bool,
}

pub const GREEN_BOUND: f64 = 0.3;
pub const YELLOW_BOUND: f64 = 0.6;
pub const VARIANCE_RISE_THRESHOLD: f64 = 0.20;
pub const AUTOCORR_RISE_THRESHOLD: f64 = 0.10;

/// D_s^(2D) 三区: <0.3 绿 / 0.3-0.6 黄 / ≥0.6 红（P22 §7.2 身体尺度）
pub fn ds_zone(d_s: f64) -> DsZone {
    let level = if d_s < GREEN_BOUND {
        "green"
    } else if d_s < YELLOW_BOUND {
        "yellow"
    } else {
        "red"
    };
    DsZone {
        level: level.to_string(),
        value: d_s,
    }
}

/// 样本方差（分母 n-1）
pub fn variance(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let s: f64 = x.iter().map(|v| (v - mean) * (v - mean)).sum();
    s / (x.len() as f64 - 1.0)
}

/// 滞后1自相关（P22 §7.5 临界慢化: 自相关上升=记忆变长=逼近临界）
pub fn autocorr_lag1(x: &[f64]) -> f64 {
    if x.len() < 3 {
        return 0.0;
    }
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..x.len() - 1 {
        num += (x[i] - mean) * (x[i + 1] - mean);
    }
    for v in x {
        den += (v - mean) * (v - mean);
    }
    if den.abs() < 1e-12 {
        return 0.0;
    }
    num / den
}

/// 临界慢化信号: RR 分前/后两段, 比较 方差/自相关 趋势（长度 < min_len → 全 false）
pub fn critical_slowing(rr: &[f64], min_len: usize) -> SlowingSignals {
    if rr.len() < min_len {
        return SlowingSignals {
            variance_prev: 0.0,
            variance_cur: 0.0,
            autocorr_prev: 0.0,
            autocorr_cur: 0.0,
            variance_rising: false,
            autocorr_rising: false,
            triggered: false,
        };
    }
    let mid = rr.len() / 2;
    let prev = &rr[..mid];
    let cur = &rr[mid..];
    let v_prev = variance(prev);
    let v_cur = variance(cur);
    let a_prev = autocorr_lag1(prev);
    let a_cur = autocorr_lag1(cur);
    let v_rise = if v_prev.abs() < 1e-12 {
        v_cur > 1e-9
    } else {
        (v_cur - v_prev) / v_prev.abs() > VARIANCE_RISE_THRESHOLD
    };
    let a_rise = (a_cur - a_prev) > AUTOCORR_RISE_THRESHOLD;
    SlowingSignals {
        variance_prev: v_prev,
        variance_cur: v_cur,
        autocorr_prev: a_prev,
        autocorr_cur: a_cur,
        variance_rising: v_rise,
        autocorr_rising: a_rise,
        triggered: v_rise || a_rise,
    }
}

/// 健康度 H=1-ED/ε_c（P22 §7.1）——近似: ED 用 D_s 替代（正相关）,
/// ε_c 用历史 D_s 均值+2σ 自适应（崩溃阈值从历史估计）。
/// history 传本会话 D_s 序列（空 → ε_c=1 中性, **不 clamp**, 与 Dart 一致）
pub fn health_from_ds(d_s: f64, history: &[f64]) -> f64 {
    if history.is_empty() {
        return 1.0 - d_s / 1.0;
    }
    let mean = history.iter().sum::<f64>() / history.len() as f64;
    let s: f64 = history.iter().map(|v| (v - mean) * (v - mean)).sum();
    let sd = if history.len() > 1 {
        (s / (history.len() as f64 - 1.0)).abs().sqrt()
    } else {
        0.0
    };
    let eps = (mean + 2.0 * sd).clamp(0.05, 1.5);
    (1.0 - d_s / eps).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ds_zone_三区边界() {
        // T-182: level 为中性 code（中文等级文案归 agent presentation）
        assert_eq!(ds_zone(0.29).level, "green");
        assert_eq!(ds_zone(0.30).level, "yellow");
        assert_eq!(ds_zone(0.59).level, "yellow");
        assert_eq!(ds_zone(0.60).level, "red");
    }

    #[test]
    fn test_variance_样本方差() {
        assert_eq!(variance(&[1.0]), 0.0, "长度<2 → 0");
        assert_eq!(variance(&[]), 0.0);
        let v = variance(&[1.0, 2.0, 3.0]);
        assert!((v - 1.0).abs() < 1e-12, "n-1 分母: {v}");
    }

    #[test]
    fn test_autocorr_短序列与常数序列() {
        assert_eq!(autocorr_lag1(&[1.0, 2.0]), 0.0, "长度<3 → 0");
        assert_eq!(autocorr_lag1(&[5.0, 5.0, 5.0, 5.0]), 0.0, "den≈0 → 0");
    }

    #[test]
    fn test_autocorr_趋势序列正相关() {
        // 单调递增 → 强正相关
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let a = autocorr_lag1(&x);
        assert!(a > 0.8, "单调序列自相关应接近1: {a}");
        // 交替序列 → 负相关
        let y: Vec<f64> = (0..20).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        assert!(autocorr_lag1(&y) < -0.8, "交替序列应负相关");
    }

    #[test]
    fn test_critical_slowing_长度不足() {
        let s = critical_slowing(&vec![1.0; 10], 40);
        assert!(!s.triggered, "长度<min_len → 不触发");
        assert_eq!(s.variance_cur, 0.0);
    }

    #[test]
    fn test_critical_slowing_方差上升触发() {
        // 前半小幅, 后半大幅 → 方差上升
        let mut rr = vec![100.0; 30];
        for i in 0..30 {
            rr.push(100.0 + if i % 2 == 0 { 20.0 } else { -20.0 });
        }
        let s = critical_slowing(&rr, 40);
        assert!(s.variance_rising, "方差应上升");
        assert!(s.triggered);
    }

    #[test]
    fn test_critical_slowing_平稳不触发() {
        let rr = vec![100.0; 60];
        let s = critical_slowing(&rr, 40);
        assert!(!s.variance_rising, "常数序列: vPrev≈0 且 vCur≈0 → 不上升");
        assert!(!s.autocorr_rising);
        assert!(!s.triggered);
    }

    #[test]
    fn test_health_from_ds_无历史() {
        // 无历史 → ε_c=1（不 clamp, 与 Dart 一致）
        assert!((health_from_ds(0.2, &[]) - 0.8).abs() < 1e-12);
        assert!((health_from_ds(1.5, &[]) - (-0.5)).abs() < 1e-12, "无 clamp");
    }

    #[test]
    fn test_health_from_ds_有历史() {
        let h = health_from_ds(0.3, &[0.2, 0.4, 0.3, 0.35]);
        assert!(h > 0.0 && h <= 1.0, "H  clamp 0-1: {h}");
        // 远超阈值 → 0
        assert_eq!(health_from_ds(10.0, &[0.2, 0.3]), 0.0);
    }

    #[test]
    fn test_序列化往返() {
        let z = ds_zone(0.45);
        let json = serde_json::to_string(&z).unwrap();
        let back: DsZone = serde_json::from_str(&json).unwrap();
        assert_eq!(back.level, "yellow");
        assert!((back.value - 0.45).abs() < 1e-12);
    }
}
