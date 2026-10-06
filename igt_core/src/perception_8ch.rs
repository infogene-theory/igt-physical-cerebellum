// SPDX-License-Identifier: BSL-1.0
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! 感知 8 通道多变量数据组织与摘要（MIGRATION-FINAL flutter 归一第九例）
//!
//! 从 flutter `lib/services/perception_8ch.dart`（第378批, V10.1 Phase 2 里程碑 M2）
//! 迁移 `build`/`summarize`——**移动不改算法**。
//!
//! 多路感知信号 → 等长窗口 → multichannelEdmd（core 复EDMD 完整数据链）→ 跨模态稳态深度/耦合。
//! 8 通道槽位: 1 HRV-RR 2 PPG-RR 3 运动幅值 4 运动Δ节律 5 对话活动 6 打卡节律
//!              7 睡眠/休息 8 昼夜相位（真实时间驱动——core 侧以 now_ms 参数注入）。
//! 无数据槽位用 0 基线 + 诚实标注; 真实通道 <2 → 多变量不可用（不静默）。

use serde::{Deserialize, Serialize};

/// 单通道数据（等长序列 + 数据源标注）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 感知通道数据 {
    pub name: String,
    pub values: Vec<f64>,
    pub source: String,
}

/// 多变量摘要（从 core 复EDMD 结果计算展示值）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 多变量摘要 {
    pub order: usize,
    /// 慢τ（τ链最大模; 空链 → None）
    pub slow_tau: Option<f64>,
    /// 快τ（τ链最小模, 需 ≥2 个模）
    pub fast_tau: Option<f64>,
    /// 非对角耦合幅度（复L 非对角元素平均模）
    pub off_diag_coupling: f64,
    pub gamma_positive: bool,
    pub unstable_count: usize,
}

/// 默认多变量窗口长度（复EDMD 需 ≥ 通道数+2）
pub const DEFAULT_WINDOW: usize = 40;

/// 等长窗口: 有数据的后部对齐, 不足前部填充 `fill`
fn pad(src: &[f64], n: usize, fill: f64) -> Vec<f64> {
    if src.len() >= n {
        return src[src.len() - n..].to_vec();
    }
    let mut out = vec![fill; n];
    let s = src.len().min(n);
    for i in 0..s {
        out[n - s + i] = src[i];
    }
    out
}

/// 组织 8 通道等长窗口（移动不改算法: RR 通道 fill=0.9 基线, 昼夜相位由 now_ms 驱动）
///
/// `now_ms`: 当前 Unix 毫秒时间戳（页面注入 `DateTime.now().millisecondsSinceEpoch`）,
/// 昼夜相位 hourFrac = now_ms/1000/86400。
pub fn build_8channels(
    rr_hrv: &[f64],
    rr_ppg: &[f64],
    motion_magnitudes: &[f64],
    dialogue_activity: f64,
    punch_rhythm: f64,
    sleep_quality: f64,
    now_ms: i64,
    window: Option<usize>,
) -> Vec<感知通道数据> {
    let n = window.unwrap_or(DEFAULT_WINDOW).max(1);
    let ch1 = pad(rr_hrv, n, 0.9);
    let ch2 = pad(rr_ppg, n, 0.9);
    let ch3 = pad(motion_magnitudes, n, 0.0);
    // 运动Δ节律（相邻差分, 节律性信息; clamp ±0.5）
    let ch4: Vec<f64> = (0..n)
        .map(|i| {
            if i == 0 {
                0.0
            } else {
                (ch3[i] - ch3[i - 1]).clamp(-0.5, 0.5)
            }
        })
        .collect();
    // 对话/打卡/睡眠: 无真实数据 → 弱基线 + 诚实标注
    let d_a = dialogue_activity.clamp(0.0, 1.0);
    let p_r = punch_rhythm.clamp(0.0, 1.0);
    let s_q = sleep_quality.clamp(0.0, 1.0);
    let ch5 = vec![d_a; n];
    let ch6: Vec<f64> = (0..n)
        .map(|i| 0.5 + 0.5 * (2.0 * std::f64::consts::PI * i as f64 / 10.0).sin() * p_r)
        .collect();
    let ch7: Vec<f64> = (0..n)
        .map(|i| 0.3 + 0.2 * (2.0 * std::f64::consts::PI * i as f64 / 20.0).sin() * s_q)
        .collect();
    // 昼夜相位（24h 节律, 真实时间驱动）
    let hour_frac = now_ms as f64 / 1000.0 / 86400.0;
    let ch8: Vec<f64> = (0..n)
        .map(|i| (2.0 * std::f64::consts::PI * (hour_frac + i as f64 / 120.0)).sin())
        .collect();

    vec![
        感知通道数据 {
            name: "HRV-RR".to_string(),
            values: ch1,
            source: "真实".to_string(),
        },
        感知通道数据 {
            name: "PPG-RR".to_string(),
            values: ch2,
            source: "真实".to_string(),
        },
        感知通道数据 {
            name: "运动幅值".to_string(),
            values: ch3,
            source: "真实".to_string(),
        },
        感知通道数据 {
            name: "运动Δ节律".to_string(),
            values: ch4,
            source: "真实".to_string(),
        },
        感知通道数据 {
            name: "对话活动".to_string(),
            values: ch5,
            source: if dialogue_activity > 0.0 {
                "真实".to_string()
            } else {
                "基线(无数据)".to_string()
            },
        },
        感知通道数据 {
            name: "打卡节律".to_string(),
            values: ch6,
            source: if punch_rhythm > 0.0 {
                "真实".to_string()
            } else {
                "基线(无数据)".to_string()
            },
        },
        感知通道数据 {
            name: "睡眠/休息".to_string(),
            values: ch7,
            source: if sleep_quality > 0.0 {
                "真实".to_string()
            } else {
                "基线(无数据)".to_string()
            },
        },
        感知通道数据 {
            name: "昼夜相位".to_string(),
            values: ch8,
            source: "真实(时间)".to_string(),
        },
    ]
}

/// 多变量摘要: τ链快慢模 + 非对角耦合幅度 + 稳定性
///
/// `complex_l`: 复L 扁平化 [re,im] 交错（同 frb MultichannelEdmdResult.complexL）
pub fn multichannel_summary(
    order: usize,
    tau_chain: &[f64],
    complex_l: &[f64],
    gamma_positive: bool,
    unstable_count: usize,
) -> 多变量摘要 {
    let slow_tau = if tau_chain.is_empty() {
        None
    } else {
        Some(tau_chain.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
    };
    let fast_tau = if tau_chain.len() >= 2 {
        Some(tau_chain.iter().cloned().fold(f64::INFINITY, f64::min))
    } else {
        None
    };
    // 非对角耦合幅度: 复L 非对角元素平均模（实虚交错 [re,im]）
    let mut off_diag = 0.0;
    let mut count = 0usize;
    if order >= 2 && complex_l.len() >= order * order * 2 {
        for i in 0..order {
            for j in 0..order {
                if i == j {
                    continue;
                }
                let idx = 2 * (i * order + j);
                let re = complex_l[idx];
                let im = complex_l[idx + 1];
                off_diag += (re * re + im * im).sqrt();
                count += 1;
            }
        }
        if count > 0 {
            off_diag /= count as f64;
        }
    }
    多变量摘要 {
        order,
        slow_tau,
        fast_tau,
        off_diag_coupling: off_diag,
        gamma_positive,
        unstable_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_不足前部填充() {
        // 窗口 5, src 长度 3 → 前部 0 填充, 后部对齐
        let out = pad(&[1.0, 2.0, 3.0], 5, 0.9);
        assert_eq!(out, vec![0.9, 0.9, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_pad_充足取后n() {
        let src: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let out = pad(&src, 5, 0.9);
        assert_eq!(out, vec![3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_build_8通道等长() {
        let chs = build_8channels(
            &vec![0.8; 50],
            &vec![0.7; 10],
            &vec![1.0; 20],
            0.5,
            0.3,
            0.6,
            1_750_000_000_000,
            None,
        );
        assert_eq!(chs.len(), 8);
        for c in &chs {
            assert_eq!(c.values.len(), DEFAULT_WINDOW, "通道 {} 应等长", c.name);
        }
        // HRV 真实; 对话有数据 → 真实
        assert_eq!(chs[0].name, "HRV-RR");
        assert_eq!(chs[4].source, "真实");
        assert_eq!(chs[5].source, "真实");
        assert_eq!(chs[7].source, "真实(时间)");
    }

    #[test]
    fn test_build_无数据基线标注() {
        let chs = build_8channels(&[], &[], &[], 0.0, 0.0, 0.0, 0, None);
        assert_eq!(chs[4].source, "基线(无数据)");
        assert_eq!(chs[5].source, "基线(无数据)");
        assert_eq!(chs[6].source, "基线(无数据)");
        // 对话弱基线 0
        assert!(chs[4].values.iter().all(|v| *v == 0.0));
        // 打卡节律: 无数据 → 0.5 + 0.5*sin*0 = 0.5 基线
        assert!((chs[5].values[0] - 0.5).abs() < 1e-12);
        // RR 通道 fill=0.9
        assert!((chs[0].values[0] - 0.9).abs() < 1e-12);
    }

    #[test]
    fn test_build_运动Δ节律() {
        let motion: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let chs = build_8channels(&[], &[], &motion, 0.0, 0.0, 0.0, 0, None);
        let ch4 = &chs[3];
        assert_eq!(ch4.name, "运动Δ节律");
        // 相邻差分 = 1, clamp ±0.5 → 0.5
        assert!((ch4.values[1] - 0.5).abs() < 1e-12);
        assert!((ch4.values[0] - 0.0).abs() < 1e-12);
        // 差分 clamp 上限
        assert!(ch4.values.iter().all(|v| *v <= 0.5 + 1e-12));
    }

    #[test]
    fn test_summarize_快慢模与耦合() {
        let tau = vec![10.0, 50.0, 3.0];
        // 2x2 复L: [re,im] 交错; 非对角 (0,1) (1,0)
        let l = vec![1.0, 0.0, 3.0, 4.0, 5.0, 0.0, 1.0, 0.0];
        // (0,1): idx=2*(0*2+1)=2 → re=3,im=4 → 模5; (1,0): idx=2*(1*2+0)=4 → re=5,im=0 → 模5
        let s = multichannel_summary(2, &tau, &l, true, 1);
        assert_eq!(s.order, 2);
        assert!((s.slow_tau.unwrap() - 50.0).abs() < 1e-12);
        assert!((s.fast_tau.unwrap() - 3.0).abs() < 1e-12);
        assert!((s.off_diag_coupling - 5.0).abs() < 1e-12);
        assert!(s.gamma_positive);
        assert_eq!(s.unstable_count, 1);
    }

    #[test]
    fn test_summarize_空链与低阶() {
        let s = multichannel_summary(1, &[], &[], false, 0);
        assert_eq!(s.slow_tau, None);
        assert_eq!(s.fast_tau, None);
        assert_eq!(s.off_diag_coupling, 0.0, "order<2 → 无耦合统计");
    }

    #[test]
    fn test_序列化往返() {
        let c = 感知通道数据 {
            name: "HRV-RR".to_string(),
            values: vec![0.9, 0.8],
            source: "真实".to_string(),
        };
        let json = serde_json::to_string(&c).unwrap();
        let back: 感知通道数据 = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "HRV-RR");
        assert_eq!(back.values.len(), 2);
    }
}
