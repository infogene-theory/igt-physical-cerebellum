// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! 个人基准椭圆统计（MIGRATION-FINAL 归一第一例）
//!
//! 从 flutter `lib/services/baseline_ellipse.dart`（第291批 Phase 2）迁移——**移动不改算法**：
//! - 统计（均值/协方差/中位数/马氏距离/绿黄红分级）下沉 core，跨端共享（flutter 经 frb 调 Rust，
//!   删 Dart 轻量版逻辑，同一能力不双实现并存——落点表 §二归一原则）
//! - 持久化（JSON 本地积累）留 UI 层（I/O 壳，非计算逻辑）
//!
//! 理论锚点: P22 §13 椭圆谱雷达——个人基准椭圆（中心均值+协方差）+ 马氏距离 → 绿(<1σ)/黄(1-2σ)/红(>2σ)。
//! 注: 本模块是"个人基准统计"，与 `nonhermitian_spectrum::椭圆系综`（P22 §6.4 单窗口椭圆拟合 D_s^(2D)）互补。

use serde::{Deserialize, Serialize};

/// 单次窗口的椭圆参数采集（个人基准样本）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 基准椭圆样本 {
    /// 窗口谱分解椭圆中心（实部）
    pub center_real: f64,
    /// 窗口谱分解椭圆中心（虚部）
    pub center_imag: f64,
    /// 角向展宽 σ_ω
    pub sigma_omega: f64,
    /// 径向展宽 σ_γ
    pub sigma_gamma: f64,
    /// 椭圆参数 τ_ellipse
    pub tau_ellipse: f64,
    /// 椭圆系综 D_s^(2D)
    pub d_s2d: f64,
    /// 分区（"身体"/"亲密"/"组织" 等）
    pub region: String,
    /// 采集时间戳（毫秒）
    pub timestamp_ms: u64,
}

/// 个人基准椭圆（7 天样本统计）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 基准椭圆 {
    pub mean_real: f64,
    pub mean_imag: f64,
    pub var_real: f64,
    pub var_imag: f64,
    pub cov_real_imag: f64,
    pub tau_ellipse_median: f64,
    pub sigma_omega_median: f64,
    pub sigma_gamma_median: f64,
    pub sample_count: usize,
}

/// 身体尺度预警（level 为中性 code）
///
/// T-182（理论侧 163 拆分线 §2）：core 只出**中性 code+数值**；面向用户的中文
/// 等级与成句文案归 agent（`igt_agent::presentation`）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 身体尺度预警 {
    /// 中性等级 code: `"green"` / `"yellow"` / `"red"`
    pub level: String,
    /// 马氏距离（-1=数据不足）
    pub mahalanobis: f64,
}

/// 统计个人基准椭圆（≥3 次采集；样本方差 /n，与原 Dart 实现一致——移动不改算法）
pub fn 统计基准椭圆(样本: &[基准椭圆样本]) -> Option<基准椭圆> {
    if 样本.len() < 3 {
        return None;
    }
    let n = 样本.len();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / n as f64;
    let cr: Vec<f64> = 样本.iter().map(|s| s.center_real).collect();
    let ci: Vec<f64> = 样本.iter().map(|s| s.center_imag).collect();
    let mr = mean(&cr);
    let mi = mean(&ci);
    let mut vr = 0.0;
    let mut vi = 0.0;
    let mut crc = 0.0;
    for i in 0..n {
        vr += (cr[i] - mr) * (cr[i] - mr);
        vi += (ci[i] - mi) * (ci[i] - mi);
        crc += (cr[i] - mr) * (ci[i] - mi);
    }
    vr /= n as f64;
    vi /= n as f64;
    crc /= n as f64;
    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[n / 2]
    };
    Some(基准椭圆 {
        mean_real: mr,
        mean_imag: mi,
        var_real: vr,
        var_imag: vi,
        cov_real_imag: crc,
        tau_ellipse_median: median(样本.iter().map(|s| s.tau_ellipse).collect()),
        sigma_omega_median: median(样本.iter().map(|s| s.sigma_omega).collect()),
        sigma_gamma_median: median(样本.iter().map(|s| s.sigma_gamma).collect()),
        sample_count: n,
    })
}

/// 当前窗口中心 vs 个人基准椭圆 → 身体尺度绿黄红（马氏距离，与原 Dart 实现一致）
pub fn 评估身体尺度(基准: &基准椭圆, center_real: f64, center_imag: f64) -> 身体尺度预警 {
    let dx = center_real - 基准.mean_real;
    let dy = center_imag - 基准.mean_imag;
    let det = 基准.var_real * 基准.var_imag - 基准.cov_real_imag * 基准.cov_real_imag;
    let d = if det.abs() < 1e-12 {
        (dx * dx + dy * dy).sqrt() / ((基准.var_real + 基准.var_imag).abs().sqrt() + 1e-9)
    } else {
        let m2 = dx * dx * 基准.var_imag + dy * dy * 基准.var_real
            - 2.0 * dx * dy * 基准.cov_real_imag;
        (m2 / det).abs().sqrt()
    };
    let level = if d < 1.0 {
        "green"
    } else if d < 2.0 {
        "yellow"
    } else {
        "red"
    };
    身体尺度预警 {
        level: level.to_string(),
        mahalanobis: d,
    }
}

/// 不可逆失稳结果（P22 §7.7，从 flutter irreversible_instability.dart 迁移——移动不改算法）
///
/// T-182（理论侧 163 拆分线 §2）：core 只出**中性 code+数值**；类型的面向用户
/// 中文名与成句文案归 agent（`igt_agent::presentation`）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 不可逆失稳结果 {
    /// 中性类型 code: `"normal"` / `"point_escape"` / `"ellipse_shrink"` /
    /// `"point_escape+ellipse_shrink"`
    pub type_name: String,
    /// 当前点 vs 基准中心 马氏距离
    pub mahalanobis: f64,
    /// σ 收缩率（1=无收缩, <1=收缩, 0.5=缩半）
    pub sigma_shrinkage: f64,
    /// 样本是否充足（≥6 才有统计意义）——中性事实；不足时 type_name 仍为 `"normal"`
    pub 样本充足: bool,
}

/// 检测不可逆失稳（样本≥6 才有统计意义；原 Dart 实现一致）
///
/// 两种失稳信号（P22 §7.7）:
/// ① 点逃逸: 当前状态点跑出基准椭圆（马氏距离 > 2σ）——可逆波动
/// ② 椭圆收缩/下沉: 椭圆系综 σ 随时间收缩（近期 σ < 早期 50%）——不可逆前兆
pub fn 检测不可逆失稳(
    样本: &[基准椭圆样本],
    center_real: f64,
    center_imag: f64,
) -> 不可逆失稳结果 {
    const ESCAPE_SIGMA: f64 = 2.0;
    const SHRINK_THRESHOLD: f64 = 0.5;

    if 样本.len() < 6 {
        return 不可逆失稳结果 {
            type_name: "normal".to_string(),
            mahalanobis: 0.0,
            sigma_shrinkage: 1.0,
            样本充足: false,
        };
    }
    let n = 样本.len();

    // ① 点逃逸: 当前点 vs 全部样本中心的马氏距离（P22 §7.7 可逆波动判据）
    let mr = 样本.iter().map(|s| s.center_real).sum::<f64>() / n as f64;
    let mi = 样本.iter().map(|s| s.center_imag).sum::<f64>() / n as f64;
    let mut vr = 0.0;
    let mut vi = 0.0;
    let mut crc = 0.0;
    for s in 样本 {
        vr += (s.center_real - mr) * (s.center_real - mr);
        vi += (s.center_imag - mi) * (s.center_imag - mi);
        crc += (s.center_real - mr) * (s.center_imag - mi);
    }
    vr /= n as f64;
    vi /= n as f64;
    crc /= n as f64;
    let dx = center_real - mr;
    let dy = center_imag - mi;
    let det = vr * vi - crc * crc;
    let mahal = if det.abs() < 1e-12 {
        (dx * dx + dy * dy).sqrt() / ((vr + vi).abs().sqrt() + 1e-9)
    } else {
        ((dx * dx * vi + dy * dy * vr - 2.0 * dx * dy * crc) / det).abs().sqrt()
    };
    let point_escape = mahal > ESCAPE_SIGMA;

    // ② 椭圆收缩: 早期 1/3 vs 近期 1/3 的 σ 均值比（P22 §7.7 不可逆前兆）
    let third = n / 3;
    if third < 1 {
        return 不可逆失稳结果 {
            type_name: if point_escape { "point_escape" } else { "normal" }.to_string(),
            mahalanobis: mahal,
            sigma_shrinkage: 1.0,
            样本充足: true,
        };
    }
    let mean_sigma = |ss: &[基准椭圆样本]| {
        let so = ss.iter().map(|s| s.sigma_omega).sum::<f64>() / ss.len() as f64;
        let sg = ss.iter().map(|s| s.sigma_gamma).sum::<f64>() / ss.len() as f64;
        (so + sg) / 2.0
    };
    let e_sigma = mean_sigma(&样本[..third]);
    let r_sigma = mean_sigma(&样本[n - third..]);
    let shrink = if e_sigma.abs() < 1e-9 {
        1.0
    } else {
        (r_sigma / e_sigma).clamp(0.0, 1.0)
    };
    let ellipse_shrink = shrink < SHRINK_THRESHOLD;

    let type_name = match (point_escape, ellipse_shrink) {
        (true, true) => "point_escape+ellipse_shrink",
        (true, false) => "point_escape",
        (false, true) => "ellipse_shrink",
        (false, false) => "normal",
    }
    .to_string();
    不可逆失稳结果 {
        type_name,
        mahalanobis: mahal,
        sigma_shrinkage: shrink,
        样本充足: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn 样本(center_real: f64, center_imag: f64) -> 基准椭圆样本 {
        基准椭圆样本 {
            center_real,
            center_imag,
            sigma_omega: 1.0,
            sigma_gamma: 0.5,
            tau_ellipse: 0.3,
            d_s2d: 0.2,
            region: "身体".to_string(),
            timestamp_ms: 1,
        }
    }

    #[test]
    fn test_不足3次_返回None() {
        assert!(统计基准椭圆(&[样本(0.0, 0.0)]).is_none());
        assert!(统计基准椭圆(&[样本(0.0, 0.0), 样本(1.0, 1.0)]).is_none());
    }

    #[test]
    fn test_3次_统计基准() {
        let s = vec![样本(0.0, 0.0), 样本(2.0, 0.0), 样本(4.0, 0.0)];
        let b = 统计基准椭圆(&s).unwrap();
        assert!((b.mean_real - 2.0).abs() < 1e-9, "均值实部=2: {}", b.mean_real);
        assert!((b.mean_imag - 0.0).abs() < 1e-9);
        assert!((b.var_real - 8.0 / 3.0).abs() < 1e-9, "样本方差/n: {}", b.var_real);
        assert_eq!(b.sample_count, 3);
    }

    #[test]
    fn test_评估_绿黄红() {
        let s = vec![样本(0.0, 0.0), 样本(2.0, 0.0), 样本(4.0, 0.0)];
        let b = 统计基准椭圆(&s).unwrap();
        // 中心点 → 绿（d≈0）
        let 绿 = 评估身体尺度(&b, 2.0, 0.0);
        assert_eq!(绿.level, "green", "中心点应绿: {:?}", 绿);
        // 远偏离 → 红（d>2σ）
        let 红 = 评估身体尺度(&b, 20.0, 20.0);
        assert_eq!(红.level, "red", "远偏离应红: {:?}", 红);
    }

    #[test]
    fn test_序列化往返() {
        let s = 样本(1.0, 2.0);
        let json = serde_json::to_string(&s).unwrap();
        let back: 基准椭圆样本 = serde_json::from_str(&json).unwrap();
        assert!((back.center_real - 1.0).abs() < 1e-9);
    }

    // ===== 不可逆失稳检测（P22 §7.7，归一第二例）=====

    fn 时间样本(sigma: f64) -> 基准椭圆样本 {
        基准椭圆样本 {
            center_real: 1.0,
            center_imag: 1.0,
            sigma_omega: sigma,
            sigma_gamma: sigma,
            tau_ellipse: 0.3,
            d_s2d: 0.2,
            region: "身体".to_string(),
            timestamp_ms: 1,
        }
    }

    #[test]
    fn test_失稳_样本不足_正常() {
        let s: Vec<基准椭圆样本> = (0..5).map(|_| 时间样本(1.0)).collect();
        let r = 检测不可逆失稳(&s, 1.0, 1.0);
        assert_eq!(r.type_name, "normal");
        // T-182: 文案归 agent；core 以中性事实位标注样本不足
        assert!(!r.样本充足);
    }

    #[test]
    fn test_失稳_点逃逸() {
        let s: Vec<基准椭圆样本> = (0..9).map(|_| 时间样本(1.0)).collect();
        let r = 检测不可逆失稳(&s, 10.0, 10.0); // 当前点远离中心 → 点逃逸
        assert_eq!(
            r.type_name, "point_escape",
            "远离中心应点逃逸: {:?}",
            r.type_name
        );
        assert!(r.mahalanobis > 2.0);
    }

    #[test]
    fn test_失稳_椭圆收缩() {
        // 早期 σ=1.0（大） → 近期 σ=0.1（小） → 收缩率 0.1 < 0.5
        let mut s: Vec<基准椭圆样本> = Vec::new();
        for i in 0..9 {
            let sigma = if i < 3 { 1.0 } else { 0.1 };
            s.push(时间样本(sigma));
        }
        let r = 检测不可逆失稳(&s, 1.0, 1.0);
        assert_eq!(
            r.type_name, "ellipse_shrink",
            "近期σ缩半应椭圆收缩: {:?}",
            r.type_name
        );
        assert!(r.sigma_shrinkage < 0.5);
    }

    #[test]
    fn test_失稳_正常() {
        let s: Vec<基准椭圆样本> = (0..9).map(|_| 时间样本(1.0)).collect();
        let r = 检测不可逆失稳(&s, 1.0, 1.0);
        assert_eq!(r.type_name, "normal");
        assert_eq!(r.sigma_shrinkage, 1.0);
    }
}
