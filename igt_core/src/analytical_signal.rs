// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! Hilbert 解析信号变换（D7, 2026-08-23, 规范 §3 Step1）
//!
//! z_i(t) = x_i(t) + i·H[x_i(t)]——把实信号转为复解析信号,
//! 使复 EDMD 的 K 成为复矩阵、L 成为一般复矩阵、Γ 可正定（熵泵必要前提）。
//!
//! 理论合法性（D7 §2 理论确认）:
//!   - |z(t)| = 原信号幅值包络, arg z(t) = 瞬时相位
//!   - 解析信号消除负频率冗余, EDMD 本征值不再强制共轭对
//!   - τ=1/γ 和 ω 物理意义完全不变（γ 是衰减率, 与频率符号无关）
//!
//! 频域实现（规范 §3 Step1 工程实现）:
//!   FFT → 正频率×2, 负频率置 0（DC/Nyquist 保持）→ 逆 FFT。

#![allow(non_snake_case)]

use nalgebra::Complex as NComplex;
use rustfft::FftPlanner;

/// 解析信号 z = x + i·H[x]（频域 Hilbert 变换）
pub fn 解析信号(x: &[f64]) -> Vec<NComplex<f64>> {
    let n = x.len();
    if n < 2 {
        return x.iter().map(|&v| NComplex::new(v, 0.0)).collect();
    }
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);
    let mut buf: Vec<NComplex<f64>> = x.iter().map(|&v| NComplex::new(v, 0.0)).collect();
    fft.process(&mut buf);
    let n2 = n / 2;
    // 正频率 ×2（DC i=0 不动）
    for i in 1..n2 {
        buf[i] *= NComplex::new(2.0, 0.0);
    }
    // 负频率置 0（Nyquist i=n2 保持, n 偶数）
    for i in (n2 + 1)..n {
        buf[i] = NComplex::new(0.0, 0.0);
    }
    ifft.process(&mut buf); // rustfft inverse 不含 1/n——手动归一化
    let 归一 = NComplex::new(n as f64, 0.0);
    for v in buf.iter_mut() {
        *v /= 归一;
    }
    buf
}

/// 希尔伯特虚部 H[x]（解析信号的虚部, 实信号 90° 相移）
pub fn 希尔伯特(x: &[f64]) -> Vec<f64> {
    解析信号(x).iter().map(|z| z.im).collect()
}

// ============================================================================
// 测试（D7 规范 §6 V1 前置性质）
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// 正弦解析信号: 虚部 = 实部 90° 相移（-cos）; |z|≈原幅值
    #[test]
    fn 测试_解析信号_正弦性质() {
        let n = 64;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * std::f64::consts::TAU / 16.0).sin()).collect();
        let z = 解析信号(&x);
        // 实部 ≈ 原信号
        for i in 0..n {
            assert!((z[i].re - x[i]).abs() < 0.05, "实部≈原信号: i={} re={} x={}", i, z[i].re, x[i]);
        }
        // 虚部 ≈ -cos（90° 相移）
        for i in 0..n {
            let 期望 = -(i as f64 * std::f64::consts::TAU / 16.0).cos();
            assert!((z[i].im - 期望).abs() < 0.05, "虚部90°相移: i={} im={} 期望={}", i, z[i].im, 期望);
        }
        // |z|≈1（幅值包络）
        for i in 0..n {
            let 幅 = (z[i].re * z[i].re + z[i].im * z[i].im).sqrt();
            assert!((幅 - 1.0).abs() < 0.05, "包络≈1: i={} |z|={}", i, 幅);
        }
    }

    /// 解析信号不改变能量（帕塞瓦尔近似）
    #[test]
    fn 测试_解析信号_能量不变() {
        let n = 128;
        let x: Vec<f64> = (0..n)
            .map(|i| (i as f64 * std::f64::consts::TAU / 20.0).sin() + 0.5 * (i as f64 * std::f64::consts::TAU / 5.0).cos())
            .collect();
        let z = 解析信号(&x);
        // 解析信号实部能量 ≈ 原信号能量（虚部为正交分量）
        let 原能量: f64 = x.iter().map(|v| v * v).sum();
        let 实部能量: f64 = z.iter().map(|v| v.re * v.re).sum();
        assert!(
            (原能量 - 实部能量).abs() / 原能量.max(1e-9) < 0.05,
            "实部能量≈原能量: {} vs {}",
            实部能量, 原能量
        );
    }

    /// 短序列退化（不 panic）
    #[test]
    fn 测试_解析信号_短序列() {
        let z = 解析信号(&[1.0]);
        assert_eq!(z.len(), 1);
        assert!((z[0].im).abs() < 1e-9, "单点虚部≈0");
    }
}
