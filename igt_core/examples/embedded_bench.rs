// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! R1.2 嵌入式性能基线基准（CODE-AUDIT-001 / DISPATCH-R1R2 R1.2）
//!
//! **只读测量**——不改动 core 任何计算逻辑, 仅产出基线数字入档 EMBEDDED_BASELINE.md。
//!
//! 运行（desktop target, release）:
//! ```text
//! cargo run --release --example embedded_bench
//! ```
//!
//! 测量项:
//!   A. `complex_dmd::复EDMD解析信号`——1024 采样点 × 通道数 {2, 8, 30}
//!      （通道数上界 30 = D8 §4.1 r≤30 边界）
//!   B. `nonhermitian_spectrum::非厄米谱求解器::分解`——矩阵维数 {32, 64, 128, 256, 512}
//!      （含求解器文档适用上限 n≤64 之外的维数, 如实记录不做优化）
//!
//! 目标（ARCH-1 v2.1 §17-4）: 手机级单窗谱分解 ≤100ms。

#![allow(non_snake_case)]

use igt_core_math::complex_dmd;
use igt_core_math::nonhermitian_spectrum::非厄米谱求解器;
use nalgebra::Complex as NComplex;
use nalgebra::DMatrix;
use std::time::Instant;

/// 构造非厄米耗散矩阵 L（对角 -1/τ + 微小振荡虚部, 相邻非对角耦合）
fn 构造耗散矩阵(n: usize) -> DMatrix<NComplex<f64>> {
    let mut l = DMatrix::<NComplex<f64>>::zeros(n, n);
    for i in 0..n {
        let τ = 1.0 + i as f64 * 0.5;
        l[(i, i)] = NComplex::new(-1.0 / τ, i as f64 * 0.01);
    }
    for i in 0..n.saturating_sub(1) {
        l[(i, i + 1)] = NComplex::new(0.1, -0.05);
        l[(i + 1, i)] = NComplex::new(0.1, 0.05);
    }
    l
}

fn main() {
    println!("=== IGT CORE 嵌入式性能基线（release, desktop）===");
    println!("目标（ARCH-1 v2.1 §17-4）: 手机级单窗谱分解 ≤100ms\n");

    // ===== A. complex_dmd（1024 采样点 × 通道数）=====
    println!("[A] complex_dmd::复EDMD解析信号（1024 采样点, 快慢双模）");
    for &通道 in &[2usize, 8, 30] {
        let n = 1024;
        let Δt = 0.1;
        let 通道组: Vec<Vec<f64>> = (0..通道)
            .map(|c| {
                (0..n)
                    .map(|k| {
                        let t = k as f64 * Δt;
                        (-t / 20.0).exp() * (0.4 * t + c as f64 * 0.1).cos()
                            + 0.5 * (-t / 5.0).exp() * (0.8 * t).cos()
                    })
                    .collect()
            })
            .collect();
        let 起点 = Instant::now();
        let 结果 = complex_dmd::复EDMD解析信号(&通道组, Δt);
        let 耗时ms = 起点.elapsed().as_secs_f64() * 1000.0;
        match 结果 {
            Some(r) => println!(
                "    通道={:>2} 复L阶={:>2} τ链={:>2}模 耗时={:>8.1}ms",
                通道,
                r.复L阶,
                r.τ链.len(),
                耗时ms
            ),
            None => println!("    通道={:>2} 失败(None) 耗时={:>8.1}ms", 通道, 耗时ms),
        }
    }

    // ===== B. nonhermitian_spectrum 谱分解（矩阵维数）=====
    println!("\n[B] 非厄米谱求解器::分解（非厄米耗散矩阵 L）");
    println!("    注: 求解器文档适用上限 n≤64（复 Schur QR 迭代）——更大维数如实记录");
    for &维 in &[32usize, 64, 128, 256, 512] {
        let l = 构造耗散矩阵(维);
        let 求解器 = 非厄米谱求解器 {
            max_iter: 300,
            ε: 1e-10,
            n_sample: 200,
        };
        let 起点 = Instant::now();
        let 结果 = 求解器.分解(&l);
        let 耗时ms = 起点.elapsed().as_secs_f64() * 1000.0;
        match 结果 {
            Ok(d) => println!(
                "    维数={:>3} 模数={:>3} 稳定模={:>3} 耗时={:>9.1}ms {}",
                维,
                d.模数,
                d.稳定模数,
                耗时ms,
                if 耗时ms <= 100.0 { "✓≤100ms" } else { "✗超目标" }
            ),
            Err(e) => println!("    维数={:>3} 失败({}) 耗时={:>9.1}ms", 维, e, 耗时ms),
        }
    }

    println!("\n=== 基线测量完成（数字入档 igt_core/EMBEDDED_BASELINE.md）===");
}
