// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! 统一稳态引擎（D5/D6 规范 §3.3, 2026-08-23）— 三大机制聚合
//!
//! "谱分离（快慢模切分）、对称破缺（分层/戈德斯通）、熵泵（长τ维持）
//! 不是并列的三种稳态判据，而是统一的稳态形成过程"——
//! 本模块把三者聚合为单一分析入口, 共享一份谱分解 + 观测者带宽。
//!
//! 依赖链（D1-D3 已落地）:
//!   - D1: 完整复L（非对角）+ 完整 v/w → 谱分解准确
//!   - D2: 观测者带宽（快/慢模分组随 τ_obs 变化）
//!   - D3: 熵泵严格定义（耗散相消干涉）
//!   - D5: 对称破缺检测（谱隙/τ分裂/戈德斯通/层级化）
//!
//! 诚实标注: 熵泵在谱分离失效（P17 湍流态/带宽不匹配）时返回 None,
//! 不静默给出错误值; 对称破缺在谱信息不足时标记, 不静默判定。

#![allow(non_snake_case)]

use nalgebra::Complex as NComplex;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use super::band_projection::三层带宽;
use super::causal_probe::因果探测结果;
use super::entropy_pump_core::{计算熵泵, 熵泵结果};
use super::nonhermitian_spectrum::非厄米谱分解;
use super::symmetry_breaking::{对称破缺检测器, 对称破缺结果};

/// 稳态分析输入（统一共享一份谱分解 + 观测者带宽）
#[derive(Debug, Clone)]
pub struct 稳态分析输入 {
    /// 完整复L（观测基, D1 修复后含非对角）
    pub l: DMatrix<NComplex<f64>>,
    /// 非厄米谱分解（完整复 v/w）
    pub 分解: 非厄米谱分解,
    /// 观测者带宽（D2）
    pub 带宽: 三层带宽,
    /// 初态扰动（熵泵 W_S/P_in 的展开系数驱动源）
    pub 初态: Vec<f64>,
    /// D1 联动: 复L 非对角是否显著（对角近似 → 对称破缺不静默）
    pub l_非对角显著: bool,
}

impl 稳态分析输入 {
    pub fn 新建(
        l: DMatrix<NComplex<f64>>,
        分解: 非厄米谱分解,
        带宽: 三层带宽,
        初态: Vec<f64>,
        l_非对角显著: bool,
    ) -> Self {
        Self { l, 分解, 带宽, 初态, l_非对角显著 }
    }
}

/// 稳态分析结果（三大机制聚合）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 稳态分析结果 {
    /// D5: 对称破缺检测（谱隙/τ分裂/戈德斯通/层级化）
    pub 对称破缺: 对称破缺结果,
    /// D3: 熵泵结果（谱分离失效 → None, 不静默）
    pub 熵泵: Option<熵泵结果>,
    /// D6: 造波因果探测（κ 方向判据 + 安全约束）
    pub 因果探测: 因果探测结果,
    /// 说明
    pub 说明: String,
}

/// 统一稳态引擎（D5/D6 §3.3）
pub struct 稳态引擎;

impl 稳态引擎 {
    /// 统一稳态分析——共享一份谱分解 + 观测者带宽
    pub fn 分析(&self, 输入: &稳态分析输入) -> 稳态分析结果 {
        let 破缺 = 对称破缺检测器::default().检测(输入.l_非对角显著, &输入.分解);
        let 熵泵 = 计算熵泵(&输入.l, &输入.分解, &输入.带宽, &输入.初态).ok();
        let 探测 = super::causal_probe::因果探测器::default().探测(&输入.l, &输入.分解);
        let 说明 = format!(
            "统一稳态分析: {} · 熵泵{} · 因果探测{}",
            破缺.说明,
            match &熵泵 {
                Some(e) => if e.熵泵条件成立 { "成立" } else { "未成立" },
                None => "失效(None)",
            },
            if 探测.谱信息充足 { "可用" } else { "信息不足" },
        );
        稳态分析结果 {
            对称破缺: 破缺,
            熵泵,
            因果探测: 探测,
            说明,
        }
    }
}

// ============================================================================
// 测试（D5/D6 规范 §3.3 聚合验证）
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::nonhermitian_spectrum::非厄米谱求解器;

    /// 构造已知 L: 快模(高耗散) + 慢模(低耗散) + Γ 非对角耦合（相位协同）
    fn 熵泵耦合L() -> DMatrix<NComplex<f64>> {
        let n = 4;
        let mut l = DMatrix::<NComplex<f64>>::zeros(n, n);
        l[(0, 0)] = NComplex::new(0.0, -2.0);
        l[(1, 1)] = NComplex::new(0.1, -1.25);
        l[(2, 2)] = NComplex::new(0.0, -0.01);
        l[(3, 3)] = NComplex::new(0.2, -0.005);
        // 快-慢耦合块（耗散相消干涉: 同相 +a+ib, a≈b → Im Σ>0）
        l[(2, 0)] = NComplex::new(0.3, 0.3);
        l[(0, 2)] = NComplex::new(0.3, 0.3);
        l[(3, 1)] = NComplex::new(0.25, 0.25);
        l[(1, 3)] = NComplex::new(0.25, 0.25);
        l
    }

    /// 统一稳态引擎聚合: 一份输入 → 三机制结果都在
    #[test]
    fn 测试_统一稳态引擎_聚合() {
        let l = 熵泵耦合L();
        let 求解 = 非厄米谱求解器::default();
        let 分解 = 求解.分解(&l).unwrap();
        let 带宽 = 三层带宽 { 快模上界: 1.0, 慢模下界: 50.0, ..Default::default() };
        let 输入 = 稳态分析输入::新建(l, 分解, 带宽, vec![1.0, 0.5, 0.0, 0.0], true);
        let 结果 = 稳态引擎.分析(&输入);
        // 对称破缺: τ 分裂（快 0.5/0.8 vs 慢 100/200）→ 破缺
        assert_ne!(结果.对称破缺.破缺判据, super::super::symmetry_breaking::破缺判据::独立相, "τ分裂应破缺: {}", 结果.对称破缺.说明);
        // 熵泵: 耦合 → 成立
        let 泵 = 结果.熵泵.expect("谱分离有效应给出熵泵结果");
        assert!(泵.熵泵条件成立, "耦合 L 熵泵成立: {}", 泵.说明);
        // 因果探测: 非对角显著 → 谱信息充足 + 方向判定非空
        assert!(结果.因果探测.谱信息充足, "非对角显著应充足");
        assert!(!结果.因果探测.方向判据.is_empty(), "应有方向判定");
        assert!(结果.说明.contains("统一稳态分析"), "{}", 结果.说明);
    }

    /// 谱分离失效 → 熵泵 None（不静默）
    #[test]
    fn 测试_统一稳态引擎_熵泵失效None() {
        let l = 熵泵耦合L();
        let 求解 = 非厄米谱求解器::default();
        let 分解 = 求解.分解(&l).unwrap();
        // 带宽把全部模归为快模 → 慢模空 → 熵泵 Err → None
        let 坏带宽 = 三层带宽 { 快模上界: 1e9, 慢模下界: 1e12, ..Default::default() };
        let 输入 = 稳态分析输入::新建(l, 分解, 坏带宽, vec![1.0, 0.5, 0.0, 0.0], true);
        let 结果 = 稳态引擎.分析(&输入);
        assert!(结果.熵泵.is_none(), "慢模空熵泵应为 None");
        assert!(结果.说明.contains("失效"), "{}", 结果.说明);
    }

    /// 谱信息不足（对角 L）→ 对称破缺不静默 + 因果探测标记
    #[test]
    fn 测试_统一稳态引擎_谱信息不足() {
        let n = 4;
        let mut l = DMatrix::<NComplex<f64>>::zeros(n, n);
        l[(0, 0)] = NComplex::new(0.0, -2.0);
        l[(1, 1)] = NComplex::new(0.1, -1.25);
        l[(2, 2)] = NComplex::new(0.0, -0.01);
        l[(3, 3)] = NComplex::new(0.2, -0.005);
        let 求解 = 非厄米谱求解器::default();
        let 分解 = 求解.分解(&l).unwrap();
        let 带宽 = 三层带宽 { 快模上界: 1.0, 慢模下界: 50.0, ..Default::default() };
        let 输入 = 稳态分析输入::新建(l, 分解, 带宽, vec![1.0, 0.5, 0.0, 0.0], false);
        let 结果 = 稳态引擎.分析(&输入);
        assert!(!结果.对称破缺.谱信息充足, "谱信息不足应标记");
        assert!(!结果.因果探测.谱信息充足, "因果探测应标记");
    }
}
