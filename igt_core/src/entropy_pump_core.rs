// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! 熵泵机制 core 化（D3, 2026-08-23）— P24 §4-5 + 理论研究团队严格定义
//!
//! 理论依据:
//!   - 《熵泵机制的 L=H-iΓ 严格数学定义》(理论研究团队 2026-08-23 补全):
//!     熵泵 = 非厄米耦合导致的耗散相消干涉——通过 H(振荡)/Γ(耗散) 非对角元
//!     相对相位协同, 使慢模有效耗散率低于裸耗散率, 延长慢模寿命（与 PT 对称同根）。
//!   - P24 §4-5: 吃短τ源 → 长τ结构 → 排长τ废热; 卡诺效率类比。
//!   - P11: 谱分离/自能 Σ(ω); P0: 三层切分; D2: 观测者带宽参数化。
//!
//! 前置依赖（必须满足, 否则熵泵错误/为 0）:
//!   - D1 R3: 完整非对角复L（对角近似 L 无 L_SF 耦合块, 熵泵=0）;
//!   - D1 R2: 完整复 v/w（块构造丢虚部 → 自能虚部失真）;
//!   - D2: 观测者带宽（快/慢模分组随 τ_obs 变化, 熵泵结果随之变化）。
//!
//! 6 步计算（理论 §六工程可计算性）:
//!   1. EDMD 提取复L（完整非对角, 由上层提供）
//!   2. 对角化按 γ 归层（观测者带宽三层切分）
//!   3. 块 L_F/L_SF/L_FS（观测基, 完整复数）
//!   4. Σ(ω_S) = L_SF(ω_S - L_F)^{-1}L_FS
//!   5. ΔΓ = (1/|S|)Tr[Im Σ(ω_S)]
//!   6. η_pump = ΔΓ·W_S / P_in, 校验卡诺上限 ≤ 1-τ_F/τ_S
//!
//! 熵泵（机制/过程量）与熵债 D_total（状态量）区分:
//!   熵泵工作 → 维持慢模长τ → D_total 降低。熵泵是"原因机制", D_total 是"结果状态"。

#![allow(non_snake_case)]

use nalgebra::Complex as NComplex;
use nalgebra::ComplexField;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use super::band_projection::三层带宽;
use super::nonhermitian_spectrum::{非厄米谱分解, 非厄米谱求解器};

/// 熵泵错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum 熵泵错误 {
    /// 谱分离失效（P17 湍流态/弱耦合近似不收敛）——不静默给出错误熵泵值
    谱分离失效(String),
    /// 数据不足
    数据不足(String),
    /// D7（2026-08-23）: 维度不足——熵泵需 ≥2 通道（跨模态耦合效应, 单通道无耦合可泵）
    /// 单通道数据只能做谱统计（D_s/椭圆系综）, 不能计算熵泵（规范 §4 硬边界）
    维度不足(String),
}

impl std::fmt::Display for 熵泵错误 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            熵泵错误::谱分离失效(s) => write!(f, "谱分离失效: {}", s),
            熵泵错误::数据不足(s) => write!(f, "数据不足: {}", s),
            熵泵错误::维度不足(s) => write!(f, "维度不足: {}", s),
        }
    }
}

/// 迹约定（D3 v1.3 契约, T-8）: K=1 单窗 vs K 窗系综平均
/// 不一致 K 的 Γ_eff/τ_FET 不得跨模块混用
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceConvention {
    /// K=1 单窗 EDMD
    SingleMode,
    /// K 窗系综平均
    EnsembleAvg,
}

impl TraceConvention {
    pub fn 名称(&self) -> &'static str {
        match self {
            TraceConvention::SingleMode => "SingleMode",
            TraceConvention::EnsembleAvg => "EnsembleAvg",
        }
    }
}

/// 域标志（D3 v1.3 契约, T-8）: 熵泵效率定义的有效域三档
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DomainFlags {
    /// 弱耦合+有限稳态域: u/Γ_S<0.9 且 Γ_-≥Γ_min=0.05——卡诺界/η_pump/τ_FET 全有效
    SafeDomain,
    /// 0.9≤u/Γ_S<0.95（SW 近似偏差 10% 量级）: 卡诺界可参考, η/τ_FET 有效但带精度警示
    MarginalCoupling,
    /// u/Γ_S>0.95 或 Γ_-<Γ_min: η_pump 发散失效（边际稳定奇点）, τ_FET 按"谱定义失效"降级,
    /// 治理决策禁止按发散大值加权
    OutsideDomain,
}

impl DomainFlags {
    pub fn 名称(&self) -> &'static str {
        match self {
            DomainFlags::SafeDomain => "SafeDomain",
            DomainFlags::MarginalCoupling => "MarginalCoupling",
            DomainFlags::OutsideDomain => "OutsideDomain",
        }
    }
}

/// 熵泵计算结果（core 层, V10.1 严格定义, **D3 v1.3 契约版**）
/// 契约地位: 本结构体是 η_pump / η_relax / Γ_eff / τ_FET 全体系唯一契约源。
/// 下游（D15治理、P33自愈、P34器官、验收）一律读字段, 禁止用 η 自算 τ 或反推 Γ。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 熵泵结果 {
    /// 熵泵条件成立: Im Σ(ω_S) > 0
    pub 熵泵条件成立: bool,
    /// 熵泵增益 ΔΓ = Γ_S - Γ_eff = (1/|S|)Tr[Im Σ(ω_S)]
    pub 熵泵增益_ΔΓ: f64,
    /// 慢模裸寿命 τ_S = 1/Γ_S
    pub 慢模裸寿命_τS: f64,
    /// 慢模有效寿命 τ_eff = 1/Γ_eff
    pub 慢模有效寿命_τeff: f64,
    /// 熵泵流量 Φ_pump = ΔΓ·W_S
    pub 熵泵流量_Φpump: f64,
    /// 快模输入功率 P_in = Σ_{n∈F} γ_n |c_n|²
    pub 快模输入功率_Pin: f64,
    /// 熵泵效率 η_pump = Φ_pump / P_in
    pub 熵泵效率_ηpump: f64,
    /// 卡诺上限 1 - τ_F/τ_S
    pub 卡诺上限: f64,
    /// 自能虚部 Im Σ（诊断用）
    pub 自能虚部_ImΣ: f64,
    /// 慢模组特征耗散 Γ_S（平均裸耗散）
    pub 慢模裸耗散_ΓS: f64,
    /// 慢模组有效耗散 Γ_eff
    pub 慢模有效耗散_Γeff: f64,
    /// 快模数 / 慢模数（诊断用）
    pub 快模数: usize,
    pub 慢模数: usize,
    /// 谱分离有效（弱耦合近似）——false 表示微扰路径失效（湍流态）
    pub 谱分离有效: bool,
    /// 说明
    pub 说明: String,

    // —— v1.3 契约字段（T-8, ED-1 R1/R2, D3 v1.3 契约块逐字）——
    /// =1/Γ_eff, 与 τ_FET 同值（不变量①）
    pub 弛豫比_ηrelax: f64,
    /// =1/τeff（携带迹约定枚举）
    pub Γeff: f64,
    /// 迹约定: SingleMode | EnsembleAvg（不一致 K 不得跨模块混用）
    pub 迹约定: TraceConvention,
    /// =τeff=1/Γ_eff; D15 债权有效期唯一来源, 禁止 τ_S/(1-η) 自算
    pub τ_FET: f64,
    /// =1-Γ_S/Γ_F, 仅 SafeDomain/MarginalCoupling 有效（域外 NaN）
    pub 卡诺上限_契约: f64,
    /// 域标志: SafeDomain | MarginalCoupling | OutsideDomain
    pub 域标志: DomainFlags,
    /// u/Γ_S 耦合强度比（<0.9安全 / 0.9~0.95偏强 / >0.95域外）
    pub 耦合强度比_u_over_ΓS: f64,
    /// 边际稳定裕度 Γ_-（须≥Γ_min=0.05, 否则边际稳定奇点）
    pub 边际稳定裕度_Γminus: f64,
    /// 非微扰流量（谱分离有效=false 时由 γ(τ) 轨填充; 规则态 None）
    pub 非微扰流量_Φpump_np: Option<f64>,
}

/// 观测基子块（完整复数）
fn 子块(
    l: &DMatrix<NComplex<f64>>,
    行: &[usize],
    列: &[usize],
) -> DMatrix<NComplex<f64>> {
    let mut m = DMatrix::<NComplex<f64>>::zeros(行.len(), 列.len());
    for (i, &r) in 行.iter().enumerate() {
        for (j, &c) in 列.iter().enumerate() {
            m[(i, j)] = l[(r, c)];
        }
    }
    m
}

/// 熵泵计算入口 — 输入完整复L + 谱分解（D1 修复后）+ 观测者带宽（D2）+ 初态扰动
///
/// # 参数
/// - `l`: 完整非对角复L（观测基, D1 R3 修复后）
/// - `分解`: 非厄米谱分解（完整复 v/w）
/// - `带宽`: 观测者带宽（D2; 快/慢模分组随 τ_obs 变化）
/// - `初态`: 初始扰动 Δx（展开系数 c_n=w†Δx 的驱动源）
pub fn 计算熵泵(
    l: &DMatrix<NComplex<f64>>,
    分解: &非厄米谱分解,
    带宽: &三层带宽,
    初态: &[f64],
) -> Result<熵泵结果, 熵泵错误> {
    // D7（2026-08-23）: 硬边界——熵泵需 ≥2 通道（状态空间维数）, 单通道无跨模态耦合
    //（放最前, 优先于模数检查——单通道必然无法熵泵）
    if l.nrows() < 2 {
        return Err(熵泵错误::维度不足(
            "维度不足: 熵泵需≥2通道（跨模态耦合效应, 单通道无耦合可泵; 规范D7§4）——单通道数据只能做谱统计".to_string(),
        ));
    }
    let n = 分解.模数;
    if n < 2 {
        return Err(熵泵错误::数据不足("模数<2, 无法构成快/慢模组".to_string()));
    }
    // 1. 模分组（按 τ 用观测者带宽三层切分）
    let 快模: Vec<usize> = 分解
        .模
        .iter()
        .enumerate()
        .filter(|(_, m)| m.τ > 0.0 && m.τ < 带宽.快模上界)
        .map(|(i, _)| i)
        .collect();
    let 慢模: Vec<usize> = 分解
        .模
        .iter()
        .enumerate()
        .filter(|(_, m)| m.τ > 带宽.慢模下界)
        .map(|(i, _)| i)
        .collect();
    if 快模.is_empty() || 慢模.is_empty() {
        return Err(熵泵错误::谱分离失效(
            "快模或慢模不足——谱分离无效（P17 湍流态/带宽不匹配）, 不静默给出熵泵值".to_string(),
        ));
    }
    let nf = 快模.len();
    let ns = 慢模.len();

    // 2. 块矩阵（观测基, 完整复数——D1 R2 依赖）
    let l_f = 子块(l, &快模, &快模);
    let l_sf = 子块(l, &慢模, &快模);
    let l_fs = 子块(l, &快模, &慢模);

    // 3. ω_S: 慢模平均频率
    let ω_s = 慢模.iter().map(|&i| 分解.模[i].ω).sum::<f64>() / ns as f64;

    // 4. 自能 Σ(ω_S) = L_SF (ω_S·I - L_F)^{-1} L_FS
    let mut 逆 = l_f.clone();
    for i in 0..nf {
        逆[(i, i)] = NComplex::new(ω_s, 0.0) - 逆[(i, i)];
    }
    let 逆 = match 逆.try_inverse() {
        Some(x) => x,
        None => {
            return Err(熵泵错误::谱分离失效(
                "L_F 求逆失败（快模子空间奇异）——弱耦合近似不收敛".to_string(),
            ))
        }
    };
    let Σ = &l_sf * 逆 * &l_fs;

    // 5. ΔΓ = (1/|S|)Tr[Im Σ(ω_S)]
    let imΣ = Σ.trace().im;
    let ΔΓ = imΣ / ns as f64;
    // Γ_S = 慢模平均裸耗散（γ_n = 1/τ_n > 0）
    let Γ_s = 慢模.iter().map(|&i| 1.0 / 分解.模[i].τ).sum::<f64>() / ns as f64;
    let Γ_eff = Γ_s - ΔΓ;

    // 6. W_S / P_in: 展开系数 c_n = w†Δx（完整复 w, D1 依赖）→ |c|²
    let 求解 = 非厄米谱求解器::default();
    let c2 = 求解.展开系数(分解, 初态); // Vec<f64>（各模 |c|², 和为 1）
    let w_s = 慢模.iter().map(|&i| c2[i]).sum::<f64>();
    let p_in = 快模.iter().map(|&i| (1.0 / 分解.模[i].τ) * c2[i]).sum::<f64>();

    // 7. 熵泵效率 + 卡诺上限
    let η = if p_in > 1e-12 { ΔΓ * w_s / p_in } else { 0.0 };
    let τ_f = 快模.iter().map(|&i| 分解.模[i].τ).sum::<f64>() / nf as f64;
    let τ_s = 慢模.iter().map(|&i| 分解.模[i].τ).sum::<f64>() / ns as f64;
    let 卡诺上限 = (1.0 - τ_f / τ_s).max(0.0);

    let 条件成立 = ΔΓ > 1e-9;
    // 第405批（D7 优化项）: 谱分离有效 = Γ_eff>0（弱耦合近似收敛）∧ η≤卡诺上限
    //（η>卡诺上限 → 近似失效/数值溢出, 不静默给出不可信熵泵值）
    let 谱分离有效 = Γ_eff > 0.0 && η <= 卡诺上限 + 1e-9;

    // ============ T-8 D3 v1.3 契约量（ED-1: η_pump/η_relax 严格区分） ============
    // η_relax = ΔΓ/Γ_S（弛豫比, 恒有界<1; τ_FET 动力学专用——不变量② 成立基础）
    let 弛豫比_ηrelax = ΔΓ / Γ_s.max(1e-12);
    // τ_FET ≡ τeff ≡ 1/Γeff（不变量①, 同一赋值派生）
    let τ_fet = 1.0 / Γ_eff.max(1e-12);
    // 耦合强度 u = ‖L_SF‖_F（快慢耦合块 Frobenius 范数, SW 近似偏差度量）
    let mut u = 0.0f64;
    for i in 0..ns {
        for j in 0..nf {
            let z = l_sf[(i, j)];
            u += z.re * z.re + z.im * z.im;
        }
    }
    let u = u.sqrt();
    let u_over_ΓS = u / Γ_s.max(1e-12);
    // 边际稳定裕度 Γ_- = Γ_eff（慢模有效耗散, D3 G-14: "W_S=Φ_in/Γ_eff 在 Γ_eff≥Γ_min 有界,
    // Γ_eff→0 发散"——边际稳定奇点以有效耗散衡量, 须≥Γ_min=0.05）
    let Γminus = Γ_eff;
    // 域标志（D3 v1.3 §3.4）: SafeDomain / MarginalCoupling / OutsideDomain
    let 域标志 = if u_over_ΓS > 0.95 || Γminus < 0.05 {
        DomainFlags::OutsideDomain
    } else if u_over_ΓS >= 0.9 {
        DomainFlags::MarginalCoupling
    } else {
        DomainFlags::SafeDomain
    };

    // 非微扰双轨（D3 v1.3 §3.6 / P40 G-13）:
    // 规则态（谱分离有效=true）→ 微扰路径有效, η_pump/卡诺上限填充;
    // 湍流态（谱分离有效=false）→ Φpump_np 由 γ(τ) 轨填充, η_pump/卡诺上限置 NaN（不变量③）
    // 域外（OutsideDomain）→ η_pump 发散失效, 同样置 NaN 不输出伪值
    let 微扰有效 = 谱分离有效 && 域标志 != DomainFlags::OutsideDomain;
    let η_输出 = if 微扰有效 { η } else { f64::NAN };
    let 卡诺_输出 = if 微扰有效 { 卡诺上限 } else { f64::NAN };
    // 非微扰流量: γ(τ) 轨简化实现——慢区 Γ 矩阵投影的耗散净减少率（P40 §2 离散模 δ 峰极限）
    let Φpump_np = if 谱分离有效 {
        None
    } else {
        Some(慢区Γ投影耗散净减(l, 分解, &慢模))
    };

    let 状态 = if 条件成立 { "成立" } else { "未成立" };
    let 说明 = format!(
        "熵泵{}[{}]: ΔΓ={:.4}, τ_S={:.2}→τ_eff={:.2}, η_pump={:.3}, η_relax={:.3}, τ_FET={:.2}, u/Γ_S={:.3}, 域={}",
        状态,
        域标志.名称(),
        ΔΓ, 1.0 / Γ_s.max(1e-12), 1.0 / Γ_eff.max(1e-12),
        η, 弛豫比_ηrelax, τ_fet, u_over_ΓS, 域标志.名称()
    );

    Ok(熵泵结果 {
        熵泵条件成立: 条件成立,
        熵泵增益_ΔΓ: ΔΓ,
        慢模裸寿命_τS: 1.0 / Γ_s.max(1e-12),
        慢模有效寿命_τeff: 1.0 / Γ_eff.max(1e-12),
        熵泵流量_Φpump: ΔΓ * w_s,
        快模输入功率_Pin: p_in,
        熵泵效率_ηpump: η_输出,
        卡诺上限,
        自能虚部_ImΣ: imΣ,
        慢模裸耗散_ΓS: Γ_s,
        慢模有效耗散_Γeff: Γ_eff,
        快模数: nf,
        慢模数: ns,
        谱分离有效,
        说明,
        弛豫比_ηrelax,
        Γeff: Γ_eff,
        迹约定: TraceConvention::SingleMode,
        τ_FET: τ_fet,
        卡诺上限_契约: 卡诺_输出,
        域标志,
        耦合强度比_u_over_ΓS: u_over_ΓS,
        边际稳定裕度_Γminus: Γminus,
        非微扰流量_Φpump_np: Φpump_np,
    })
}

/// T-8 D3 v1.3 非微扰（P40 G-13 简化轨）: 慢区 Γ 投影的耗散净减少率
/// Φ_pump_np ≈ |S|·(Γ_S^裸 - Γ_S^有效_投影), 其中
///   Γ_S^裸 = Σ_{n∈S}γ_n/|S|（已算）; Γ_S^有效_投影 = Tr(P_S Γ P_S)/|S|,
///   P_S = Σ_{n∈S}|v_n⟩⟨w_n|（双正交投影, D1 依赖）——湍流态 Σ(ω) 失效时
///   Γ 矩阵投影仍可计算（Γ 正定由 D7 保证）
fn 慢区Γ投影耗散净减(
    l: &DMatrix<NComplex<f64>>,
    分解: &非厄米谱分解,
    慢模: &[usize],
) -> f64 {
    let n = l.nrows();
    // Γ = i(L†-L)/2
    let mut gamma = DMatrix::<NComplex<f64>>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            gamma[(i, j)] = NComplex::new(0.0, 1.0) * (l[(j, i)].conjugate() - l[(i, j)]) / NComplex::new(2.0, 0.0);
        }
    }
    // P_S = Σ_{n∈S} v_n w_n†（双正交投影）
    let mut ps = DMatrix::<NComplex<f64>>::zeros(n, n);
    for &idx in 慢模 {
        for a in 0..n {
            let v_a = crate::nonhermitian_spectrum::取右向量(&分解.模[idx], a);
            for b in 0..n {
                let w_b = crate::nonhermitian_spectrum::取左向量(&分解.模[idx], b);
                ps[(a, b)] += v_a * w_b.conjugate();
            }
        }
    }
    let pΓp = &ps * &gamma * &ps;
    let 投影耗散 = pΓp.trace().re / 慢模.len().max(1) as f64;
    // 裸耗散 Γ_S = Σγ_n/|S|
    let 裸耗散 = 慢模.iter().map(|&i| 1.0 / 分解.模[i].τ).sum::<f64>() / 慢模.len().max(1) as f64;
    (裸耗散 - 投影耗散).max(0.0) * 慢模.len() as f64
}

// ============================================================================
// 测试（D3 规范 §7 V1-V5）
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// 构造已知 L: 快模(高耗散) + 慢模(低耗散) + Γ 非对角耦合（相位协同）
    ///   H 非对角 0.3（振荡耦合）, Γ 非对角 -i·0.2（耗散耦合同相位）
    fn 熵泵耦合L(对角: bool) -> DMatrix<NComplex<f64>> {
        let n = 4;
        let mut l = DMatrix::<NComplex<f64>>::zeros(n, n);
        // 快模组: τ=0.5, 0.8（高耗散 γ=2, 1.25）
        l[(0, 0)] = NComplex::new(0.0, -2.0);
        l[(1, 1)] = NComplex::new(0.1, -1.25);
        // 慢模组: τ=100, 200（低耗散 γ=0.01, 0.005）
        l[(2, 2)] = NComplex::new(0.0, -0.01);
        l[(3, 3)] = NComplex::new(0.2, -0.005);
        // 快-慢耦合块（耗散相消干涉: 同相 +a+ib, a≈b → Im Σ>0, 慢模有效耗散降低）
        // L = H - iΓ: 实部 H 振荡耦合 + 虚部 -iΓ 耗散耦合; 同相使自能虚部为正。
        // 对角=1 关耦合（D3-V2/V5 无泵对照）。
        if !对角 {
            l[(2, 0)] = NComplex::new(0.3, 0.3);
            l[(0, 2)] = NComplex::new(0.3, 0.3);
            l[(3, 1)] = NComplex::new(0.25, 0.25);
            l[(1, 3)] = NComplex::new(0.25, 0.25);
        }
        l
    }

    /// 带宽: 快界 1.0 / 慢界 50（快模 τ<1, 慢模 τ>50）
    fn 测试带宽() -> 三层带宽 {
        三层带宽 { 快模上界: 1.0, 慢模下界: 50.0, ..Default::default() }
    }

    /// D3-V1: 含 Γ 非对角耦合 → 熵泵条件成立（Im Σ>0, Γ_eff<Γ_S, τ_eff>τ_S）
    #[test]
    fn 测试_D3V1_熵泵条件成立() {
        let l = 熵泵耦合L(false);
        let 求解 = 非厄米谱求解器::default();
        let 分解 = 求解.分解(&l).unwrap();
        let 结果 = 计算熵泵(&l, &分解, &测试带宽(), &[1.0, 0.5, 0.0, 0.0]).unwrap();
        assert!(结果.熵泵条件成立, "Γ 非对角相位协同应成立熵泵: {}", 结果.说明);
        assert!(结果.自能虚部_ImΣ > 0.0, "Im Σ>0: {}", 结果.自能虚部_ImΣ);
        assert!(
            结果.慢模有效耗散_Γeff < 结果.慢模裸耗散_ΓS,
            "Γ_eff<Γ_S: {} vs {}",
            结果.慢模有效耗散_Γeff, 结果.慢模裸耗散_ΓS
        );
        assert!(
            结果.慢模有效寿命_τeff > 结果.慢模裸寿命_τS,
            "τ_eff>τ_S: {} vs {}",
            结果.慢模有效寿命_τeff, 结果.慢模裸寿命_τS
        );
    }

    /// 第405批（D7 优化项）: η 超卡诺上限 → 谱分离有效=false（近似失效不静默）
    #[test]
    fn 测试_D7_η超卡诺谱分离无效() {
        let 求解 = 非厄米谱求解器::default();
        let mut l = 熵泵耦合L(false);
        // 强耦合: 放大耦合块 → ΔΓ 增大 → η 可能超卡诺上限（弱耦合近似失效）
        l[(2, 0)] = NComplex::new(0.9, 0.9);
        l[(0, 2)] = NComplex::new(0.9, 0.9);
        l[(3, 1)] = NComplex::new(0.8, 0.8);
        l[(1, 3)] = NComplex::new(0.8, 0.8);
        let 分解 = 求解.分解(&l).unwrap();
        if let Ok(泵) = 计算熵泵(&l, &分解, &测试带宽(), &[1.0, 0.5, 0.5, 1.0]) {
            // 强耦合下要么谱分离失效 Err（诚实）, 要么 η 超卡诺/Γ_eff<0 → 谱分离有效=false
            assert!(
                !泵.谱分离有效,
                "强耦合近似失效应标记（η 超卡诺或 Γ_eff<0）: 说明={}",
                泵.说明
            );
        }
        // Err（近似不收敛）也诚实——两者都不静默给出可信熵泵值
    }

    /// D3-V2: Γ 对角（无耦合）→ 无熵泵（ΔΓ≈0 或负向）
    #[test]
    fn 测试_D3V2_无泵情形() {
        let l = 熵泵耦合L(true);
        let 求解 = 非厄米谱求解器::default();
        let 分解 = 求解.分解(&l).unwrap();
        let 结果 = 计算熵泵(&l, &分解, &测试带宽(), &[1.0, 0.5, 0.0, 0.0]).unwrap();
        assert!(!结果.熵泵条件成立, "无耦合不应成立熵泵: {}", 结果.说明);
        assert!(
            结果.熵泵增益_ΔΓ.abs() < 1e-6,
            "对角L 无耦合 → ΔΓ≈0: {}",
            结果.熵泵增益_ΔΓ
        );
    }

    /// D3-V3+V4: 卡诺上限校验（η_pump ≤ 1-τ_F/τ_S）+ 极限 τ_F≪τ_S → η_max≈1
    /// T-8 更新: 当前测试 L 强耦合（u/Γ_S 大）→ OutsideDomain 时 η 诚实降级 NaN（D3 v1.3）
    #[test]
    fn 测试_D3V3_卡诺上限() {
        let l = 熵泵耦合L(false);
        let 求解 = 非厄米谱求解器::default();
        let 分解 = 求解.分解(&l).unwrap();
        let 结果 = 计算熵泵(&l, &分解, &测试带宽(), &[1.0, 0.5, 0.0, 0.0]).unwrap();
        if 结果.熵泵效率_ηpump.is_nan() {
            assert_eq!(结果.域标志, DomainFlags::OutsideDomain, "η NaN 应域外");
            assert!(结果.卡诺上限_契约.is_nan(), "域外卡诺契约字段 NaN");
        } else {
            assert!(
                结果.熵泵效率_ηpump <= 结果.卡诺上限 + 1e-6,
                "η≤η_max: {} ≤ {}",
                结果.熵泵效率_ηpump, 结果.卡诺上限
            );
            assert!(
                结果.卡诺上限 > 0.9,
                "τ_F≪τ_S → η_max≈1: {}",
                结果.卡诺上限
            );
            assert!(结果.熵泵效率_ηpump >= 0.0, "η 非负: {}", 结果.熵泵效率_ηpump);
        }
    }

    /// D3-V5: D1 依赖联动——复L 对角近似（L_SF=0）→ 熵泵=0（防静默错误）
    /// T-8 更新: 对角 L 慢模耗散小（Γ_eff=Γ_s<Γ_min）→ OutsideDomain, η 诚实降级（不输出伪值）
    #[test]
    fn 测试_D3V5_对角L无泵() {
        let l = 熵泵耦合L(true);
        let 求解 = 非厄米谱求解器::default();
        let 分解 = 求解.分解(&l).unwrap();
        let 结果 = 计算熵泵(&l, &分解, &测试带宽(), &[1.0, 0.5, 0.0, 0.0]).unwrap();
        assert_eq!(结果.熵泵流量_Φpump, 0.0, "对角L 无耦合 → Φ=0");
        // 无泵不输出伪值: η=0（域内）或 NaN（域外诚实降级）
        assert!(
            结果.熵泵效率_ηpump == 0.0 || 结果.熵泵效率_ηpump.is_nan(),
            "对角L 无泵不输出伪值: {}",
            结果.熵泵效率_ηpump
        );
    }

    /// 谱分离失效: 快/慢模不足（带宽不匹配）→ Err
    #[test]
    fn 测试_D3_谱分离失效() {
        let l = 熵泵耦合L(false);
        let 求解 = 非厄米谱求解器::default();
        let 分解 = 求解.分解(&l).unwrap();
        // 带宽把全部模归为快模（慢界巨大）→ 慢模空 → Err
        let 坏带宽 = 三层带宽 { 快模上界: 1e9, 慢模下界: 1e12, ..Default::default() };
        let 结果 = 计算熵泵(&l, &分解, &坏带宽, &[1.0, 0.5, 0.0, 0.0]);
        assert!(结果.is_err(), "慢模空应 Err");
        assert_eq!(结果.err().unwrap(), 熵泵错误::谱分离失效("快模或慢模不足——谱分离无效（P17 湍流态/带宽不匹配）, 不静默给出熵泵值".to_string()));
    }

    /// D7-V5（2026-08-23, 规范 §4 硬边界）: 单通道（1×1 L）→ 维度不足, 不静默
    #[test]
    fn 测试_D7V5_维度不足() {
        let 求解 = 非厄米谱求解器::default();
        let l = DMatrix::<NComplex<f64>>::from_element(1, 1, NComplex::new(0.0, -1.0));
        let 分解 = 求解.分解(&l).unwrap();
        let 结果 = 计算熵泵(&l, &分解, &测试带宽(), &[1.0]);
        match 结果 {
            Err(熵泵错误::维度不足(_)) => {}
            other => panic!("单通道应维度不足, 实际: {:?}", other.map(|r| r.熵泵效率_ηpump)),
        }
        // 2×2 正常（非维度不足）
        let l2 = 熵泵耦合L(false);
        let 分解2 = 求解.分解(&l2).unwrap();
        let 结果2 = 计算熵泵(&l2, &分解2, &测试带宽(), &[1.0, 0.5, 0.0, 0.0]);
        assert!(结果2.is_ok(), "2×2 不应维度不足: {:?}", 结果2.err());
    }

    // ==================== T-8 D3 v1.3 契约（派发单 §5 测试要求） ====================

    /// 弱耦合基 L（T-8 SafeDomain 构造）: 快模 τ=0.5/0.8, 慢模 τ=10/20（Γ_s=0.075≥Γ_min）
    fn 弱耦合基L() -> DMatrix<NComplex<f64>> {
        let n = 4;
        let mut l = DMatrix::<NComplex<f64>>::zeros(n, n);
        l[(0, 0)] = NComplex::new(0.0, -2.0); // 快 τ=0.5
        l[(1, 1)] = NComplex::new(0.1, -1.25); // 快 τ=0.8
        l[(2, 2)] = NComplex::new(0.0, -0.1); // 慢 τ=10
        l[(3, 3)] = NComplex::new(0.2, -0.05); // 慢 τ=20
        l
    }

    /// 弱耦合带宽: 快界 1.0 / 慢界 5
    fn 弱耦合带宽() -> 三层带宽 {
        三层带宽 { 快模上界: 1.0, 慢模下界: 5.0, ..Default::default() }
    }

    /// T-8 契约字段填充 + 不变量①τ_FET≡τeff≡1/Γeff + 不变量②τ_S/(1-η_relax)≡τ_FET
    #[test]
    fn 测试_T8_契约字段与不变量() {
        let mut l = 弱耦合基L();
        // 弱耦合块（u/Γ_S≈0.34 <0.9 → SafeDomain）
        l[(2, 0)] = NComplex::new(0.01, 0.01);
        l[(0, 2)] = NComplex::new(0.01, 0.01);
        l[(3, 1)] = NComplex::new(0.008, 0.008);
        l[(1, 3)] = NComplex::new(0.008, 0.008);
        let 求解 = 非厄米谱求解器::default();
        let 分解 = 求解.分解(&l).unwrap();
        let 结果 = 计算熵泵(&l, &分解, &弱耦合带宽(), &[0.5, 0.5, 1.0, 0.5]).unwrap();
        assert_eq!(结果.域标志, DomainFlags::SafeDomain, "{}", 结果.说明);
        // 不变量①: τ_FET ≡ τeff ≡ 1/Γeff（同一赋值派生）
        assert!((结果.τ_FET - 结果.慢模有效寿命_τeff).abs() < 1e-9, "τ_FET≡τeff");
        assert!((结果.τ_FET - 1.0 / 结果.Γeff.max(1e-12)).abs() < 1e-9, "τ_FET≡1/Γeff");
        assert!((结果.Γeff - 结果.慢模有效耗散_Γeff).abs() < 1e-12, "Γeff≡Γ_eff");
        // 不变量②: τ_S/(1-η_relax) ≡ τ_FET（仅弛豫比成立）
        assert!(结果.弛豫比_ηrelax < 1.0, "η_relax<1: {}", 结果.弛豫比_ηrelax);
        let 推导 = 结果.慢模裸寿命_τS / (1.0 - 结果.弛豫比_ηrelax);
        assert!(
            (推导 - 结果.τ_FET).abs() / 结果.τ_FET.max(1e-9) < 1e-6,
            "τ_S/(1-η_relax)≡τ_FET: 推导={} τ_FET={}",
            推导, 结果.τ_FET
        );
        // SafeDomain: η/卡诺 有效（非 NaN）
        assert!(!结果.熵泵效率_ηpump.is_nan(), "SafeDomain η 有效");
        assert!(!结果.卡诺上限_契约.is_nan(), "SafeDomain 卡诺有效");
        assert!(结果.熵泵效率_ηpump <= 结果.卡诺上限_契约 + 1e-6, "η≤卡诺(域内)");
        assert!(结果.卡诺上限_契约 > 0.9, "τ_F≪τ_S → 卡诺≈1: {}", 结果.卡诺上限_契约);
        // 迹约定: 单窗默认
        assert_eq!(结果.迹约定, TraceConvention::SingleMode);
        // 规则态无 Φpump_np
        assert!(结果.非微扰流量_Φpump_np.is_none());
    }

    /// T-8 η_relax 恒有界 + 域判定单调: 扫描耦合强度 0.005→0.3
    /// 初态快模主导（[1,0.8,0.2,0.1]）: 避免慢模主导致 P_in 极小 → η_pump 人为偏大
    #[test]
    fn 测试_T8_ηrelax恒有界与三域() {
        let 求解 = 非厄米谱求解器::default();
        for 耦合 in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3] {
            let mut l = 弱耦合基L();
            for (i, j) in [(2, 0), (0, 2), (3, 1), (1, 3)] {
                l[(i, j)] = NComplex::new(耦合, 耦合);
            }
            let 分解 = 求解.分解(&l).unwrap();
            let 结果 = 计算熵泵(&l, &分解, &弱耦合带宽(), &[1.0, 0.8, 0.2, 0.1]).unwrap();
            // η_relax 恒有界性（D3 v1.3）: 规则态（微扰有效, 非域外非湍流）时恒<1——
            // Γ_eff=Γ_S(1-η_relax)>0 ⟺ η_relax<1; 域外/湍流态 η_pump 诚实降级 NaN
            if 结果.熵泵效率_ηpump.is_nan() {
                // 域外（η 发散）或湍流态（谱分离失效）: 不输出伪值
                if !结果.谱分离有效 {
                    assert!(结果.非微扰流量_Φpump_np.is_some(), "湍流态 Φpump_np 应填充: 耦合={}", 耦合);
                } else {
                    assert_eq!(结果.域标志, DomainFlags::OutsideDomain, "η NaN 且谱分离有效应域外: 耦合={}", 耦合);
                }
            } else {
                assert!(
                    结果.弛豫比_ηrelax < 1.0,
                    "规则态 η_relax<1: 耦合={} η_relax={}",
                    耦合, 结果.弛豫比_ηrelax
                );
                assert!(
                    结果.熵泵效率_ηpump <= 结果.卡诺上限_契约 + 1e-6,
                    "规则态 η≤卡诺: 耦合={} η={} η_max={}",
                    耦合, 结果.熵泵效率_ηpump, 结果.卡诺上限_契约
                );
            }
            // 弱耦合 Safe / 强耦合 Outside 单调
            if 耦合 <= 0.01 {
                assert_eq!(结果.域标志, DomainFlags::SafeDomain, "弱耦合 Safe: {}", 耦合);
            }
            if 耦合 >= 0.2 {
                assert_eq!(结果.域标志, DomainFlags::OutsideDomain, "强耦合 Outside: {}", 耦合);
            }
        }
    }

    /// T-8 非微扰双轨: 中等强耦合（快慢模仍在但谱分离失效）→ Φpump_np 填充 + η NaN 不输出伪值
    #[test]
    fn 测试_T8_非微扰双轨() {
        // 弱耦合基 L + 中等强耦合（0.4）: 快慢模仍在（非 Err）, 但 Γ_eff<0 → 谱分离失效
        let mut l = 弱耦合基L();
        for (i, j) in [(2, 0), (0, 2), (3, 1), (1, 3)] {
            l[(i, j)] = NComplex::new(0.4, 0.4);
        }
        let 求解 = 非厄米谱求解器::default();
        let 分解 = 求解.分解(&l).unwrap();
        let 结果 = 计算熵泵(&l, &分解, &弱耦合带宽(), &[1.0, 0.8, 0.2, 0.1]).unwrap();
        // 湍流态: η NaN（不输出伪值）, Φpump_np 由 γ(τ) 轨填充
        assert!(结果.熵泵效率_ηpump.is_nan(), "湍流态 η 不输出伪值: {}", 结果.熵泵效率_ηpump);
        assert!(结果.卡诺上限_契约.is_nan(), "湍流态卡诺 NaN");
        assert!(结果.非微扰流量_Φpump_np.is_some(), "Φpump_np 应填充");
        let np = 结果.非微扰流量_Φpump_np.unwrap();
        assert!(np.is_finite() && np >= 0.0, "Φpump_np 有限非负: {}", np);
        assert!(!结果.谱分离有效, "强耦合谱分离应失效");
        // 域标志: 强耦合 → OutsideDomain（η 发散区）
        assert_eq!(结果.域标志, DomainFlags::OutsideDomain, "强耦合应域外");
    }
}
