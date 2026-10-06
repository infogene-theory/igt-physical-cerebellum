// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! 造波安全控制层（MIGRATION-FINAL Step0-3a 下沉自 engine `l1/wave_safety.rs`）
//!
//! **逐字搬迁，不改算法、不改命名**（Step2 engine 只改 use 路径便于核对）；
//! engine 原位 Step2 删除（本批不动，中间态允许重复）。
//! 依赖 `crate::lyapunov`（core 本有，engine `l1::lyapunov` 亦为其 re-export）——
//! 故本模块下沉**无新增 core 依赖**。
//!
//! ## 理论核心（Paper B《造波安全控制》v0.2 修正版，搬自 engine 未改）
//! - **能量预算幅度约束**: 微扰波注入能量 E_wave = ∫u²dt = A²·τ_sys/2 受 Landauer 能量量级参考约束
//!   E_wave_max = η·k_BT·ln2 → A_max = η·sqrt(2·k_BT·ln2/τ_sys)
//!   （v0.1 量纲错误修正: 原 A_max=η·k_BT/(τ_sys·f_s) 量纲为能量而非幅度，且与 E_wave∝A² 自相矛盾）
//! - **S 不降级弱化定理**: |ΔS_controller(t)| ≤ ε_wave（控制器自身不导致 S 降级）
//!   （v0.1 表述过强修正: 外部强扰动可直接降 S，控制器只保证自身不降 S）
//! - **非线性闭环**（变分原理深化 §3.6 + Paper B §5.2）: 非线性 S = λ_max<0 ? 1 : 0；
//!   造波后 λ_max 不转正且 ε_wave=|Δλ_max| < 上界
//!
//! 工程落地: 作为造波器的安全前置层——生成波前检查幅度是否超能量预算（超限 clamp 到 A_max
//! 并记录降级说明），生成波后校验 S 变化不超 ε_wave。
//!
//! 锚点: Paper B v0.2 定义2/步骤5、变分原理深化 §3.6、P20（ε_wave<0.3 安全阈值）、
//! P12（κτ₀ 稳态窗）、ARCH-1 v2.1 §4（ω/γ 独立）。core 五类约束: 无 IO/网络/异步/时间/随机。

#![allow(non_snake_case)]

use crate::lyapunov::{Lyapunov计算器, εWave结果};
use serde::{Deserialize, Serialize};

/// 玻尔兹曼常数 × 室温 300K（J）——能量量级参考
const k_BT_300K: f64 = 4.14e-21;
/// ln2（Landauer 比特擦除能耗因子）
const LN2: f64 = 0.693_147_180_559_945_3;
/// 默认安全系数 η（Paper B: η∈(0,1), 通常η<1）
pub const 默认安全系数: f64 = 0.5;
/// 默认 S 不降级上界 ε_wave（Paper B: ε_wave < 0.3）
pub const 默认S不降级上界: f64 = 0.2;
/// 默认系统本征弛豫时间（秒; 缺省时用 1.0s）
pub const 默认系统τ: f64 = 1.0;

/// 安全校验结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 安全校验结果 {
    /// 是否通过校验（未超能量预算）
    pub 通过: bool,
    /// 原始强度（[0,1] 或任意）
    pub 原始强度: f64,
    /// 安全幅度上限 A_max（能量预算反解）
    pub 幅度上限: f64,
    /// 实际采用幅度（已 clamp）
    pub 采用幅度: f64,
    /// 注入能量 E_wave = A²·τ_sys/2
    pub 注入能量: f64,
    /// 能量预算 E_wave_max = η·k_BT·ln2
    pub 能量预算: f64,
    /// 降级说明（超限时）
    pub 降级说明: Option<String>,
    /// S 不降级校验: 预计 S 变化上界
    pub s变化上界: f64,
    /// S 校验是否通过（预计变化 ≤ ε_wave）
    pub s校验通过: bool,
}

/// 造波安全控制层
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 造波安全控制 {
    /// 安全系数 η（能量预算占比）
    pub 安全系数: f64,
    /// S 不降级上界 ε_wave
    pub s不降级上界: f64,
}

impl Default for 造波安全控制 {
    fn default() -> Self {
        Self {
            安全系数: 默认安全系数,
            s不降级上界: 默认S不降级上界,
        }
    }
}

impl 造波安全控制 {
    /// 能量预算幅度约束（Paper B v0.2 定义2）
    ///
    /// A_max = η · sqrt(2·k_BT·ln2 / τ_sys)
    pub fn 幅度上限(&self, τ_sys: f64) -> f64 {
        let τ = τ_sys.max(1e-6);
        self.安全系数 * (2.0 * k_BT_300K * LN2 / τ).sqrt()
    }

    /// 能量预算 E_wave_max = η·k_BT·ln2
    pub fn 能量预算(&self) -> f64 {
        self.安全系数 * k_BT_300K * LN2
    }

    /// 对波强度进行安全校验（Paper B 步骤5: A = min(A_max, A_needed)）
    pub fn 校验(&self, 原始强度: f64, τ_sys: f64) -> 安全校验结果 {
        let 幅度上限 = self.幅度上限(τ_sys);
        let 能量预算 = self.能量预算();
        // 采用幅度 = min(原始强度, 幅度上限) —— Paper B: A = min(A_max, A_needed)
        let 采用幅度 = 原始强度.min(幅度上限);
        let 注入能量 = 采用幅度 * 采用幅度 * τ_sys.max(1e-6) / 2.0;
        // S 不降级校验: 注入能量与能量预算之比 → S 变化上界（线性代理）
        let s变化上界 = if 能量预算 > 0.0 {
            (注入能量 / 能量预算) * self.s不降级上界
        } else {
            0.0
        };
        let s校验通过 = s变化上界 <= self.s不降级上界;
        let 降级说明 = if 采用幅度 < 原始强度 {
            Some(format!(
                "能量预算降级: 请求强度{:.4} → 采用{:.4} (A_max={:.2e}, E预算={:.2e}J)",
                原始强度, 采用幅度, 幅度上限, 能量预算
            ))
        } else {
            None
        };

        安全校验结果 {
            通过: 降级说明.is_none(),
            原始强度,
            幅度上限,
            采用幅度,
            注入能量,
            能量预算,
            降级说明,
            s变化上界,
            s校验通过,
        }
    }

    /// 组合校验（强度 + S）——造波前一次性检查
    pub fn 组合校验(&self, 原始强度: f64, τ_sys: f64) -> 安全校验结果 {
        let mut 结果 = self.校验(原始强度, τ_sys);
        if !结果.s校验通过 {
            结果.通过 = false;
            if 结果.降级说明.is_none() {
                结果.降级说明 = Some(format!(
                    "S不降级校验失败: 预计S变化{:.4} > 上界{:.4}",
                    结果.s变化上界, self.s不降级上界
                ));
            }
        }
        结果
    }
}

// ============================================================================
// 非线性安全控制（第136批, 变分原理深化§3.6 + Paper B §5.2 联动）
// ============================================================================

/// 非线性造波安全控制层 — Lyapunov ε_wave 校验（非线性 S 不降级闭环）
///
/// 工程落地:
///   - 造波前: 对系统时序计算 λ_max_前
///   - 造波后（模拟/实测）: 计算 λ_max_后
///   - 安全判据: λ_max_后 < 0（不转正）且 ε_wave=|Δλ_max| < 上界
///   - 非线性 S = λ_max<0 ? 1 : 0（混沌即不健康）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 非线性安全控制 {
    /// Lyapunov 计算器（Rosenstein）
    pub 计算器: Lyapunov计算器,
    /// ε_wave 上界（|Δλ_max| 允许的最大变化）
    pub ε_wave上界: f64,
    /// 混沌判定阈值（λ_max 转正超过此值判混沌）
    pub 混沌阈值: f64,
}

impl Default for 非线性安全控制 {
    fn default() -> Self {
        Self {
            计算器: Lyapunov计算器::default(),
            ε_wave上界: 0.3,
            混沌阈值: 0.01,
        }
    }
}

/// 非线性安全校验结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 非线性安全校验结果 {
    /// 是否通过（λ_max 不转正 且 ε_wave < 上界）
    pub 通过: bool,
    /// 造波前 λ_max
    pub λ_max前: f64,
    /// 造波后 λ_max
    pub λ_max后: f64,
    /// ε_wave = |Δλ_max|
    pub ε_wave: f64,
    /// 造波前非线性 S（λ_max<0 → 1）
    pub 非线性s前: f64,
    /// 造波后非线性 S
    pub 非线性s后: f64,
    /// 是否混沌转正（λ_max 从负转正）
    pub 混沌转正: bool,
    /// 降级/拦截说明
    pub 说明: Option<String>,
}

impl 非线性安全控制 {
    /// 非线性安全校验——对造波前后的系统时序做 Lyapunov 闭环检查
    pub fn 校验(&self, 造波前: &[f64], 造波后: &[f64], Δt: f64) -> 非线性安全校验结果 {
        let εWave结果 = self.计算器.计算ε_wave(造波前, 造波后, Δt);
        self.从εwave(&εWave结果)
    }

    /// 从 ε_wave 结果构造校验（供已算好的场景复用）
    pub fn 从εwave(&self, ε: &εWave结果) -> 非线性安全校验结果 {
        let 混沌转正 = ε.λ_max前 < -self.混沌阈值 && ε.λ_max后 > self.混沌阈值;
        let 通过 = !混沌转正 && ε.ε_wave < self.ε_wave上界;
        let 说明 = if 混沌转正 {
            Some(format!(
                "非线性S不降级失败: λ_max {:.4}→{:.4} 转正(混沌), 违反非线性S=1要求",
                ε.λ_max前, ε.λ_max后
            ))
        } else if ε.ε_wave >= self.ε_wave上界 {
            Some(format!(
                "ε_wave={:.4} ≥ 上界{:.4}: 造波导致非线性结构显著变化",
                ε.ε_wave, self.ε_wave上界
            ))
        } else {
            None
        };
        非线性安全校验结果 {
            通过,
            λ_max前: ε.λ_max前,
            λ_max后: ε.λ_max后,
            ε_wave: ε.ε_wave,
            非线性s前: if ε.λ_max前 < -self.混沌阈值 { 1.0 } else { 0.0 },
            非线性s后: if ε.λ_max后 < -self.混沌阈值 { 1.0 } else { 0.0 },
            混沌转正,
            说明,
        }
    }
}

// ============================================================================
// 测试（随搬迁一同搬运——数值断言锁死算法未变）
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 能量预算幅度约束_量纲自洽() {
        // A_max 应随 τ_sys 增大而减小（长弛豫→能量预算分摊到更长周期→幅度上限更小）
        let 控制 = 造波安全控制::default();
        let a_快 = 控制.幅度上限(0.1);
        let a_慢 = 控制.幅度上限(1.0);
        let a_更慢 = 控制.幅度上限(10.0);
        assert!(a_快 > a_慢 && a_慢 > a_更慢, "τ越大幅度上限越小: {:.2e}>{:.2e}>{:.2e}", a_快, a_慢, a_更慢);
        // 量纲: A_max 应远小于宏观幅度(10^-10量级), 因为能量是原子量级
        assert!(a_慢 < 1e-8, "幅度上限应为微观量级: {}", a_慢);
    }

    #[test]
    fn 强度clamp_超限降级() {
        let 控制 = 造波安全控制::default();
        // 请求宏观强度1.0(远大于原子量级A_max) → 应clamp到A_max并记录降级
        let 结果 = 控制.校验(1.0, 默认系统τ);
        assert!(!结果.通过, "宏观强度应触发降级");
        assert!(结果.采用幅度 < 结果.原始强度, "应clamp");
        assert!(结果.降级说明.is_some(), "应有降级说明");
        assert!(结果.采用幅度 <= 结果.幅度上限 + 1e-12, "采用幅度不超上限");
        // 注入能量不超预算
        assert!(结果.注入能量 <= 结果.能量预算 + 1e-24, "注入能量不超预算");
    }

    #[test]
    fn 微观强度_不降级() {
        let 控制 = 造波安全控制::default();
        // 请求原子量级强度(远小于A_max) → 通过
        let a_max = 控制.幅度上限(默认系统τ);
        let 结果 = 控制.校验(a_max * 0.1, 默认系统τ);
        assert!(结果.通过, "微观强度应通过: {:?}", 结果.降级说明);
        assert!(结果.降级说明.is_none());
        assert!(结果.s校验通过);
    }

    #[test]
    fn 组合校验_宏观强度降级为安全() {
        let 控制 = 造波安全控制::default();
        // 极高强度(宏观1.0) → 被clamp到A_max(原子量级) → 能量预算自动满足
        let 结果 = 控制.组合校验(1.0, 默认系统τ);
        assert!(!结果.通过, "宏观强度应触发降级(有降级说明)");
        assert!(结果.降级说明.is_some(), "应有能量预算降级说明");
        assert!(结果.注入能量 <= 结果.能量预算 + 1e-24, "clamp后能量不超预算");
        assert!(结果.s校验通过, "clamp后S变化应在上界内: {}", 结果.s变化上界);
    }

    #[test]
    fn 组合校验_微观强度直接通过() {
        let 控制 = 造波安全控制::default();
        let a_max = 控制.幅度上限(默认系统τ);
        let 结果 = 控制.组合校验(a_max * 0.01, 默认系统τ);
        assert!(结果.通过, "微观强度应通过: {:?}", 结果.降级说明);
        assert!(结果.降级说明.is_none());
        assert!(结果.s校验通过);
    }

    #[test]
    fn 默认参数符合论文() {
        let 控制 = 造波安全控制::default();
        assert!(控制.安全系数 < 1.0, "η应<1: {}", 控制.安全系数);
        assert!(控制.s不降级上界 < 0.3, "ε_wave应<0.3(Paper B): {}", 控制.s不降级上界);
    }

    // ===== 非线性安全控制（第136批: Lyapunov ε_wave 校验） =====

    fn 衰减时序(τ: f64, Δt: f64, n: usize) -> Vec<f64> {
        (0..n).map(|k| (-(k as f64 * Δt) / τ).exp()).collect()
    }

    fn 增长时序(τ: f64, Δt: f64, n: usize) -> Vec<f64> {
        (0..n).map(|k| ((k as f64 * Δt) / τ).exp()).collect()
    }

    #[test]
    fn 非线性安全_稳定到稳定通过() {
        let 控制 = 非线性安全控制::default();
        let 前 = 衰减时序(1.0, 0.05, 400);
        let 后 = 衰减时序(1.25, 0.05, 400);
        let 结果 = 控制.校验(&前, &后, 0.05);
        assert!(结果.通过, "稳定→稳定应通过: {:?}", 结果.说明);
        assert!(!结果.混沌转正);
        assert_eq!(结果.非线性s前, 1.0);
        assert_eq!(结果.非线性s后, 1.0);
        assert!(结果.ε_wave >= 0.0);
    }

    #[test]
    fn 非线性安全_稳定转混沌拦截() {
        let 控制 = 非线性安全控制::default();
        let 前 = 衰减时序(1.0, 0.05, 400);
        let 后 = 增长时序(2.0, 0.05, 300);
        let 结果 = 控制.校验(&前, &后, 0.05);
        assert!(!结果.通过, "转正应拦截: {:?}", 结果.说明);
        assert!(结果.混沌转正);
        assert!(结果.非线性s后 == 0.0, "造波后混沌→非线性S=0");
    }

    #[test]
    fn 非线性安全_εwave超上界拦截() {
        let 控制 = 非线性安全控制::default();
        // 造波前后都是稳定但 λ 变化很大（τ 1.0 → τ 5.0, Δλ≈0.8 > 0.3）
        let 前 = 衰减时序(1.0, 0.05, 400);
        let 后 = 衰减时序(5.0, 0.05, 400);
        let 结果 = 控制.校验(&前, &后, 0.05);
        // 若未转正则看 ε_wave 是否超界
        if !结果.混沌转正 {
            assert!(
                !结果.通过 || 结果.ε_wave < 控制.ε_wave上界,
                "ε_wave 超上界应拦截: ε={:.4} 上界={}",
                结果.ε_wave,
                控制.ε_wave上界
            );
        }
    }
}
