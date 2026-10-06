// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! 三层耗散联通引擎 — 第282批 P22 v10.1 §2.6
//!
//! 理论依据:
//!   - 《P22 IGT 全息系统：τ工程化架构与接口》§2.6 三层耗散联通机制:
//!     core/agent/engine 不是三个独立系统, 是**同一非厄米动力学图在三个
//!     带宽下的投影**, 通过谱分离(P11)粗粒化联通:
//!       - core 快模: τ~毫秒-秒, EDMD/对角化, 积分成 agent 有效参数
//!       - agent 中速模: τ~秒-小时, 监测/预警/造波/验证, 用户直接感知
//!       - engine 慢模: τ~天-年, 基因蒸馏/共享/自由能配置, agent 视为
//!         "冻结背景"(P0 带宽切分)
//!     三层通过自能 Σ(ω) 耦合: engine 共享基因改变 agent 的 L_eff 参数;
//!     agent 造波数据回流 engine 改变共享基因库。
//!   - 数据流闭环: 传感器→agent感知→core(EDMD→L→谱分离→逆元)→agent决策
//!     →输出→新数据回流core验证(c_n(t) 是否按 e^{-t/τ_n} 回归)。
//!   - 设计铁律: core 永远不知道 engine 存在——本模块 100% 纯计算,
//!     输入时间序列/矩阵, 输出谱结果, 零网络零领域知识。
//!
//! 核心定位: §2.6 最小可验证闭环的三块拼图:
//!   ① 三层带宽投影: 同一 L 的本征值按 τ 归层, 各层 L_eff + 谱隙
//!   ② 跨层自能耦合: engine 慢模背景对 agent 中速层 L_eff 的 Σ(ω) 修正
//!   ③ 造波回归验证: 模态展开系数 c_n(t) 按 e^{-t/τ_n} 指数衰减拟合
//!     （数据流"回流 core 验证"的第 1 块）

#![allow(non_snake_case)]

use nalgebra::Complex as NComplex;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use super::nonhermitian_spectrum::{非厄米模, 非厄米谱分解, 非厄米谱求解器};

// ============================================================================
// 三层带宽（P0 带宽切分）
// ============================================================================

/// 三层带宽 — 同一动力学图按 τ 切分为三层投影（D2 2026-08-23 观测者参数化）
///
/// 零破坏兼容: 新增字段均 `#[serde(default)]`, Default 保持旧值（1.0/3600）;
/// 旧 JSON 反序列化不失败。
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct 三层带宽 {
    /// core 快模上界 τ_fast（τ<此值 → core 快模, 毫秒-秒级）
    pub 快模上界: f64,
    /// engine 慢模下界 τ_slow（τ>此值 → engine 慢模冻结背景, 天-年级）
    pub 慢模下界: f64,
    /// D2: 观测者等级（O0-O4; None=无观测者上下文）
    #[serde(default)]
    pub 观测者等级: Option<u8>,
    /// D2: 观测者特征时间 τ_obs（由 from_observer 填充）
    #[serde(default)]
    pub 观测者τ_obs: Option<f64>,
    /// D2: 窗函数参数 σ_O（v2 优化项, v1.0 不启用）
    #[serde(default)]
    pub 窗参数σ_O: Option<f64>,
}

impl Default for 三层带宽 {
    fn default() -> Self {
        // P22 §2.6 工程部署: core 快模 秒级 / agent 中速 秒-小时 / engine 慢模 天+
        //（无观测者上下文时的硬编码回退, 现有行为不变, 零破坏）
        Self { 快模上界: 1.0, 慢模下界: 3600.0, 观测者等级: None, 观测者τ_obs: None, 窗参数σ_O: None }
    }
}

impl 三层带宽 {
    /// 快模上界须 < 慢模下界（带宽有序）
    pub fn 合法(&self) -> bool {
        self.快模上界 > 0.0 && self.慢模下界 > self.快模上界
    }

    /// D2（P0 测量相对性原理落地）: 由观测者 τ_obs 推导带宽
    ///   τ_fast = τ_obs（观测者自身时间尺度以下为 core 快模）
    ///   τ_slow = 100·τ_obs（百倍 τ_obs 以上为 engine 慢模冻结背景）
    /// 三层比例约 1:100:1000（与 P0 的 0.1~1000 带宽对齐）。
    pub fn from_observer(τ_obs: f64, 等级: u8) -> Self {
        Self {
            快模上界: τ_obs,
            慢模下界: 100.0 * τ_obs,
            观测者等级: Some(等级),
            观测者τ_obs: Some(τ_obs),
            窗参数σ_O: None,
        }
    }

    /// D2: 观测者分辨范围（P0）: τ_min=0.1τ_obs, τ_max=1000τ_obs
    pub fn 观测者范围(&self) -> Option<(f64, f64)> {
        let τ_obs = self.观测者τ_obs?;
        Some((0.1 * τ_obs, 1000.0 * τ_obs))
    }

    /// 谱自适应带宽（第402批, D7 优化项）: 基于复L 谱分解模 τ 分布确定快/慢界——
    /// 快界 = 最快模×1.5, 慢界 = 快界×3（谱隙约 3 倍）。固定 100·τ_obs（from_observer）
    /// 对尺度差 <100 的信号（如快慢 6 倍）慢模不归组 → 熵泵谱分离失效 → V8 回退。
    pub fn 谱自适应(模τ: &[f64]) -> Self {
        if 模τ.len() < 2 {
            return Self::default();
        }
        let 最小 = 模τ.iter().fold(f64::MAX, |a, b| a.min(*b));
        let 最大 = 模τ.iter().fold(0.0f64, |a, b| a.max(*b));
        if 最小 <= 0.0 || 最大 <= 最小 {
            return Self::default();
        }
        let 快界 = 最小 * 1.5;
        Self {
            快模上界: 快界,
            慢模下界: 快界 * 3.0, // 谱隙约 3 倍（快模 < 快界, 慢模 > 3·快界）
            ..Default::default()
        }
    }
}

/// D2: 观测者上下文 — core 谱计算的统一入参（带宽由 from_observer 推导）
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct 观测者上下文 {
    /// 观测者等级 O0-O4（O5 预留）
    pub 等级: u8,
    /// 观测者特征时间
    pub 观测者τ_obs: f64,
    /// 由 from_observer 推导的带宽
    pub 带宽: 三层带宽,
    /// 窗函数参数（v2 优化项, v1.0 不启用）
    #[serde(default)]
    pub 窗参数σ_O: Option<f64>,
}

impl 观测者上下文 {
    /// 新建观测者上下文（带宽由 τ_obs 推导）
    pub fn 新建(等级: u8, τ_obs: f64) -> Self {
        Self {
            等级,
            观测者τ_obs: τ_obs,
            带宽: 三层带宽::from_observer(τ_obs, 等级),
            窗参数σ_O: None,
        }
    }
}

// ============================================================================
// 层投影
// ============================================================================

/// 单个带宽层的投影 — 该层子空间的有效算子 + 谱摘要
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 层投影 {
    /// 层名: core快模 / agent中速 / engine慢模
    pub 层名: String,
    /// 该层模索引（在分解模列表中的位置）
    pub 模索引: Vec<usize>,
    /// 该层本征值 (ω, λ_im)——λ_im=Im λ（统一令 §5：本征值权威表示；γ=−λ_im 耗散率）
    pub 本征值: Vec<(f64, f64)>,
    /// 该层有效算子 L_eff（该层子空间投影, 原基）
    pub l_eff: Vec<Vec<f64>>,
    /// 该层平均 τ（秒）
    pub 平均τ: f64,
    /// 该层谱隙（层内最大速率-最小速率, 表征带宽）
    pub 谱隙: f64,
    /// 冻结背景（engine 慢模层=true, 视为冻结参数不随快动力学演化）
    pub 冻结背景: bool,
}

// ============================================================================
// 三层投影结果
// ============================================================================

/// 三层投影结果 — 同一动力学图在三个带宽下的投影
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 三层投影结果 {
    /// 带宽切分
    pub 带宽: 三层带宽,
    /// core 快模层（积分成 agent 有效参数）
    pub core层: 层投影,
    /// agent 中速层（监测/预警/造波, 用户直接感知）
    pub agent层: 层投影,
    /// engine 慢模层（冻结背景, 基因/自由能配置）
    pub engine层: 层投影,
    /// 总模数（谱重归一化守恒验证: 三层模数和 = 总模数）
    pub 总模数: usize,
    /// 守恒: core.模数 + agent.模数 + engine.模数 == 总模数
    pub 守恒: bool,
    /// 说明
    pub 说明: String,
}

// ============================================================================
// 跨层自能结果
// ============================================================================

/// 跨层自能 Σ(ω) — engine 慢模背景对 agent 中速层的修正
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 跨层自能结果 {
    /// 计算频率 ω（rad/s）
    pub ω: f64,
    /// 自能 Σ(ω)（原基, 复矩阵）
    pub 自能: Vec<Vec<f64>>,
    /// 修正后 Γ_eff = -Im Σ(ω) 对角
    pub Γ_eff对角: Vec<f64>,
    /// 因果性通过（Γ_eff 全 >0 = 慢模积分有效, 冻结背景成立）
    pub 因果性通过: bool,
    /// 说明
    pub 说明: String,
}

// ============================================================================
// 造波回归验证结果
// ============================================================================

/// 单个模的回归验证
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 模回归 {
    /// 模索引
    pub 模索引: usize,
    /// 理论 τ_n = 1/|γ_n|（秒）
    pub 理论τ: f64,
    /// 拟合 τ（从 ln|c(t)| 斜率, 秒）
    pub 拟合τ: f64,
    /// 相对误差 |拟合-理论|/理论
    pub 相对误差: f64,
    /// 通过（相对误差 < 容差）
    pub 通过: bool,
    /// 判定
    pub 判定: String,
}

/// 造波回归验证结果 — c_n(t) 是否按 e^{-t/τ_n} 回归
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 回归验证结果 {
    /// 每模验证
    pub 每模: Vec<模回归>,
    /// 总体通过（全部模通过）
    pub 总体通过: bool,
    /// 容差（默认 0.15）
    pub 容差: f64,
    /// 说明
    pub 说明: String,
}

// ============================================================================
// 三层联通引擎
// ============================================================================

/// 三层耗散联通引擎 — §2.6 最小可验证闭环
#[derive(Debug, Clone, Copy, Default)]
pub struct 三层联通引擎 {
    /// 造波回归验证容差（相对误差, 默认 0.15）
    pub 回归容差: f64,
}

impl 三层联通引擎 {
    /// 新建（回归容差 0.15）
    pub fn 新建() -> Self {
        Self { 回归容差: 0.15 }
    }

    /// ① 三层带宽投影 — 同一 L 的本征值按 τ 归层
    ///
    /// - core 层: τ < 快模上界（快模, 积分成 agent 有效参数）
    /// - agent 层: 快模上界 ≤ τ ≤ 慢模下界（中速模, 监测/造波）
    /// - engine 层: τ > 慢模下界 或 τ 非有限（慢模冻结背景）
    /// 每层 L_eff = V (P_层 L_基 P_层) V^{-1}（该层子空间投影, 原基）。
    /// 守恒: 三层模数和 = 总模数（谱重归一化守恒验证防切分错误）。
    pub fn 三层投影(
        &self,
        l: &DMatrix<NComplex<f64>>,
        带宽: 三层带宽,
    ) -> Result<三层投影结果, String> {
        if !带宽.合法() {
            return Err(format!(
                "带宽须 0 < 快模上界({}) < 慢模下界({})",
                带宽.快模上界, 带宽.慢模下界
            ));
        }
        let 求解器 = 非厄米谱求解器::default();
        let 分解 = 求解器.分解(l)?;
        let n = l.nrows();
        let m = 分解.模数;

        // 归层（分解已按 τ 降序: 慢模在前, τ=INF 在最前）
        let mut core_idx = Vec::new();
        let mut agent_idx = Vec::new();
        let mut engine_idx = Vec::new();
        for (i, 模) in 分解.模.iter().enumerate() {
            if 模.τ.is_finite() {
                if 模.τ < 带宽.快模上界 {
                    core_idx.push(i);
                } else if 模.τ <= 带宽.慢模下界 {
                    agent_idx.push(i);
                } else {
                    engine_idx.push(i);
                }
            } else {
                // τ=INF（虚部≈0）= 冻结背景 → engine 慢模
                engine_idx.push(i);
            }
        }

        // 模基算子 L_基 = V^{-1} L V（近似对角）
        let v矩阵 = Self::右向量矩阵(&分解, n);
        let v逆 = match v矩阵.clone().try_inverse() {
            Some(inv) => inv,
            None => DMatrix::<NComplex<f64>>::zeros(n, m),
        };
        let l基 = &v逆 * l * &v矩阵;

        // 各层投影算子（模基）+ 层内 L_eff（原基）
        let 层_eff = |idx: &[usize]| -> (Vec<Vec<f64>>, f64, f64) {
            let mut p = DMatrix::<NComplex<f64>>::zeros(m, m);
            for &i in idx {
                p[(i, i)] = NComplex::new(1.0, 0.0);
            }
            let eff基 = &p * &l基 * &p;
            let eff原 = &v矩阵 * &eff基 * &v逆;
            let 矩阵: Vec<Vec<f64>> = (0..n)
                .map(|i| (0..n).map(|j| eff原[(i, j)].re).collect())
                .collect();
            // 层内 τ 列表 → 平均τ + 谱隙（最大速率-最小速率）
            let τ列表: Vec<f64> = idx.iter().filter_map(|&i| {
                let t = 分解.模[i].τ;
                if t.is_finite() { Some(t) } else { None }
            }).collect();
            let 平均τ = if τ列表.is_empty() { 0.0 } else {
                τ列表.iter().sum::<f64>() / τ列表.len() as f64
            };
            let 谱隙 = if τ列表.len() >= 2 {
                let 快_最大率 = τ列表.iter().map(|t| 1.0 / t.max(1e-9)).fold(0.0, f64::max);
                let 慢_最小率 = τ列表.iter().map(|t| 1.0 / t.max(1e-9)).fold(f64::INFINITY, f64::min);
                快_最大率 - 慢_最小率
            } else {
                0.0
            };
            (矩阵, 平均τ, 谱隙)
        };

        let (core_eff, core_τ, core_隙) = 层_eff(&core_idx);
        let (agent_eff, agent_τ, agent_隙) = 层_eff(&agent_idx);
        let (engine_eff, engine_τ, engine_隙) = 层_eff(&engine_idx);

        let 本征值_取 = |idx: &[usize]| -> Vec<(f64, f64)> {
            // 统一令 §5：本征值权威表示 (ω, λ_im)；γ=−λ_im 仅在物理判据/对外报告处使用
            idx.iter().map(|&i| (分解.模[i].ω, 分解.模[i].λ_im)).collect()
        };

        let core层 = 层投影 {
            层名: "core快模".to_string(),
            模索引: core_idx.clone(),
            本征值: 本征值_取(&core_idx),
            l_eff: core_eff,
            平均τ: core_τ,
            谱隙: core_隙,
            冻结背景: false,
        };
        let agent层 = 层投影 {
            层名: "agent中速".to_string(),
            模索引: agent_idx.clone(),
            本征值: 本征值_取(&agent_idx),
            l_eff: agent_eff,
            平均τ: agent_τ,
            谱隙: agent_隙,
            冻结背景: false,
        };
        let engine层 = 层投影 {
            层名: "engine慢模".to_string(),
            模索引: engine_idx.clone(),
            本征值: 本征值_取(&engine_idx),
            l_eff: engine_eff,
            平均τ: engine_τ,
            谱隙: engine_隙,
            冻结背景: true,
        };

        let 守恒 = core_idx.len() + agent_idx.len() + engine_idx.len() == m;
        let 说明 = format!(
            "三层带宽投影: core快模{}个(τ<{}s, 积分成agent参数) / agent中速{}个({}≤τ≤{}s) / engine慢模{}个(τ>{}s冻结背景); 守恒={}",
            core_idx.len(), 带宽.快模上界,
            agent_idx.len(), 带宽.快模上界, 带宽.慢模下界,
            engine_idx.len(), 带宽.慢模下界, 守恒
        );

        Ok(三层投影结果 {
            带宽,
            core层,
            agent层,
            engine层,
            总模数: m,
            守恒,
            说明,
        })
    }

    /// ② 跨层自能 — engine 慢模背景对 agent 中速层的 Σ(ω) 修正
    ///
    /// Σ = V (P_agent L_基 P_engine (ω - P_engine L_基 P_engine)^{-1}
    ///      P_engine L_基 P_agent) V^{-1}
    /// Γ_eff = -Im Σ(ω) 对角全 >0 = 因果性通过（慢模积分有效, 冻结背景成立）。
    pub fn 跨层自能(
        &self,
        l: &DMatrix<NComplex<f64>>,
        agent_idx: &[usize],
        engine_idx: &[usize],
        ω: f64,
    ) -> Result<跨层自能结果, String> {
        let 求解器 = 非厄米谱求解器::default();
        let 分解 = 求解器.分解(l)?;
        let n = l.nrows();
        let m = 分解.模数;
        let v矩阵 = Self::右向量矩阵(&分解, n);
        let v逆 = match v矩阵.clone().try_inverse() {
            Some(inv) => inv,
            None => DMatrix::<NComplex<f64>>::zeros(n, m),
        };
        let l基 = &v逆 * l * &v矩阵;
        let 投影 = |idx: &[usize]| -> DMatrix<NComplex<f64>> {
            let mut p = DMatrix::<NComplex<f64>>::zeros(m, m);
            for &i in idx {
                p[(i, i)] = NComplex::new(1.0, 0.0);
            }
            p
        };
        let pa = 投影(agent_idx);
        let pe = 投影(engine_idx);
        // Σ_基 = P_A L_基 P_E (ω - P_E L_基 P_E)^{-1} P_E L_基 P_A
        let lp_s = &pa * &l基;      // P_A L_基
        let l_pf = &l基 * &pe;      // L_基 P_E
        let pf_l_pf = &pe * &l基 * &pe;
        let 中括号 = &pf_l_pf - &DMatrix::<NComplex<f64>>::from_element(m, m, NComplex::new(ω, 0.0));
        let 中括号逆 = match 中括号.clone().try_inverse() {
            Some(inv) => inv,
            None => DMatrix::<NComplex<f64>>::zeros(m, m),
        };
        let 自能基 = &lp_s * &中括号逆 * &l_pf;
        // Σ_原 = V Σ_基 V^{-1}
        let 自能原 = &v矩阵 * &自能基 * &v逆;
        let 自能: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| 自能原[(i, j)].re).collect())
            .collect();
        let Γ_eff对角: Vec<f64> = (0..n).map(|i| -自能原[(i, i)].im).collect();
        // 因果性: Γ_eff ≥ 0 通过——0=无耦合可精确分离（Σ=0 无修正, 正确）;
        // >0=弱耦合积分有效; <0=因果性破坏（冻结背景不成立）
        let 因果性通过 = !Γ_eff对角.is_empty() && Γ_eff对角.iter().all(|g| *g >= 0.0);
        let 说明 = if 因果性通过 {
            let 非零修正 = Γ_eff对角.iter().filter(|g| **g > 1e-12).count();
            format!(
                "engine慢模背景积分有效（因果性通过, Γ_eff≥0, {}个非零修正）——慢模冻结为agent的L_eff修正, Σ(ω={:.2})",
                非零修正, ω
            )
        } else {
            format!(
                "engine慢模背景积分后 Γ_eff 为负（因果性破坏——冻结背景不成立, 需增大 慢模下界 或 增加慢模样本）"
            )
        };
        Ok(跨层自能结果 { ω, 自能, Γ_eff对角, 因果性通过, 说明 })
    }

    /// ③ 造波回归验证 — c_n(t) 是否按 e^{-t/τ_n} 指数回归
    ///
    /// 对每个模拟合 ln|c(t)| 线性斜率 → τ拟合 = -1/斜率; 与理论 τ_n=1/|γ_n|
    /// 比较, 相对误差 < 容差(默认 0.15) → 通过。数据流"回流 core 验证"闭环。
    /// 输入: 模态系数序列（每模一个时序, 可取 |c_n(t)| 或 c_n(t)）。
    pub fn 造波回归验证(
        &self,
        模态系数: &[Vec<f64>],
        理论τs: &[f64],
        容差: Option<f64>,
    ) -> Result<回归验证结果, String> {
        let 容差 = 容差.unwrap_or(self.回归容差);
        if 模态系数.is_empty() || 模态系数.len() != 理论τs.len() {
            return Err(format!(
                "模态系数数({})与理论τ数({})不匹配或为空",
                模态系数.len(),
                理论τs.len()
            ));
        }
        let mut 每模 = Vec::new();
        for (i, (序列, 理论τ)) in 模态系数.iter().zip(理论τs.iter()).enumerate() {
            // ln|c(t)| 线性回归（t 索引作时间, Δt=1 归一化——斜率量纲需与 τ 一致）
            let 幅值: Vec<f64> = 序列.iter().map(|c| c.abs()).collect();
            let 有效: Vec<f64> = 幅值.iter().filter(|a| **a > 1e-12).cloned().collect();
            let (拟合τ, 判定, 通过) = if 有效.len() < 2 {
                (0.0, "样本不足(需≥2个非零幅值), 无法验证".to_string(), false)
            } else {
                // 线性回归 y=ln|a| vs x=样本索引
                let n = 有效.len() as f64;
                let x̄ = (0..有效.len()).map(|x| x as f64).sum::<f64>() / n;
                let ȳ = 有效.iter().map(|a| a.ln()).sum::<f64>() / n;
                let mut 分子 = 0.0;
                let mut 分母 = 0.0;
                for (k, a) in 有效.iter().enumerate() {
                    let x = k as f64;
                    分子 += (x - x̄) * (a.ln() - ȳ);
                    分母 += (x - x̄) * (x - x̄);
                }
                if 分母.abs() < 1e-12 {
                    (0.0, "时序无变化(斜率为0), 无法验证".to_string(), false)
                } else {
                    let 斜率 = 分子 / 分母;
                    if 斜率 >= 0.0 {
                        // 非衰减（指数增长或平台）→ 非 e^{-t/τ} 回归
                        (0.0, format!("非指数回归(斜率{:.3}≥0, 应衰减)", 斜率), false)
                    } else {
                        let τ拟合 = -1.0 / 斜率;
                        let 相对误差 = (*理论τ).max(1e-9).min(1e9);
                        let 误差 = (τ拟合 - *理论τ).abs() / 相对误差;
                        let 通过 = 误差 < 容差;
                        let 判定 = if 通过 {
                            format!(
                                "按 e^(-t/τ) 回归: 拟合τ={:.2}s vs 理论τ={:.2}s, 误差={:.3}<容差{:.2}",
                                τ拟合, 理论τ, 误差, 容差
                            )
                        } else {
                            format!(
                                "偏离理论: 拟合τ={:.2}s vs 理论τ={:.2}s, 误差={:.3}≥容差{:.2}",
                                τ拟合, 理论τ, 误差, 容差
                            )
                        };
                        (τ拟合, 判定, 通过)
                    }
                }
            };
            每模.push(模回归 {
                模索引: i,
                理论τ: *理论τ,
                拟合τ,
                相对误差: if 拟合τ > 0.0 { (拟合τ - *理论τ).abs() / (*理论τ).max(1e-9) } else { f64::INFINITY },
                通过,
                判定,
            });
        }
        let 总体通过 = !每模.is_empty() && 每模.iter().all(|m| m.通过);
        let 说明 = if 总体通过 {
            format!("全部 {} 个模按 e^(-t/τ_n) 回归——造波后新数据回流验证通过", 每模.len())
        } else {
            format!(
                "{}/{} 个模通过回归验证（失败的模偏离理论衰减——可能数据噪声/模态混叠/造波后系统偏离）",
                每模.iter().filter(|m| m.通过).count(),
                每模.len()
            )
        };
        Ok(回归验证结果 { 每模, 总体通过, 容差, 说明 })
    }

    /// 便捷: 全流程 — 三层投影 + 跨层自能(默认 ω=0.1) + 造波回归验证
    pub fn 联通(
        &self,
        l: &DMatrix<NComplex<f64>>,
        带宽: 三层带宽,
        模态系数: Option<&[Vec<f64>]>,
        理论τs: Option<&[f64]>,
    ) -> Result<联通输出, String> {
        let 投影 = self.三层投影(l, 带宽)?;
        // 跨层自能: agent 为 S, engine 为 F
        let 跨层自能 = if 投影.agent层.模索引.is_empty() || 投影.engine层.模索引.is_empty() {
            None
        } else {
            match self.跨层自能(l, &投影.agent层.模索引, &投影.engine层.模索引, 0.1) {
                Ok(结果) => Some(结果),
                Err(_) => None,
            }
        };
        // 造波回归验证（理论 τ = engine+agent 层的 |1/γ|, 用输入理论τs 或取 投影 agent 层平均）
        let 回归验证 = match (模态系数, 理论τs) {
            (Some(系数), Some(τs)) => match self.造波回归验证(系数, τs, None) {
                Ok(结果) => Some(结果),
                Err(_) => None,
            },
            _ => None,
        };
        let 说明 = format!(
            "{}; 跨层自能: {}; 造波回归验证: {}",
            投影.说明,
            跨层自能.as_ref().map(|s| s.说明.clone()).unwrap_or_else(|| "无(agent或engine层为空)".to_string()),
            回归验证.as_ref().map(|r| r.说明.clone()).unwrap_or_else(|| "无(未提供模态系数)".to_string())
        );
        Ok(联通输出 {
            投影,
            跨层自能,
            回归验证,
            说明,
        })
    }

    /// 右向量矩阵 V（列=右向量, 从分解模的 v 字段重建）
    fn 右向量矩阵(分解: &非厄米谱分解, n: usize) -> DMatrix<NComplex<f64>> {
        let m = 分解.模数;
        let mut v = DMatrix::<NComplex<f64>>::zeros(n, m);
        for (j, 模) in 分解.模.iter().enumerate() {
            for i in 0..n.min(模.v.len()) {
                v[(i, j)] = NComplex::new(模.v[i], 0.0);
            }
        }
        v
    }
}

/// 三层联通输出（全流程汇总）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 联通输出 {
    /// 三层带宽投影
    pub 投影: 三层投影结果,
    /// 跨层自能（engine 慢模 → agent 中速）
    pub 跨层自能: Option<跨层自能结果>,
    /// 造波回归验证（模态系数按 e^{-t/τ_n}）
    pub 回归验证: Option<回归验证结果>,
    /// 汇总说明
    pub 说明: String,
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Complex as NComplex;

    /// 构造 n×n 对角复矩阵（diag = ω - i/τ）
    fn 对角L(对角: &[(f64, f64)]) -> DMatrix<NComplex<f64>> {
        let n = 对角.len();
        let mut l = DMatrix::<NComplex<f64>>::zeros(n, n);
        for (i, (ω, γ)) in 对角.iter().enumerate() {
            l[(i, i)] = NComplex::new(*ω, *γ);
        }
        l
    }

    /// 已知多尺度 L: 快模 τ=0.5 / 中速 τ=5,50 / 慢模 τ=5000,1e9(冻结)
    fn 多尺度L() -> DMatrix<NComplex<f64>> {
        对角L(&[
            (0.0, -1.0 / 0.5),   // 快模 τ=0.5
            (0.1, -1.0 / 5.0),   // agent 中速 τ=5
            (0.2, -1.0 / 50.0),  // agent 中速 τ=50
            (0.0, -1.0 / 5000.0), // engine 慢模 τ=5000
            (0.0, -1.0 / 1e9),   // engine 慢模 τ=1e9(近似冻结)
        ])
    }

    /// 三层投影带宽切分: 已知多尺度 L → 三层各含对应模
    #[test]
    fn 测试_三层投影_带宽切分() {
        let 引擎 = 三层联通引擎::新建();
        let 带宽 = 三层带宽 { 快模上界: 1.0, 慢模下界: 3600.0, ..Default::default() };
        let 结果 = 引擎.三层投影(&多尺度L(), 带宽).unwrap();
        assert_eq!(结果.core层.模索引.len(), 1, "core应含τ=0.5快模");
        assert_eq!(结果.agent层.模索引.len(), 2, "agent应含τ=5,50中速");
        assert_eq!(结果.engine层.模索引.len(), 2, "engine应含τ=5000,1e9慢模");
        assert!(结果.守恒, "三层模数和应等于总模数");
        assert_eq!(结果.总模数, 5);
        assert!(结果.engine层.冻结背景, "engine层应冻结");
        assert!(!结果.core层.冻结背景 && !结果.agent层.冻结背景);
        assert!(结果.说明.contains("守恒=true"), "{}", 结果.说明);
    }

    /// 谱自适应带宽（第402批）: 基于模 τ 分布（快界=最快×1.5, 慢界=快界×3）——
    /// 快慢尺度差 6 倍的模正确分组（from_observer 固定 100·τ_obs 会漏慢模）
    #[test]
    fn 测试_谱自适应带宽() {
        let 模τ = [0.5, 5.0, 50.0, 5000.0, 1e9];
        let 带宽 = 三层带宽::谱自适应(&模τ);
        assert!((带宽.快模上界 - 0.75).abs() < 1e-9, "快界=最快×1.5: {}", 带宽.快模上界);
        assert!((带宽.慢模下界 - 2.25).abs() < 1e-9, "慢界=快界×3: {}", 带宽.慢模下界);
        // 分组验证: 快模 0.5 归快, 5/50/5000/1e9 归慢
        let 快: Vec<f64> = 模τ.iter().copied().filter(|t| *t < 带宽.快模上界).collect();
        let 慢: Vec<f64> = 模τ.iter().copied().filter(|t| *t > 带宽.慢模下界).collect();
        assert_eq!(快, vec![0.5], "快模组: {:?}", 快);
        assert_eq!(慢.len(), 4, "慢模组含 5/50/5000/1e9: {:?}", 慢);
        // 不足 2 模 → 默认带宽
        let 默认 = 三层带宽::谱自适应(&[5.0]);
        assert!(默认.快模上界 > 0.0);
    }

    /// 三层投影守恒: 带宽改变 → 守恒仍成立（各层模数随带宽变化）
    #[test]
    fn 测试_三层投影_守恒随带宽() {
        let 引擎 = 三层联通引擎::新建();
        // 慢模下界 100 → engine 只含 τ=5000,1e9; agent 含 τ=5,50; core 含 τ=0.5
        let 带宽a = 三层带宽 { 快模上界: 1.0, 慢模下界: 100.0, ..Default::default() };
        let 结果a = 引擎.三层投影(&多尺度L(), 带宽a).unwrap();
        assert!(结果a.守恒 && 结果a.总模数 == 5);
        assert_eq!(结果a.agent层.模索引.len(), 2, "agent应含τ=5,50");
        assert_eq!(结果a.engine层.模索引.len(), 2, "engine应含τ=5000,1e9");
        // 快模上界 100 → core 含 τ=0.5,5,50
        let 带宽b = 三层带宽 { 快模上界: 100.0, 慢模下界: 3600.0, ..Default::default() };
        let 结果b = 引擎.三层投影(&多尺度L(), 带宽b).unwrap();
        assert!(结果b.守恒 && 结果b.总模数 == 5);
        assert_eq!(结果b.core层.模索引.len(), 3, "core应含τ=0.5,5,50");
        assert_eq!(结果b.agent层.模索引.len(), 0, "100≤τ≤3600 无模");
    }

    /// 跨块弱耗散耦合 L（agent τ=5,10 / engine τ=5000,1e6, 跨块耦合 -0.0001i）
    ///
    /// 耦合取**虚部**（耗散耦合）: 实部耦合会与对角耗散竞争产生实根/失稳。
    /// 稳定条件 κ² < |γ_a|·|γ_e|（耦合后本征值不越过零点）——engine γ 取
    /// -2e-4/-1e-6, κ=1e-4 → κ²=1e-8 < |γ_a||γ_e|(4e-5/1e-7) ✓ 全稳定且分层清晰。
    fn 耦合L() -> DMatrix<NComplex<f64>> {
        let mut l = 对角L(&[
            (0.0, -1.0 / 5.0),
            (0.0, -1.0 / 10.0),
            (0.0, -1.0 / 5000.0),
            (0.0, -1.0 / 1e6),
        ]);
        l[(0, 2)] = NComplex::new(0.0, -0.0001);
        l[(2, 0)] = NComplex::new(0.0, -0.0001);
        l[(1, 3)] = NComplex::new(0.0, -0.0001);
        l[(3, 1)] = NComplex::new(0.0, -0.0001);
        l
    }

    /// 跨层自能因果性: engine 慢模背景 → agent 中速 L_eff 修正 Γ>0
    ///
    /// 需跨块耦合（Σ 才非零）; ω 对齐 engine 慢模频率(0.0) → (ω-λ_E)^{-1} 近纯虚
    /// → Γ_eff=-Im Σ>0 因果性通过。
    #[test]
    fn 测试_跨层自能_因果性() {
        let 引擎 = 三层联通引擎::新建();
        let l = 耦合L();
        let 带宽 = 三层带宽 { 快模上界: 1.0, 慢模下界: 3600.0, ..Default::default() };
        let 投影 = 引擎.三层投影(&l, 带宽).unwrap();
        assert_eq!(投影.agent层.模索引.len(), 2, "agent应含τ=5,10");
        assert_eq!(投影.engine层.模索引.len(), 2, "engine应含τ=5000,1e6");
        let 结果 = 引擎.跨层自能(&l, &投影.agent层.模索引, &投影.engine层.模索引, 0.0).unwrap();
        assert!(结果.因果性通过, "耦合L+ω对齐慢模 → 因果性应通过: {}", 结果.说明);
        assert_eq!(结果.Γ_eff对角.len(), 4, "Γ_eff 应为 n 维");
        assert!(结果.自能.len() == 4 && 结果.自能[0].len() == 4);
        // engine 层空 → 跨层自能不可算（联通 返回 None）
        let 全快 = 对角L(&[(0.0, -1.0 / 0.5), (0.0, -1.0 / 0.8)]);
        let 投影2 = 引擎.三层投影(&全快, 三层带宽 { 快模上界: 1.0, 慢模下界: 3600.0, ..Default::default() }).unwrap();
        assert!(投影2.engine层.模索引.is_empty());
        let 联通 = 引擎.联通(&全快, 三层带宽 { 快模上界: 1.0, 慢模下界: 3600.0, ..Default::default() }, None, None).unwrap();
        assert!(联通.跨层自能.is_none(), "无engine层应无跨层自能");
    }

    /// 造波回归验证: c(t)=e^{-t/2} → 拟合τ≈2 误差<15% 通过
    #[test]
    fn 测试_造波回归验证_指数() {
        let 引擎 = 三层联通引擎::新建();
        // 指数衰减 τ=2 的离散序列（ln 线性）
        let 序列: Vec<f64> = (0..20).map(|k| (-k as f64 / 2.0).exp()).collect();
        let 结果 = 引擎.造波回归验证(&[序列], &[2.0], None).unwrap();
        assert!(结果.总体通过, "{}", 结果.说明);
        assert!(结果.每模[0].通过, "{}", 结果.每模[0].判定);
        assert!((结果.每模[0].拟合τ - 2.0).abs() < 0.3, "拟合τ应≈2: {}", 结果.每模[0].拟合τ);
    }

    /// 造波回归验证: c(t)=sin(t) 非指数衰减 → 拒绝
    #[test]
    fn 测试_造波回归验证_非指数() {
        let 引擎 = 三层联通引擎::新建();
        let 序列: Vec<f64> = (0..20).map(|k| (k as f64).sin()).collect();
        let 结果 = 引擎.造波回归验证(&[序列], &[2.0], None).unwrap();
        assert!(!结果.每模[0].通过, "非指数衰减应拒绝: {}", 结果.每模[0].判定);
        assert!(!结果.总体通过);
        // 样本不足
        let 结果2 = 引擎.造波回归验证(&[vec![1.0]], &[2.0], None).unwrap();
        assert!(!结果2.每模[0].通过, "样本不足应失败");
    }

    /// 造波回归验证: 输入保护（系数与理论τ数不匹配）
    #[test]
    fn 测试_造波回归验证_输入保护() {
        let 引擎 = 三层联通引擎::新建();
        assert!(引擎.造波回归验证(&[], &[], None).is_err(), "空应Err");
        assert!(引擎.造波回归验证(&[vec![1.0, 0.5]], &[1.0, 2.0], None).is_err(), "数量不匹配应Err");
    }

    // ==================== D2（2026-08-23）: 观测者等级 O0-O5 贯穿 + 带宽参数化 ====================

    /// D2-V1: from_observer 带宽推导（P0 测量相对性原理落地）
    #[test]
    fn 测试_D2V1_from_observer带宽推导() {
        let 带宽 = 三层带宽::from_observer(1.0, 2); // O2 生理级 τ_obs=1s
        assert!((带宽.快模上界 - 1.0).abs() < 1e-12, "τ_fast=τ_obs: {}", 带宽.快模上界);
        assert!((带宽.慢模下界 - 100.0).abs() < 1e-12, "τ_slow=100τ_obs: {}", 带宽.慢模下界);
        assert_eq!(带宽.观测者等级, Some(2));
        assert_eq!(带宽.观测者τ_obs, Some(1.0));
        // P0 分辨范围: τ_min=0.1τ_obs, τ_max=1000τ_obs
        let (最小, 最大) = 带宽.观测者范围().unwrap();
        assert!((最小 - 0.1).abs() < 1e-12 && (最大 - 1000.0).abs() < 1e-12, "范围({},{})", 最小, 最大);
        // O3 自指级 τ_obs=10s → 边界按比例移动
        let 带宽3 = 三层带宽::from_observer(10.0, 3);
        assert!((带宽3.快模上界 - 10.0).abs() < 1e-12);
        assert!((带宽3.慢模下界 - 1000.0).abs() < 1e-12);
        assert!(带宽3.合法());
    }

    /// D2-V3: default() 零破坏兼容（1.0/3600 + 无观测者字段）
    #[test]
    fn 测试_D2V3_default零破坏() {
        let 带宽 = 三层带宽::default();
        assert!((带宽.快模上界 - 1.0).abs() < 1e-12);
        assert!((带宽.慢模下界 - 3600.0).abs() < 1e-12);
        assert_eq!(带宽.观测者等级, None);
        assert_eq!(带宽.观测者τ_obs, None);
        assert!(带宽.合法());
        // 旧 JSON 反序列化兼容（无新字段）
        let 旧json = r#"{"快模上界": 1.0, "慢模下界": 3600.0}"#;
        let 反: 三层带宽 = serde_json::from_str(旧json).expect("旧 JSON 应反序列化成功（serde default）");
        assert!((反.快模上界 - 1.0).abs() < 1e-12);
        assert_eq!(反.观测者等级, None);
    }

    /// D2-V2: 观测者相关性——同一动力学图, 不同 τ_obs 三段切分边界移动
    #[test]
    fn 测试_D2V2_观测者相关性() {
        let 引擎 = 三层联通引擎::新建();
        let l = 多尺度L(); // τ=0.5,5,50,5000,1e9
        // O2 生理级 τ_obs=1s: 快界1/慢界100
        let 带宽O2 = 三层带宽::from_observer(1.0, 2);
        let 结果O2 = 引擎.三层投影(&l, 带宽O2).unwrap();
        // O4 集体级 τ_obs=100s: 快界100/慢界1e4
        let 带宽O4 = 三层带宽::from_observer(100.0, 4);
        let 结果O4 = 引擎.三层投影(&l, 带宽O4).unwrap();
        // 边界移动 → 快模/慢模集合不同（P0"不同 τ_O 看到不同世界"）
        assert_ne!(结果O2.core层.模索引, 结果O4.core层.模索引, "快模集合应随观测者变化");
        assert_ne!(结果O2.engine层.模索引, 结果O4.engine层.模索引, "慢模集合应随观测者变化");
        // O2 下 τ=50 在带内（agent）; O4 下 τ=50 在 core（<100）
        assert!(结果O2.agent层.模索引.contains(&2), "O2 下 τ=50 应属 agent");
        assert!(结果O4.core层.模索引.contains(&2), "O4 下 τ=50 应属 core 快模");
        assert!(结果O2.守恒 && 结果O4.守恒, "两种观测者带宽均守恒");
    }

    /// D2: 观测者上下文 新建推导带宽
    #[test]
    fn 测试_D2_观测者上下文() {
        let 上下文 = 观测者上下文::新建(3, 10.0);
        assert_eq!(上下文.等级, 3);
        assert!((上下文.带宽.快模上界 - 10.0).abs() < 1e-12);
        assert!((上下文.带宽.慢模下界 - 1000.0).abs() < 1e-12);
        assert_eq!(上下文.窗参数σ_O, None);
    }
}
