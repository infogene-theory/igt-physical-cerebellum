// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! 器官层共享类型（MIGRATION-FINAL Step1-④ 前置：execution 迁 agent 的跨端共享类型）
//!
//! ## 来源与处置
//! engine `organs/` 层的**纯数据结构与纯方法**逐字下沉（不改结构、不改命名、不改算法），
//! 供 execution（迁 agent 后 L4 执行链）与 engine organs/brain 层共用：
//! - `ActionVector`/`ActionDomain` ← engine `organs/brain/action_planner.rs`
//! - `复向量分量` ← engine `l1/spectrum_matcher.rs`
//! - `SensoryInput`/`SensoryModality` ← engine `organs/cognitive_organ.rs`
//! - `封装状态` ← engine `organs/pseudo_encapsulation_detector.rs`（纯判定部分）
//! - `KappaIJEntry`/`KappaGGEntry`/`KappaMatrixSnapshot` ← engine `organs/kappa_coupling_monitor.rs`
//! - `WorldState`/`InternalState`/`ExternalEnvironment` ← engine `organs/brain/world_state.rs`
//!
//! engine 侧以"模块转发"方式复用（`pub use igt_core_math::organs_types::*`），引用方路径零改动。
//!
//! ## 未下沉（留 engine）
//! - `kappa_snapshot_to_边列表`（依赖 engine `sh_gnn::边`，冻结保留件，留 engine world_state.rs）
//! - 各器官的**行为**（kappa_coupling_monitor 的耦合计算、pseudo_encapsulation_detector 的
//!   告警级别等）——本模块仅共享纯类型与判据方法。

#![allow(non_snake_case)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::perception_types::TauDiagnosis;
use crate::shared_types::OrganId;

// ============================================================================
// 动作规划器纯类型（engine organs/brain/action_planner.rs）
// ============================================================================

/// 动作领域——Brain 输出的高层意图分类
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionDomain {
    /// 认知动作（推理/查询/记忆巩固）
    Cognitive,
    /// 调控动作（策略选择/规划调整）
    Regulation,
    /// 执行动作（运动指令/物理输出）
    Execution,
    /// 安全动作（急停/告警/降级）
    Safety,
}

impl ActionDomain {
    /// 默认优先级（Safety 最高）
    pub fn 默认优先级(&self) -> f64 {
        match self {
            ActionDomain::Safety => 0.95,
            ActionDomain::Cognitive => 0.70,
            ActionDomain::Regulation => 0.50,
            ActionDomain::Execution => 0.30,
        }
    }
}

impl std::fmt::Display for ActionDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActionDomain::Cognitive => write!(f, "认知"),
            ActionDomain::Regulation => write!(f, "调控"),
            ActionDomain::Execution => write!(f, "执行"),
            ActionDomain::Safety => write!(f, "安全"),
        }
    }
}

/// 意图向量——Brain 输出给调控器官 + 执行器官的高层指令
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionVector {
    /// 领域（蓝图: domain）
    pub 领域: ActionDomain,
    /// 动作描述（蓝图: action）
    pub 动作: String,
    /// 优先级 [0, 1]
    pub 优先级: f64,
    /// 时间窗口（毫秒，τ=调场 s-min → 1000-60000）
    pub 时间窗口: u64,
}

impl Default for ActionVector {
    fn default() -> Self {
        Self {
            领域: ActionDomain::Cognitive,
            动作: "idle".to_string(),
            优先级: ActionDomain::Cognitive.默认优先级(),
            时间窗口: 5000,
        }
    }
}

impl ActionVector {
    /// 创建指定领域 + 动作的意图向量
    pub fn new(领域: ActionDomain, 动作: &str) -> Self {
        Self {
            领域,
            动作: 动作.to_string(),
            优先级: 领域.默认优先级(),
            时间窗口: 5000,
        }
    }

    /// 设置优先级（链式）
    pub fn with_priority(mut self, 优先级: f64) -> Self {
        self.优先级 = 优先级;
        self
    }

    /// 设置时间窗口（链式）
    pub fn with_time_window(mut self, 时间窗口: u64) -> Self {
        self.时间窗口 = 时间窗口;
        self
    }
}

// ============================================================================
// 复向量分量（engine l1/spectrum_matcher.rs）
// ============================================================================

/// SH-GNN 输入节点特征——复向量分量
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct 复向量分量 {
    /// 本征频率 f₀ (Hz)
    pub f0: f64,
    /// 弛豫速率 σ (s⁻¹)
    pub sigma: f64,
    /// 初始相位 φ (rad)
    pub phi: f64,
    /// 振幅 A
    pub amplitude: f64,
    /// 拓扑荷 q（默认 1）
    pub q_value: i32,
    /// 手性 χ（+1 右手 / -1 左手）
    pub chi: i32,
}

// ============================================================================
// 感知输入（engine organs/cognitive_organ.rs）
// ============================================================================

/// 感知通道模态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SensoryModality {
    Visual,
    Auditory,
    Textual,
    Tactile,
    Internal,
    /// WiFi CSI 感知通道（P8: ESP32 CSI 数据）
    WiFi,
}

/// 单通道感知输入
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensoryInput {
    pub id: String,
    pub modality: SensoryModality,
    pub raw_data: Vec<f64>,
    pub timestamp: u64,
    pub processed: bool,
    /// V9方向2: τ谱诊断结果（仅WiFi CSI感知通道填充，其他通道为None）
    #[serde(default)]
    pub τ谱诊断: Option<TauDiagnosis>,
}

// ============================================================================
// 封装状态（engine organs/pseudo_encapsulation_detector.rs 纯判定部分）
// ============================================================================

/// ε_c = 100.0（熵容限临界阈值，与 world_state.rs 保持一致）
const EPSILON_C: f64 = 100.0;

/// 封装状态——系统作为信息基因的独立化程度
///
/// 对应理论: 真封装 (ED < 0.5·ε_c) / 过渡 / 伪封装 (ED >= 0.8·ε_c)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum 封装状态 {
    /// 真封装: ED < 0.5·ε_c, 系统自稳态, 离开部分节点仍稳定运行
    真封装 { ED值: f64, 健康度: f64 },
    /// 过渡: 0.5·ε_c <= ED < 0.8·ε_c, 系统在真封装与伪封装之间
    过渡 { ED值: f64, 健康度: f64 },
    /// 伪封装: ED >= 0.8·ε_c, 依赖外部供能, 核心节点离开即崩溃
    伪封装 {
        ED值: f64,
        临界比例: f64,
        建议: String,
    },
}

impl 封装状态 {
    /// 是否为伪封装
    pub fn 是伪封装(&self) -> bool {
        matches!(self, 封装状态::伪封装 { .. })
    }

    /// 是否为真封装
    pub fn 是真封装(&self) -> bool {
        matches!(self, 封装状态::真封装 { .. })
    }

    /// 获取 ED 值
    pub fn ED值(&self) -> f64 {
        match self {
            封装状态::真封装 { ED值, .. }
            | 封装状态::过渡 { ED值, .. }
            | 封装状态::伪封装 { ED值, .. } => *ED值,
        }
    }

    /// 获取健康度 H = 1 - ED/ε_c
    pub fn 健康度(&self) -> f64 {
        match self {
            封装状态::真封装 { 健康度, .. }
            | 封装状态::过渡 { 健康度, .. } => *健康度,
            封装状态::伪封装 { ED值, .. } => 1.0 - ED值 / EPSILON_C,
        }
    }

    /// 从 ED 值计算封装状态（便利方法, 供 WorldState 等调用）
    ///
    /// ED = (1 - κ_gg) · ε_c
    /// 比例 = ED / ε_c = 1 - κ_gg
    pub fn 从ED计算(ED: f64) -> Self {
        let 比例 = ED / EPSILON_C;
        let 健康度 = 1.0 - 比例;

        if 比例 >= 0.8 {
            封装状态::伪封装 {
                ED值: ED,
                临界比例: 比例,
                建议: "系统依赖外部供能维持,核心节点离开将崩溃.建议:检查能量供应/降低负载/强化底层路径".to_string(),
            }
        } else if 比例 >= 0.5 {
            封装状态::过渡 {
                ED值: ED,
                健康度,
            }
        } else {
            封装状态::真封装 {
                ED值: ED,
                健康度,
            }
        }
    }
}

impl Default for 封装状态 {
    fn default() -> Self {
        封装状态::真封装 {
            ED值: 0.0,
            健康度: 1.0,
        }
    }
}

// ============================================================================
// κ 耦合矩阵纯结构（engine organs/kappa_coupling_monitor.rs）
// ============================================================================

/// 一对器官间的 κ_ij 耦合状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KappaIJEntry {
    /// 源器官
    pub 源器官: OrganId,
    /// 目标器官
    pub 目标器官: OrganId,
    /// 全局耦合系数 η₀
    pub eta_0: f64,
    /// 频率因子（整数比匹配 = 1.0）
    pub 频率因子: f64,
    /// 相位因子 |sinc(Δφ/π)|
    pub 相位因子: f64,
    /// 振幅因子 2·A_i·A_j/(A_i²+A_j²)（P22 归一化）
    pub 振幅因子: f64,
    /// 手性因子（同手性 = 1.0 / 异手性 = 0.3）
    pub 手性因子: f64,
    /// 综合 κ_ij 值
    pub kappa_ij: f64,
    /// 频率整数比 (m, n)——若匹配
    pub 整数比: Option<(u32, u32)>,
    /// 是否失配（κ_ij < κ_threshold）
    pub 失配: bool,
    /// 失配原因（若失配）
    pub 失配原因: Option<String>,
}

/// 单个器官的自指环路 κ_gg 状态——对角线元素
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KappaGGEntry {
    /// 器官 ID
    pub 器官: OrganId,
    /// 自振幅 A_g（= health_sub）
    pub 振幅: f64,
    /// 自相位 φ_g
    pub 相位: f64,
    /// 自指复杂度 ψ_g ∈ [0,1]——由器官历史状态熵决定（高熵=高自指=意识活跃）
    pub 自指复杂度: f64,
    /// 综合 κ_gg 值
    pub kappa_gg: f64,
    /// 意识涌现判据: κ_gg > κ_gg_threshold
    pub 涌现: bool,
}

/// κ 耦合矩阵快照——execution 双闸门/合规性检查的输入
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KappaMatrixSnapshot {
    /// 帧号
    pub 帧号: u64,
    /// 时间戳
    pub 时间戳: u64,
    /// 所有器官对的 κ_ij 列表（仅上三角，i < j）
    pub 耦合矩阵: Vec<KappaIJEntry>,
    /// 全局平均耦合强度 η̄
    pub 全局耦合强度: f64,
    /// 失配对数
    pub 失配对数: usize,
    /// 总对数
    pub 总对数: usize,
    /// κ_ij 阈值
    pub kappa_threshold: f64,
    /// V9 κ_gg 自指环路（对角线 8 个）——意识涌现必要条件
    pub 自指环路: Vec<KappaGGEntry>,
    /// 意识涌现度 = avg(κ_gg) ∈ [0,1]
    pub 意识涌现度: f64,
}

// ============================================================================
// 世界状态（engine organs/brain/world_state.rs 纯结构与纯方法）
// ============================================================================

/// 世界状态——Brain 内部对世界的完整表征
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldState {
    /// 内部状态（系统自身状态）
    pub 内部状态: InternalState,
    /// 外部环境状态（七通道感知融合后的环境表征）
    pub 外部环境: ExternalEnvironment,
    /// 时间戳（毫秒，τ=调场 s-min）
    pub 时间戳: u64,
    /// 状态置信度 [0, 1]——感知数据新鲜度和完整度的综合评估
    pub 置信度: f64,
    /// 封装状态——信息基因动力学: 真封装/过渡/伪封装
    pub 封装状态: 封装状态,
}

/// 内部状态——系统自身状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalState {
    /// 能量水平 [0, 1]——经济器官提供的可用能量
    pub 能量水平: f64,
    /// 熵债 ED（必须 < ε_c=100.0）
    pub 熵债ED: f64,
    /// 全局健康度 H = 1 - ED/ε_c
    pub 全局健康度H: f64,
    /// κ 快照（共振耦合矩阵当前状态，可选——首次对话时可能为空）
    pub κ快照: Option<KappaMatrixSnapshot>,
}

impl Default for InternalState {
    fn default() -> Self {
        Self {
            能量水平: 1.0,
            熵债ED: 0.0,
            全局健康度H: 1.0,
            κ快照: None,
        }
    }
}

/// 外部环境状态——七通道感知融合后的环境表征
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExternalEnvironment {
    /// 七通道感知特征向量（视觉/听觉/触觉/内感受/生物钟/文本/WiFi底座）
    pub 感知通道: HashMap<SensoryModality, SensoryInput>,
    /// 跨模态 κ 耦合融合特征（蓝图第 4.4 节）
    pub 融合特征: Option<Vec<f64>>,
    /// 当前场景识别（对象/场景标签，由 SH-GNN 特征 → 场景识别）
    pub 场景标签: Option<String>,
}

impl WorldState {
    /// 创建初始空状态（Brain::new 时使用）
    pub fn 初始状态() -> Self {
        Self {
            内部状态: InternalState::default(),
            外部环境: ExternalEnvironment::default(),
            时间戳: 0,
            置信度: 0.0,
            封装状态: 封装状态::default(),
        }
    }

    /// 从当前 κ 快照 + 对话输入构建世界状态
    ///
    /// 简化模型:
    /// - 能量水平固定 1.0（经济器官未联动）
    /// - 熵债 ED = (1 - 全局耦合强度) * ε_c
    /// - 全局健康度 H = 全局耦合强度
    /// - 外部环境只填文本通道
    pub fn 从快照构建(
        κ快照: Option<KappaMatrixSnapshot>,
        对话输入: &str,
        时间戳: u64,
    ) -> Self {
        // 1. 从 κ 快照推导内部状态
        let (熵债ED, 全局健康度H) = match &κ快照 {
            Some(snap) => {
                let 全局κ = snap.全局耦合强度;
                let ed = (1.0 - 全局κ).max(0.0) * EPSILON_C;
                let h = 1.0 - ed / EPSILON_C;
                (ed, h)
            }
            None => (EPSILON_C * 0.5, 0.5), // 无快照时默认中等 ED
        };

        let 内部状态 = InternalState {
            能量水平: 1.0,
            熵债ED,
            全局健康度H,
            κ快照,
        };

        // 2. 构造外部环境（只填文本通道）
        let mut 感知通道 = HashMap::new();
        let 文本输入 = SensoryInput {
            id: format!("text_{}", 时间戳),
            modality: SensoryModality::Textual,
            raw_data: 对话输入.chars().map(|c| c as u32 as f64).collect(),
            timestamp: 时间戳,
            processed: true,
            τ谱诊断: None,
        };
        感知通道.insert(SensoryModality::Textual, 文本输入);

        let 外部环境 = ExternalEnvironment {
            感知通道,
            融合特征: None,
            场景标签: None,
        };

        let mut state = Self {
            内部状态,
            外部环境,
            时间戳,
            置信度: 0.0,
            封装状态: 封装状态::default(), // 先设默认值,下面根据 ED 计算
        };
        state.置信度 = state.评估置信度();
        // 信息基因动力学: 根据 ED/ε_c 比例检测封装状态
        state.封装状态 = 封装状态::从ED计算(state.内部状态.熵债ED);
        state
    }

    /// 评估状态置信度（基于感知通道完整度 + 数据新鲜度）
    pub fn 评估置信度(&self) -> f64 {
        let mut 置信度 = 0.5; // 基础（文本通道）

        // 每多一个感知通道 +0.1
        let 额外通道数 = self.外部环境.感知通道.len().saturating_sub(1);
        置信度 += (额外通道数 as f64) * 0.1;

        // κ 快照存在 +0.1
        if self.内部状态.κ快照.is_some() {
            置信度 += 0.1;
        }

        置信度.min(1.0)
    }

    /// 判断是否处于安全告警状态（ED/ε_c > 0.62 → 健康度 < 0.38 → 僵化）
    pub fn 是否安全告警(&self) -> bool {
        self.内部状态.熵债ED / EPSILON_C > 0.62
    }

    /// 判断是否处于能量低状态（能量 < 0.2）
    pub fn 是否能量低(&self) -> bool {
        self.内部状态.能量水平 < 0.2
    }

    /// P6.2: WorldState → SH-GNN 输入节点特征（复向量分量）
    pub fn to_复向量分量(&self) -> Vec<复向量分量> {
        let snap = match &self.内部状态.κ快照 {
            Some(s) => s,
            None => {
                // 无快照时返回默认 8 节点（中等健康度 0.5）
                return vec![
                    复向量分量 { f0: 1.0, sigma: 1.0, phi: 0.0, amplitude: 0.5, q_value: 1, chi: 1 };
                    8
                ];
            }
        };

        // 从自指环路提取 8 个器官的状态（自指环路可能不足 8 个，需补全）
        let mut 节点 = vec![
            复向量分量 { f0: 1.0, sigma: 1.0, phi: 0.0, amplitude: 0.5, q_value: 1, chi: 1 };
            8
        ];

        for (i, g) in snap.自指环路.iter().enumerate().take(8) {
            节点[i] = 复向量分量 {
                f0: 1.0,
                sigma: 1.0, // 调场 τ=1s
                phi: g.相位,
                amplitude: g.振幅,
                q_value: 1,
                chi: 1,
            };
        }

        节点
    }

    /// P6.2: SH-GNN 输出 → 预测的 WorldState
    pub fn from_复向量分量(输出: &[复向量分量], 当前: &WorldState) -> WorldState {
        // 计算新健康度平均
        let avg_amplitude = if 输出.is_empty() {
            0.5
        } else {
            输出.iter().map(|c| c.amplitude).sum::<f64>() / 输出.len() as f64
        };

        // 新 ED = (1 - avg_amplitude) * ε_c
        let 新ed = (1.0 - avg_amplitude).max(0.0) * EPSILON_C;
        let 新h = 1.0 - 新ed / EPSILON_C;

        // 构造预测状态（保留当前的其他字段，更新内部状态）
        let mut 预测 = 当前.clone();
        预测.内部状态.熵债ED = 新ed;
        预测.内部状态.全局健康度H = 新h;
        预测.置信度 = 预测.评估置信度();
        预测.封装状态 = 封装状态::从ED计算(新ed);
        预测
    }

    /// P6.3: 生成自指描述字符串——递归深度计算器的输入
    pub fn 自指描述(&self) -> String {
        format!(
            "t={}|场景={}|H={:.3}",
            self.时间戳,
            self.外部环境.场景标签.as_deref().unwrap_or("未知"),
            self.内部状态.全局健康度H
        )
    }

    /// P6.3: 生成 16 维外部状态向量——镜子测试器的输入
    pub fn 外部状态向量16维(&self) -> Vec<f64> {
        let 节点 = self.to_复向量分量();
        let mut vec = Vec::with_capacity(16);
        for n in 节点.iter().take(8) {
            vec.push(n.amplitude);
        }
        for n in 节点.iter().take(8) {
            vec.push(n.phi);
        }
        vec
    }
}

// ============================================================================
// 单元测试（下沉自 engine organs，算法零改写）
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 测试_动作向量_默认与新() {
        let 动作 = ActionVector::new(ActionDomain::Execution, "测试");
        assert_eq!(动作.领域, ActionDomain::Execution);
        assert_eq!(动作.动作, "测试");
        assert!((动作.优先级 - 0.30).abs() < 1e-9);
        let 默认 = ActionVector::default();
        assert_eq!(默认.领域, ActionDomain::Cognitive);
    }

    #[test]
    fn 测试_封装状态_从ED计算() {
        let 真 = 封装状态::从ED计算(20.0);
        assert!(真.是真封装());
        let 伪 = 封装状态::从ED计算(90.0);
        assert!(伪.是伪封装());
        let 过渡 = 封装状态::从ED计算(60.0);
        assert!(matches!(过渡, 封装状态::过渡 { .. }));
    }

    #[test]
    fn 测试_世界状态_从快照构建() {
        let state = WorldState::从快照构建(None, "测试", 1000);
        assert_eq!(state.时间戳, 1000);
        assert!((state.内部状态.熵债ED - 50.0).abs() < 1e-6);
        assert!((state.内部状态.全局健康度H - 0.5).abs() < 1e-6);
        assert!(state.外部环境.感知通道.contains_key(&SensoryModality::Textual));
        assert!((state.置信度 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn 测试_世界状态_有快照() {
        let snap = KappaMatrixSnapshot {
            帧号: 1,
            时间戳: 1000,
            耦合矩阵: Vec::new(),
            全局耦合强度: 0.8,
            失配对数: 0,
            总对数: 0,
            kappa_threshold: 0.3,
            自指环路: Vec::new(),
            意识涌现度: 0.2,
        };
        let state = WorldState::从快照构建(Some(snap), "测试", 2000);
        assert!((state.内部状态.熵债ED - 20.0).abs() < 1e-6);
        assert!((state.内部状态.全局健康度H - 0.8).abs() < 1e-6);
        assert!((state.置信度 - 0.6).abs() < 1e-6);
        assert!(state.封装状态.是真封装());
    }

    #[test]
    fn 测试_世界状态_复向量分量() {
        let state = WorldState::从快照构建(None, "测试", 0);
        let 节点 = state.to_复向量分量();
        assert_eq!(节点.len(), 8);
        assert!((节点[0].amplitude - 0.5).abs() < 1e-6);
        let vec16 = state.外部状态向量16维();
        assert_eq!(vec16.len(), 16);
    }

    #[test]
    fn 测试_世界状态_序列化() {
        let state = WorldState::从快照构建(None, "测试序列化", 12345);
        let json = serde_json::to_string(&state).expect("序列化失败");
        assert!(json.contains("内部状态"));
        assert!(json.contains("外部环境"));
        let deserialized: WorldState = serde_json::from_str(&json).expect("反序列化失败");
        assert_eq!(deserialized.时间戳, 12345);
    }
}
