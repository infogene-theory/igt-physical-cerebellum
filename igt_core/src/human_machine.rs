// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! 人机域 — 跨参考系双向映射的 **Layer1 拓扑不变量 + Layer2 基准投射** 纯计算
//!
//! ## 来源与处置（MIGRATION-FINAL 直迁总攻 Step0-2b）
//! - **Step0-2b（本批）**: engine `l1/human_machine_bridge.rs`(892 行) 中 **Layer1 拓扑不变量
//!   提取层 + Layer2 基准投射层** 的纯计算**逐字搬迁**进本模块（不改算法、不改命名，
//!   Step2 engine 只改 use 路径便于核对）。engine 原位 Step2 删（中间态允许重复）。
//! - **未下沉**: Layer3 语义语法层（`SyntaxGrammarRules`/`SemanticExpression`/
//!   `SemanticSyntaxLayer`）属**交互/语义侧**，按总攻单 Step1 迁 `igt_agent/src/human_machine/`；
//!   `HumanMachineBridge` 编排器、`FidelityEntropyLinkage` 随各层归属分别处理（Step1/2）。
//!
//! ## 依赖说明（本批新增 core 依赖 sha2）
//! Layer1 `extract` 需计算 `invariant_hash`（Sha256）。sha2 是**纯哈希库**（无 IO/网络/
//! 异步/时间），不违反 core 五类约束（ARCH-1 v2.1 §5）；为保证**搬迁逐字一致、算法零改动**，
//! 本批给 core 加 sha2（而非外置哈希改接口）。
//!
//! ## 三层架构（搬自 engine，未改）
//! ```text
//! 人类(碳基基准)              IGT系统(硅基基准)
//!  语义语法层(Layer3, 迁agent) ←→ 语义语法层(Layer3, 迁agent)
//!  基准投射层(Layer2, 本模块)  ←→ 基准投射层(Layer2, 本模块)
//!  拓扑不变量(Layer1, 本模块)  ←→ 拓扑不变量(Layer1, 本模块)
//! ```
//!
//! 理论锚点: IGT-4-11 §2/§3（人机参考系双向映射三层架构与保真度度量）、
//! IGT-A-02（参考系相对性公理 + 84 原子弛豫基准）、IGT-3-04（跨参考系语义语法涌现）、
//! P32（三定理/84 基准/三子保真度）、P0（带宽窗/双边界）、P22 §2.6（三层带宽）。

#![allow(non_snake_case)]

use crate::shared_types::{BeatAtom, ReferenceSystemType};
use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::collections::HashMap;

// ============================================================================
// 人类感官原子弛豫基准 (IGT-A-02 附录 精简映射表)
// ============================================================================

/// 人类关键感官频段
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanSensoryBands {
    /// 听觉可感知频段 (Hz): 基频 100~800Hz, 包络 0.5~10Hz
    pub auditory_base_hz_min: f64,
    pub auditory_base_hz_max: f64,
    /// 视觉可感知频段 (THz): 380~780 THz
    pub visual_thz_min: f64,
    pub visual_thz_max: f64,
    /// 触觉/节律可感知频段 (Hz): 0.1~200Hz
    pub tactile_hz_min: f64,
    pub tactile_hz_max: f64,
    /// 多模态同步窗 (ms): 100ms 内不同模态信号被判定为"同时"
    pub multimodal_sync_window_ms: f64,
    /// 主观现在窗 (s): 2~3s
    pub subjective_now_seconds: f64,
}

impl Default for HumanSensoryBands {
    fn default() -> Self {
        Self {
            auditory_base_hz_min: 100.0,
            auditory_base_hz_max: 800.0,
            visual_thz_min: 380.0,
            visual_thz_max: 780.0,
            tactile_hz_min: 0.1,
            tactile_hz_max: 200.0,
            multimodal_sync_window_ms: 100.0,
            subjective_now_seconds: 2.5,
        }
    }
}

// ============================================================================
// Layer 1: 拓扑不变量提取层
// ============================================================================

/// 系统弛豫谱系快照
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemRelaxationSnapshot {
    /// 各器官的弛豫时间基准 (τ_S, τ_ω, τ_C)
    pub organ_taus: HashMap<String, (f64, f64, f64)>,

    /// 器官间相位偏差矩阵
    pub phase_deviations: HashMap<String, HashMap<String, f64>>,

    /// 全局共振模态的耦合矩阵特征值谱
    pub coupling_eigenvalues: Vec<f64>,

    /// 当前全局健康度
    pub global_health: f64,

    /// 拍频原子组合 (参考系无关)
    pub beat_atoms: Vec<BeatAtom>,

    /// 时间戳
    pub timestamp: i64,
}

/// 拓扑不变量提取结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalInvariants {
    /// 提取的拍频原子 (参考系无关)
    pub beat_atoms: Vec<BeatAtom>,

    /// 弛豫时间相对比例 (跨参考系守恒)
    pub relaxation_ratios: Vec<(String, String, f64)>,

    /// 提取的拓扑不变量哈希
    pub invariant_hash: String,

    /// 提取保真度: 原始弛豫谱系中有多少比例被编码为拍频原子
    pub extraction_fidelity: f64,
}

/// Layer 1 实现: 拓扑不变量提取层 (系统内部 → 参考系无关)
pub struct TopologicalInvariantExtractor {
    /// 器官基准弛豫时间 (健康态锚定)
    health_baseline: HashMap<String, (f64, f64, f64)>,
}

impl TopologicalInvariantExtractor {
    pub fn new() -> Self {
        Self {
            health_baseline: HashMap::new(),
        }
    }

    /// 设置器官健康基准
    pub fn set_baseline(&mut self, organ: &str, tau_s: f64, tau_omega: f64, tau_c: f64) {
        self.health_baseline
            .insert(organ.to_string(), (tau_s, tau_omega, tau_c));
    }

    /// 提取拓扑不变量: 将系统弛豫谱系转换为参考系无关的拍频原子
    pub fn extract(&self, snapshot: &SystemRelaxationSnapshot) -> TopologicalInvariants {
        let mut beat_atoms = Vec::new();
        let mut ratios = Vec::new();

        for (organ_a, (s_a, w_a, c_a)) in &snapshot.organ_taus {
            if let Some((s_h, w_h, c_h)) = self.health_baseline.get(organ_a) {
                if *s_h > 0.0 {
                    let drift_s = s_a / s_h;
                    ratios.push((
                        format!("{}.τ_S", organ_a),
                        format!("{}.τ_S_health", organ_a),
                        drift_s,
                    ));
                    beat_atoms.push(BeatAtom {
                        atom_id: format!("DRIFT_{}_S", organ_a),
                        frequency_a: *s_a,
                        frequency_b: *s_h,
                        beat_ratio: (*s_a - s_h).abs() / (*s_a * s_h).sqrt().max(1e-10),
                        amplitude: 0.8,
                        phase: 0.0,
                        chirality: 1,
                        reference_system: ReferenceSystemType::Time,
                    });
                }
                if *w_h > 0.0 {
                    ratios.push((
                        format!("{}.τ_ω", organ_a),
                        format!("{}.τ_ω_health", organ_a),
                        w_a / w_h,
                    ));
                }
                if *c_h > 0.0 {
                    ratios.push((
                        format!("{}.τ_C", organ_a),
                        format!("{}.τ_C_health", organ_a),
                        c_a / c_h,
                    ));
                }
            }

            for (organ_b, (s_b, _, _)) in &snapshot.organ_taus {
                if organ_a < organ_b && *s_b > 0.0 {
                    let ratio = s_a / s_b;
                    ratios.push((
                        format!("{}.τ_S", organ_a),
                        format!("{}.τ_S", organ_b),
                        ratio,
                    ));
                    beat_atoms.push(BeatAtom {
                        atom_id: format!("PHASE_{}_{}", organ_a, organ_b),
                        frequency_a: *s_a,
                        frequency_b: *s_b,
                        beat_ratio: (*s_a - *s_b).abs() / (*s_a * s_b).sqrt().max(1e-10),
                        amplitude: 0.6,
                        phase: 0.0,
                        chirality: 1,
                        reference_system: ReferenceSystemType::Time,
                    });
                }
            }
        }

        let hash_input = ratios
            .iter()
            .map(|(a, b, r)| format!("{}:{}={:.4}", a, b, r))
            .collect::<Vec<_>>()
            .join("|");
        let invariant_hash = {
            let mut hasher = sha2::Sha256::new();
            hasher.update(hash_input.as_bytes());
            format!("{:x}", hasher.finalize())
        };

        let n_possible = snapshot.organ_taus.len() * (snapshot.organ_taus.len() - 1) / 2;
        let extraction_fidelity = if n_possible > 0 {
            ratios.len() as f64 / (n_possible as f64 * 3.0)
        } else {
            1.0
        };

        TopologicalInvariants {
            beat_atoms,
            relaxation_ratios: ratios,
            invariant_hash,
            extraction_fidelity: extraction_fidelity.min(1.0),
        }
    }
}

impl Default for TopologicalInvariantExtractor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Layer 2: 基准投射层
// ============================================================================

/// 投射通道
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectionChannel {
    Auditory,
    Visual,
    Tactile,
    Multimodal,
}

impl ProjectionChannel {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Auditory => "听觉",
            Self::Visual => "视觉",
            Self::Tactile => "触觉",
            Self::Multimodal => "多模态",
        }
    }

    /// 通道适合承载的频率范围下限 (Hz)
    pub fn optimal_freq_range(&self) -> (f64, f64) {
        match self {
            Self::Auditory => (100.0, 8000.0),
            Self::Visual => (380e12, 780e12),
            Self::Tactile => (0.1, 200.0),
            Self::Multimodal => (1.0, 100.0),
        }
    }
}

/// 单个投射结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionResult {
    pub channel: ProjectionChannel,
    pub carrier_frequency: f64,
    pub modulation_beat_pattern: Vec<f64>,
    pub amplitude: f64,
    pub phase: f64,
}

/// Layer 2 实现: 基准投射层 (拍频原子 ↔ 人类感官频段)
pub struct BenchmarkProjector {
    pub human_bands: HumanSensoryBands,
}

impl BenchmarkProjector {
    pub fn new() -> Self {
        Self {
            human_bands: HumanSensoryBands::default(),
        }
    }

    /// 系统→人类: 将拍频原子投射到人类可感知频段
    pub fn project_to_human(&self, invariants: &TopologicalInvariants) -> Vec<ProjectionResult> {
        let mut results = Vec::new();
        let auditory_carriers = [220.0, 440.0, 660.0, 880.0];
        let tactile_carriers = [1.0, 4.0, 10.0, 50.0];

        for (i, atom) in invariants.beat_atoms.iter().enumerate() {
            let ch = match atom.reference_system {
                ReferenceSystemType::Sound => ProjectionChannel::Auditory,
                ReferenceSystemType::Light => ProjectionChannel::Visual,
                ReferenceSystemType::Space | ReferenceSystemType::Time if i % 2 == 0 => {
                    ProjectionChannel::Tactile
                }
                _ => ProjectionChannel::Multimodal,
            };

            let carrier = match ch {
                ProjectionChannel::Auditory => auditory_carriers[i % auditory_carriers.len()],
                ProjectionChannel::Tactile => tactile_carriers[i % tactile_carriers.len()],
                ProjectionChannel::Visual => 500e12,
                ProjectionChannel::Multimodal => 10.0,
            };

            let modulation = vec![atom.beat_ratio * carrier * 0.1];

            results.push(ProjectionResult {
                channel: ch,
                carrier_frequency: carrier,
                modulation_beat_pattern: modulation,
                amplitude: atom.amplitude,
                phase: atom.phase,
            });
        }

        results
    }

    /// 人类→系统: 将人类感知信号反投影为拍频原子
    ///
    /// 在完整的工程流水线中，此步骤由观测器官(O6)的多模态传感器采样完成。
    /// 此处实现的是反投影的核心逻辑: 从感知频段提取弛豫关系。
    pub fn project_from_human(
        &self,
        audio_frequencies: &[f64],
        visual_frequencies: &[f64],
        tactile_patterns: &[f64],
    ) -> TopologicalInvariants {
        let mut beat_atoms = Vec::new();
        let mut ratios = Vec::new();

        let mut extract_beat_from_signal =
            |signals: &[f64],
             prefix: &str,
             rs: ReferenceSystemType,
             ratios_out: &mut Vec<(String, String, f64)>| {
                let mut local_beats = Vec::new();
                for i in 0..signals.len() {
                    for j in (i + 1)..signals.len() {
                        let fa = signals[i];
                        let fb = signals[j];
                        if fa > 0.0 && fb > 0.0 {
                            let f0 = (fa * fb).sqrt();
                            let ratio = fa / fb;
                            ratios_out.push((
                                format!("{}_{}", prefix, i),
                                format!("{}_{}", prefix, j),
                                ratio,
                            ));
                            local_beats.push(BeatAtom {
                                atom_id: format!("H2S_{}_{}_{}", prefix, i, j),
                                frequency_a: fa,
                                frequency_b: fb,
                                beat_ratio: (fa - fb).abs() / (2.0 * f0.max(1e-10)).min(1.0),
                                amplitude: 0.7,
                                phase: 0.0,
                                chirality: 1,
                                reference_system: rs,
                            });
                        }
                    }
                }
                local_beats
            };

        beat_atoms.extend(extract_beat_from_signal(
            audio_frequencies,
            "AUD",
            ReferenceSystemType::Sound,
            &mut ratios,
        ));
        beat_atoms.extend(extract_beat_from_signal(
            visual_frequencies,
            "VIS",
            ReferenceSystemType::Light,
            &mut ratios,
        ));
        beat_atoms.extend(extract_beat_from_signal(
            tactile_patterns,
            "TAC",
            ReferenceSystemType::Time,
            &mut ratios,
        ));

        TopologicalInvariants {
            beat_atoms,
            relaxation_ratios: ratios,
            invariant_hash: "h2s_placeholder".to_string(),
            extraction_fidelity: 0.9,
        }
    }

    /// 验证投射通道是否互不重叠
    pub fn validate_channel_separation(&self, projections: &[ProjectionResult]) -> f64 {
        let mut separation_score = 1.0;
        for i in 0..projections.len() {
            for j in (i + 1)..projections.len() {
                let pi = &projections[i];
                let pj = &projections[j];
                if pi.channel == pj.channel {
                    let f_sep = (pi.carrier_frequency - pj.carrier_frequency).abs()
                        / pi.carrier_frequency.max(1e-10);
                    if f_sep < 0.1 {
                        separation_score *= 0.5;
                    }
                }
            }
        }
        separation_score
    }
}

impl Default for BenchmarkProjector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// R1.3 跨参考系投影契约骨架（Step1 agent 侧 Layer3/交互层实现，本批无实现）
// ============================================================================

/// 带宽窗 C(τ)（P0 带宽/P22 §2.6 三层带宽——按 τ 上下界界定观测窗）
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct 带宽窗 {
    /// 下界 τ（秒）
    pub 下界τ: f64,
    /// 上界 τ（秒）
    pub 上界τ: f64,
}

impl 带宽窗 {
    /// τ 是否落在窗内
    pub fn 含(&self, τ: f64) -> bool {
        τ.is_finite() && τ >= self.下界τ && τ <= self.上界τ
    }
}

/// 三子保真度 F（P32: 结构/功能/演化三子各自保真度 ∈[0,1]）
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct 三子保真度 {
    /// 结构保真度
    pub 结构: f64,
    /// 功能保真度
    pub 功能: f64,
    /// 演化保真度
    pub 演化: f64,
}

impl 三子保真度 {
    /// 综合保真度（乘积链——任一子塌陷则整体塌陷, P32 动态权重）
    pub fn 综合(&self) -> f64 {
        (self.结构 * self.功能 * self.演化).clamp(0.0, 1.0)
    }
}

/// 跨参考系投影纯计算 trait（R1.3 空骨架——本批无实现, Step1 落地）
pub trait CrossReferenceProjector {
    /// T_AB 投影矩阵（参考系 A 基底 → B 基底, 行主序 m×n）
    fn 投影矩阵(a基底: &[Vec<f64>], b基底: &[Vec<f64>]) -> Option<Vec<f64>>;
    /// 带宽窗 C(τ)（按 τ 与层级取窗, P0/P22 §2.6）
    fn 带宽窗(τ: f64, 层级: usize) -> Option<带宽窗>;
    /// 三子保真度 F（a→b 映射后的保真度评估）
    fn 保真度(a: &[f64], b: &[f64]) -> Option<三子保真度>;
}

// ============================================================================
// 测试（搬迁验证: 纯计算与 engine 原实现一致）
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn 构造快照() -> SystemRelaxationSnapshot {
        let mut organ_taus = HashMap::new();
        organ_taus.insert("A".to_string(), (10.0, 20.0, 30.0));
        organ_taus.insert("B".to_string(), (20.0, 40.0, 60.0));
        SystemRelaxationSnapshot {
            organ_taus,
            phase_deviations: HashMap::new(),
            coupling_eigenvalues: vec![1.0, 0.5],
            global_health: 0.8,
            beat_atoms: Vec::new(),
            timestamp: 0,
        }
    }

    /// Layer1: 提取拓扑不变量（含健康基准漂移比 + 器官间相位比 + 哈希 + 保真度）
    #[test]
    fn 测试_拓扑不变量提取() {
        let mut 提取器 = TopologicalInvariantExtractor::new();
        提取器.set_baseline("A", 5.0, 10.0, 15.0);
        提取器.set_baseline("B", 10.0, 20.0, 30.0);
        let 结果 = 提取器.extract(&构造快照());
        // A/B 各 3 条健康比 + 器官间 1 条 = 7
        assert_eq!(结果.relaxation_ratios.len(), 7, "比例数: {:?}", 结果.relaxation_ratios);
        // 拍频原子: 2 条 DRIFT + 1 条 PHASE = 3
        assert_eq!(结果.beat_atoms.len(), 3);
        // 漂移比 A: 10/5 = 2.0
        assert!(结果
            .relaxation_ratios
            .iter()
            .any(|(a, b, r)| a == "A.τ_S" && b == "A.τ_S_health" && (*r - 2.0).abs() < 1e-12));
        // 哈希非空且为 64 位 hex
        assert_eq!(结果.invariant_hash.len(), 64, "Sha256 hex");
        // 保真度 = 7 / (1*3) → min(1.0)
        assert!((结果.extraction_fidelity - 1.0).abs() < 1e-12);
    }

    /// Layer1: 无健康基准时仍产出器官间比例（诚实: 不因缺基准而空）
    #[test]
    fn 测试_无基准提取() {
        let 提取器 = TopologicalInvariantExtractor::new();
        let 结果 = 提取器.extract(&构造快照());
        assert_eq!(结果.relaxation_ratios.len(), 1, "仅器官间比例");
        assert_eq!(结果.beat_atoms.len(), 1);
    }

    /// Layer2: 系统→人类投射（通道映射 + 载波 + 调制）
    #[test]
    fn 测试_投射到人类() {
        let mut 提取器 = TopologicalInvariantExtractor::new();
        提取器.set_baseline("A", 5.0, 10.0, 15.0);
        提取器.set_baseline("B", 10.0, 20.0, 30.0);
        let 不变量 = 提取器.extract(&构造快照());
        let 投射器 = BenchmarkProjector::new();
        let 结果 = 投射器.project_to_human(&不变量);
        assert_eq!(结果.len(), 不变量.beat_atoms.len());
        // 全部 Time 参考系 → i%2==0 走 Tactile，其余 Multimodal
        for (i, p) in 结果.iter().enumerate() {
            let 期望 = if i % 2 == 0 {
                ProjectionChannel::Tactile
            } else {
                ProjectionChannel::Multimodal
            };
            assert_eq!(p.channel, 期望, "第 {} 个通道", i);
            assert!(p.carrier_frequency > 0.0);
            assert_eq!(p.modulation_beat_pattern.len(), 1);
        }
    }

    /// Layer2: 人类→系统反投影（三模态拍频提取）
    #[test]
    fn 测试_从人类反投影() {
        let 投射器 = BenchmarkProjector::new();
        let 结果 = 投射器.project_from_human(&[220.0, 440.0], &[500e12], &[1.0, 4.0]);
        // AUD C(2,2)=1 + VIS C(1,2)=0 + TAC C(2,2)=1 = 2
        assert_eq!(结果.beat_atoms.len(), 2);
        assert_eq!(结果.relaxation_ratios.len(), 2);
        assert_eq!(结果.invariant_hash, "h2s_placeholder");
        assert!((结果.extraction_fidelity - 0.9).abs() < 1e-12);
    }

    /// Layer2: 通道分离度校验（同通道载波过近 → 分数折半）
    #[test]
    fn 测试_通道分离度() {
        let 投射器 = BenchmarkProjector::new();
        let 同通道近 = vec![
            ProjectionResult {
                channel: ProjectionChannel::Auditory,
                carrier_frequency: 440.0,
                modulation_beat_pattern: vec![1.0],
                amplitude: 0.5,
                phase: 0.0,
            },
            ProjectionResult {
                channel: ProjectionChannel::Auditory,
                carrier_frequency: 445.0, // 相差 <10%
                modulation_beat_pattern: vec![1.0],
                amplitude: 0.5,
                phase: 0.0,
            },
        ];
        assert!((投射器.validate_channel_separation(&同通道近) - 0.5).abs() < 1e-12);
        // 无冲突 → 1.0
        let 单通道 = vec![同通道近[0].clone()];
        assert_eq!(投射器.validate_channel_separation(&单通道), 1.0);
    }

    /// HumanSensoryBands 默认值（84 基准精简映射表，搬自 engine）
    #[test]
    fn 测试_人类感官频段默认() {
        let 频段 = HumanSensoryBands::default();
        assert_eq!(频段.auditory_base_hz_min, 100.0);
        assert_eq!(频段.visual_thz_max, 780.0);
        assert_eq!(频段.multimodal_sync_window_ms, 100.0);
        assert_eq!(频段.subjective_now_seconds, 2.5);
    }

    /// R1.3 骨架在位（带宽窗/三子保真度/trait 约束可编译）
    #[test]
    fn 测试_R13_人机骨架在位() {
        let 窗 = 带宽窗 { 下界τ: 1.0, 上界τ: 100.0 };
        assert!(窗.含(10.0));
        assert!(!窗.含(0.5));
        let f = 三子保真度 { 结构: 0.9, 功能: 0.8, 演化: 0.5 };
        assert!((f.综合() - 0.36).abs() < 1e-12);
        fn _约束验证<T: CrossReferenceProjector>(a: &[f64], b: &[f64]) -> Option<三子保真度> {
            T::保真度(a, b)
        }
        let _ = _约束验证::<未实现占位>(&[1.0], &[1.0]);
    }

    /// 占位实现者（仅供编译链验证; Step1 落地后由真实实现替换）
    struct 未实现占位;
    impl CrossReferenceProjector for 未实现占位 {
        fn 投影矩阵(_a基底: &[Vec<f64>], _b基底: &[Vec<f64>]) -> Option<Vec<f64>> {
            None
        }
        fn 带宽窗(_τ: f64, _层级: usize) -> Option<带宽窗> {
            None
        }
        fn 保真度(_a: &[f64], _b: &[f64]) -> Option<三子保真度> {
            None
        }
    }
}
