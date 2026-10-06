// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! Perception 纯类型层（MIGRATION-FINAL Step0-3c-1）
//!
//! engine `l1/perception/csi_data_format.rs` 与 `tau_spectrum.rs` 的**纯数据结构与纯方法**
//! **逐字搬迁**（不改结构、不改命名、不改算法），供谱计算层（Step0-3c-2）与 agent 感知域复用。
//!
//! ## 本批包含（纯）
//! - 常量: TAU谱分辨率/熵值阈值/熵流路径最大节点数/稳态基准学习率/基准最小样本数/谱熵归一化上限
//! - CSI 载体: `CsiComplex`（幅度/相位纯计算）、`CsiFrame`、`Esp32NodeType`（子载波数/载波频率/带宽/构造测试帧）
//! - τ 谱载体: `TauSpectrum`/`EntropyLevel`/`EntropyFlowPath`/`拓扑类型`/`TopologyFeature`/`TauDiagnosis`
//! - 稳态基准: `BaselineTauAnchor`（新建/更新基准 EMA/基准已建立）
//!
//! ## 本批追加（MIGRATION-FINAL Step1-②，纯计算下沉）
//! - `CsiFrame::触发τ谱诊断`（简化版，调 core `perception_spectrum::τ谱诊断`）
//! - `检测急停`（滑动窗口方差判据）
//!
//! ## 未下沉（分别处置）
//! - `CsiFrame::触发τ谱诊断_带sh_gnn` + `τ谱诊断_带sh_gnn`（依赖 engine `l1::sh_gnn::SHGNN层`，
//!   非纯——留 engine，sh_gnn 为冻结保留件）
//! - `csi矩阵_到_复向量分量`/`构造csi边列表`（依赖 engine `spectrum_matcher::复向量分量`/
//!   `sh_gnn::边`，随其类型下沉后接线）
//! - `计算τ谱`/`计算熵值水平`/`识别拓扑特征`/`计算熵流路径`/`τ谱诊断`（Step0-3c-2 谱计算层）
//! - `csi_to_sh_gnn_input`/`csi_to_tau_aware_input`/`检测急停`（随对应依赖处置）
//!
//! 锚点: 论文 §1.1（τ0 分布/熵值水平/熵流路径/辨拓扑）、论文 §4（五层稳态模型本征 τ0 与 Ω 锚）、
//! ARCH-1 v2.1 §4（ω/γ 独立）。core 五类约束: 无 IO/网络/异步/时间/随机。

#![allow(non_snake_case)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// 常量定义
// ============================================================================

/// τ谱分辨率（FFT点数）——对齐ESP32-S3的64子载波
pub const TAU谱分辨率: usize = 64;

/// 熵值阈值——超过此值判定为"偏离稳态"
pub const 熵值阈值: f64 = 0.3;

/// 熵流路径最大节点数
pub const 熵流路径最大节点数: usize = 8;

/// 稳态基准学习率（指数移动平均）
pub const 稳态基准学习率: f64 = 0.01;

/// 基准建立所需最小样本数
pub const 基准最小样本数: u64 = 10;

/// 谱熵归一化上限（ln(64)≈4.16）
pub(crate) const 谱熵归一化上限: f64 = 4.16;

// ============================================================================
// CSI 载体（下沉自 csi_data_format.rs:29-170，结构/方法纯）
// ============================================================================

/// CSI 复数——实部 + 虚部
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CsiComplex {
    /// 实部
    pub 实部: i16,
    /// 虚部
    pub 虚部: i16,
}

impl CsiComplex {
    /// 计算幅度（|z| = sqrt(re² + im²)）
    pub fn 幅度(&self) -> f64 {
        ((self.实部 as f64).powi(2) + (self.虚部 as f64).powi(2)).sqrt()
    }

    /// 计算相位（弧度，atan2(im, re)）
    pub fn 相位(&self) -> f64 {
        (self.虚部 as f64).atan2(self.实部 as f64)
    }
}

/// ESP32 节点类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Esp32NodeType {
    /// ESP32-S3: 2.4GHz, 20MHz 带宽, 64 子载波
    S3,
    /// ESP32-C6: 5GHz, 80MHz 带宽, 256 子载波
    C6,
}

impl Esp32NodeType {
    /// 默认子载波数
    pub fn 子载波数(&self) -> usize {
        match self {
            Esp32NodeType::S3 => 64,
            Esp32NodeType::C6 => 256,
        }
    }

    /// 默认载波频率（Hz）
    pub fn 载波频率(&self) -> u64 {
        match self {
            Esp32NodeType::S3 => 2_412_000_000, // 2.4GHz channel 1
            Esp32NodeType::C6 => 5_180_000_000, // 5GHz channel 36
        }
    }

    /// 默认带宽（Hz）
    pub fn 带宽(&self) -> u32 {
        match self {
            Esp32NodeType::S3 => 20_000_000, // 20MHz
            Esp32NodeType::C6 => 80_000_000, // 80MHz
        }
    }

    /// 构造默认测试 CSI 帧
    pub fn 构造测试帧(&self, 节点ID: &str, 时间戳: u64, 序列号: u32) -> CsiFrame {
        let 子载波数 = self.子载波数();
        CsiFrame {
            节点ID: 节点ID.to_string(),
            时间戳,
            序列号,
            csi矩阵: vec![CsiComplex { 实部: 100, 虚部: 50 }; 子载波数],
            载波频率: self.载波频率(),
            带宽: self.带宽(),
            采样率: 100,
            rssi: -45,
            噪声底: -92,
            天线对数: 1,
            τ谱诊断: None,
        }
    }
}

/// CSI 数据帧
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsiFrame {
    /// 节点 ID（如 "esp32-s3-01"）
    pub 节点ID: String,
    /// 时间戳（毫秒，Unix epoch）
    pub 时间戳: u64,
    /// 序列号（用于丢包检测）
    pub 序列号: u32,
    /// CSI 矩阵（子载波 × 天线对，复数）
    pub csi矩阵: Vec<CsiComplex>,
    /// 载波频率（Hz）—— 2.4G / 5G
    pub 载波频率: u64,
    /// 带宽（Hz）—— 20M / 80M
    pub 带宽: u32,
    /// 采样率（Hz）—— 100
    pub 采样率: u32,
    /// RSSI（信号强度 dBm）
    pub rssi: i16,
    /// 噪声底（dBm）
    pub 噪声底: i16,
    /// 天线对数量
    pub 天线对数: u8,
    /// V9方向2: τ谱诊断结果（首次接收到时计算并填充，向后兼容旧数据反序列化为None）
    #[serde(default)]
    pub τ谱诊断: Option<TauDiagnosis>,
}

impl CsiFrame {
    /// V9方向2: 触发τ谱诊断（如果尚未计算）
    ///
    /// [MIGRATION-FINAL Step1-②] 下沉自 engine `csi_data_format.rs`（纯计算：
    /// 判断未诊断 → 调 core `perception_spectrum::τ谱诊断` → 填充）。
    /// `触发τ谱诊断_带sh_gnn`（完整版）依赖 engine `sh_gnn`（冻结保留件），非纯、留 engine。
    pub fn 触发τ谱诊断(&mut self, 基准表: &HashMap<String, BaselineTauAnchor>) {
        if self.τ谱诊断.is_some() {
            return;
        }
        let 诊断 = crate::perception_spectrum::τ谱诊断(
            &self.csi矩阵,
            &self.节点ID,
            基准表,
            self.时间戳,
        );
        self.τ谱诊断 = Some(诊断);
    }
}

/// L0 急停阈值算法——基于 CSI 振幅突变检测
///
/// [MIGRATION-FINAL Step1-②] 下沉自 engine `csi_data_format.rs`（纯计算，算法零改写）。
/// 蓝图阶段1 验证目标: 遮挡/暗光下 30ms 内急停。
/// 算法: 滑动窗口内平均幅度方差 > 阈值 → 触发急停
///
/// 参数:
/// - 历史帧: 最近采集的 CSI 帧序列（按时间顺序）
/// - 窗口大小: 滑动窗口帧数（如 3 = 30ms @ 100Hz）
/// - 方差阈值: 触发急停的方差下限（经验值，如 100.0）
pub fn 检测急停(历史帧: &[CsiFrame], 窗口大小: usize, 方差阈值: f64) -> bool {
    if 历史帧.len() < 2 || 窗口大小 < 2 {
        return false;
    }

    // 1. 取最近 窗口大小 帧
    let 取帧数 = 窗口大小.min(历史帧.len());
    let 最近帧 = &历史帧[历史帧.len() - 取帧数..];

    // 2. 计算每帧的平均幅度
    let 平均幅度序列: Vec<f64> = 最近帧
        .iter()
        .map(|frame| {
            if frame.csi矩阵.is_empty() {
                0.0
            } else {
                frame.csi矩阵.iter().map(|c| c.幅度()).sum::<f64>() / frame.csi矩阵.len() as f64
            }
        })
        .collect();

    // 3. 计算方差
    let n = 平均幅度序列.len() as f64;
    if n < 2.0 {
        return false;
    }
    let 均值 = 平均幅度序列.iter().sum::<f64>() / n;
    let 方差 = 平均幅度序列
        .iter()
        .map(|&m| (m - 均值).powi(2))
        .sum::<f64>()
        / n;

    方差 > 方差阈值
}

// ============================================================================
// τ 谱载体（下沉自 tau_spectrum.rs:47-145，结构/方法纯）
// ============================================================================

/// τ谱——本征弛豫时间分布
///
/// 论文§1.1: "τ0分布回答系统有哪些本征节律，各占多大权重"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TauSpectrum {
    /// τ0峰值列表（单位：秒），按权重降序
    pub τ0峰值: Vec<f64>,
    /// 各峰值对应的权重（和为1）
    pub 权重: Vec<f64>,
    /// 谱熵（τ0分布的Shannon熵，反映节律多样性）
    pub 谱熵: f64,
    /// 计算时间戳
    pub 时间戳: u64,
}

/// 熵值水平——偏离稳态的程度（标量）
///
/// 论文§1.1: "熵值水平回答系统当前偏离稳态多远"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyLevel {
    /// 熵值 [0, 1]——0=稳态，1=完全失稳
    pub 熵值: f64,
    /// 是否偏离稳态
    pub 偏离稳态: bool,
    /// 主要偏离维度（"τ0分布"|"谱熵"|"幅度"|"基准未建立"|"无基准"）
    pub 偏离维度: String,
}

/// 熵流路径——熵扰动在节点间的传导方向
///
/// 论文§1.1: "熵流路径回答扰动从哪个节点来、向哪个节点去"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyFlowPath {
    /// 源节点ID（扰动发起点）
    pub 源节点: String,
    /// 目标节点ID列表（扰动传导方向）
    pub 目标节点: Vec<String>,
    /// 传导强度 [0, 1]
    pub 传导强度: f64,
    /// 传导延迟（毫秒）
    pub 传导延迟_ms: u64,
}

/// 拓扑类型——CSI矩阵的吸引子/连通性特征
///
/// 论文§1.1: "辨拓扑——CSI矩阵的吸引子/连通性特征"
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum 拓扑类型 {
    /// 层流——稳态期，平稳层流
    层流,
    /// 涡旋——涡旋期，探索新稳态
    涡旋,
    /// 混沌——失稳，熵值极高
    混沌,
}

/// 拓扑结构识别结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyFeature {
    /// 吸引子数量（系统稳态数量）
    pub 吸引子数: usize,
    /// 连通性 [0, 1]——节点间耦合强度
    pub 连通性: f64,
    /// 拓扑类型
    pub 拓扑类型: 拓扑类型,
}

/// 空间谱特征（单帧 CSI 子载波的空间频率结构）
///
/// [Step0-5 正名, PM 物理裁决 2026-08-29] 由原 `计算τ谱` 更名而来——单帧子载波
/// 相位是**空间序列**，其频率是**空间尺度非弛豫 τ**（铁原则: **τ 只来自时间衰减 γ，
/// 空间频率永不冒充 τ**）。本结构输出空间结构量，**不进 τ 主链/造波/熵泵**；
/// 时间弛豫 τ 由 `计算τ谱_复L`（多帧时间序列 → 复 EDMD → τ=1/γ）提供。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 空间谱特征 {
    /// 空间频率峰值列表（按权重降序）
    pub 空间频率: Vec<f64>,
    /// 各峰值对应权重（和为1）
    pub 权重: Vec<f64>,
    /// 空间谱熵（频率分布 Shannon 熵, 反映空间结构多样性）
    pub 空间熵: f64,
    /// 计算时间戳
    pub 时间戳: u64,
}

/// τ谱诊断结果——三问的完整答案
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TauDiagnosis {
    /// 时间弛豫 τ 谱（复 L 供 τ=1/γ；单帧无时间序列/数据不足时为空——
    /// [Step0-5] 空间频率不得冒充弛豫 τ）
    pub τ谱: TauSpectrum,
    /// [Step0-5] 空间谱特征（单帧子载波的空间频率结构, 非弛豫 τ）
    #[serde(default)]
    pub 空间谱: Option<空间谱特征>,
    pub 熵值水平: EntropyLevel,
    /// 单节点观测时为None，多节点聚合时才有
    pub 熵流路径: Option<EntropyFlowPath>,
    pub 拓扑特征: TopologyFeature,
    pub 诊断时间戳: u64,
}

// ============================================================================
// 稳态基准锚定（下沉自 tau_spectrum.rs:127-188，方法纯）
// ============================================================================

/// 稳态基准谱系——每节点的本征τ0锚
///
/// 论文§4: "五层稳态模型存储各层本征τ0与Ω锚"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineTauAnchor {
    pub 节点ID: String,
    /// 基准τ0峰值（学习得到，初始化为空）
    pub 基准τ0峰值: Vec<f64>,
    /// 基准权重
    pub 基准权重: Vec<f64>,
    /// 基准谱熵
    pub 基准谱熵: f64,
    /// 学习样本数
    pub 学习样本数: u64,
    /// 上次更新时间戳
    pub 上次更新: u64,
}

impl BaselineTauAnchor {
    pub fn 新建(节点ID: &str) -> Self {
        Self {
            节点ID: 节点ID.to_string(),
            基准τ0峰值: Vec::new(),
            基准权重: Vec::new(),
            基准谱熵: 0.0,
            学习样本数: 0,
            上次更新: 0,
        }
    }

    /// 用新观测更新基准（指数移动平均）
    pub fn 更新基准(&mut self, 观测τ谱: &TauSpectrum) {
        if self.基准τ0峰值.is_empty() {
            // 首次学习——直接采用
            self.基准τ0峰值 = 观测τ谱.τ0峰值.clone();
            self.基准权重 = 观测τ谱.权重.clone();
            self.基准谱熵 = 观测τ谱.谱熵;
        } else {
            // EMA更新——lr小则基准稳定，lr大则适应快
            let lr = 稳态基准学习率;
            self.基准谱熵 = (1.0 - lr) * self.基准谱熵 + lr * 观测τ谱.谱熵;
            // 峰值数量一致时才EMA，否则保留原基准（避免维度错配）
            if self.基准τ0峰值.len() == 观测τ谱.τ0峰值.len() {
                for (i, &τ0) in 观测τ谱.τ0峰值.iter().enumerate() {
                    self.基准τ0峰值[i] = (1.0 - lr) * self.基准τ0峰值[i] + lr * τ0;
                }
                for (i, &w) in 观测τ谱.权重.iter().enumerate() {
                    self.基准权重[i] = (1.0 - lr) * self.基准权重[i] + lr * w;
                }
            }
        }
        self.学习样本数 += 1;
        self.上次更新 = 观测τ谱.时间戳;
    }

    /// 判断基准是否已建立（至少学习10个样本）
    pub fn 基准已建立(&self) -> bool {
        self.学习样本数 >= 基准最小样本数
    }
}

// ============================================================================
// 测试（搬迁验证: 结构与纯方法与 engine 原实现一致）
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 测试_csi复数_幅度相位() {
        let z = CsiComplex { 实部: 3, 虚部: 4 };
        assert!((z.幅度() - 5.0).abs() < 1e-12, "|3+4i|=5");
        assert!((z.相位() - 0.9273).abs() < 1e-3, "atan2(4,3)");
        let 负 = CsiComplex { 实部: -3, 虚部: 0 };
        assert!((负.相位() - std::f64::consts::PI).abs() < 1e-12, "π");
    }

    #[test]
    fn 测试_esp32节点默认参数() {
        assert_eq!(Esp32NodeType::S3.子载波数(), 64);
        assert_eq!(Esp32NodeType::C6.子载波数(), 256);
        assert_eq!(Esp32NodeType::S3.载波频率(), 2_412_000_000);
        assert_eq!(Esp32NodeType::C6.带宽(), 80_000_000);
        let 帧 = Esp32NodeType::S3.构造测试帧("esp32-s3-01", 100, 1);
        assert_eq!(帧.节点ID, "esp32-s3-01");
        assert_eq!(帧.csi矩阵.len(), 64);
        assert_eq!(帧.采样率, 100);
        assert!(帧.τ谱诊断.is_none());
    }

    #[test]
    fn 测试_基准锚_更新与建立() {
        let mut 锚 = BaselineTauAnchor::新建("node-1");
        assert!(!锚.基准已建立(), "0 样本未建立");
        // 首次学习直接采用
        let 谱 = TauSpectrum { τ0峰值: vec![10.0, 20.0], 权重: vec![0.6, 0.4], 谱熵: 0.5, 时间戳: 1 };
        锚.更新基准(&谱);
        assert_eq!(锚.基准τ0峰值, vec![10.0, 20.0], "首次直接采用");
        // EMA 更新（lr=0.01）
        let 谱2 = TauSpectrum { τ0峰值: vec![11.0, 21.0], 权重: vec![0.7, 0.3], 谱熵: 0.6, 时间戳: 2 };
        锚.更新基准(&谱2);
        assert!((锚.基准谱熵 - 0.501).abs() < 1e-9, "EMA 熵: {}", 锚.基准谱熵);
        assert!((锚.基准τ0峰值[0] - 10.01).abs() < 1e-9, "EMA τ0[0]: {}", 锚.基准τ0峰值[0]);
        assert_eq!(锚.学习样本数, 2);
        assert_eq!(锚.上次更新, 2);
        // 维度错配时保留原基准
        let 谱3 = TauSpectrum { τ0峰值: vec![30.0], 权重: vec![1.0], 谱熵: 0.7, 时间戳: 3 };
        锚.更新基准(&谱3);
        assert_eq!(锚.基准τ0峰值.len(), 2, "维度错配保留原基准");
        // 10 样本后建立
        for i in 4..12 {
            let 谱i = TauSpectrum { τ0峰值: vec![10.0, 20.0], 权重: vec![0.6, 0.4], 谱熵: 0.5, 时间戳: i };
            锚.更新基准(&谱i);
        }
        assert!(锚.基准已建立(), "11 样本已建立");
    }

    #[test]
    fn 测试_载体序列化roundtrip() {
        let 帧 = Esp32NodeType::C6.构造测试帧("esp32-c6-01", 200, 7);
        let json = serde_json::to_string(&帧).unwrap();
        let 还原: CsiFrame = serde_json::from_str(&json).unwrap();
        assert_eq!(还原.节点ID, "esp32-c6-01");
        assert_eq!(还原.csi矩阵.len(), 256);
        assert!(还原.τ谱诊断.is_none());
        // τ 谱载体
        let 诊断 = TauDiagnosis {
            τ谱: TauSpectrum { τ0峰值: vec![10.0], 权重: vec![1.0], 谱熵: 0.2, 时间戳: 1 },
            空间谱: Some(空间谱特征 {
                空间频率: vec![2.0],
                权重: vec![1.0],
                空间熵: 0.2,
                时间戳: 1,
            }),
            熵值水平: EntropyLevel { 熵值: 0.1, 偏离稳态: false, 偏离维度: "无".to_string() },
            熵流路径: None,
            拓扑特征: TopologyFeature { 吸引子数: 1, 连通性: 0.5, 拓扑类型: 拓扑类型::层流 },
            诊断时间戳: 1,
        };
        let json2 = serde_json::to_string(&诊断).unwrap();
        let 还原2: TauDiagnosis = serde_json::from_str(&json2).unwrap();
        assert_eq!(还原2.拓扑特征.拓扑类型, 拓扑类型::层流);
    }

    #[test]
    fn 测试_常量对齐() {
        assert_eq!(TAU谱分辨率, 64);
        assert!((熵值阈值 - 0.3).abs() < 1e-12);
        assert_eq!(熵流路径最大节点数, 8);
        assert!((稳态基准学习率 - 0.01).abs() < 1e-12);
        assert_eq!(基准最小样本数, 10);
    }

    // ===== MIGRATION-FINAL Step1-② 下沉：触发τ谱诊断 + 检测急停 =====

    #[test]
    fn 测试_触发τ谱诊断_填充() {
        let mut 帧 = Esp32NodeType::S3.构造测试帧("esp32-s3-01", 100, 1);
        let 基准表 = HashMap::new();
        帧.触发τ谱诊断(&基准表);
        let 诊断 = 帧.τ谱诊断.expect("应填充τ谱诊断");
        // 无基准 → 熵值水平"无基准"，拓扑特征简化版吸引子数=1
        assert_eq!(诊断.拓扑特征.吸引子数, 1, "简化版吸引子数应=1");
        assert_eq!(诊断.熵值水平.偏离维度, "无基准");
    }

    #[test]
    fn 测试_触发τ谱诊断_幂等() {
        let mut 帧 = Esp32NodeType::S3.构造测试帧("esp32-s3-01", 100, 1);
        let 基准表 = HashMap::new();
        帧.触发τ谱诊断(&基准表);
        帧.触发τ谱诊断(&基准表);
        let 诊断 = 帧.τ谱诊断.expect("应有τ谱诊断");
        assert_eq!(诊断.诊断时间戳, 100);
    }

    #[test]
    fn 测试_检测急停_正常场景() {
        let 帧1 = Esp32NodeType::S3.构造测试帧("esp32-s3-01", 0, 0);
        let 帧2 = Esp32NodeType::S3.构造测试帧("esp32-s3-01", 1, 1);
        let 帧3 = Esp32NodeType::S3.构造测试帧("esp32-s3-01", 2, 2);
        assert!(!检测急停(&[帧1, 帧2, 帧3], 3, 100.0), "正常帧不应急停");
    }

    #[test]
    fn 测试_检测急停_突变场景() {
        let 帧1 = Esp32NodeType::S3.构造测试帧("esp32-s3-01", 0, 0);
        let 帧2 = Esp32NodeType::S3.构造测试帧("esp32-s3-01", 1, 1);
        // 第3帧振幅突变（实部/虚部 ×5）
        let mut 帧3 = Esp32NodeType::S3.构造测试帧("esp32-s3-01", 2, 2);
        for c in &mut 帧3.csi矩阵 {
            c.实部 = 500;
            c.虚部 = 250;
        }
        assert!(检测急停(&[帧1, 帧2, 帧3], 3, 100.0), "振幅突变应触发急停");
    }

    #[test]
    fn 测试_检测急停_帧数不足() {
        let 帧 = Esp32NodeType::S3.构造测试帧("esp32-s3-01", 0, 0);
        assert!(!检测急停(&[帧], 3, 100.0), "帧数不足不应急停");
    }
}
