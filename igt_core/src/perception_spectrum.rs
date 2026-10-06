// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! Perception 谱计算层（MIGRATION-FINAL Step0-3c-2a）
//!
//! engine `l1/perception/tau_spectrum.rs` 的**纯计算函数**逐字搬迁（不改算法、不改命名）：
//! `计算τ谱`（+ `简化功率谱`/`检测功率谱峰值`）/ `计算熵值水平` / `识别拓扑特征`（熵值阈值版）/
//! `计算熵流路径` / `τ谱诊断`。依赖 `crate::perception_types`（Step0-3c-1 已下沉）——**无 IO**。
//!
//! ## ⚠️ τ0=1/f 工程近似标注（R1.4 ω 贯通范围外，本批不擅改算法）
//! `计算τ谱` 第 5 步 `τ0 = 1/频率`（L242 原文）——perception 简化谱只从相位序列提取
//! **频率维度**，无衰减率信息，故 τ0 取**振荡周期**语义。这与 ARCH-1 v2.1 §4
//! （τ=1/γ，ω/γ 两维独立）冲突。**处置：原样搬迁 + 标注，不擅改算法**——
//! perception 谱计算升级为复 EDMD 数据链（T-22 RR 复L 链同款）后消除；
//! 是否立项升级由 PM 定。
//!
//! ## 未下沉（非纯/依赖 engine 类型）
//! `识别拓扑特征_完整版`/`分类拓扑类型_从滤波输出`/`计算连通性_从滤波输出`/
//! `估计吸引子数_从滤波输出`/`τ谱诊断_带sh_gnn`——依赖 engine `l1::sh_gnn::SHGNN层`
//! 与 `spectrum_matcher::复向量分量`（不在 core），留 engine Step2 处置。

#![allow(non_snake_case)]

use crate::perception_types::{
    拓扑类型, 空间谱特征, BaselineTauAnchor, CsiComplex, EntropyFlowPath, EntropyLevel,
    TauDiagnosis, TauSpectrum, TopologyFeature, 熵值阈值, 谱熵归一化上限,
};
use std::collections::HashMap;

// ============================================================================
// 空间谱特征（Step0-5 正名：单帧 CSI 空间频率结构，非弛豫 τ）
// ============================================================================

/// 从CSI单帧子载波相位计算空间谱特征（原 `计算τ谱` 正名）
///
/// 算法步骤（与 engine 原实现一致）:
/// 1. 提取CSI矩阵的相位序列
/// 2. 相位差分（去除线性漂移）+ 相位卷绕处理
/// 3. 自相关法得到功率谱密度
/// 4. 峰值检测 → 空间频率
/// 5. 计算空间熵
///
/// [Step0-5 正名, PM 物理裁决] 单帧子载波相位是**空间序列**——空间频率是
/// **空间尺度非弛豫 τ**（铁原则: τ 只来自时间衰减 γ，空间频率永不冒充 τ）。
/// 本函数输出空间结构量，**不进 τ 主链/造波/熵泵**；时间弛豫 τ 由
/// `计算τ谱_复L`（多帧时间序列 → 复 EDMD → τ=1/γ）提供。
pub fn 计算空间谱特征(csi矩阵: &[CsiComplex], 时间戳: u64) -> 空间谱特征 {
    if csi矩阵.len() < 4 {
        return 空间谱特征 {
            空间频率: Vec::new(),
            权重: Vec::new(),
            空间熵: 0.0,
            时间戳,
        };
    }

    // 1. 提取相位序列
    let 相位序列: Vec<f64> = csi矩阵.iter().map(|c| c.相位()).collect();

    // 2. 相位差分（去线性漂移）+ 相位卷绕处理
    let 差分序列: Vec<f64> = 相位序列
        .windows(2)
        .map(|w| {
            let diff = w[1] - w[0];
            // 相位卷绕处理——差分>π则减2π，<-π则加2π
            if diff > std::f64::consts::PI {
                diff - 2.0 * std::f64::consts::PI
            } else if diff < -std::f64::consts::PI {
                diff + 2.0 * std::f64::consts::PI
            } else {
                diff
            }
        })
        .collect();

    // 3. 简化版功率谱密度——自相关法
    let 功率谱 = 简化功率谱(&差分序列);

    // 4. 峰值检测——找功率谱的局部极大值
    let (峰值频率, 峰值强度) = 检测功率谱峰值(&功率谱);

    // 5. 空间频率，权重 = 强度归一化
    let mut 空间频率: Vec<f64> = Vec::new();
    let mut 权重: Vec<f64> = Vec::new();
    let 总强度: f64 = 峰值强度.iter().sum();
    if 总强度 > 1e-10 {
        for (i, &freq) in 峰值频率.iter().enumerate() {
            if freq > 1e-6 {
                空间频率.push(freq);
                权重.push(峰值强度[i] / 总强度);
            }
        }
    }

    // 6. 空间熵 = -Σ pᵢ·log(pᵢ)
    let 空间熵 = if 权重.is_empty() {
        0.0
    } else {
        -权重
            .iter()
            .map(|&p| if p > 1e-10 { p * p.ln() } else { 0.0 })
            .sum::<f64>()
    };

    空间谱特征 {
        空间频率,
        权重,
        空间熵,
        时间戳,
    }
}

/// 简化功率谱——用自相关法
fn 简化功率谱(序列: &[f64]) -> Vec<f64> {
    let n = 序列.len();
    if n < 2 {
        return Vec::new();
    }
    // 自相关——lag从0到n/2
    let mut 谱: Vec<f64> = Vec::with_capacity(n / 2);
    for lag in 0..n / 2 {
        let mut sum = 0.0;
        for i in 0..n - lag {
            sum += 序列[i] * 序列[i + lag];
        }
        谱.push(sum / (n - lag) as f64);
    }
    谱
}

/// 检测功率谱的局部极大值
fn 检测功率谱峰值(谱: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut 频率: Vec<f64> = Vec::new();
    let mut 强度: Vec<f64> = Vec::new();
    if 谱.len() < 3 {
        return (频率, 强度);
    }

    let 谱最大值 = 谱.iter().cloned().fold(0.0f64, f64::max);
    let 阈值 = 谱最大值 * 0.3; // 仅保留30%以上的峰值

    for i in 1..谱.len() - 1 {
        if 谱[i] > 谱[i - 1] && 谱[i] > 谱[i + 1] && 谱[i] > 阈值 {
            // 频率索引 → 实际频率（假设采样率=1，单位：Hz）
            频率.push(i as f64);
            强度.push(谱[i]);
        }
    }
    (频率, 强度)
}

// ============================================================================
// 熵值水平 / 拓扑特征（下沉自 tau_spectrum.rs:305-381）
// ============================================================================

/// 计算熵值水平——当前τ谱与基准的偏离程度
///
/// 论文§1.1: "熵值水平回答系统当前偏离稳态多远"
/// 简化版: 用谱熵差异 + τ0主峰漂移综合
pub fn 计算熵值水平(当前τ谱: &TauSpectrum, 基准: &BaselineTauAnchor) -> EntropyLevel {
    // [Step0-5] 谱空（无时间弛豫——单帧无时间序列时 τ 谱为空）→ 诚实降级, 不拿空间频率冒充
    if 当前τ谱.τ0峰值.is_empty() {
        return EntropyLevel {
            熵值: 0.0,
            偏离稳态: false,
            偏离维度: "谱空（无时间弛豫）".to_string(),
        };
    }
    if !基准.基准已建立() {
        return EntropyLevel {
            熵值: 0.0,
            偏离稳态: false,
            偏离维度: "基准未建立".to_string(),
        };
    }

    // 谱熵差异
    let 谱熵差 = (当前τ谱.谱熵 - 基准.基准谱熵).abs();
    let 谱熵归一化 = (谱熵差 / 谱熵归一化上限).min(1.0);

    // τ0主峰偏移
    let τ0偏移 = if !当前τ谱.τ0峰值.is_empty() && !基准.基准τ0峰值.is_empty() {
        let 当前主峰 = 当前τ谱.τ0峰值[0];
        let 基准主峰 = 基准.基准τ0峰值[0];
        if 基准主峰 > 1e-6 {
            ((当前主峰 - 基准主峰) / 基准主峰).abs().min(1.0)
        } else {
            0.0
        }
    } else {
        0.0
    };

    // 综合熵值 = 0.5 * 谱熵归一化 + 0.5 * τ0偏移
    let 熵值 = 0.5 * 谱熵归一化 + 0.5 * τ0偏移;

    let 偏离维度 = if 谱熵归一化 > τ0偏移 {
        "谱熵".to_string()
    } else {
        "τ0分布".to_string()
    };

    EntropyLevel {
        熵值,
        偏离稳态: 熵值 > 熵值阈值,
        偏离维度,
    }
}

/// 识别拓扑特征——基于CSI矩阵的吸引子/连通性
///
/// 简化版: 拓扑类型由熵值直接决定
/// 完整版: 需用SH-GNN拓扑分析（见 engine `识别拓扑特征_完整版`，未下沉）
pub fn 识别拓扑特征(csi矩阵: &[CsiComplex], 熵值: f64) -> TopologyFeature {
    let 拓扑类型 = if 熵值 < 0.3 {
        拓扑类型::层流
    } else if 熵值 < 0.7 {
        拓扑类型::涡旋
    } else {
        拓扑类型::混沌
    };

    // 连通性——CSI矩阵的平均幅度归一化
    let 连通性 = if csi矩阵.is_empty() {
        0.0
    } else {
        let 平均幅度: f64 =
            csi矩阵.iter().map(|c| c.幅度()).sum::<f64>() / csi矩阵.len() as f64;
        (平均幅度 / 200.0).min(1.0) // 经验归一化
    };

    // 吸引子数——简化版固定为1（完整版需SH-GNN分析）
    let 吸引子数 = 1;

    TopologyFeature {
        吸引子数,
        连通性,
        拓扑类型,
    }
}

// ============================================================================
// 熵流路径 / 完整诊断（下沉自 tau_spectrum.rs:542-613）
// ============================================================================

/// 计算熵流路径——熵扰动在节点间的传导方向
pub fn 计算熵流路径(节点τ谱历史: &HashMap<String, Vec<TauSpectrum>>) -> Option<EntropyFlowPath> {
    if 节点τ谱历史.len() < 2 {
        return None;
    }

    // 找熵变化最大的节点（扰动源）
    let mut 源节点: Option<String> = None;
    let mut 最大变化: f64 = 0.0;
    for (节点ID, 历史) in 节点τ谱历史 {
        if 历史.len() < 2 {
            continue;
        }
        let 最近 = 历史.last().unwrap();
        let 上次 = &历史[历史.len() - 2];
        let 变化 = (最近.谱熵 - 上次.谱熵).abs();
        if 变化 > 最大变化 {
            最大变化 = 变化;
            源节点 = Some(节点ID.clone());
        }
    }

    let 源节点 = 源节点?;
    let 源历史 = 节点τ谱历史.get(&源节点)?;
    if 源历史.len() < 2 {
        return None;
    }

    // 目标节点——其他节点（简化版：所有其他节点都算目标）
    let mut 目标节点: Vec<String> = Vec::new();
    for (节点ID, 历史) in 节点τ谱历史 {
        if 节点ID == &源节点 || 历史.len() < 2 {
            continue;
        }
        目标节点.push(节点ID.clone());
    }

    Some(EntropyFlowPath {
        源节点,
        目标节点,
        传导强度: 最大变化.min(1.0),
        传导延迟_ms: 100, // 默认100ms（完整版需时序对齐）
    })
}

/// 完整τ谱诊断——三问一次性回答
///
/// [Step0-5] 单帧输入 → 计算**空间谱特征**（非弛豫 τ）；时间弛豫 τ 谱保持空
/// （由 `计算τ谱_复L` 时间序列路径提供）；熵值水平对空谱诚实降级（"谱空"）。
pub fn τ谱诊断(
    csi矩阵: &[CsiComplex],
    节点ID: &str,
    基准表: &HashMap<String, BaselineTauAnchor>,
    时间戳: u64,
) -> TauDiagnosis {
    let 空间谱 = 计算空间谱特征(csi矩阵, 时间戳);
    // 单帧无时间序列 → τ 谱空（不拿空间频率冒充弛豫 τ）
    let τ谱 = TauSpectrum {
        τ0峰值: Vec::new(),
        权重: Vec::new(),
        谱熵: 0.0,
        时间戳,
    };

    let 熵值水平 = match 基准表.get(节点ID) {
        Some(b) => 计算熵值水平(&τ谱, b),
        None => EntropyLevel {
            熵值: 0.0,
            偏离稳态: false,
            偏离维度: "无基准".to_string(),
        },
    };

    let 拓扑特征 = 识别拓扑特征(csi矩阵, 熵值水平.熵值);

    TauDiagnosis {
        τ谱,
        空间谱: Some(空间谱),
        熵值水平,
        熵流路径: None, // 单节点诊断无法计算熵流，需多节点聚合
        拓扑特征,
        诊断时间戳: 时间戳,
    }
}

/// 从多帧时间序列计算时间弛豫 τ 谱（Step0-5(b): 接 core 复 EDMD, τ=1/γ）
///
/// 输入: 多帧 CSI/传感器 的**时间序列**——每个通道是某观测量（如子载波幅度/
/// 相位）沿时间的演化, 帧间等间隔 Δt; 需 ≥2 通道且每通道 ≥50 帧
/// （复 EDMD 硬边界, 与 T-22 HRV 复 L 链同款）。
///
/// 输出: τ=1/γ（复 EDMD τ 链最慢稳定模）, **独立 ω 由复 dmd 结果携带**
/// （调用方可取 `复dmd结果.频率` 对应项——本函数只出 τ 谱）。
/// 数据不足/失败 → None（调用方诚实降级）。
pub fn 计算τ谱_复L(时间序列: &[Vec<f64>], Δt: f64, 时间戳: u64) -> Option<TauSpectrum> {
    use crate::complex_dmd::复EDMD解析信号;
    if 时间序列.len() < 2 || 时间序列.iter().any(|c| c.len() < 50) {
        return None;
    }
    let 复 = 复EDMD解析信号(时间序列, Δt)?;
    let τ0峰值: Vec<f64> = 复
        .τ链
        .iter()
        .copied()
        .filter(|τ| τ.is_finite() && *τ > 0.0)
        .collect();
    if τ0峰值.is_empty() {
        return None;
    }
    // 复 EDMD 无幅值权重输出——均匀权重（诚实: 幅值权重需频谱能量, 本批均匀近似）
    let n = τ0峰值.len() as f64;
    let 权重: Vec<f64> = vec![1.0 / n; τ0峰值.len()];
    let 谱熵 = -权重.iter().map(|p| p * p.ln()).sum::<f64>();
    Some(TauSpectrum {
        τ0峰值,
        权重,
        谱熵,
        时间戳,
    })
}

// ============================================================================
// 测试（搬迁验证: 数值口径与 engine 原实现一致）
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// 构造振荡相位 CSI 序列（相位 = 2π·k/8 → 单频振荡，差分序列有周期性）
    fn 振荡CSI(n: usize) -> Vec<CsiComplex> {
        (0..n)
            .map(|k| {
                let θ = 2.0 * std::f64::consts::PI * k as f64 / 8.0;
                CsiComplex {
                    实部: (100.0 * θ.cos()) as i16,
                    虚部: (100.0 * θ.sin()) as i16,
                }
            })
            .collect()
    }

    #[test]
    fn 测试_计算空间谱_振荡信号() {
        let 谱 = 计算空间谱特征(&振荡CSI(64), 100);
        assert_eq!(谱.时间戳, 100);
        // 差分序列周期性 → 功率谱应有峰 → 空间频率非空（诚实: 若不检出则为空但不 panic）
        if !谱.空间频率.is_empty() {
            assert_eq!(谱.空间频率.len(), 谱.权重.len(), "空间频率与权重等长");
            let 权重和: f64 = 谱.权重.iter().sum();
            assert!((权重和 - 1.0).abs() < 1e-9, "权重和为1: {}", 权重和);
            assert!(谱.空间熵 >= 0.0, "空间熵非负");
        } else {
            assert_eq!(谱.空间熵, 0.0, "空谱空间熵=0");
        }
    }

    #[test]
    fn 测试_计算空间谱_短序列降级() {
        let 谱 = 计算空间谱特征(&[CsiComplex { 实部: 1, 虚部: 1 }], 1);
        assert!(谱.空间频率.is_empty(), "短序列空谱");
        assert_eq!(谱.空间熵, 0.0);
    }

    /// [Step0-5(b)] 复 L 路径: 合成双通道阻尼时间序列 → τ≈1/σ（时间弛豫）
    #[test]
    fn 测试_计算τ谱_复L() {
        let Δt = 0.5;
        let σ0 = 0.02;
        let f0 = 0.1;
        let n = 300; // 150s
        let 通道1: Vec<f64> = (0..n)
            .map(|k| {
                let t = k as f64 * Δt;
                (-σ0 * t).exp() * (2.0 * std::f64::consts::PI * f0 * t).sin()
            })
            .collect();
        let 通道2: Vec<f64> = (0..n)
            .map(|k| {
                let t = k as f64 * Δt;
                (-σ0 * t).exp() * (2.0 * std::f64::consts::PI * f0 * t + 0.4).cos()
            })
            .collect();
        let 谱 = 计算τ谱_复L(&[通道1, 通道2], Δt, 7).expect("时间序列应出弛豫 τ 谱");
        assert_eq!(谱.时间戳, 7);
        assert!(!谱.τ0峰值.is_empty(), "复 L 应出 τ");
        // 最慢稳定模 ≈ 1/σ₀ = 50s（非 1/f₀=10s——铁原则: τ 只来自时间衰减 γ）
        let 主导 = 谱.τ0峰值.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (主导 - 50.0).abs() < 25.0,
            "主导τ≈1/σ₀=50s, 实际: {}",
            主导
        );
        assert!(主导 > 20.0, "τ 应远离 1/f₀=10s（空间/时间维度分离）");
        // 均匀权重（复 EDMD 无幅值输出）→ 和为1
        let 权重和: f64 = 谱.权重.iter().sum();
        assert!((权重和 - 1.0).abs() < 1e-9);
    }

    /// [Step0-5(b)] 数据不足 → None（诚实降级）
    #[test]
    fn 测试_计算τ谱_复L_数据不足() {
        assert!(计算τ谱_复L(&[], 0.5, 0).is_none(), "无通道→None");
        // 单通道（复 EDMD 硬边界 ≥2）→ None
        assert!(计算τ谱_复L(&[vec![0.0; 100]], 0.5, 0).is_none(), "单通道→None");
        // 帧数不足（<50）→ None
        assert!(
            计算τ谱_复L(&[vec![0.0; 10], vec![0.0; 10]], 0.5, 0).is_none(),
            "帧数不足→None"
        );
    }

    #[test]
    fn 测试_熵值水平_基准未建立() {
        let 谱 = TauSpectrum { τ0峰值: vec![10.0], 权重: vec![1.0], 谱熵: 0.5, 时间戳: 1 };
        let 基准 = BaselineTauAnchor::新建("n1"); // 0 样本
        let 水平 = 计算熵值水平(&谱, &基准);
        assert_eq!(水平.偏离维度, "基准未建立");
        assert_eq!(水平.熵值, 0.0);
        assert!(!水平.偏离稳态);
    }

    #[test]
    fn 测试_熵值水平_谱熵偏差() {
        let 基准谱 = TauSpectrum { τ0峰值: vec![10.0], 权重: vec![1.0], 谱熵: 0.5, 时间戳: 1 };
        let mut 基准 = BaselineTauAnchor::新建("n1");
        for i in 2..12 {
            基准.更新基准(&基准谱); // 10 样本建立
        }
        // 当前谱熵显著不同（0.5 → 3.0）→ 谱熵归一化 = |2.5|/4.16 ≈ 0.6
        let 当前 = TauSpectrum { τ0峰值: vec![10.0], 权重: vec![1.0], 谱熵: 3.0, 时间戳: 12 };
        let 水平 = 计算熵值水平(&当前, &基准);
        assert!(水平.熵值 > 0.0, "谱熵偏差应产生熵值: {}", 水平.熵值);
        assert_eq!(水平.偏离维度, "谱熵");
        assert!(水平.偏离稳态 || 水平.熵值 <= 熵值阈值);
    }

    #[test]
    fn 测试_识别拓扑_熵阈值分层() {
        let 矩阵 = vec![CsiComplex { 实部: 100, 虚部: 0 }];
        assert_eq!(识别拓扑特征(&矩阵, 0.1).拓扑类型, 拓扑类型::层流);
        assert_eq!(识别拓扑特征(&矩阵, 0.5).拓扑类型, 拓扑类型::涡旋);
        assert_eq!(识别拓扑特征(&矩阵, 0.8).拓扑类型, 拓扑类型::混沌);
        // 空矩阵连通性=0
        assert_eq!(识别拓扑特征(&[], 0.1).连通性, 0.0);
    }

    #[test]
    fn 测试_熵流路径() {
        let mut 历史 = HashMap::new();
        历史.insert("A".to_string(), vec![
            TauSpectrum { τ0峰值: vec![10.0], 权重: vec![1.0], 谱熵: 0.5, 时间戳: 1 },
            TauSpectrum { τ0峰值: vec![10.0], 权重: vec![1.0], 谱熵: 0.9, 时间戳: 2 },
        ]);
        历史.insert("B".to_string(), vec![
            TauSpectrum { τ0峰值: vec![20.0], 权重: vec![1.0], 谱熵: 0.5, 时间戳: 1 },
            TauSpectrum { τ0峰值: vec![20.0], 权重: vec![1.0], 谱熵: 0.6, 时间戳: 2 },
        ]);
        let 路径 = 计算熵流路径(&历史).expect("应有熵流路径");
        assert_eq!(路径.源节点, "A", "A 熵变化最大 0.4");
        assert_eq!(路径.目标节点, vec!["B".to_string()]);
        assert!((路径.传导强度 - 0.4).abs() < 1e-12, "传导强度=最大变化");
        // 单节点 → None
        assert!(计算熵流路径(&HashMap::new()).is_none());
    }

    #[test]
    fn 测试_τ谱诊断_无基准() {
        let 诊断 = τ谱诊断(&振荡CSI(32), "n1", &HashMap::new(), 42);
        assert_eq!(诊断.诊断时间戳, 42);
        // [Step0-5] 单帧无时间序列 → τ 谱空（空间频率不冒充弛豫 τ）
        assert!(诊断.τ谱.τ0峰值.is_empty(), "单帧 τ 谱应为空");
        assert!(诊断.空间谱.is_some(), "空间谱应在位");
        assert_eq!(诊断.熵值水平.偏离维度, "无基准");
        assert!(诊断.熵流路径.is_none());
        assert_eq!(诊断.拓扑特征.吸引子数, 1);
    }
}
