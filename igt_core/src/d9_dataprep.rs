// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! D9 数据接入与预处理层（P22 §3/§4 工程化，2026-08-25 第421批）
//!
//! 依据: `IGT_数据接入与预处理层工程规范_v1.0.md`（D9）——P22 §3（数据接入层）/§4（预处理层）
//!
//! ## 核心内容
//!
//! 1. **TimeSeries 标准格式**（P22 §3.2）：所有输入统一转"实体-时间-变量"三元组
//! 2. **数据质量门槛**（P22 §3.3）：
//!    - 时间戳连续性：缺失率 <5% 线性插值；5%-20% 标记降质量；>20% 重采样/拒绝
//!    - 时间跨度 ≥ 3×最长τ（Cramér-Rao 充分条件：覆盖 95% 回归过程，τ 误差<10%）
//!    - 采样率 ≥ 2×最高目标频率（Nyquist）
//!    - 变量数 ≥ 10（P5 群论表示论：6 维直积下界 + 4 维冗余余量）
//! 3. **预处理保 τ**（P22 §4，IGT τ 第一原则）：任何预处理不得改变 τ 谱

#![allow(non_snake_case)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// §2 TimeSeries 标准格式（P22 §3.2）
// ============================================================================

/// 单个变量采样点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 变量 {
    pub 名称: String,
    pub 值: f64,
    /// 数据质量 0-1
    pub 质量: f64,
}

/// 标准时间序列（P22 §3.2 "实体-时间-变量"三元组）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    /// 系统ID（用户/公司/设备/区域）
    pub entity_id: String,
    /// 时间戳（统一到毫秒）
    pub timestamp: Vec<u64>,
    /// 变量名:值序列（每个时间戳一个）
    pub variables: Vec<Vec<变量>>,
    /// 领域/来源/单位/采样率
    pub metadata: HashMap<String, String>,
}

impl TimeSeries {
    /// 变量数（以第一个时间戳的变量数为准）
    pub fn 变量数(&self) -> usize {
        self.variables.first().map_or(0, |v| v.len())
    }

    /// 时间跨度（毫秒）
    pub fn 时间跨度_ms(&self) -> u64 {
        if self.timestamp.len() < 2 {
            return 0;
        }
        self.timestamp.last().unwrap().saturating_sub(self.timestamp[0])
    }

    /// 平均采样间隔（毫秒）
    pub fn 平均采样间隔_ms(&self) -> f64 {
        if self.timestamp.len() < 2 {
            return 0.0;
        }
        self.时间跨度_ms() as f64 / (self.timestamp.len() - 1) as f64
    }

    /// 数据缺失率（0-1）——用最小相邻间隔作为期望采样间隔
    /// 期望点数 = 时间跨度 / 期望间隔 + 1；缺失率 = 1 - 实际点数/期望点数
    pub fn 缺失率(&self) -> f64 {
        let n = self.timestamp.len();
        if n < 2 {
            return 1.0;
        }
        let span = self.时间跨度_ms();
        if span == 0 {
            return 1.0;
        }
        // 最小相邻间隔 = 期望采样间隔（缺失不改变期望间隔）
        let 期望间隔 = self
            .timestamp
            .windows(2)
            .map(|w| w[1].saturating_sub(w[0]))
            .filter(|d| *d > 0)
            .min()
            .unwrap_or(0);
        if 期望间隔 == 0 {
            return 1.0;
        }
        let 期望点数 = span as f64 / 期望间隔 as f64 + 1.0;
        (1.0 - n as f64 / 期望点数).max(0.0)
    }

    /// 提取指定变量的数值序列（不含缺失点）
    pub fn 变量序列(&self, 名称: &str) -> Vec<(u64, f64)> {
        self.timestamp
            .iter()
            .zip(self.variables.iter())
            .filter_map(|(t, vars)| {
                vars.iter()
                    .find(|v| v.名称 == 名称)
                    .map(|v| (*t, v.值))
            })
            .collect()
    }
}

// ============================================================================
// §3 数据质量门槛
// ============================================================================

/// 质量门槛评估结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 质量评估 {
    /// 缺失率检查：缺失率 <5% 可插值；5%-20% 降质量；>20% 拒绝
    pub 缺失率: f64,
    pub 缺失处理: 缺失处理,
    /// 时间跨度 ≥ 3×最长τ 检查
    pub 跨度充足: bool,
    /// 实际跨度 / (3×最长τ)
    pub 跨度比: f64,
    /// 采样率 ≥ 2×最高目标频率 检查（Nyquist）
    pub 采样充足: bool,
    /// 变量数 ≥ 10 检查（P5 群论表示论）
    pub 变量数: usize,
    pub 变量充足: bool,
    /// 综合是否通过（所有硬门槛）
    pub 通过: bool,
}

/// 缺失处理策略（D9 §3.1）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum 缺失处理 {
    /// <5%：可线性插值
    线性插值,
    /// 5%-20%：标记降质量
    降质量,
    /// >20%：重采样或拒绝
    拒绝,
}

/// 执行质量门槛评估（D9 §3）
///
/// - `最长τ_ms`: 估计的最长弛豫时间（毫秒）
/// - `最高频率_hz`: 最高目标频率
pub fn 评估质量(
    ts: &TimeSeries,
    最长τ_ms: f64,
    最高频率_hz: f64,
) -> 质量评估 {
    let 缺失率 = ts.缺失率();
    let 缺失处理 = if 缺失率 < 0.05 {
        缺失处理::线性插值
    } else if 缺失率 <= 0.20 {
        缺失处理::降质量
    } else {
        缺失处理::拒绝
    };

    let 跨度比 = if 最长τ_ms > 0.0 {
        ts.时间跨度_ms() as f64 / (3.0 * 最长τ_ms)
    } else {
        0.0
    };
    let 跨度充足 = 跨度比 >= 1.0;

    let 采样间隔_s = ts.平均采样间隔_ms() / 1000.0;
    let 采样充足 = 采样间隔_s > 0.0 && (1.0 / 采样间隔_s) >= 2.0 * 最高频率_hz;

    let 变量数 = ts.变量数();
    let 变量充足 = 变量数 >= 10;

    let 通过 = 缺失处理 != 缺失处理::拒绝 && 跨度充足 && 采样充足 && 变量充足;

    质量评估 {
        缺失率,
        缺失处理,
        跨度充足,
        跨度比,
        采样充足,
        变量数,
        变量充足,
        通过,
    }
}

// ============================================================================
// §4 预处理保 τ（P22 §4）
// ============================================================================

/// 预处理选项
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct 预处理配置 {
    /// 去趋势（中心化）——保 τ（常数偏移不改变衰减指数）
    pub 去中心化: bool,
    /// 归一化（标准差缩放）——保 τ（线性缩放不改变 τ）
    pub 归一化: bool,
    /// 3σ 离群点处理（标记不静默删除，D9 §3.2）
    pub 离群点标记: bool,
}

impl Default for 预处理配置 {
    fn default() -> Self {
        Self {
            去中心化: true,
            归一化: false,
            离群点标记: true,
        }
    }
}

/// 预处理结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 预处理结果 {
    /// 处理后的变量序列（保持时间戳对齐）
    pub 处理后: Vec<Vec<变量>>,
    /// 被标记的离群点（时间戳, 变量名, 原值, z值）——保留审计（D9 §3.2 不静默删除）
    pub 离群点: Vec<(u64, String, f64, f64)>,
    /// 是否保持 τ（去中心化/归一化均为线性变换，不改变弛豫指数）
    pub 保τ: bool,
    /// 说明
    pub 说明: String,
}

/// 执行预处理（P22 §4，保 τ 原则）
///
/// 去中心化（减均值）与归一化（除标准差）都是线性变换：x'(t)=ax(t)+b 不改变
/// 弛豫指数 τ（自相关 R(Δt) 不受仿射变换影响）——保 τ。
pub fn 预处理(ts: &TimeSeries, 配置: &预处理配置) -> 预处理结果 {
    // 按变量名分组统计（跨时间戳）
    let mut 变量统计: HashMap<String, (f64 /*sum*/, f64 /*sum2*/, usize)> = HashMap::new();
    for vars in &ts.variables {
        for v in vars {
            let e = 变量统计.entry(v.名称.clone()).or_insert((0.0, 0.0, 0));
            e.0 += v.值;
            e.1 += v.值 * v.值;
            e.2 += 1;
        }
    }
    let 均值: HashMap<String, f64> = 变量统计
        .iter()
        .map(|(k, (s, _, n))| (k.clone(), s / *n as f64))
        .collect();
    let 标准差: HashMap<String, f64> = 变量统计
        .iter()
        .map(|(k, (_, s2, n))| {
            let m = 均值[k];
            let var = (s2 / *n as f64 - m * m).max(0.0);
            (k.clone(), var.sqrt().max(1e-12))
        })
        .collect();

    let mut 离群点 = Vec::new();
    let mut 处理后 = Vec::with_capacity(ts.variables.len());

    for (t, vars) in ts.timestamp.iter().zip(ts.variables.iter()) {
        let mut 行 = Vec::with_capacity(vars.len());
        for v in vars {
            let m = 均值.get(&v.名称).copied().unwrap_or(0.0);
            let s = 标准差.get(&v.名称).copied().unwrap_or(1.0);
            let 去中心 = if 配置.去中心化 { v.值 - m } else { v.值 };
            let 归一 = if 配置.归一化 { 去中心 / s } else { 去中心 };
            // 3σ 离群点标记（D9 §3.2：不静默删除，保留审计）
            if 配置.离群点标记 {
                let z = (去中心 / s).abs();
                if z > 3.0 {
                    离群点.push((*t, v.名称.clone(), v.值, z));
                }
            }
            行.push(变量 {
                名称: v.名称.clone(),
                值: 归一,
                质量: v.质量,
            });
        }
        处理后.push(行);
    }

    let 保τ = true; // 线性变换不改变弛豫指数
    let 离群点数 = 离群点.len();
    预处理结果 {
        处理后,
        离群点,
        保τ,
        说明: format!(
            "去中心化={}, 归一化={}, 离群点标记={} 个 —— 线性变换保 τ（D9 §4）",
            配置.去中心化,
            配置.归一化,
            离群点数
        ),
    }
}

/// 时间跨度充分性检查：跨度 ≥ 3×最长τ（Cramér-Rao 充分条件，D9 §3.3）
pub fn 跨度检查(时间跨度_ms: u64, 最长τ_ms: f64) -> (bool, f64) {
    if 最长τ_ms <= 0.0 {
        return (false, 0.0);
    }
    let 比 = 时间跨度_ms as f64 / (3.0 * 最长τ_ms);
    (比 >= 1.0, 比)
}

/// 变量数充分性检查：≥10（P5 群论表示论，D9 §3.4）
pub fn 变量数检查(变量数: usize) -> bool {
    变量数 >= 10
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn 合成序列(n: usize, 缺失: &[usize]) -> TimeSeries {
        let mut timestamp = Vec::new();
        let mut variables = Vec::new();
        for i in 0..n {
            if 缺失.contains(&i) {
                continue;
            }
            timestamp.push(i as u64 * 100);
            let mut 行 = Vec::new();
            for v in 0..12 {
                行.push(变量 {
                    名称: format!("v{}", v),
                    值: (i as f64 * 0.1 + v as f64).sin(),
                    质量: 1.0,
                });
            }
            variables.push(行);
        }
        TimeSeries {
            entity_id: "test".to_string(),
            timestamp,
            variables,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn d9_timeSeries_格式与变量数() {
        let ts = 合成序列(100, &[]);
        assert_eq!(ts.变量数(), 12);
        assert!(ts.变量数() >= 10, "≥10 变量（P5 群论表示论）");
        assert_eq!(ts.时间跨度_ms(), 9900);
    }

    #[test]
    fn d9_缺失率_分类() {
        let ts_full = 合成序列(100, &[]);
        assert!(ts_full.缺失率() < 0.05, "完整序列缺失率≈0: {}", ts_full.缺失率());

        let ts_mid = 合成序列(100, &(5..15).collect::<Vec<_>>());
        let mid = 评估质量(&ts_mid, 3000.0, 1.0);
        assert!(mid.缺失率 > 0.05, "10% 缺失应>5%: {}", mid.缺失率);
        assert_eq!(mid.缺失处理, 缺失处理::降质量);
    }

    #[test]
    fn d9_跨度充分性_3tau() {
        // 100 点 × 100ms = 9900ms；3×τ=3000ms → 跨度充足
        let ts = 合成序列(100, &[]);
        let 评估 = 评估质量(&ts, 3000.0, 1.0);
        assert!(评估.跨度充足, "跨度比={:.2}", 评估.跨度比);
        assert!(评估.跨度比 >= 1.0, "9900/9000=1.1 ≥ 1.0");

        // 短序列 + 长 τ → 跨度不足
        let ts_short = 合成序列(20, &[]);
        let 评估2 = 评估质量(&ts_short, 3000.0, 1.0);
        assert!(!评估2.跨度充足, "1900ms < 3×3000ms");
    }

    #[test]
    fn d9_变量数门槛_10() {
        assert!(变量数检查(10));
        assert!(变量数检查(12));
        assert!(!变量数检查(6), "6 维仅理论下界（P5）");
    }

    #[test]
    fn d9_预处理_保τ() {
        let ts = 合成序列(100, &[]);
        let 结果 = 预处理(&ts, &预处理配置::default());
        assert!(结果.保τ, "线性变换不改变弛豫指数");
        assert_eq!(结果.处理后.len(), 100);
        // 去中心化后均值≈0
        let 均值0: f64 = 结果.处理后.iter().map(|r| r[0].值).sum::<f64>() / 100.0;
        assert!(均值0.abs() < 0.1, "去中心化后均值≈0: {}", 均值0);
    }

    #[test]
    fn d9_离群点_标记不删除() {
        // 注入一个 10σ 离群点
        let mut ts = 合成序列(50, &[]);
        ts.variables[25][0].值 = 100.0;
        let 结果 = 预处理(&ts, &预处理配置::default());
        assert!(!结果.离群点.is_empty(), "10σ 离群点应被标记");
        assert_eq!(结果.处理后.len(), 50, "标记不删除，长度不变");
    }

    #[test]
    fn d9_质量综合判定() {
        let ts = 合成序列(200, &[]);
        let 评估 = 评估质量(&ts, 5000.0, 0.5);
        assert!(评估.通过, "200点×100ms=19900ms ≥ 3×5000；缺失≈0；12 变量");
    }
}
