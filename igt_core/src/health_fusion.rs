// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! 双判据/三判据融合预警（MIGRATION-FINAL flutter 归一第三例）
//!
//! 从 flutter `lib/services/health_fusion.dart`（第292批 Phase 2 + 第318/362批）迁移——**移动不改算法**：
//! - 图健康度（三层联通/耗散逆元平衡/结构完整）+ 身体尺度（基准椭圆马氏距离）+ 感知通道（HRV/PPG/运动）
//! - 融合权重在线学习（第318批：用户反馈 → 梯度式权重更新，clamp [0.3, 0.7]）
//! - 三判据融合（第362批：图/身体按 w 分配 0.8，感知通道固定 0.2）
//!
//! 理论锚点: P22 §13 四指标 + P19 结构权重 0.6。

use serde::{Deserialize, Serialize};

/// 图节点最小接口（layer 分层 + gamma 耗散率）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 图节点 {
    pub layer: String,
    pub gamma: f64,
}

/// 图健康度（结构判据，P22 §13）
///
/// T-182（理论侧163拆分线 §2）：core 只出**中性等级 code + 数值**；面向用户的中文
/// 等级与成句文案归 agent（`igt_agent::presentation`）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 图健康度 {
    pub three_band_connected: bool,
    pub inverse_balance: bool,
    pub structure_complete: bool,
    pub node_count: usize,
    pub red_count: usize,
    pub layer_count: usize,
    pub score: f64,
    /// 中性等级 code: `"green"` / `"yellow"` / `"red"`
    pub level: String,
}

/// 从节点列表计算图健康度（P22 §13 四指标→图健康度）
pub fn 图健康度从节点(nodes: &[图节点]) -> 图健康度 {
    if nodes.is_empty() {
        return 图健康度 {
            three_band_connected: false,
            inverse_balance: false,
            structure_complete: false,
            node_count: 0,
            red_count: 0,
            layer_count: 0,
            score: 0.0,
            level: "red".to_string(),
        };
    }
    let mut layers: Vec<String> = Vec::new();
    let mut red = 0usize;
    for n in nodes {
        if !layers.contains(&n.layer) {
            layers.push(n.layer.clone());
        }
        if n.gamma > 1e-9 {
            red += 1;
        }
    }
    let three_band = layers.contains(&"core快模".to_string())
        && layers.contains(&"agent中速".to_string())
        && layers.contains(&"engine慢模".to_string());
    let inverse = red == 0;
    let structure = !nodes.is_empty();
    // 分数: 三层 0.4 + 逆元 0.4 + 结构 0.2
    let score = (if three_band { 0.4 } else { 0.0 })
        + (if inverse { 0.4 } else { 0.0 })
        + (if structure { 0.2 } else { 0.0 });
    let level = if score >= 0.8 {
        "green"
    } else if score >= 0.5 {
        "yellow"
    } else {
        "red"
    }
    .to_string();
    图健康度 {
        three_band_connected: three_band,
        inverse_balance: inverse,
        structure_complete: structure,
        node_count: nodes.len(),
        red_count: red,
        layer_count: layers.len(),
        score,
        level,
    }
}

/// 身体判据级别分数
///
/// T-182：core 只出**中性等级 code + 分数**；面向用户的成句 message 归 agent
/// （`igt_agent::presentation::身体判据文案`）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 身体判据分数 {
    /// 中性等级 code: `"green"` / `"yellow"` / `"red"` / `"gray"`
    pub level: String,
    pub score: f64,
}

/// 身体判据级别 → 分数（green 1.0 / yellow 0.5 / red 0.0 / 其余 gray 0.5 中性）
pub fn 身体判据分数(level: &str) -> 身体判据分数 {
    match level {
        "green" => 身体判据分数 {
            level: "green".to_string(),
            score: 1.0,
        },
        "yellow" => 身体判据分数 {
            level: "yellow".to_string(),
            score: 0.5,
        },
        "red" => 身体判据分数 {
            level: "red".to_string(),
            score: 0.0,
        },
        _ => 身体判据分数 {
            level: "gray".to_string(),
            score: 0.5,
        },
    }
}

/// 融合权重（第318批 在线学习）——图判据权重 w∈[0.3, 0.7], 椭圆权重=1-w
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 融合权重 {
    pub w_graph: f64,
}

impl Default for 融合权重 {
    fn default() -> Self {
        Self { w_graph: 0.6 }
    }
}

impl 融合权重 {
    pub const MIN_W: f64 = 0.3;
    pub const MAX_W: f64 = 0.7;
    const LR: f64 = 0.05;

    /// 在线学习（梯度式）: 基于用户反馈调整主导判据权重（移动不改算法）
    pub fn 学习(&mut self, graph_score: f64, body_score: f64, correct: bool) {
        let graph_dominant = graph_score >= body_score;
        let step = if correct { Self::LR / 2.0 } else { Self::LR };
        if graph_dominant {
            self.w_graph = if correct {
                (self.w_graph + step).clamp(Self::MIN_W, Self::MAX_W)
            } else {
                (self.w_graph - step).clamp(Self::MIN_W, Self::MAX_W)
            };
        } else {
            self.w_graph = if correct {
                (self.w_graph - step).clamp(Self::MIN_W, Self::MAX_W)
            } else {
                (self.w_graph + step).clamp(Self::MIN_W, Self::MAX_W)
            };
        }
    }
}

/// 双判据融合预警
///
/// T-182：core 只出**中性等级 code + 数值**；成句 summary 归 agent
/// （`igt_agent::presentation::双判据融合文案`）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 融合预警 {
    pub graph_score: f64,
    pub ellipse_score: f64,
    pub fused_score: f64,
    pub w_graph: f64,
    /// 中性等级 code: `"green"` / `"yellow"` / `"red"`
    pub level: String,
}

/// 双判据融合（0.6 图 + 0.4 椭圆; 可传学习权重）
pub fn 融合(graph: &图健康度, body: &身体判据分数, w_graph: f64) -> 融合预警 {
    let g = graph.score;
    let e = body.score;
    let fused = w_graph * g + (1.0 - w_graph) * e;
    let level = if fused >= 0.8 {
        "green"
    } else if fused >= 0.5 {
        "yellow"
    } else {
        "red"
    }
    .to_string();
    融合预警 {
        graph_score: g,
        ellipse_score: e,
        fused_score: fused,
        w_graph,
        level,
    }
}

/// 感知通道判据（第362批: HRV 谱分解 + PPG RR + 运动通道完整性）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 感知通道分数 {
    pub hrv_active: bool,
    pub ppg_active: bool,
    pub motion_active: bool,
}

impl 感知通道分数 {
    /// 通道完整性 0..1（HRV 0.4 + PPG 0.3 + 运动 0.3）
    pub fn score(&self) -> f64 {
        let mut s = 0.0;
        if self.hrv_active {
            s += 0.4;
        }
        if self.ppg_active {
            s += 0.3;
        }
        if self.motion_active {
            s += 0.3;
        }
        s
    }

    /// 中性等级 code: `"green"` / `"yellow"` / `"gray"`（中文等级与成句文案归 agent）
    pub fn level(&self) -> String {
        let s = self.score();
        if s >= 0.7 {
            "green".to_string()
        } else if s >= 0.4 {
            "yellow".to_string()
        } else {
            "gray".to_string()
        }
    }
}

/// 三判据融合结果（第362批）
///
/// T-182：core 只出**中性等级 code + 数值**；成句 summary 归 agent
/// （`igt_agent::presentation::三判据融合文案`）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 三判据融合结果 {
    pub graph_score: f64,
    pub ellipse_score: f64,
    pub sensor_score: f64,
    pub fused_score: f64,
    pub w_graph: f64,
    /// 中性等级 code: `"green"` / `"yellow"` / `"red"`
    pub level: String,
}

/// 三判据融合（图/身体按 w 分配 0.8, 感知通道固定 0.2）
pub fn 三判据融合(
    graph: &图健康度,
    body: &身体判据分数,
    sensor: &感知通道分数,
    w_graph: f64,
) -> 三判据融合结果 {
    let g = graph.score;
    let e = body.score;
    let s = sensor.score();
    let fused = 0.8 * (w_graph * g + (1.0 - w_graph) * e) + 0.2 * s;
    let level = if fused >= 0.8 {
        "green"
    } else if fused >= 0.5 {
        "yellow"
    } else {
        "red"
    }
    .to_string();
    三判据融合结果 {
        graph_score: g,
        ellipse_score: e,
        sensor_score: s,
        fused_score: fused,
        w_graph,
        level,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn 节点(layer: &str, gamma: f64) -> 图节点 {
        图节点 {
            layer: layer.to_string(),
            gamma,
        }
    }

    #[test]
    fn test_空节点_红() {
        // T-182: level 为中性 code（中文等级/文案归 agent presentation）
        let h = 图健康度从节点(&[]);
        assert_eq!(h.level, "red");
        assert_eq!(h.score, 0.0);
    }

    #[test]
    fn test_三层联通_绿() {
        let nodes = vec![
            节点("core快模", 0.0),
            节点("agent中速", 0.0),
            节点("engine慢模", 0.0),
        ];
        let h = 图健康度从节点(&nodes);
        assert!(h.three_band_connected);
        assert!(h.inverse_balance);
        assert_eq!(h.level, "green");
        assert!((h.score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_有红节点_逆元失稳() {
        let nodes = vec![节点("core快模", 0.5), 节点("agent中速", 0.0), 节点("engine慢模", 0.0)];
        let h = 图健康度从节点(&nodes);
        assert!(!h.inverse_balance);
        assert_eq!(h.red_count, 1);
        assert_eq!(h.level, "yellow");
    }

    #[test]
    fn test_融合_权重() {
        let graph = 图健康度从节点(&[
            节点("core快模", 0.0),
            节点("agent中速", 0.0),
            节点("engine慢模", 0.0),
        ]);
        let body = 身体判据分数("green");
        let f = 融合(&graph, &body, 0.6);
        assert_eq!(f.level, "green");
        assert!((f.fused_score - 1.0).abs() < 1e-9, "全绿融合=1: {}", f.fused_score);
    }

    #[test]
    fn test_权重学习_误报降图() {
        let mut w = 融合权重::default();
        w.学习(1.0, 0.0, false); // 图主导 + 误报 → 降图权重
        assert!(w.w_graph < 0.6, "误报应降图权重: {}", w.w_graph);
        assert!(w.w_graph >= 0.3, "clamp 下限: {}", w.w_graph);
    }

    #[test]
    fn test_三判据融合() {
        let graph = 图健康度从节点(&[节点("core快模", 0.0)]);
        let body = 身体判据分数("green");
        let sensor = 感知通道分数 {
            hrv_active: true,
            ppg_active: true,
            motion_active: true,
        };
        let f = 三判据融合(&graph, &body, &sensor, 0.6);
        assert!(
            (f.sensor_score - 1.0).abs() < 1e-9,
            "全通道感知=1: {}",
            f.sensor_score
        );
        assert_eq!(f.level, "green");
    }

    #[test]
    fn test_感知通道分数() {
        let s = 感知通道分数 {
            hrv_active: true,
            ppg_active: false,
            motion_active: true,
        };
        assert!((s.score() - 0.7).abs() < 1e-9);
        assert_eq!(s.level(), "green");
    }

    #[test]
    fn test_身体判据分数_中性code() {
        assert_eq!(身体判据分数("green").score, 1.0);
        assert_eq!(身体判据分数("yellow").score, 0.5);
        assert_eq!(身体判据分数("red").score, 0.0);
        let g = 身体判据分数("未知");
        assert_eq!(g.level, "gray");
        assert_eq!(g.score, 0.5);
    }
}
