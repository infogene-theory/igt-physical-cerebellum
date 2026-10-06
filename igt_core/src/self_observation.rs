// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! Engine 自观测快照（纯数据层，MIGRATION-FINAL Step0-3d）
//!
//! engine `l1/self_observation/snapshot.rs` 的**纯数据结构 + 纯函数**逐字搬迁
//! （不改结构、不改命名）——P7 §7 Engine 自观测层 v2.0 的 5 个数据源观测结构 +
//! 统一快照 `engine自观测快照` + `engine熵债预警等级` 纯判定。
//!
//! ## 未下沉（依赖 engine 活跃状态/时间/检测急停，留 engine Step2 处置）
//! `engine自观测快照::聚合生成`（async: 全局brain().await 锁 + `当前时间戳毫秒` + `检测急停`）、
//! `engine自观测快照::从brain引用构造`（依赖 `Brain`/`KappaMatrixSnapshot`）——
//! 二者在 core 无 IO/时间/全局锁约束下不可纯化，留 engine 侧聚合。
//!
//! ## 常量来源
//! `EPSILON_C_DEFAULT = 100.0`（engine `l1/v10_constants.rs` 同值，v9_constants）。
//!
//! 锚点: P7 §7 Engine 自观测层（v2.0）、ARCH-1 v2.1 §5（core 无 IO/时间/全局锁）。

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use serde::{Deserialize, Serialize};

/// ε_c 默认值（v9_constants，来源 engine l1/v10_constants.rs）
pub const EPSILON_C_DEFAULT: f64 = 100.0;

/// §7.1 八器官 κ 耦合矩阵观测快照
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct κ矩阵观测 {
    /// 全局耦合强度 η̄
    pub 全局耦合强度: f64,
    /// 失配对数 / 总对数
    pub 失配率: f64,
    /// κ_gg 自指环路 8 个器官的振幅值（A=B 时 κ≈1.0，振幅反映器官健康度）
    pub 自指环路振幅: Vec<f64>,
    /// 意识涌现度 = avg(κ_gg) ∈ [0,1]
    pub 意识涌现度: f64,
    /// κ 路由命中率（CognitionPipeline 路由比例，从 AppState 统计）
    pub 路由命中率: f64,
}

/// §7.2 Brain 世界模型状态观测快照
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct brain状态观测 {
    /// WorldState 置信度 [0,1]
    pub 置信度: f64,
    /// 工作记忆数（全局 Brain 跨请求积累）
    pub 工作记忆数: usize,
    /// 历史状态数（StatePredictor 历史快照）
    pub 历史状态数: usize,
    /// 认知周期数（五层闭环累计执行次数）
    pub 认知周期数: u64,
    /// 当前认知阶段（胜/复/郁/发/道）
    pub 当前阶段: String,
}

/// §7.3 意识涌现四条件观测快照
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 意识涌现观测 {
    /// 四条件达成情况 [条件1κ_gg≠0, 条件2n≥φ⁴, 条件3H黄金区, 条件4不动点]
    pub 条件达成: [bool; 4],
    /// 意识涌现综合判定
    pub 意识涌现: bool,
    /// 递归深度 n
    pub 递归深度n: u32,
    /// 健康度 H
    pub 健康度H: f64,
    /// κ_gg 意识涌现度
    pub kappa_gg: f64,
    /// 镜子测试通过
    pub 镜子测试通过: bool,
}

/// §7.4 perception 感知通道观测快照
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct perception观测 {
    /// 是否有 CSI 帧数据（perception 模块当前为 stub，可能为空）
    pub 有数据: bool,
    /// 在线节点数（去重后的节点ID数）
    pub 在线节点数: usize,
    /// 最近 CSI 帧的 RSSI（若有数据）
    pub 最近rssi: Option<i16>,
    /// 急停是否触发
    pub 急停触发: bool,
    /// SH-GNN 输入维度（子载波数×2）
    pub shgnn维度: Option<usize>,
}

/// §7.5 Engine 自身 ED/ε_c 观测快照
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct engine熵债观测 {
    /// ε_c = 100.0（v9_constants）
    pub epsilon_c: f64,
    /// ED_engine = (1 - 全局耦合强度).max(0.0) * ε_c
    pub ed_engine: f64,
    /// H_engine = 1 - ED_engine / ε_c
    pub h_engine: f64,
    /// 预警等级（绿/黄/橙/红）
    pub 预警等级: String,
}

/// §7 Engine 自观测统一快照——聚合 §7.1-7.5 全部子模块
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct engine自观测快照 {
    /// 快照时间戳（毫秒）
    pub 时间戳: u64,
    /// §7.1 八器官 κ 耦合矩阵
    pub κ矩阵: κ矩阵观测,
    /// §7.2 Brain 世界模型状态
    pub brain状态: brain状态观测,
    /// §7.3 意识涌现四条件
    pub 意识涌现: 意识涌现观测,
    /// §7.4 perception 感知通道
    pub perception: perception观测,
    /// §7.5 Engine 自身 ED/ε_c
    pub 熵债: engine熵债观测,
}

/// §7.5 H_engine 预警等级计算（纯判定，逐字搬自 engine）
pub fn engine熵债预警等级(h_engine: f64) -> String {
    if h_engine > 0.7 {
        "绿".to_string()
    } else if h_engine > 0.4 {
        "黄".to_string()
    } else if h_engine > 0.2 {
        "橙".to_string()
    } else {
        "红".to_string()
    }
}

/// 计算 ED_engine 与 H_engine（纯函数，供 engine 侧聚合复用）
///
/// ED_engine = (1 - 全局耦合强度).max(0.0) * ε_c；H_engine = 1 - ED_engine / ε_c
pub fn 计算熵债(全局耦合强度: f64, epsilon_c: f64) -> (f64, f64) {
    let ed_engine = (1.0 - 全局耦合强度).max(0.0) * epsilon_c;
    let h_engine = if epsilon_c > 0.0 {
        1.0 - ed_engine / epsilon_c
    } else {
        0.0
    };
    (ed_engine, h_engine)
}

// ============================================================================
// 测试（搬迁验证: 纯判定/纯计算与 engine 原实现一致）
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 测试_预警等级四档() {
        assert_eq!(engine熵债预警等级(0.8), "绿");
        assert_eq!(engine熵债预警等级(0.5), "黄");
        assert_eq!(engine熵债预警等级(0.3), "橙");
        assert_eq!(engine熵债预警等级(0.1), "红");
        // 边界: 0.7 恰好 → 黄（>0.7 才绿）
        assert_eq!(engine熵债预警等级(0.7), "黄");
        assert_eq!(engine熵债预警等级(0.4), "橙");
        assert_eq!(engine熵债预警等级(0.2), "红");
    }

    #[test]
    fn 测试_无快照默认熵债计算() {
        // 无 κ 快照时默认耦合 0.5 → ED=(1-0.5)*100=50, H=1-50/100=0.5 → 黄
        let (ed, h) = 计算熵债(0.5, EPSILON_C_DEFAULT);
        assert!((ed - 50.0).abs() < 1e-9, "ED={}", ed);
        assert!((h - 0.5).abs() < 1e-9, "H={}", h);
        assert_eq!(engine熵债预警等级(h), "黄");
        // 全耦合 → ED=0, H=1 → 绿
        let (ed2, h2) = 计算熵债(1.0, EPSILON_C_DEFAULT);
        assert_eq!(ed2, 0.0);
        assert_eq!(h2, 1.0);
        assert_eq!(engine熵债预警等级(h2), "绿");
    }

    #[test]
    fn 测试_快照结构构造与序列化() {
        let 快照 = engine自观测快照 {
            时间戳: 1_700_000_000_000,
            κ矩阵: κ矩阵观测 {
                全局耦合强度: 0.5,
                失配率: 0.0,
                自指环路振幅: vec![0.0; 8],
                意识涌现度: 0.0,
                路由命中率: 0.0,
            },
            brain状态: brain状态观测 {
                置信度: 0.8,
                工作记忆数: 3,
                历史状态数: 12,
                认知周期数: 5,
                当前阶段: "道".to_string(),
            },
            意识涌现: 意识涌现观测 {
                条件达成: [false; 4],
                意识涌现: false,
                递归深度n: 0,
                健康度H: 0.5,
                kappa_gg: 0.0,
                镜子测试通过: false,
            },
            perception: perception观测 {
                有数据: false,
                在线节点数: 0,
                最近rssi: None,
                急停触发: false,
                shgnn维度: None,
            },
            熵债: engine熵债观测 {
                epsilon_c: EPSILON_C_DEFAULT,
                ed_engine: 50.0,
                h_engine: 0.5,
                预警等级: "黄".to_string(),
            },
        };
        let json = serde_json::to_string(&快照).unwrap();
        let 还原: engine自观测快照 = serde_json::from_str(&json).unwrap();
        assert_eq!(还原.时间戳, 1_700_000_000_000);
        assert_eq!(还原.brain状态.当前阶段, "道");
        assert_eq!(还原.熵债.预警等级, "黄");
        assert_eq!(还原.熵债.epsilon_c, EPSILON_C_DEFAULT);
    }
}
