// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! storage 域 — τ 谱数据结构 + 序列化（第214批，IGT CORE 六域扩充）
//!
//! 定位: IGT CORE 的 storage 域——定义 τ 谱核心数据结构与序列化格式，
//! 供 IGT AGENT（本地个人 τ 谱）与 IGT ENGINE（公共 τ 谱）共享。
//! 纯数据+纯计算，无 UI/网络/平台依赖。

use serde::{Deserialize, Serialize};

// ============================================================================
// τ 谱核心数据结构
// ============================================================================

/// τ 链节点 — 单个弛豫模式
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct τ链节点 {
    /// 弛豫时间 τ（秒）——τ=1/γ（衰减率维度）
    pub τ: f64,
    /// 振荡频率 ω=Re(λ)（弧度/秒）——**独立维度，勿由 τ 反推**（ARCH-1 v2.1 §4: λ=ω−iγ,
    /// ω=Re(λ) 振荡频率、γ=1/τ 衰减率, ω≠2π/τ）
    ///
    /// [R1.4 ω 贯通, 2026-08-29] 已取消 2π/τ 换算——本字段由复 L 谱分解供真 ω（Re(λ)），
    /// 调用方无独立 ω 信息时显式传 0.0（无振荡/未知, 诚实标注）。
    pub ω: f64,
    /// 幅值（贡献权重）
    pub 幅值: f64,
}

impl τ链节点 {
    /// 新建 τ 链节点（**接收独立振荡频率 ω**——ARCH-1 v2.1 §4: ω 独立维度, 勿由 τ 反推）
    ///
    /// 复 L 谱分解供真 ω=Re(λ)；调用方无 ω 信息时显式传 0.0（无振荡/未知）。
    /// [R1.4] 原实现按 ω=2π/τ 换算（τ=1/f 违例残余）已删除——ω=2π/τ 在 core 清零。
    pub fn 新建(τ: f64, ω: f64, 幅值: f64) -> Self {
        Self { τ, ω, 幅值 }
    }

    /// 新建 τ 链节点（按频率 Hz 换算 ω=2π·f）——频率→角频率是**物理正确换算**
    /// （ω=2πf 与 τ=1/f 无涉），供调用方只有频率(Hz)信息时使用
    pub fn 新建带频率(τ: f64, 频率Hz: f64, 幅值: f64) -> Self {
        Self {
            τ,
            ω: 2.0 * std::f64::consts::PI * 频率Hz,
            幅值,
        }
    }
}

/// τ 链 — 系统的完整弛豫谱
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct τ链 {
    /// 节点列表（按 τ 降序，长τ在前）
    pub 节点: Vec<τ链节点>,
}

impl τ链 {
    /// 新建空 τ 链
    pub fn 新建() -> Self {
        Self { 节点: vec![] }
    }

    /// 添加节点并排序（τ 降序）
    pub fn 添加(&mut self, 节点: τ链节点) {
        self.节点.push(节点);
        self.节点.sort_by(|a, b| b.τ.partial_cmp(&a.τ).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// τ 链长度
    pub fn 长度(&self) -> usize {
        self.节点.len()
    }

    /// 主导 τ（最大 τ）
    pub fn 主导τ(&self) -> Option<f64> {
        self.节点.first().map(|n| n.τ)
    }

    /// 主导模态（最大幅值）
    pub fn 主导模态(&self) -> Option<τ链节点> {
        self.节点.iter().cloned().max_by(|a, b| {
            a.幅值.partial_cmp(&b.幅值).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// τ 跨度比（最大τ/最小τ）
    pub fn 跨度比(&self) -> Option<f64> {
        if self.节点.is_empty() {
            return None;
        }
        let 最大 = self.节点.iter().map(|n| n.τ).fold(f64::NEG_INFINITY, f64::max);
        let 最小 = self.节点.iter().map(|n| n.τ).fold(f64::INFINITY, f64::min);
        if 最小 > 1e-12 {
            Some(最大 / 最小)
        } else {
            None
        }
    }
}

// ============================================================================
// τ 指纹
// ============================================================================

/// τ 指纹 — 用于检索/分类/对比的紧凑表示
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct τ指纹 {
    /// 指纹哈希（16进制字符串）
    pub 哈希: String,
    /// 对称性类（GOE/GUE/Poisson/未知）
    pub 对称性类: String,
    /// β 值（Dyson 参数，1/2/4，0=未知）
    pub β: u8,
    /// 六维指纹分量（最短τ/主导τ/跨度比/复杂度/稳定度/健康度）
    pub 分量: [f64; 6],
}

impl τ指纹 {
    /// 从 τ 链生成六维指纹分量
    pub fn 从τ链(链: &τ链) -> [f64; 6] {
        if 链.长度() == 0 {
            return [0.0; 6];
        }
        let 最短τ = 链.节点.iter().map(|n| n.τ).fold(f64::INFINITY, f64::min);
        let 主导τ = 链.主导τ().unwrap_or(0.0);
        let 跨度比 = 链.跨度比().unwrap_or(0.0);
        // 复杂度 = 模态数归一化（≤10 映射到 [0,1]）
        let 复杂度 = (链.长度() as f64 / 10.0).min(1.0);
        // 稳定度 = 主模态占谱比例（幅值主导度）
        let 总幅值: f64 = 链.节点.iter().map(|n| n.幅值).sum();
        let 稳定度 = if 总幅值 > 1e-9 {
            (链.主导模态().map(|n| n.幅值).unwrap_or(0.0) / 总幅值).clamp(0.0, 1.0)
        } else {
            0.0
        };
        // 健康度 = 1 - 熵债近似（τ跨度越广越健康，1 - 1/(1+跨度比)）
        let 健康度 = 1.0 - 1.0 / (1.0 + 跨度比);
        [最短τ, 主导τ, 跨度比, 复杂度, 稳定度, 健康度]
    }

    /// 生成稳定指纹哈希（主导τ+跨度比+复杂度 确定性编码）
    pub fn 生成哈希(链: &τ链) -> String {
        let 主导τ = 链.主导τ().unwrap_or(0.0);
        let 跨度比 = 链.跨度比().unwrap_or(0.0);
        let 复杂度 = 链.长度();
        // 确定性格哈希（f64位+usize转字节）
        let mut buf = Vec::new();
        for x in [主导τ, 跨度比] {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        buf.extend_from_slice(&复杂度.to_le_bytes());
        // 简单确定性哈希（FNV-1a 64位）
        let mut hash: u64 = 0xcbf29ce484222325;
        for &b in &buf {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        format!("{:016x}", hash)
    }
}

// ============================================================================
// 谱动力学结果 DTO（Agent/Engine 共享）
// ============================================================================

/// 谱动力学结果 DTO — 跨组件传递的统一格式
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct 谱动力学DTO {
    /// 对称性类（GOE/GUE/Poisson/未知）
    pub 对称性类: String,
    /// β 值
    pub β: u8,
    /// GUE 拟合度 D_s^WD
    pub ds_wd: f64,
    /// 混沌判据 p 值
    pub p值: f64,
    /// 混沌指标 η = D_s^P/D_s^WD
    pub η: f64,
    /// τ 链（核心输出）
    pub τ链: τ链,
    /// 指纹
    pub 指纹: τ指纹,
}

impl 谱动力学DTO {
    /// 新建 DTO（自动生成指纹）
    pub fn 新建(对称性类: String, β: u8, ds_wd: f64, p值: f64, η: f64, 链: τ链) -> Self {
        let 分量 = τ指纹::从τ链(&链);
        let 哈希 = τ指纹::生成哈希(&链);
        let 指纹类 = 对称性类.clone();
        Self {
            对称性类,
            β,
            ds_wd,
            p值,
            η,
            τ链: 链,
            指纹: τ指纹 {
                哈希,
                对称性类: 指纹类,
                β,
                分量,
            },
        }
    }

    /// 序列化为 JSON 字符串
    pub fn 到json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// 从 JSON 字符串反序列化
    pub fn 从json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// [R1.4] τ链节点 接收独立 ω（真 ω 语义，删 2π/τ 断言）+ 新建带频率（ω=2πf 物理换算）
    #[test]
    fn 测试_τ链节点_独立ω与带频率() {
        let n = τ链节点::新建(1.0, 0.5, 2.0);
        assert!((n.τ - 1.0).abs() < 1e-12);
        assert!((n.ω - 0.5).abs() < 1e-12, "独立 ω 直接接收: {}", n.ω);
        assert!((n.幅值 - 2.0).abs() < 1e-12);
        // ω=0.0 = 无振荡/未知（调用方无 ω 信息时的诚实标注）
        let 无振荡 = τ链节点::新建(3.0, 0.0, 1.0);
        assert_eq!(无振荡.ω, 0.0);
        // 新建带频率: ω=2π·f（物理正确换算，非 τ=1/f 违例）
        let 带频率 = τ链节点::新建带频率(2.0, 0.25, 1.0);
        assert!((带频率.ω - 0.5 * std::f64::consts::PI).abs() < 1e-12, "ω=2π·0.25");
    }

    #[test]
    fn 测试_τ链_排序与主导() {
        let mut 链 = τ链::新建();
        链.添加(τ链节点::新建(2.0, 0.0, 1.0));
        链.添加(τ链节点::新建(5.0, 0.0, 3.0));
        链.添加(τ链节点::新建(0.5, 0.0, 2.0));
        assert_eq!(链.长度(), 3);
        assert!((链.主导τ().unwrap() - 5.0).abs() < 1e-12, "τ降序主导应为5.0");
        assert!((链.主导模态().unwrap().τ - 5.0).abs() < 1e-12, "最大幅值模态τ=5");
        assert!((链.跨度比().unwrap() - 10.0).abs() < 1e-9, "跨度比5/0.5=10");
    }

    #[test]
    fn 测试_τ指纹_确定性哈希() {
        let mut 链 = τ链::新建();
        链.添加(τ链节点::新建(1.0, 0.0, 1.0));
        链.添加(τ链节点::新建(2.0, 0.0, 2.0));
        let h1 = τ指纹::生成哈希(&链);
        let h2 = τ指纹::生成哈希(&链);
        assert_eq!(h1, h2, "确定性哈希应相同");
        assert_eq!(h1.len(), 16, "64位哈希=16hex字符");
    }

    #[test]
    fn 测试_谱动力学DTO_JSON往返() {
        let mut 链 = τ链::新建();
        链.添加(τ链节点::新建(1.0, 0.0, 1.0));
        let dto = 谱动力学DTO::新建("GUE".to_string(), 2, 0.2, 0.5, 2.5, 链);
        let json = dto.到json().unwrap();
        let 还原 = 谱动力学DTO::从json(&json).unwrap();
        assert_eq!(还原.对称性类, "GUE");
        assert_eq!(还原.β, 2);
        assert!((还原.τ链.主导τ().unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(还原.指纹.哈希, dto.指纹.哈希, "指纹哈希应保留");
    }

    #[test]
    fn 测试_空τ链_边界() {
        let 链 = τ链::新建();
        assert_eq!(链.主导τ(), None);
        assert_eq!(链.跨度比(), None);
        let 分量 = τ指纹::从τ链(&链);
        assert_eq!(分量[3], 0.0, "空链复杂度=0");
    }
}
