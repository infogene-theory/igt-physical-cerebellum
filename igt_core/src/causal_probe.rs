// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! 造波因果探测（D6, 2026-08-23）— P22 §8.6 + P20
//!
//! 理论依据:
//!   - P22 §8.6: 主动造波打破被动观测对称性——注入已知扰动 ε_wave(<0.3),
//!     观测各模态响应, κ 非对称性直接可读
//!   - P20 §3.4: 注入 f(t)=ε·v_n·e^{iω_w t}e^{-t/τ_w}, 复匹配(ω_w≈ω_n, τ_w≈τ_n)
//!   - P22a §7.2: 样本门槛（50 条最低/100 条安全/20 条欠定禁止判向）
//!   - D1: κ_ij = w_i†δL v_j（需完整复 v/w 双正交）
//!
//! ★ 基的约定（T-74 / P39 §14.2 三类对象, 2026-09-14）:
//!   - **本文件"内部非对称方向"判据 A 属类 I · EP 裸参数**——读**裸基非对角元**
//!     `|L[i,j]| vs |L[j,i]|`（物理通道基）, **不做本征矢投影**（强混合下投影非对称比
//!     会塌缩、掩盖真实方向, 见 `测试_P39G12_强混合裸元与投影分道扬镳守卫`）。
//!   - 判据 C 的共同慢模投影 `|w_i†v_S|²` 是"模身份重叠", 非耦合测量, 正当保留。
//!   - **类 II 外扰造波**（基线 L0＋ΔL, κ=w0†ΔL v0, 费米黄金规则）＝G-12 第二步, 待 T-72 后。
//!   - 类 III 集体模图边（w†δL v, w/v 取完整 L）走 `kappa_channel`, 勿与本判据混用。
//!
//! κ 非对称方向判据（v0.1 候选统计判据, 2σ 显著性待理论侧确认后升级 v1.0）:
//!   |L[i,j]| > |L[j,i]|·(1+显著性) → i→j 方向耦合
//!   |L[i,j]| ≈ |L[j,i]|          → 共同慢模驱动（不误判为方向耦合）
//!
//! 安全约束（P20/P12）: ε_wave<0.3 安全区; 0.3~1 过渡区谨慎; ε_wave≥1 危险区拒绝。

#![allow(non_snake_case)]

use nalgebra::Complex as NComplex;
use nalgebra::ComplexField;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use super::nonhermitian_spectrum::{非厄米谱分解, 非厄米谱求解器};

/// 方向判定类型（T-7 升级: P39 四态, D5+D6 v1.1 §5.2）
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum 方向类型 {
    /// i→j 方向耦合（A+B+C 全满足）
    方向耦合,
    /// 共同慢模驱动（C 不满足, i/j 同挂慢模 v_S——投影几何伪因果）
    共同驱动,
    /// 对称/无耦合（A 不满足, R_ij≈1）
    对称无耦合,
    /// 样本不足（M < (3/|κ|τ)², 弱耦合需 M≥100）
    样本不足,
}

impl 方向类型 {
    pub fn 名称(&self) -> &'static str {
        match self {
            方向类型::方向耦合 => "方向耦合",
            方向类型::共同驱动 => "共同驱动",
            方向类型::对称无耦合 => "对称无耦合",
            方向类型::样本不足 => "样本不足",
        }
    }
}

/// 单对模方向判定（T-7 升级: 三重判据 A+B+C + 样本量）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 方向判定 {
    pub 通道i: usize,
    pub 通道j: usize,
    /// |κ_ij|（w_i†δL v_j 模幅）
    pub κ_ij: f64,
    /// |κ_ji|
    pub κ_ji: f64,
    /// P39 判据A: 非对称比 R_ij=|κ_ij|/(|κ_ji|+ε)（T-7 新增）
    pub 非对称比R_ij: f64,
    /// P39 判据B: 响应时序差 Δt_ij=t_j^peak-t_i^peak（T-7 新增）
    pub 时序差Δt_ij: f64,
    /// P39 判据A 满足（R_ij>3, 10dB 非对称）
    pub 判据A非对称比: bool,
    /// P39 判据B 满足（Δt_ij>2σ_t, 95%置信）
    pub 判据B时序: bool,
    /// P39 判据C 满足（排除共同慢模驱动）
    pub 判据C排除共同慢模: bool,
    /// 样本数 M（T-7 新增）
    pub 样本数: usize,
    /// 方向判据
    pub 方向: 方向类型,
    /// 描述（i→j / j→i / 共同驱动 / 对称无耦合 / 样本不足）
    pub 描述: String,
}

/// 因果探测结果（T-7 升级: 新增响应延迟表/样本数）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 因果探测结果 {
    /// κ_ij = w_i†δL v_j（完整复数, D1 依赖; 存 (实,虚) 对, 兼容序列化）
    pub κ矩阵: Vec<Vec<(f64, f64)>>,
    /// 方向判据列表（模对）
    pub 方向判据: Vec<方向判定>,
    /// 响应延迟表 Δt_ij（P39 判据B, T-7 新增）
    pub 响应延迟表: Vec<Vec<f64>>,
    /// 样本数 M（P39 §9, T-7 新增）
    pub 样本数: usize,
    /// 谱信息充足（D1 联动: 复L 对角近似 → false, 不静默给方向）
    pub 谱信息充足: bool,
    /// 说明
    pub 说明: String,
}

/// 因果探测器（κ 非对称方向判据, v0.1）
pub struct 因果探测器 {
    /// 显著性阈值（κ 非对称占比, 默认 0.1）
    pub 显著性: f64,
    /// 谱信息阈值（δL 非对角最小显著值）
    pub 非对角阈值: f64,
}

impl Default for 因果探测器 {
    fn default() -> Self {
        Self { 显著性: 0.1, 非对角阈值: 1e-6 }
    }
}

impl 因果探测器 {
    /// 造波安全校验（P20/P12）: ε_wave≥1 危险区拒绝
    pub fn 安全校验(ε_wave: f64) -> Result<(), String> {
        if ε_wave >= 1.0 {
            return Err(format!(
                "ε_wave={} 处于危险区（P12 三级崩溃, 禁止）——造波因果探测安全约束: ε_wave<0.3 安全区, 0.3~1 过渡区谨慎",
                ε_wave
            ));
        }
        if ε_wave >= 0.3 {
            return Err(format!(
                "ε_wave={} 处于过渡区（0.3~1 需谨慎, P20 安全区 <0.3）——建议降低扰动强度",
                ε_wave
            ));
        }
        Ok(())
    }

    /// 因果探测入口
    /// `l`: 完整复L（观测基, D1 修复后含非对角）
    /// `分解`: 非厄米谱分解（完整复 v/w）
    pub fn 探测(&self, l: &DMatrix<NComplex<f64>>, 分解: &非厄米谱分解) -> 因果探测结果 {
        let n = 分解.模数;
        // 谱信息充足: δL（L 非对角部分）是否显著（D1 依赖联动）
        let mut 非对角最大 = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let z = l[(i, j)];
                    非对角最大 = 非对角最大.max((z.re * z.re + z.im * z.im).sqrt());
                }
            }
        }
        let 谱信息充足 = 非对角最大 > self.非对角阈值;

        if !谱信息充足 {
            return 因果探测结果 {
                κ矩阵: vec![vec![(0.0, 0.0); n]; n],
                方向判据: vec![],
                响应延迟表: vec![vec![0.0; n]; n],
                样本数: 0,
                谱信息充足: false,
                说明: "谱信息不足（D1 未修复态: 对角近似复L）——不静默给出因果方向".to_string(),
            };
        }

        // 类 I（G-12 快照版, P39 §14.2）: 读"内部"非对称方向直接取**裸基非对角元**
        //   κ_ij=L[i,j]（物理通道基）, 不做本征矢投影——弱耦合下与投影数值一致,
        //   强混合/EP 邻域投影会被本征矢旋转扭曲（与 G-11 同口径, 见强混合反例守卫）。
        //   判据 C 的"共同慢模投影 |w_i†v_S|²"是模身份重叠、仍在 v10 里保留（正当用途）。
        let mut κ = DMatrix::<NComplex<f64>>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    κ[(i, j)] = l[(i, j)]; // 对角置 0（δL 口径）
                }
            }
        }

        // 方向判据（v0.1）: |κ_ij| vs |κ_ji|
        let mut 判据列表 = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let zij = κ[(i, j)];
                let zji = κ[(j, i)];
                let κij = (zij.re * zij.re + zij.im * zij.im).sqrt();
                let κji = (zji.re * zji.re + zji.im * zji.im).sqrt();
                let 方向 = if κij > κji * (1.0 + self.显著性) {
                    方向类型::方向耦合
                } else if κji > κij * (1.0 + self.显著性) {
                    方向类型::方向耦合
                } else {
                    方向类型::共同驱动
                };
                let 描述 = match 方向 {
                    方向类型::方向耦合 => {
                        if κij > κji {
                            format!("i{}→j{}（κ_ij={:.4} > κ_ji={:.4}）", i, j, κij, κji)
                        } else {
                            format!("j{}→i{}（κ_ji={:.4} > κ_ij={:.4}）", j, i, κji, κij)
                        }
                    }
                    _ => format!("i{}-j{} 共同驱动（κ_ij={:.4}≈κ_ji={:.4}）", i, j, κij, κji),
                };
                判据列表.push(方向判定 {
                    通道i: i,
                    通道j: j,
                    κ_ij: κij,
                    κ_ji: κji,
                    // T-7: v0.1 兼容构造——R 可算, Δt/样本未提供时判据 B 不满足
                    非对称比R_ij: κij / (κji + 1e-9),
                    时序差Δt_ij: 0.0,
                    判据A非对称比: κij > κji * (1.0 + self.显著性),
                    判据B时序: false,
                    判据C排除共同慢模: true,
                    样本数: 0,
                    方向,
                    描述,
                });
            }
        }

        let κ序列化: Vec<Vec<(f64, f64)>> = (0..n)
            .map(|i| (0..n).map(|j| (κ[(i, j)].re, κ[(i, j)].im)).collect())
            .collect();
        let 判据数 = 判据列表.len();
        因果探测结果 {
            κ矩阵: κ序列化,
            方向判据: 判据列表,
            响应延迟表: vec![vec![0.0; n]; n],
            样本数: 0,
            谱信息充足: true,
            说明: format!("因果探测完成(v0.1兼容): {} 对模方向判定", 判据数),
        }
    }

    /// P39 G-12 三重统计判据因果探测（D5+D6 v1.1 §5.2/§5.3）:
    /// 判据A 非对称比 R_ij=|κ_ij|/(|κ_ji|+ε)>R_c=3
    /// 判据B 响应时序 Δt_ij=t_j^peak-t_i^peak>2σ_t
    /// 判据C 排除共同慢模 NOT(|w_i†v_S|²>θ_S 且 |w_j†v_S|²>θ_S), θ_S≈0.5
    /// 四态输出（方向耦合/共同驱动/对称无耦合/样本不足）
    /// 样本量（P39 §9）: |κ|τ>0.5 M≥10 强耦合 / 弱耦合 M≥100; M<(3/|κ|τ)² 标注样本不足
    /// D1 依赖标注: 复L 对角近似 → 谱信息充足=false, 不静默给方向
    ///
    /// `响应延迟`: Δt_ij = t_j^peak - t_i^peak（峰值响应时序差, 由应用层注入）
    /// `峰值定位误差σt`: 时序定位误差（判据B 阈值, P1b 时间箭头）
    /// `样本数M`: 重复造波次数（P39 §9 样本量）
    pub fn 探测_v10(
        &self,
        l: &DMatrix<NComplex<f64>>,
        分解: &非厄米谱分解,
        响应延迟: &std::collections::HashMap<(usize, usize), f64>,
        峰值定位误差σt: f64,
        样本数M: usize,
    ) -> 因果探测结果 {
        let n = 分解.模数;
        // D1 联动: δL 非对角显著
        let mut 非对角最大 = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let z = l[(i, j)];
                    非对角最大 = 非对角最大.max((z.re * z.re + z.im * z.im).sqrt());
                }
            }
        }
        let 谱信息充足 = 非对角最大 > self.非对角阈值;
        if !谱信息充足 {
            return 因果探测结果 {
                κ矩阵: vec![vec![(0.0, 0.0); n]; n],
                方向判据: vec![],
                响应延迟表: vec![vec![0.0; n]; n],
                样本数: 样本数M,
                谱信息充足: false,
                说明: "谱信息不足（D1 未修复态: 对角近似复L）——不静默给出因果方向".to_string(),
            };
        }

        // 类 I（G-12 快照版, P39 §14.2）: 判据 A 的"内部非对称方向"读**裸基非对角元**
        //   κ_ij=L[i,j]、κ_ji=L[j,i]（物理通道基）, 不做本征矢投影; 判据 C 的共同慢模
        //   投影 |w_i†v_S|² 属"模身份重叠"、正当, 下方保留。
        let mut κ = DMatrix::<NComplex<f64>>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    κ[(i, j)] = l[(i, j)];
                }
            }
        }

        // 共同慢模 v_S = τ 最长模（统一令 §4.1：按 **λ_im 最大**〔= 旧"γ 最大"同一数值序，
        // λ_im→0⁻ 即 γ→0⁺〕; 若含失稳模则与旧实现同样落到增长最快模——选择逻辑不变）; 慢模投影 |w_i†v_S|²
        let 慢模索引 = (0..n)
            .max_by(|&a, &b| 分解.模[a].λ_im.partial_cmp(&分解.模[b].λ_im).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0);
        let θ_S = 0.5; // P39 §7 建议阈值
        let 慢模投影 = |i: usize| -> f64 {
            let mut 累 = NComplex::new(0.0, 0.0);
            for a in 0..n {
                let w_i = NComplex::new(
                    分解.模[i].w[a],
                    分解.模[i].w虚.get(a).copied().unwrap_or(0.0),
                );
                let v_s = NComplex::new(
                    分解.模[慢模索引].v[a],
                    分解.模[慢模索引].v虚.get(a).copied().unwrap_or(0.0),
                );
                累 += w_i.conjugate() * v_s;
            }
            累.re * 累.re + 累.im * 累.im
        };

        let mut 判据列表 = Vec::new();
        let mut 延迟表 = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let zij = κ[(i, j)];
                let zji = κ[(j, i)];
                let κij = (zij.re * zij.re + zij.im * zij.im).sqrt();
                let κji = (zji.re * zji.re + zji.im * zji.im).sqrt();

                // 判据A: R_ij > 3（10dB 非对称）
                let r_ij = κij / (κji + 1e-9);
                let 判据A = r_ij > 3.0;

                // 判据B: Δt_ij = t_j^peak - t_i^peak > 2σ_t
                let δt = 响应延迟.get(&(i, j)).copied().unwrap_or(0.0);
                延迟表[i][j] = δt;
                延迟表[j][i] = -δt;
                let 判据B = δt > 2.0 * 峰值定位误差σt;

                // 判据C: 排除共同慢模
                let 投影i = 慢模投影(i);
                let 投影j = 慢模投影(j);
                let 判据C = !(投影i > θ_S && 投影j > θ_S);

                // 样本量（P39 §9）: M >= (3/|κ|τ)²
                let κτ = κij.max(κji) * 分解.模[j].τ.max(1e-9);
                let 所需样本 = (3.0 / κτ).powi(2);
                let 样本充足 = (样本数M as f64) >= 所需样本;

                // 四态判断
                let (方向, 描述) = if !样本充足 {
                    (
                        方向类型::样本不足,
                        format!(
                            "i{}-j{} 样本不足: M={} < (3/|κ|τ)²={:.0}",
                            i, j, 样本数M, 所需样本
                        ),
                    )
                } else if !判据C {
                    (
                        方向类型::共同驱动,
                        format!(
                            "i{}-j{} 共同慢模驱动（v_S 投影 {:.2}/{:.2} > θ_S=0.5, 伪因果排除）",
                            i, j, 投影i, 投影j
                        ),
                    )
                } else if 判据A && 判据B {
                    (
                        方向类型::方向耦合,
                        format!(
                            "i{}→j{}（A:R_ij={:.2}>3, B:Δt_ij={:.3}>2σ_t={:.3}, C:排除慢模）",
                            i, j, r_ij, δt, 2.0 * 峰值定位误差σt
                        ),
                    )
                } else if (1.0 / (r_ij + 1e-9)) > 3.0 && δt < -2.0 * 峰值定位误差σt {
                    // j→i 方向（R_ji>3 且 Δt_ji>2σ_t）
                    (
                        方向类型::方向耦合,
                        format!(
                            "j{}→i{}（A:R_ji={:.2}>3, B:Δt_ji={:.3}>2σ_t={:.3}, C:排除慢模）",
                            j, i, 1.0 / (r_ij + 1e-9), -δt, 2.0 * 峰值定位误差σt
                        ),
                    )
                } else {
                    (
                        方向类型::对称无耦合,
                        format!("i{}-j{} 对称/无耦合（R_ij={:.2}≈1）", i, j, r_ij),
                    )
                };

                判据列表.push(方向判定 {
                    通道i: i,
                    通道j: j,
                    κ_ij: κij,
                    κ_ji: κji,
                    非对称比R_ij: r_ij,
                    时序差Δt_ij: δt,
                    判据A非对称比: 判据A,
                    判据B时序: 判据B,
                    判据C排除共同慢模: 判据C,
                    样本数: 样本数M,
                    方向,
                    描述,
                });
            }
        }

        let κ序列化: Vec<Vec<(f64, f64)>> = (0..n)
            .map(|i| (0..n).map(|j| (κ[(i, j)].re, κ[(i, j)].im)).collect())
            .collect();
        因果探测结果 {
            κ矩阵: κ序列化,
            方向判据: 判据列表,
            响应延迟表: 延迟表,
            样本数: 样本数M,
            谱信息充足: true,
            说明: format!(
                "三重判据因果探测完成(v1.0): M={} 次, 慢模v_S=模{}, σ_t={}",
                样本数M, 慢模索引, 峰值定位误差σt
            ),
        }
    }
}

// ============================================================================
// 测试（D6 规范 §8 V3/V4/V5/V7）
// ============================================================================
// ============================================================================
// T-82 · G-12 第二步：**类 II 外扰造波版 κ**（未扰基线 L0 ＋ 外扰 ΔL）
// ============================================================================
// 理论依据（理论侧续43 两步法 ＋ 续44 三类对象取法表 ＋ T-72 裁决）：
//   · 类 II 对象 ＝「**基线 L0 ＋ 外扰 ΔL**」；`κ_II(i,j) = w0_i† ΔL v0_j`（费米黄金规则）；
//   · ⛔【**禁自指**】`w/v` **必须取未扰基线 L0 的本征基**。若改用"已含 ΔL 的完整 L"分解后
//     再投影 ΔL **自身**：因 `W†LV=Λ` ⇒ i≠j 时 `w_i†L v_j ≡ 0` ⇒ `w_i†δL v_j = −w_i†L_diag v_j`，
//     测到的是**对角失谐在本征基上的投影**、**不是非对角耦合**；简并/EP 邻域解析归零＝**脆弱假绿**
//     （这正是 G-12 的病根，见 T-74 / P39 §14.2）。取 L0 基线即彻底避开。
//   · 【**相位规范**】逐模自由相位 `v_j→e^{iθ_j}v_j`、`w_i→e^{−iθ_i}w_i` ⇒ `κ_ij→e^{i(θ_j−θ_i)}κ_ij`：
//     对角 `κ_jj` 相位自抵消（Re＝频移／Im＝线宽，**规范不变、物理**）；
//     **非对角单边 Re/Im 随任意 θ_j 变、不是物理量** ⇒ **跨模路径在类型层不放 Re/Im 字段**
//     （同 `kappa_channel` 类 III 做法）；**"甲"（绝对相位／同相-正交）唯一合法位置
//     ＝ 单模对角相对已知物理外扰 φ_d**（T-72 裁决）。
// ============================================================================

/// 类 II **跨模**（i≠j）κ 结果——**⚠️ 类型层挡死：本结构体故意不含 Re/Im 字段**。
///
/// 依据（`kappa_channel.rs:97-102`）：非对角 κ 带模间自由相位 `e^{i(θ_j−θ_i)}`，
/// 单边实/虚部**不是物理量** ⇒ 跨模只呈现**规范不变量**。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct 类II跨模不变量 {
    pub i: usize,
    pub j: usize,
    /// `|κ_II(i,j)| = |w0_i† ΔL v0_j|`
    pub κ模ij: f64,
    /// `|κ_II(j,i)|`
    pub κ模ji: f64,
    /// 判据A 非对称比 `R_ij = |κ_ij| / (|κ_ji| + ε)`（与类 I 同口径）
    pub 非对称比R_ij: f64,
    /// 耦合强度×寿命 `|κ_ij|·τ_i`（P39 §9 样本量口径；τ 取 L0 第 i 模）
    pub κτ: f64,
    /// 判据A 满足（R_ij 或 1/R_ij 超 `1+显著性`）
    pub 判据A非对称比: bool,
    /// 判据C：i/j **是否同挂共同慢模**（`|w0_i†v_S|²`、`|w0_j†v_S|²` 均 > θ_S ⇒ 共同驱动嫌疑）
    /// （**以 L0 的 w0/v0 为基准**；共同慢模投影是"模身份重叠"、非耦合测量，正当保留）
    pub 判据C排除共同慢模: bool,
    /// 方向判读（i→j／j→i 记 `方向耦合`；同挂慢模记 `共同驱动`；其余 `对称无耦合`）
    pub 方向: 方向类型,
    /// 人读描述（只含不变量，不带单边相位）
    pub 描述: String,
}

/// 类 II **单模**（i=j）响应。
///
/// `φ_d = None` ⇒ `同相/正交` 均为 `None`（**不暴露任何同相/正交量**）；
/// `φ_d = Some(φ)` ⇒ 给 `Re/Im[e^{−iφ}·w0_i†ΔL v0_i]`（**"甲"的唯一合法位置**）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct 类II单模响应 {
    pub i: usize,
    /// `|κ_II(i,i)| = |w0_i† ΔL v0_i|`（对角模量，恒可给、无相位依赖）
    pub κ模: f64,
    /// 已知物理外扰相位 φ_d（rad）；`None` ＝ 未提供 ⇒ 不给同相/正交
    pub φ_d: Option<f64>,
    /// `Re[e^{−iφ_d}·κ_ii]`（相对 φ_d 的**同相**分量）
    pub 同相: Option<f64>,
    /// `Im[e^{−iφ_d}·κ_ii]`（相对 φ_d 的**正交**分量）
    pub 正交: Option<f64>,
}

/// 类 II 外扰造波 κ 结果。
///
/// **成果面**：`跨模` 只有不变量（类型层无可达的单边 Re/Im），`单模` 仅在给了 φ_d 时才带同相/正交。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct 类II结果 {
    /// 模数（＝L0 本征模数）
    pub 模数: usize,
    /// 跨模（i<j）不变量
    pub 跨模: Vec<类II跨模不变量>,
    /// 单模（全部 i）响应
    pub 单模: Vec<类II单模响应>,
    /// **判据B（响应时序差 Δt_ij）不可得原因**：类 II 是**纯函数**（只吃 L0/ΔL），
    /// 时序须造波**执行后**观测；**造波执行永久归 agent 侧（铁律5）** ⇒ 本路径恒"未提供"，
    /// **不造数**（宁缺勿造）。非空字符串即为如实标注。
    pub 判据B未提供原因: String,
    /// 基来源说明（**禁自指**取证：须为"未扰基线 L0"）
    pub 基来源: String,
    /// 说明
    pub 说明: String,
}

/// 类 II 外扰造波 κ 探测器（G-12 第二步；纯函数，无 IO、无造波执行）
#[derive(Debug, Clone)]
pub struct 类II探测器 {
    /// 方向显著性阈值（`R_ij > 1+显著性` 判据A 满足；与类 I 同口径）
    pub 显著性: f64,
    /// 共同慢模投影阈值 θ_S（`|w0_i†v_S|² > θ_S` 记"同挂慢模"）
    pub 共同慢模阈值θS: f64,
    /// 外扰显著性阈值：`ΔL` 全零（≤ 阈值）⇒ **如实报"无外扰"、不给方向**（禁零输入充数）
    pub 外扰阈值: f64,
}

impl Default for 类II探测器 {
    fn default() -> Self {
        Self { 显著性: 0.1, 共同慢模阈值θS: 0.5, 外扰阈值: 1e-12 }
    }
}

impl 类II探测器 {
    /// **类 II κ 纯函数**：`(未扰基线 L0, 外扰矩阵 ΔL, 可选 φ_d)` → κ 不变量 ＋ 单模响应。
    ///
    /// - `w/v` **取 L0 的本征基**（⛔不对 `L0+ΔL` 分解——那是 G-12 自指病）；
    /// - `φ_d = None` ⇒ 单模不给同相/正交；
    /// - 判据 A／C **一律以 L0 的 w0/v0 为基准**；判据 B 不可得（见 `判据B未提供原因`）。
    pub fn 探测_类II(
        &self,
        l0: &DMatrix<NComplex<f64>>,
        δl: &DMatrix<NComplex<f64>>,
        φ_d: Option<f64>,
    ) -> Result<类II结果, String> {
        let n = l0.nrows();
        if n == 0 || l0.ncols() != n {
            return Err("L0 须为非空方阵".to_string());
        }
        if δl.nrows() != n || δl.ncols() != n {
            return Err(format!(
                "ΔL 维数（{}×{}）须与 L0（{}×{}）一致",
                δl.nrows(),
                δl.ncols(),
                n,
                n
            ));
        }
        // 外扰显著性：ΔL 全零 ⇒ 无信号可探，如实报错（禁"零输入充数"）
        let mut δl_max = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                δl_max = δl_max.max(δl[(i, j)].norm());
            }
        }
        if δl_max <= self.外扰阈值 {
            return Err(format!(
                "ΔL 全零（最大元 {:.3e} ≤ 阈值 {:.1e}）——类 II 无外扰信号可探，不给出任何方向（禁零输入充数）",
                δl_max, self.外扰阈值
            ));
        }

        // ★★ 禁自指：分解**未扰基线 L0**（绝不用 L0+ΔL 的基投影 ΔL 自身）
        let 分解0 = 非厄米谱求解器::default().分解(l0)?;
        if 分解0.模数 != n {
            return Err(format!(
                "L0 谱分解模数 {} ≠ n {}（本征结构退化）——不静默给类 II 方向",
                分解0.模数, n
            ));
        }
        let 复矢 = |re: &[f64], im: &[f64]| -> Vec<NComplex<f64>> {
            (0..n)
                .map(|k| NComplex::new(re[k], im.get(k).copied().unwrap_or(0.0)))
                .collect()
        };
        let v0: Vec<Vec<NComplex<f64>>> =
            分解0.模.iter().map(|m| 复矢(&m.v, &m.v虚)).collect();
        let w0: Vec<Vec<NComplex<f64>>> =
            分解0.模.iter().map(|m| 复矢(&m.w, &m.w虚)).collect();

        // κ_II(i,j) = w0_i† ΔL v0_j（费米黄金规则；左矢取共轭）
        let κ = |i: usize, j: usize| -> NComplex<f64> {
            let mut 累 = NComplex::new(0.0, 0.0);
            for k in 0..n {
                let mut 内 = NComplex::new(0.0, 0.0);
                for m in 0..n {
                    内 += δl[(k, m)] * v0[j][m];
                }
                累 += w0[i][k].conj() * 内;
            }
            累
        };
        // 共同慢模投影 |w0_i† v_S|²（v_S ＝ L0 最慢模；"模身份重叠"，正当保留）
        let 慢模投影 = |i: usize| -> f64 {
            let mut 累 = NComplex::new(0.0, 0.0);
            for k in 0..n {
                累 += w0[i][k].conj() * v0[0][k];
            }
            累.norm_sqr()
        };

        let τ: Vec<f64> = 分解0.模.iter().map(|m| m.τ).collect();
        let mut 跨模 = Vec::with_capacity(n * n / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let κij = κ(i, j).norm();
                let κji = κ(j, i).norm();
                let r = κij / (κji + 1e-9);
                let 判据A = r > 1.0 + self.显著性 || (1.0 / (r + 1e-12)) > 1.0 + self.显著性;
                let 共同 = 慢模投影(i) > self.共同慢模阈值θS
                    && 慢模投影(j) > self.共同慢模阈值θS;
                let 方向 = if 共同 {
                    方向类型::共同驱动
                } else if κij > κji * (1.0 + self.显著性) {
                    方向类型::方向耦合
                } else if κji > κij * (1.0 + self.显著性) {
                    方向类型::方向耦合
                } else {
                    方向类型::对称无耦合
                };
                let 描述 = if 共同 {
                    format!("i{}-j{} 共同驱动（|κ|={:.6}≈{:.6}，两模同挂慢模）", i, j, κij, κji)
                } else if κij > κji {
                    format!("i{}→j{}（|κ_ij|={:.6} > |κ_ji|={:.6}, R={:.4}）", i, j, κij, κji, r)
                } else if κji > κij {
                    format!("j{}→i{}（|κ_ji|={:.6} > |κ_ij|={:.6}）", j, i, κji, κij)
                } else {
                    format!("i{}-j{} 对称无耦合（|κ|={:.6}≈{:.6}）", i, j, κij, κji)
                };
                跨模.push(类II跨模不变量 {
                    i,
                    j,
                    κ模ij: κij,
                    κ模ji: κji,
                    非对称比R_ij: r,
                    判据A非对称比: 判据A,
                    判据C排除共同慢模: !共同,
                    κτ: κij * τ[i].abs(),
                    方向,
                    描述,
                });
            }
        }

        // 单模对角：**仅** 给了 φ_d 才暴露同相/正交（"甲"的唯一合法位置）
        let 单模: Vec<类II单模响应> = (0..n)
            .map(|i| {
                let z = κ(i, i);
                match φ_d {
                    None => 类II单模响应 {
                        i,
                        κ模: z.norm(),
                        φ_d: None,
                        同相: None,
                        正交: None,
                    },
                    Some(φ) => {
                        // e^{−iφ}·κ_ii
                        let 旋转 = NComplex::new(φ.cos(), -φ.sin());
                        let y = 旋转 * z;
                        类II单模响应 {
                            i,
                            κ模: z.norm(),
                            φ_d: Some(φ),
                            同相: Some(y.re),
                            正交: Some(y.im),
                        }
                    }
                }
            })
            .collect();

        let φ说明 = match φ_d {
            Some(φ) => format!(
                "单模对角相对已知物理外扰 φ_d={:.6} rad 给 Re/Im[e^{{−iφ_d}}κ_ii]（同相/正交——甲的唯一合法位置）",
                φ
            ),
            None => "未提供 φ_d ⇒ 单模亦**不暴露**同相/正交（只给对角模量）".to_string(),
        };
        Ok(类II结果 {
            模数: n,
            跨模,
            单模,
            判据B未提供原因: "判据B（响应时序差 Δt_ij）须造波**执行后**的时序观测方可测；类 II 为纯函数（只吃 L0/ΔL），且造波执行永久归 agent 侧（铁律5）⇒ 本路径如实「未提供」，不造数".to_string(),
            基来源: "L0（未扰基线）本征基 —— 禁自指取证：w0/v0 全部取自 L0 分解，未对 L0+ΔL 分解".to_string(),
            说明: format!(
                "类 II 外扰造波 κ（G-12 第二步）: κ_II(i,j)=w0_i†ΔL v0_j, w/v 取**未扰基线 L0**；跨模仅不变量（|κ|/R_ij/κτ 等，类型层无 Re/Im）；{}；判据A/C 以 L0 的 w0/v0 为基准",
                φ说明
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use crate::nonhermitian_spectrum::非厄米谱求解器;

    /// 已知 L: 2 模 + 非对称耦合 δL（L_01 > L_10 → 0→1 方向）
    fn 非对称耦合L() -> DMatrix<NComplex<f64>> {
        let n = 2;
        let mut l = DMatrix::<NComplex<f64>>::zeros(n, n);
        l[(0, 0)] = NComplex::new(0.0, -2.0);
        l[(1, 1)] = NComplex::new(0.1, -0.5);
        // 非对称耦合（弱: 本征向量近标准基）: L_01(0→1) = 0.3, L_10(1→0) = 0.05 → 0→1 方向
        l[(0, 1)] = NComplex::new(0.3, 0.1);
        l[(1, 0)] = NComplex::new(0.05, 0.02);
        l
    }

    /// D6-V3: 方向耦合——非对称 δL → κ 非对称 → 方向耦合判定
    ///（具体 i→j/j→i 由模基排序决定, 应用层按注入通道映射; 本测试验证非对称→方向）
    #[test]
    fn 测试_D6V3_方向耦合() {
        let l = 非对称耦合L();
        let 求解 = 非厄米谱求解器::default();
        let 分解 = 求解.分解(&l).unwrap();
        let 结果 = 因果探测器::default().探测(&l, &分解);
        assert!(结果.谱信息充足, "非对角显著应充足");
        let 判定 = &结果.方向判据[0];
        assert_eq!(判定.方向, 方向类型::方向耦合, "{}", 判定.描述);
        assert!(
            (判定.κ_ij - 判定.κ_ji).abs() > 1e-6,
            "非对称耦合 κ 应非对称: κ_ij={} κ_ji={}",
            判定.κ_ij, 判定.κ_ji
        );
        assert!(判定.描述.contains("→"), "方向描述应含箭头: {}", 判定.描述);
    }

    /// D6-V4: 共同慢模驱动——对称耦合 → 不误判方向
    #[test]
    fn 测试_D6V4_共同驱动() {
        let n = 2;
        let mut l = DMatrix::<NComplex<f64>>::zeros(n, n);
        l[(0, 0)] = NComplex::new(0.0, -2.0);
        l[(1, 1)] = NComplex::new(0.1, -0.5);
        // 对称耦合 → κ_01≈κ_10 → 共同驱动
        l[(0, 1)] = NComplex::new(0.3, 0.1);
        l[(1, 0)] = NComplex::new(0.3, 0.1);
        let 求解 = 非厄米谱求解器::default();
        let 分解 = 求解.分解(&l).unwrap();
        let 结果 = 因果探测器::default().探测(&l, &分解);
        let 判定 = &结果.方向判据[0];
        assert_eq!(判定.方向, 方向类型::共同驱动, "对称耦合不误判方向: {}", 判定.描述);
        assert!((判定.κ_ij - 判定.κ_ji).abs() < 1e-6, "κ_ij≈κ_ji: {} vs {}", 判定.κ_ij, 判定.κ_ji);
    }

    /// D6-V5: 谱信息不足（对角近似 L）→ 不静默给方向
    #[test]
    fn 测试_D6V5_谱信息不足() {
        let n = 2;
        let mut l = DMatrix::<NComplex<f64>>::zeros(n, n);
        l[(0, 0)] = NComplex::new(0.0, -2.0);
        l[(1, 1)] = NComplex::new(0.1, -0.5);
        let 求解 = 非厄米谱求解器::default();
        let 分解 = 求解.分解(&l).unwrap();
        let 结果 = 因果探测器::default().探测(&l, &分解);
        assert!(!结果.谱信息充足, "对角 L 应谱信息不足");
        assert!(结果.方向判据.is_empty(), "不足时无方向判定");
        assert!(结果.说明.contains("谱信息不足"), "{}", 结果.说明);
    }

    /// D6-V7: 安全约束——ε_wave≥0.3 拒绝, <0.3 通过
    #[test]
    fn 测试_D6V7_安全约束() {
        assert!(因果探测器::安全校验(0.2).is_ok(), "0.2 安全区应通过");
        assert!(因果探测器::安全校验(0.5).is_err(), "0.5 过渡区应拒绝");
        assert!(因果探测器::安全校验(1.2).is_err(), "1.2 危险区应拒绝");
    }

    // ==================== T-7 P39 G-12 三重判据（D5+D6 v1.1 §8 V3a/V4/V5/V7） ====================

    /// V3a: 三重判据 A+B+C 全满足 → 方向耦合（非对称 R≈6 + 响应时序 + 样本充足）
    /// 注: 分解按 τ 降序（慢模在前 P22 §6.2）——模索引与输入 L 索引错位,
    ///     故先预探测确定 κ 非对称方向, 再沿该方向注入响应时序 Δt（与 R 一致）
    #[test]
    fn 测试_P39V3a_三重判据方向耦合() {
        let l = 非对称耦合L(); // 输入 L[0][1]=0.3 强（快→慢）, L[1][0]=0.05 弱 → 方向耦合
        let 求解 = 非厄米谱求解器::default();
        let 分解 = 求解.分解(&l).unwrap();
        // 预探测（Δt=0）: 确定 κ 非对称方向
        let 空延迟 = std::collections::HashMap::new();
        let 预查 = 因果探测器::default().探测_v10(&l, &分解, &空延迟, 0.1, 500);
        let 预判定 = &预查.方向判据[0];
        assert!(
            预判定.非对称比R_ij > 3.0 || 1.0 / (预判定.非对称比R_ij + 1e-9) > 3.0,
            "κ 应非对称: R={:.2}", 预判定.非对称比R_ij
        );
        // 沿 R 方向设置 Δt: R_ij>3 → i→j（Δt_ij>0）; 1/R>3 → j→i（Δt_ij<0）
        let 方向符号 = if 预判定.非对称比R_ij >= 1.0 { 1.0 } else { -1.0 };
        let mut 延迟 = std::collections::HashMap::new();
        延迟.insert((0, 1), 方向符号 * 0.5);
        let 结果 = 因果探测器::default().探测_v10(&l, &分解, &延迟, 0.1, 500);
        assert!(结果.谱信息充足);
        let 判定 = &结果.方向判据[0];
        let 判据A = 判定.判据A非对称比 || (1.0 / (判定.非对称比R_ij + 1e-9)) > 3.0;
        let 判据B = 判定.判据B时序 || 判定.时序差Δt_ij < -2.0 * 0.1;
        assert!(判据A, "判据A（任一方向）: R_ij={:.2}", 判定.非对称比R_ij);
        assert!(判据B, "判据B（任一方向）: |Δt|={:.3}>2σ_t", 判定.时序差Δt_ij.abs());
        assert!(判定.判据C排除共同慢模, "判据C: 排除共同慢模");
        assert_eq!(判定.方向, 方向类型::方向耦合, "{}", 判定.描述);
        assert!(判定.描述.contains("→"), "应含方向箭头: {}", 判定.描述);
    }

    /// V3a 补充: 样本不足——M=3 < (3/|κ|τ)² → 样本不足
    #[test]
    fn 测试_P39V3a_样本不足() {
        let n = 2;
        let mut l = DMatrix::<NComplex<f64>>::zeros(n, n);
        l[(0, 0)] = NComplex::new(0.0, -0.5);   // 慢模
        l[(1, 1)] = NComplex::new(0.1, -2.0);   // 快模
        l[(0, 1)] = NComplex::new(0.3, 0.1);
        l[(1, 0)] = NComplex::new(0.05, 0.02);
        let 求解 = 非厄米谱求解器::default();
        let 分解 = 求解.分解(&l).unwrap();
        let mut 延迟 = std::collections::HashMap::new();
        延迟.insert((0, 1), 0.5);
        let 结果 = 因果探测器::default().探测_v10(&l, &分解, &延迟, 0.1, 3);
        let 判定 = &结果.方向判据[0];
        assert_eq!(判定.方向, 方向类型::样本不足, "{}", 判定.描述);
        assert_eq!(判定.样本数, 3);
    }

    /// V4: 共同慢模不误判——对称耦合（R≈1）判据A 不满足 → 不误判方向
    #[test]
    fn 测试_P39V4_对称不误判() {
        let n = 2;
        let mut l = DMatrix::<NComplex<f64>>::zeros(n, n);
        l[(0, 0)] = NComplex::new(0.0, -2.0);   // 快模
        l[(1, 1)] = NComplex::new(0.1, -0.5);   // 慢模
        // 对称耦合 → R≈1
        l[(0, 1)] = NComplex::new(0.3, 0.1);
        l[(1, 0)] = NComplex::new(0.3, 0.1);
        let 求解 = 非厄米谱求解器::default();
        let 分解 = 求解.分解(&l).unwrap();
        let mut 延迟 = std::collections::HashMap::new();
        延迟.insert((0, 1), 0.2);
        let 结果 = 因果探测器::default().探测_v10(&l, &分解, &延迟, 0.05, 500);
        let 判定 = &结果.方向判据[0];
        assert!(!判定.判据A非对称比, "R≈1 判据A 不满足: R={:.3}", 判定.非对称比R_ij);
        assert_eq!(判定.方向, 方向类型::对称无耦合, "对称耦合判对称无耦合: {}", 判定.描述);
    }

    /// V5: D1 联动——对角近似 L → 谱信息充足=false（探测_v10 不静默给方向）
    #[test]
    fn 测试_P39V5_探测v10谱信息不足() {
        let n = 2;
        let mut l = DMatrix::<NComplex<f64>>::zeros(n, n);
        l[(0, 0)] = NComplex::new(0.0, -2.0);
        l[(1, 1)] = NComplex::new(0.1, -0.5);
        let 求解 = 非厄米谱求解器::default();
        let 分解 = 求解.分解(&l).unwrap();
        let 延迟 = std::collections::HashMap::new();
        let 结果 = 因果探测器::default().探测_v10(&l, &分解, &延迟, 0.1, 30);
        assert!(!结果.谱信息充足, "对角 L 应谱信息不足");
        assert!(结果.方向判据.is_empty(), "不足时无方向判定");
    }

    /// V7: 安全约束——ε_wave≥1 危险区拒绝（三重判据入口前置校验）
    #[test]
    fn 测试_P39V7_安全约束() {
        assert!(因果探测器::安全校验(0.2).is_ok(), "0.2 安全区应通过");
        assert!(因果探测器::安全校验(0.5).is_err(), "0.5 过渡区应拒绝");
        assert!(因果探测器::安全校验(1.0).is_err(), "1.0 危险区应拒绝");
        assert!(因果探测器::安全校验(1.2).is_err(), "1.2 危险区应拒绝");
    }

    /// ★ T-74 G-12 第一步反例守卫（强混合, P39 §14.2 类 I 快照版必须读裸元）:
    /// 近简并两模（detune≈0.05）＋强非对称耦合 |L01|=0.5、|L10|=0.05（**裸元比 10:1**）。
    /// 本征矢被强耦合显著旋转后, **本征基投影非对称比塌缩到 ≈1.29（近对称）** ⇒
    /// 若判据 A 用投影, 会把真实 10:1 的内部方向耦合误判成"共同驱动"。故快照版读裸元。
    /// （判据 C 的 |w_i†v_S|² 慢模身份重叠是另一用途, 不受此影响、仍保留。）
    #[test]
    fn 测试_P39G12_强混合裸元与投影分道扬镳守卫() {
        let mut l = DMatrix::<NComplex<f64>>::zeros(2, 2);
        l[(0, 0)] = NComplex::new(0.0, -1.0);
        l[(1, 1)] = NComplex::new(0.05, -1.0); // 近简并（detune≈0.05）→ 强混合
        l[(0, 1)] = NComplex::new(0.5, 0.0); // 0→1 强
        l[(1, 0)] = NComplex::new(0.05, 0.0); // 1→0 弱

        // ① 裸基: 非对称比 10:1, 快照版（读裸元）判方向耦合
        let 求解 = 非厄米谱求解器::default();
        let 分解 = 求解.分解(&l).unwrap();
        let 结果 = 因果探测器::default().探测(&l, &分解);
        let 判定 = &结果.方向判据[0];
        assert_eq!(判定.方向, 方向类型::方向耦合, "裸元 10:1 应判方向耦合: {}", 判定.描述);
        assert!(判定.非对称比R_ij > 5.0 || 1.0 / (判定.非对称比R_ij + 1e-9) > 5.0,
            "裸元非对称比应≈10: {:.3}", 判定.非对称比R_ij);

        // ② 含扰本征基投影 w†δL v 的非对称比（同样输入）——强混合下塌缩到近对称
        let 投影 = |i: usize, j: usize| {
            let mut s = NComplex::new(0.0, 0.0);
            for x in 0..2 {
                for y in 0..2 {
                    if x == y { continue; }
                    let wi = NComplex::new(分解.模[i].w[x], 分解.模[i].w虚[x]);
                    let vj = NComplex::new(分解.模[j].v[y], 分解.模[j].v虚[y]);
                    s += wi.conjugate() * l[(x, y)] * vj;
                }
            }
            s.norm()
        };
        let p01 = 投影(0, 1);
        let p10 = 投影(1, 0);
        let 投影比 = p01 / (p10 + 1e-30);
        assert!(
            投影比 < 2.0,
            "强混合下投影非对称比应塌缩到近对称（实测≈1.29）, 证明快照版不可用投影: {:.3}", 投影比
        );
    }

    // ══════════════════════════════════════════════════════════════════
    // T-82 · G-12 第二步：类 II 外扰造波版 κ（禁自指 ＋ 解析真值对账）
    // ══════════════════════════════════════════════════════════════════
    // 对账原理：**对角 L0** ⇒ v0_j = w0_j = e_j（标准基，双正交 w†v=1）
    //   ⇒ **解析 κ_II(i,j) = ΔL[i,j]**（模序 = 通道序）。
    //   ⚠️ 对角元取 **λ_k = i·γ_k（γ_k<0 稳定模）**：若取**纯实** λ（γ=0）则 τ=−1/γ=∞、
    //   `κτ` 无意义（本项目 γ 为耗散率口径）。

    fn 对角L0(γs: &[f64]) -> DMatrix<NComplex<f64>> {
        let n = γs.len();
        let mut l = DMatrix::<NComplex<f64>>::zeros(n, n);
        for (k, γ) in γs.iter().enumerate() {
            l[(k, k)] = NComplex::new(0.0, *γ);
        }
        l
    }

    fn 矩阵(n: usize, 元: &[(usize, usize, f64, f64)]) -> DMatrix<NComplex<f64>> {
        let mut m = DMatrix::<NComplex<f64>>::zeros(n, n);
        for &(i, j, re, im) in 元 {
            m[(i, j)] = NComplex::new(re, im);
        }
        m
    }

    fn 跨模项<'a>(r: &'a 类II结果, i: usize, j: usize) -> &'a 类II跨模不变量 {
        r.跨模.iter().find(|e| e.i == i && e.j == j).expect("跨模项缺失")
    }

    fn 相对误差(实测: f64, 真值: f64) -> f64 {
        if 真值.abs() < 1e-300 {
            实测.abs()
        } else {
            (实测 - 真值).abs() / 真值.abs()
        }
    }

    /// ① 解析真值对账 · 组 1/5：n=2 实非对角 ⇒ |κ_ij| = |ΔL[i,j]|（机器精度）
    #[test]
    fn t82_解析对账_组1_n2实非对角() {
        let l0 = 对角L0(&[-0.1, -0.4]);
        let δl = 矩阵(2, &[(0, 1, 0.7, 0.0), (1, 0, 0.2, 0.0)]);
        let r = 类II探测器::default().探测_类II(&l0, &δl, None).expect("类 II 应成功");
        let e01 = 跨模项(&r, 0, 1);
        assert!(相对误差(e01.κ模ij, 0.7) < 1e-12, "|κ_01| 应 = |ΔL[0,1]|=0.7, 实测 {}", e01.κ模ij);
        assert!(相对误差(e01.κ模ji, 0.2) < 1e-12, "|κ_10| 应 = 0.2, 实测 {}", e01.κ模ji);
        // R 口径与类 I 同族：|κ_ij|/(|κ_ji|+ε=1e-9) ⇒ 与 3.5 相差 ε/κ_ji ≈ 5e-9（容差取 1e-7）
        assert!(相对误差(e01.非对称比R_ij, 0.7 / 0.2) < 1e-7, "R_ij≈3.5, 实测 {}", e01.非对称比R_ij);
        assert!(e01.判据A非对称比, "R=3.5 超 1+0.1 应判据A 满足");
        // 对角 L0 ⇒ v_S = e_0：仅模0 挂慢模、模1 不挂 ⇒ 非共同驱动，按 κ 大小判方向
        assert!(e01.判据C排除共同慢模, "仅一个模挂慢模 ⇒ 排除共同驱动");
        assert_eq!(e01.方向, 方向类型::方向耦合, "κ_01>κ_10 应记方向耦合");
    }

    /// ② 解析真值对账 · 组 2/5：n=3 复非对角 ⇒ |κ_ij| = |ΔL[i,j]|（机器精度）
    #[test]
    fn t82_解析对账_组2_n3复非对角() {
        let l0 = 对角L0(&[-0.1, -0.2, -0.4]);
        let 元 = [
            (0usize, 1usize, 0.30f64, 0.40f64), // |0.3+0.4i| = 0.5
            (1usize, 0usize, 0.05f64, -0.12f64), // |0.05-0.12i| = 0.13
            (0usize, 2usize, -0.60f64, 0.80f64), // = 1.0
            (2usize, 1usize, 0.09f64, 0.00f64),
        ];
        let δl = 矩阵(3, &元);
        let r = 类II探测器::default().探测_类II(&l0, &δl, None).expect("类 II 应成功");
        assert_eq!(r.跨模.len(), 3, "n=3 ⇒ i<j 共 3 对");
        for &(i, j, re, im) in &元 {
            let 真值 = (re * re + im * im).sqrt();
            let e = 跨模项(&r, i.min(j), i.max(j));
            let 实测 = if i < j { e.κ模ij } else { e.κ模ji };
            assert!(
                相对误差(实测, 真值) < 1e-12,
                "|κ_{}{}| 应 = {:.12}, 实测 {}",
                i,
                j,
                真值,
                实测
            );
        }
    }

    /// ③ 解析真值对账 · 组 3/5：n=4 复 ΔL（含对角元）⇒ 对角 |κ_ii| 也须 = |ΔL[i,i]|
    #[test]
    fn t82_解析对账_组3_n4含对角() {
        let l0 = 对角L0(&[-0.1, -0.2, -0.4, -0.8]);
        let 元 = [
            (0usize, 0usize, 0.25f64, 0.10f64),
            (1usize, 1usize, -0.15f64, 0.20f64),
            (2usize, 3usize, 0.44f64, -0.33f64),
            (3usize, 0usize, 0.01f64, 0.02f64),
        ];
        let δl = 矩阵(4, &元);
        let r = 类II探测器::default().探测_类II(&l0, &δl, None).expect("类 II 应成功");
        for &(i, j, re, im) in &元 {
            let 真值 = (re * re + im * im).sqrt();
            let 实测 = if i == j {
                r.单模[i].κ模
            } else {
                let e = 跨模项(&r, i.min(j), i.max(j));
                if i < j { e.κ模ij } else { e.κ模ji }
            };
            assert!(相对误差(实测, 真值) < 1e-12, "({},{}) 应 = {:.12}, 实测 {}", i, j, 真值, 实测);
        }
    }

    /// ④ 解析真值对账 · 组 4/5：ΔL = 单位阵元 E_pq（**分量级**解析真值
    ///    κ_ij = conj(w0_i[p])·v0_j[q]，对角 L0 下 = δ_{i,p}·δ_{j,q}，可手核）
    #[test]
    fn t82_解析对账_组4_单位阵元() {
        let l0 = 对角L0(&[-0.1, -0.2, -0.4]);
        // ΔL = E_{0,2}（唯一非零元在 (0,2)=1）⇒ 仅 κ_02 = 1，其余全 0
        let δl = 矩阵(3, &[(0, 2, 1.0, 0.0)]);
        let r = 类II探测器::default().探测_类II(&l0, &δl, None).expect("类 II 应成功");
        assert!(相对误差(跨模项(&r, 0, 1).κ模ij, 0.0) < 1e-12);
        assert!(相对误差(跨模项(&r, 1, 2).κ模ij, 0.0) < 1e-12);
        assert!(相对误差(跨模项(&r, 0, 2).κ模ij, 1.0) < 1e-12, "E_02 ⇒ κ_02=1");
        assert!(相对误差(跨模项(&r, 0, 2).κ模ji, 0.0) < 1e-12, "反向仍 0（E_02 非对称）");
        // 对角元全零 ⇒ 单模模量 0
        for i in 0..3 {
            assert!(相对误差(r.单模[i].κ模, 0.0) < 1e-12, "E_02 对角响应应为 0");
        }
    }

    /// ⑤ 解析真值对账 · 组 5/5：n=2 复 ΔL 全元件（双向非对称）＋ 判据C 慢模投影
    #[test]
    fn t82_解析对账_组5_n2复全元件与判据C() {
        let l0 = 对角L0(&[-0.1, -0.9]); // τ=[10, 1.11] ⇒ 索引0 为唯一慢模
        let δl = 矩阵(2, &[(0, 0, 0.5, -0.5), (0, 1, 0.3, 0.9), (1, 0, 0.6, -0.1), (1, 1, -0.2, 0.0)]);
        let r = 类II探测器::default().探测_类II(&l0, &δl, None).expect("类 II 应成功");
        assert!(相对误差(r.单模[0].κ模, (0.5f64 * 0.5 + 0.5 * 0.5).sqrt()) < 1e-12);
        assert!(相对误差(r.单模[1].κ模, 0.2) < 1e-12);
        assert!(相对误差(跨模项(&r, 0, 1).κ模ij, (0.09f64 + 0.81).sqrt()) < 1e-12);
        assert!(相对误差(跨模项(&r, 0, 1).κ模ji, (0.36f64 + 0.01).sqrt()) < 1e-12);
        // 判据C：对角 L0 下 w0_i†v_S = w0_i†e_0 = δ_{i,0} ⇒ 仅模0 挂慢模、模1 不挂 ⇒ 排除共同慢模
        assert!(跨模项(&r, 0, 1).判据C排除共同慢模, "仅一个模挂慢模 ⇒ 非共同驱动");
    }

    /// ★ 相位硬测试（对齐 T-72 族）：ΔL 做**确定性随机相位旋转** ⇒ **不变量逐位不变**
    #[test]
    fn t82_相位旋转_不变量逐位不变() {
        let l0 = 对角L0(&[-0.1, -0.2, -0.4]);
        let δl = 矩阵(
            3,
            &[
                (0, 1, 0.30, 0.40),
                (1, 0, 0.05, -0.12),
                (2, 2, -0.11, 0.07),
                (0, 2, 0.60, -0.80),
            ],
        );
        let 探测 = 类II探测器::default();
        let 基准 = 探测.探测_类II(&l0, &δl, None).unwrap();
        // 确定性"随机"相位（不引 rand, 保证可复现）
        for k in 0..5u32 {
            let θ = (k as f64 * 2.399963229728653).sin() * std::f64::consts::PI;
            let 相位 = NComplex::new(θ.cos(), θ.sin());
            let δl2 = δl.map(|z| z * 相位);
            let r2 = 探测.探测_类II(&l0, &δl2, None).unwrap();
            for (a, b) in 基准.跨模.iter().zip(r2.跨模.iter()) {
                assert_eq!(a.i, b.i);
                assert_eq!(a.j, b.j);
                // 全局相位 e^{iθ} 下 |κ| 为**严格不变量**（仅浮点舍入级差异 ⇒ 相对误差打到机器精度）
                assert!(相对误差(b.κ模ij, a.κ模ij) < 1e-13, "|κ_ij| 不变量（θ={}）", θ);
                assert!(相对误差(b.κ模ji, a.κ模ji) < 1e-13, "|κ_ji| 不变量（θ={}）", θ);
                // κτ = |κ|·τ：小 κ 项做相对比较会灾难性放大 ⇒ 用绝对+相对混合容差
                assert!(
                    (b.κτ - a.κτ).abs() <= 1e-13 * (1.0 + a.κτ.abs()),
                    "κτ 不变量（θ={}, {} vs {}）",
                    θ,
                    a.κτ,
                    b.κτ
                );
                assert_eq!(a.判据A非对称比, b.判据A非对称比);
                assert_eq!(a.方向, b.方向);
            }
            for (a, b) in 基准.单模.iter().zip(r2.单模.iter()) {
                assert!(相对误差(b.κ模, a.κ模) < 1e-13, "对角模量不变量");
            }
        }
    }

    /// ★ "甲"的唯一合法位置：单模对角**相对已知 φ_d** 的 Re/Im 按预期旋转；
    ///   未给 φ_d ⇒ **不暴露任何同相/正交**（跨模路径本来就拿不到单边量）
    #[test]
    fn t82_单模φd_同相正交旋转_未给则不暴露() {
        let l0 = 对角L0(&[-0.1, -0.2]);
        // ΔL[0,0] = 3 + 4i ⇒ |κ_00| = 5
        let δl = 矩阵(2, &[(0, 0, 3.0, 4.0), (1, 0, 0.1, 0.0)]);
        let 探测 = 类II探测器::default();
        // 未给 φ_d ⇒ 单模只有模量, 无 Re/Im
        let r0 = 探测.探测_类II(&l0, &δl, None).unwrap();
        assert!(r0.单模.iter().all(|m| m.φ_d.is_none() && m.同相.is_none() && m.正交.is_none()),
                "未给 φ_d 必须不暴露同相/正交");
        assert!(相对误差(r0.单模[0].κ模, 5.0) < 1e-12);
        // φ_d = 0 ⇒ (同相, 正交) = (Re κ, Im κ) = (3, 4)
        let r1 = 探测.探测_类II(&l0, &δl, Some(0.0)).unwrap();
        assert!(相对误差(r1.单模[0].同相.unwrap(), 3.0) < 1e-12, "φ_d=0 ⇒ 同相=Re κ");
        assert!(相对误差(r1.单模[0].正交.unwrap(), 4.0) < 1e-12, "φ_d=0 ⇒ 正交=Im κ");
        // φ_d = π/2 ⇒ e^{−iπ/2}κ = −i(3+4i) = 4 − 3i ⇒ (同相, 正交) = (4, −3)
        let r2 = 探测.探测_类II(&l0, &δl, Some(std::f64::consts::FRAC_PI_2)).unwrap();
        assert!(相对误差(r2.单模[0].同相.unwrap(), 4.0) < 1e-12, "φ_d=π/2 同相应=4");
        assert!(相对误差(r2.单模[0].正交.unwrap(), -3.0) < 1e-12, "φ_d=π/2 正交应=−3");
        // φ_d = π ⇒ (−3, −4)
        let r3 = 探测.探测_类II(&l0, &δl, Some(std::f64::consts::PI)).unwrap();
        assert!(相对误差(r3.单模[0].同相.unwrap(), -3.0) < 1e-12);
        assert!(相对误差(r3.单模[0].正交.unwrap(), -4.0) < 1e-12);
        // 模量不随 φ_d 变（相位只旋转、不改模）
        for r in [&r1, &r2, &r3] {
            assert!(相对误差(r.单模[0].κ模, 5.0) < 1e-12);
        }
    }

    /// ★★ 禁自指守卫：类 II（取 **L0** 基）与"对含扰完整 L 分解后投影 ΔL 自身"
    ///   （**G-12 的病**）在**简并/EP 邻域**必须**分道扬镳** —— 后者给错值。
    #[test]
    fn t82_禁自指_EP邻域两法分道扬镳() {
        // L0 = diag(0, −1)（实负非正：0 为慢模）, ΔL = [[0,1],[0,0]]（Jordan 型非对角）
        let l0 = 对角L0(&[0.0, -1.0]);
        let δl = 矩阵(2, &[(0, 1, 1.0, 0.0)]);
        let 探测 = 类II探测器::default();
        // ── 类 II（正确路径）：w/v 取 L0 基 ⇒ 解析真值 |κ_01| = |ΔL[0,1]| = 1
        let r = 探测.探测_类II(&l0, &δl, None).expect("类 II 应成功");
        assert!(相对误差(跨模项(&r, 0, 1).κ模ij, 1.0) < 1e-12, "类 II 应给解析真值 1.0");
        // ── 自指做法（**G-12 的病**，仅测试内复现, 不进生产代码）：
        //    对 L = L0 + ΔL 分解, 再用其 w/v 投影 δL（对角置 0）自身
        let l_full = &l0 + &δl;
        let 分解完整 = 非厄米谱求解器::default().分解(&l_full).expect("完整 L 应可分解");
        let n = 2;
        let 取复矢 = |re: &[f64], im: &[f64]| -> Vec<NComplex<f64>> {
            (0..n).map(|k| NComplex::new(re[k], im.get(k).copied().unwrap_or(0.0))).collect()
        };
        let v_f: Vec<Vec<NComplex<f64>>> = 分解完整.模.iter().map(|m| 取复矢(&m.v, &m.v虚)).collect();
        let w_f: Vec<Vec<NComplex<f64>>> = 分解完整.模.iter().map(|m| 取复矢(&m.w, &m.w虚)).collect();
        let 自指κ01 = {
            let mut 累 = NComplex::new(0.0, 0.0);
            for k in 0..n {
                let mut 内 = NComplex::new(0.0, 0.0);
                for m in 0..n {
                    if k != m {
                        内 += δl[(k, m)] * v_f[1][m];
                    }
                }
                累 += w_f[0][k].conj() * 内;
            }
            累.norm()
        };
        assert!(
            相对误差(自指κ01, 1.0) > 1e-6,
            "自指做法（对含扰 L 分解后投影 ΔL 自身）在 EP 邻域应给出**错值**（≠ 1.0）, 实测 {}",
            自指κ01
        );
    }

    /// ★ 禁平凡验收：ΔL 全零 ⇒ 如实报"无外扰"、**不给方向**（禁零输入充数）
    #[test]
    fn t82_零外扰_如实拒绝不充数() {
        let l0 = 对角L0(&[-0.1, -0.2]);
        let δl = 矩阵(2, &[]);
        let e = 类II探测器::default().探测_类II(&l0, &δl, None).expect_err("零 ΔL 必须报错");
        assert!(e.contains("全零"), "错误须点明零外扰: {}", e);
        // 维数不一致也要挡
        let 错维 = 矩阵(3, &[(0, 1, 1.0, 0.0)]);
        assert!(类II探测器::default().探测_类II(&l0, &错维, None).is_err(), "ΔL 维数须校验");
        // 非方阵 L0 也要挡
        let mut 非方 = DMatrix::<NComplex<f64>>::zeros(2, 3);
        非方[(0, 0)] = NComplex::new(-0.1, 0.0);
        assert!(类II探测器::default().探测_类II(&非方, &δl, None).is_err(), "L0 须为方阵");
    }

    /// ★ 禁自指取证：结果的 `基来源` 必须自陈"取 L0"
    #[test]
    fn t82_基来源自陈取L0() {
        let l0 = 对角L0(&[-0.1, -0.2]);
        let δl = 矩阵(2, &[(0, 1, 0.5, 0.0)]);
        let r = 类II探测器::default().探测_类II(&l0, &δl, None).unwrap();
        assert!(r.基来源.contains("L0"), "须自陈取 L0 基线: {}", r.基来源);
        assert!(r.基来源.contains("禁自指") || r.基来源.contains("未对"), "须含禁自指取证");
        assert!(!r.判据B未提供原因.is_empty(), "判据B 不可得须如实标注原因");
        assert!(r.判据B未提供原因.contains("时序"), "原因须点明缺时序");
    }


    /// ★ **对账表打印**（供 PM 独立复算留痕）：
    /// `cargo test --lib t82_对账表 -- --nocapture` 抓取「解析真值 vs 实测 vs 相对误差」。
    /// **默认不写文件、无副作用**（纯打印）。
    #[test]
    fn t82_对账表_打印() {
        type 组 = (
            &'static str,
            DMatrix<NComplex<f64>>,
            DMatrix<NComplex<f64>>,
            Vec<(usize, usize, f64, f64)>,
        );
        let 用例: Vec<组> = vec![
            (
                "组1 n=2 实非对角",
                对角L0(&[-0.05, -0.2]),
                矩阵(2, &[(0, 1, 0.7, 0.0), (1, 0, 0.2, 0.0)]),
                vec![(0, 1, 0.7, 0.0), (1, 0, 0.2, 0.0)],
            ),
            (
                "组2 n=3 复非对角",
                对角L0(&[-0.05, -0.1, -0.2]),
                矩阵(3, &[(0, 1, 0.30, 0.40), (1, 0, 0.05, -0.12), (0, 2, -0.60, 0.80), (2, 1, 0.09, 0.0)]),
                vec![(0, 1, 0.30, 0.40), (1, 0, 0.05, -0.12), (0, 2, -0.60, 0.80), (2, 1, 0.09, 0.0)],
            ),
            (
                "组3 n=4 含对角",
                对角L0(&[-0.05, -0.1, -0.2, -0.4]),
                矩阵(4, &[(0, 0, 0.25, 0.10), (1, 1, -0.15, 0.20), (2, 3, 0.44, -0.33), (3, 0, 0.01, 0.02)]),
                vec![(0, 0, 0.25, 0.10), (1, 1, -0.15, 0.20), (2, 3, 0.44, -0.33), (3, 0, 0.01, 0.02)],
            ),
            (
                "组4 n=3 单位阵元 E02",
                对角L0(&[-0.05, -0.1, -0.2]),
                矩阵(3, &[(0, 2, 1.0, 0.0)]),
                vec![(0, 1, 0.0, 0.0), (1, 2, 0.0, 0.0), (0, 2, 1.0, 0.0)],
            ),
            (
                "组5 n=2 复全元件",
                对角L0(&[-0.1, -0.9]),
                矩阵(2, &[(0, 0, 0.5, -0.5), (0, 1, 0.3, 0.9), (1, 0, 0.6, -0.1), (1, 1, -0.2, 0.0)]),
                vec![(0, 0, 0.5, -0.5), (1, 1, -0.2, 0.0), (0, 1, 0.3, 0.9), (1, 0, 0.6, -0.1)],
            ),
        ];
        let 探测 = 类II探测器::default();
        let mut 最大误差 = 0.0f64;
        for (名, l0, δl, 真值表) in &用例 {
            let r = 探测.探测_类II(l0, δl, None).expect("类 II 应成功");
            for &(i, j, re, im) in 真值表 {
                let 真值 = (re * re + im * im).sqrt();
                let 实测 = if i == j {
                    r.单模[i].κ模
                } else {
                    let e = 跨模项(&r, i.min(j), i.max(j));
                    if i < j { e.κ模ij } else { e.κ模ji }
                };
                let 误 = 相对误差(实测, 真值);
                最大误差 = 最大误差.max(误);
                println!(
                    "T82RECON|{}|({},{})|解析={:.17e}|实测={:.17e}|相对误差={:.3e}",
                    名, i, j, 真值, 实测, 误
                );
            }
        }
        println!("T82RECON|最大相对误差={:.3e}", 最大误差);
        assert!(最大误差 < 1e-12, "全部对账组须 ≤1e-12（实测最大 {:.3e}）", 最大误差);
    }


    // ══════════════════════════════════════════════════════════════════
    // T-82 · F3 验收补强（PM 复算补强：验收"证伪力"差的两格）
    // ══════════════════════════════════════════════════════════════════
    // 背景：原 16 组对账**全部用对角 L0**（L0_对角γ）⇒ w0_j=v0_j=e_j ⇒
    //   **"非平凡本征基 ＋ 双正交归一"路径从未被行使**（跳过 L0 直接返回 |ΔL[i,j]|
    //   的**退化实现会通过全部 16 组**）⇒ 本段补两格。

    /// 构造**非对角** L0 = [[λ1, β], [0, λ2]]（上三角；λ 互异 ⇒ 本征基**非平凡**：
    /// v0_0 = e_0、v0_1 ∝ [β, λ1−λ2]）——真实数据的 L0 必非对角，这才是 w0/v0 起作用的场合。
    fn 非对角L0(λ1: NComplex<f64>, λ2: NComplex<f64>, β: f64) -> DMatrix<NComplex<f64>> {
        let mut l = DMatrix::<NComplex<f64>>::zeros(2, 2);
        l[(0, 0)] = λ1;
        l[(1, 1)] = λ2;
        l[(0, 1)] = NComplex::new(β, 0.0);
        l
    }

    fn 上三角L0默认() -> DMatrix<NComplex<f64>> {
        非对角L0(NComplex::new(0.0, -0.05), NComplex::new(0.0, -0.4), 0.7)
    }

    /// ★ F3① **非对角 L0 解析恒等式**：`ΔL = L0` ⇒ `κ_II(i,j) = w0_i†L0 v0_j = λ_j·δ_ij`。
    /// 真值**精确可解析**（不依赖归一化细节，仅靠双正交 w0_i†v0_j=δ_ij）。
    /// ⚠️ **判别力**：退化实现（跳过 L0、直接返回 `ΔL[i,j]`）会给 `κ_01 = L0[0,1] = 0.7 ≠ 0` ⇒ **必红**。
    #[test]
    fn t82_F3a_非对角L0_解析恒等式_退化实现必红() {
        let l0 = 上三角L0默认();
        assert!(l0[(0, 1)].norm() > 1e-6, "本组必须用非对角 L0（否则退回已覆盖的对角情形）");
        let r = 类II探测器::default()
            .探测_类II(&l0, &l0.clone(), None)
            .expect("ΔL=L0 应成功");
        // κ_jj = λ_j（对角精确）
        assert!(相对误差(r.单模[0].κ模, l0[(0, 0)].norm()) < 1e-10, "κ_00 应=|λ1|, 实测 {}", r.单模[0].κ模);
        assert!(相对误差(r.单模[1].κ模, l0[(1, 1)].norm()) < 1e-10, "κ_11 应=|λ2|, 实测 {}", r.单模[1].κ模);
        // κ_ij = 0（i≠j）
        assert!(相对误差(跨模项(&r, 0, 1).κ模ij, 0.0) < 1e-10, "ΔL=L0 ⇒ 非对角 κ 必为 0");
        assert!(相对误差(跨模项(&r, 0, 1).κ模ji, 0.0) < 1e-10);
        // 判别力取证：退化实现会在此给 0.7 ⇒ 本条断言把"是否会红"钉死
        assert!(l0[(0, 1)].norm() > 0.5, "退化实现在本组会给 |κ_01|=0.7（≠0）⇒ 必红");
    }

    /// ★ F3①b：非对角 L0 ＋ **单位阵元 ΔL = E_00** ⇒ 元件级手推（第二实现独立重算）＋退化判别。
    /// 真实现：`κ_01 = w0_0†E_00 v0_1 = conj(w0_0[0])·v0_1[0]`（v0_1 第一分量 ∝ β ≠ 0 ⇒ **非零**）；
    /// 退化实现：会返回 `ΔL[0,1] = 0` ⇒ **必红**。
    #[test]
    fn t82_F3b_非对角L0_元件级_退化给零必红() {
        let l0 = 上三角L0默认();
        let δl = 矩阵(2, &[(0, 0, 1.0, 0.0)]);
        let r = 类II探测器::default().探测_类II(&l0, &δl, None).expect("应成功");
        // 第二实现（独立手算：按定义式 w0_i†ΔL v0_j，显式共轭）——不共用实现内部变量
        let 分解 = 非厄米谱求解器::default().分解(&l0).expect("L0 应可分解");
        let n = 2;
        let 取复矢 = |re: &[f64], im: &[f64]| -> Vec<NComplex<f64>> {
            (0..n).map(|k| NComplex::new(re[k], im.get(k).copied().unwrap_or(0.0))).collect()
        };
        let v0: Vec<Vec<NComplex<f64>>> = 分解.模.iter().map(|m| 取复矢(&m.v, &m.v虚)).collect();
        let w0: Vec<Vec<NComplex<f64>>> = 分解.模.iter().map(|m| 取复矢(&m.w, &m.w虚)).collect();
        let mut 手算 = NComplex::new(0.0, 0.0);
        for k in 0..n {
            let mut 内 = NComplex::new(0.0, 0.0);
            for m in 0..n {
                内 += δl[(k, m)] * v0[1][m];
            }
            手算 += w0[0][k].conj() * 内;
        }
        let e = 跨模项(&r, 0, 1);
        assert!(相对误差(e.κ模ij, 手算.norm()) < 1e-12, "实现与独立手算须一致: {} vs {}", e.κ模ij, 手算.norm());
        assert!(e.κ模ij > 1e-6, "非对角 L0 下真实现应给非零（退化实现会给 ΔL[0,1]=0）⇒ 必红");
        // 反向取证：ΔL[0,1] 确为 0（退化实现的值）
        assert!(δl[(0, 1)].norm() < 1e-300, "本组 ΔL[0,1]=0 ⇒ 退化实现在此给 0");
    }

    /// ★ F3② **酉基变换硬测试**（T-72 族真判别器，替换"全局相位"这类**零信息**测试）：
    /// `L0→U L0 U†`、`ΔL→U ΔL U†`（任意酉 U ⊂ 实正交族）⇒ κ 的**不变量应不变**。
    ///
    /// 为何必须换：`ΔL→e^{iθ}ΔL` 下 `|e^{iθ}κ|=|κ|` 对**任何线性实现**都成立
    /// （**连自指做法也照样通过**）⇒ 挡不住本单要挡的错；酉基变换才真检验"基的正确性"。
    #[test]
    fn t82_F3c_酉基变换_不变量不变() {
        let l0 = 上三角L0默认();
        let δl = 矩阵(2, &[(0, 0, 0.25, 0.10), (0, 1, 0.30, 0.40), (1, 0, 0.05, -0.12), (1, 1, -0.15, 0.20)]);
        let 探测 = 类II探测器::default();
        let 基准 = 探测.探测_类II(&l0, &δl, None).expect("基准应成功");
        // 实正交（⊂ 酉）：U = [[c, −s], [s, c]]，c²+s²=1
        for &(c, s) in &[(1.0f64, 0.0f64), (0.6, 0.8), (0.0, 1.0), (0.28, 0.96), (-0.8, 0.6)] {
            let mut u = DMatrix::<NComplex<f64>>::zeros(2, 2);
            u[(0, 0)] = NComplex::new(c, 0.0);
            u[(0, 1)] = NComplex::new(-s, 0.0);
            u[(1, 0)] = NComplex::new(s, 0.0);
            u[(1, 1)] = NComplex::new(c, 0.0);
            let uh = u.adjoint();
            let l0b = &u * &l0 * &uh;
            let δlb = &u * &δl * &uh;
            let r = 探测.探测_类II(&l0b, &δlb, None).expect("酉变换后应成功");
            for (a, b) in 基准.跨模.iter().zip(r.跨模.iter()) {
                assert_eq!(a.i, b.i);
                assert_eq!(a.j, b.j);
                assert!(
                    相对误差(b.κ模ij, a.κ模ij) < 1e-9,
                    "酉变换下 |κ_ij| 应不变（c={}, s={}）: {} vs {}",
                    c, s, a.κ模ij, b.κ模ij
                );
                assert!(
                    相对误差(b.κ模ji, a.κ模ji) < 1e-9,
                    "酉变换下 |κ_ji| 应不变（c={}, s={}）",
                    c, s
                );
                assert_eq!(a.判据A非对称比, b.判据A非对称比, "判据A 应不变");
                assert_eq!(a.方向, b.方向, "方向应不变");
            }
            for (a, b) in 基准.单模.iter().zip(r.单模.iter()) {
                assert!(相对误差(b.κ模, a.κ模) < 1e-9, "酉变换下 |κ_jj| 应不变（c={}, s={}）", c, s);
            }
        }
    }

}
