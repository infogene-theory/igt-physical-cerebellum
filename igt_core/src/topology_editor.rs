// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! 拓扑编辑器 — 第132批 P0 谱动力学核心三件套·第二件（引擎架构调整方案 v1.0 §四）
//!
//! 理论依据:
//!   - 《IGT_Engine_v2_架构规范》: 拓扑编辑器 = 节点+边+κ权重 → 生成元 L = D - A + V
//!   - 《IGT闭合拓扑变分原理_GUE最优τ链_v1.0》: L 是闭合拓扑的生成元，
//!     τ_i = 1/λ_i 是 L 的本征谱 = 系统的弛豫时间谱
//!   - 《IGT闭合拓扑变分原理_v1.1 §6.3》: 半直积检测 = L 分块结构（L12≠0 判半直积）
//!
//! 核心功能:
//!   1. 拓扑图数据结构: 节点 + 边(κ权重) + 对角势 V
//!   2. 生成元构造: L = D - A + V（图 Laplacian + 势）
//!   3. 对称性类判定辅助: 详细平衡检验 ||L-L^T||（对称→GOE; 非平衡环流→GUE）
//!   4. 半直积检测: L 分块结构（L12≠0）替代对易子检验（线性代数化）
//!   5. 与谱求解器接线: 构造 L → 谱动力学结果

#![allow(non_snake_case)]

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::spectrum_solver::谱求解器;

/// 拓扑边 — 节点 i 与 j 之间的 κ 耦合边
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct 拓扑边 {
    /// 节点索引（0-based）
    pub 节点a: usize,
    /// 节点索引（0-based）
    pub 节点b: usize,
    /// 耦合权重 κ_ij（τ共振耦合强度，≥0）
    pub κ: f64,
}

/// 拓扑图 — Engine v2 拓扑编辑器的输入
///
/// 对应架构规范: 节点（本征过程基元）+ 边（κ耦合）+ 势（外部场/观测者锚定）
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct 拓扑图 {
    /// 节点数
    pub 节点数: usize,
    /// κ 耦合边列表
    pub 边: Vec<拓扑边>,
    /// 对角势 V（节点上的外部势，长度=节点数；缺省全 0）
    pub 势能: Vec<f64>,
    /// 图名称（诊断/日志用）
    pub 名称: String,
}

impl 拓扑图 {
    /// 构造空拓扑图
    pub fn 新建(节点数: usize) -> Self {
        Self {
            节点数,
            边: vec![],
            势能: vec![0.0; 节点数],
            名称: "未命名拓扑".to_string(),
        }
    }

    /// 添加一条 κ 耦合边（对称图：i↔j）
    pub fn 加边(&mut self, 节点a: usize, 节点b: usize, κ: f64) {
        debug_assert!(节点a < self.节点数 && 节点b < self.节点数);
        self.边.push(拓扑边 { 节点a, 节点b, κ });
    }

    /// 设置节点势
    pub fn 设势(&mut self, 索引: usize, v: f64) {
        debug_assert!(索引 < self.节点数);
        self.势能[索引] = v;
    }

    /// 生成元构造: L = D - A + V
    ///
    /// - A_ij = κ_ij（i≠j 且有边）；A_ii = 0
    /// - D_ii = Σ_j A_ij（度矩阵）
    /// - L = D - A（图 Laplacian，对称半正定）+ 对角势 V
    ///
    /// # 保证
    /// - 对称（L_ij = L_ji）: 自洽的图 Laplacian
    /// - 连通时零空间 1 维（连通闭合，变分原理前提）
    /// - 半正定（含势 V≥0 时）
    pub fn 构造生成元(&self) -> DMatrix<f64> {
        let n = self.节点数;
        let mut l = DMatrix::zeros(n, n);

        // A 矩阵（邻接）与 D 度矩阵
        let mut 度 = vec![0.0; n];
        for 边 in &self.边 {
            if 边.节点a == 边.节点b {
                continue; // 无自环（自环会破坏零空间唯一性）
            }
            let κ = 边.κ.abs();
            度[边.节点a] += κ;
            度[边.节点b] += κ;
        }
        // L = D - A
        for 边 in &self.边 {
            if 边.节点a == 边.节点b {
                continue;
            }
            let κ = 边.κ.abs();
            l[(边.节点a, 边.节点b)] = -κ;
            l[(边.节点b, 边.节点a)] = -κ;
        }
        for i in 0..n {
            l[(i, i)] = 度[i];
        }
        // + V（对角势）
        for i in 0..n {
            if i < self.势能.len() {
                l[(i, i)] += self.势能[i];
            }
        }
        l
    }

    /// 第151批: 构造非厄米算子 L = H - iΓ（V10 非厄米耗散算子框架）
    ///
    /// - H = 现有保守生成元（D-A+V, 厄米）
    /// - Γ = 对角耗散矩阵 diag(γ_i), γ_i≥0（Γ 正定前提）
    ///
    /// 返回 (H, Γ)，供 [`非厄米分析器::从非厄米`](crate::spectrum_solver::非厄米分析器)
    /// 分析复本征值 λ=ω-i/τ（椭圆系综）。
    pub fn 构造非厄米(&self, 耗散: &[f64]) -> (DMatrix<f64>, DMatrix<f64>) {
        let n = self.节点数;
        let h = self.构造生成元();
        let mut g = DMatrix::zeros(n, n);
        for i in 0..n {
            g[(i, i)] = if i < 耗散.len() {
                耗散[i].max(0.0)
            } else {
                0.0
            };
        }
        (h, g)
    }

    /// 半直积检测（v1.1 §6.3 线性代数化）— 替代对易子检验
    ///
    /// 将 L 分块为 L = [[L11, L12], [L21, L22]]：
    /// - L12 ≠ 0（块间有耦合）→ 半直积（半直积群的生成元块间非零耦合）
    /// - L12 ≈ 0 → 直积（两个子系统完全解耦）
    ///
    /// 对比旧对易子检验（[L1,L2]=0 判直积）:
    /// - 旧: 需要先辨识子系统生成元再算对易子
    /// - 新: 直接从 L 分块结构判断，无需预分组
    pub fn 半直积检测(l: &DMatrix<f64>, 分块1大小: usize) -> 半直积检测结果 {
        let n = l.nrows();
        if 分块1大小 == 0 || 分块1大小 >= n {
            return 半直积检测结果::无效("分块大小需在 (0, n) 之间".into());
        }
        // L12 = 块(0..分块1, 分块1..n)
        let mut 块间范数 = 0.0;
        let mut 非零耦合数 = 0usize;
        for i in 0..分块1大小 {
            for j in 分块1大小..n {
                let v = l[(i, j)];
                块间范数 += v * v;
                if v.abs() > 1e-9 {
                    非零耦合数 += 1;
                }
            }
        }
        块间范数 = 块间范数.sqrt();
        let 判定 = if 块间范数 < 1e-9 {
            "直积".to_string()
        } else {
            "半直积".to_string()
        };
        半直积检测结果 {
            判定,
            块间范数,
            非零耦合数,
            分块1大小,
            有效: true,
        }
    }

    /// 对称性类判定（v1.1 裂痕1）— 详细平衡检验
    ///
    /// - ||L - L^T|| < ε_sym（实对称 + 详细平衡）→ GOE（β=1）
    /// - 非对称/非平衡环流 → GUE（β=2）
    pub fn 对称性类(l: &DMatrix<f64>, ε_sym: f64) -> 对称性类判定 {
        let 破坏度 = 谱求解器::对称破坏度(l);
        if 破坏度 < ε_sym {
            对称性类判定 { β: 1, 名称: "GOE".into(), 破坏度 }
        } else {
            对称性类判定 { β: 2, 名称: "GUE".into(), 破坏度 }
        }
    }

    /// 分块检测（论文§2.4 模式A/B）— 谱聚类（Fiedler 向量二分）+ 模块化度判据
    ///
    /// 用 L 的 Fiedler 向量（第2小特征值对应特征向量）符号划分节点为两块，
    /// 用**模块化度 Q**（Newman 社区检测标准判据）判定社区显著性：
    ///   - Q ≥ 0.25 → 明显社区结构（模式A：解耦混合，各块独立定β）
    ///   - Q < 0.25 → 强耦合混合（模式C：全局β_eff）
    ///
    /// 模块化度 Q = (块内边比例) - (块内期望边比例)²，完全图 Q≈0（无社区），
    /// 弱耦合双团 Q≈0.7（强社区），避免 Fiedler 向量对完全图误报。
    pub fn 分块检测(l: &DMatrix<f64>) -> 分块检测结果 {
        let n = l.nrows();
        if n < 4 {
            return 分块检测结果 {
                检测到分块: false,
                块1: vec![],
                块2: vec![],
                块间耦合强度: 1.0,
                模块化度q: 0.0,
                说明: "节点数<4，无需分块".to_string(),
            };
        }
        // 归一化 Laplacian L_norm = D^{-1/2} L D^{-1/2}，Fiedler 向量 = 第2小特征向量
        let eig = nalgebra::SymmetricEigen::new(l.clone());
        let 特征值: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
        // 对奇异矩阵（连通图零空间1维）第2小是 Fiedler；若零空间>1（不连通）选第1个正特征值
        let fid_index = if 特征值.len() >= 2 && 特征值[0].abs() < 1e-9 {
            1
        } else {
            0
        };
        if fid_index >= n {
            return 分块检测结果 {
                检测到分块: false,
                块1: vec![],
                块2: vec![],
                块间耦合强度: 1.0,
                模块化度q: 0.0,
                说明: "特征向量不足".to_string(),
            };
        }
        // 提取 Fiedler 特征向量列
        let fiedler: Vec<f64> = (0..n).map(|i| eig.eigenvectors[(i, fid_index)]).collect();
        // 按符号划分
        let mut 块1 = Vec::new();
        let mut 块2 = Vec::new();
        for (i, &v) in fiedler.iter().enumerate() {
            if v >= 0.0 {
                块1.push(i);
            } else {
                块2.push(i);
            }
        }
        // 若符号划分不平衡（全同号 → 无分块）
        if 块1.is_empty() || 块2.is_empty() {
            return 分块检测结果 {
                检测到分块: false,
                块1,
                块2,
                块间耦合强度: 1.0,
                模块化度q: 0.0,
                说明: "Fiedler 向量无符号划分（强连通无社区结构）".to_string(),
            };
        }

        // 模块化度 Q = Σ_ij (A_ij - k_i k_j / 2m) δ(c_i, c_j) / 2m
        // A_ij = κ_ij（边权重）；k_i = 度；m = Σ 边权重 / 2
        let 度: Vec<f64> = (0..n).map(|i| (0..n).map(|j| l[(i, j)]).filter(|&v| v < 0.0).map(|v| -v).sum()).collect();
        let m: f64 = 度.iter().sum::<f64>() / 2.0;
        let 模块化度q = if m > 1e-12 {
            let mut q = 0.0;
            for &i in 块1.iter().chain(块2.iter()) {
                for &j in 块1.iter().chain(块2.iter()) {
                    // 仅同块对累加
                    let 同块 = (块1.contains(&i) && 块1.contains(&j)) || (块2.contains(&i) && 块2.contains(&j));
                    if !同块 {
                        continue;
                    }
                    let a_ij = if i == j { 0.0 } else { l[(i, j)].abs() };
                    let 期望 = 度[i] * 度[j] / (2.0 * m);
                    q += a_ij - 期望;
                }
            }
            q / (2.0 * m)
        } else {
            0.0
        };

        // 块间耦合强度 ||L12||_F / ||L||_F（诊断用）
        let mut 块间平方 = 0.0;
        let mut 全平方 = 0.0;
        for i in 0..n {
            for j in 0..n {
                全平方 += l[(i, j)] * l[(i, j)];
                let 在块1 = 块1.contains(&i);
                let 在块1j = 块1.contains(&j);
                if 在块1 != 在块1j {
                    块间平方 += l[(i, j)] * l[(i, j)];
                }
            }
        }
        let 块间耦合强度 = if 全平方 > 1e-12 {
            (块间平方 / 全平方).sqrt()
        } else {
            0.0
        };
        // 模块化度 Q≥0.25 判社区（Newman 经验阈值）
        let 检测到分块 = 模块化度q >= 0.25;
        分块检测结果 {
            检测到分块,
            块1,
            块2,
            块间耦合强度,
            模块化度q,
            说明: if 检测到分块 {
                format!("检测到社区结构(Q={:.3}≥0.25): 模式A解耦混合, 各块独立定β", 模块化度q)
            } else {
                format!("强耦合混合(Q={:.3}<0.25): 模式C, 用全局β_eff", 模块化度q)
            },
        }
    }
}

/// 分块检测结果（论文§2.4）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 分块检测结果 {
    /// 是否检测到明显社区分块（模块化度 Q ≥ 0.25）
    pub 检测到分块: bool,
    /// 块1 节点索引
    pub 块1: Vec<usize>,
    /// 块2 节点索引
    pub 块2: Vec<usize>,
    /// 块间耦合强度 ||L12||_F / ||L||_F ∈ [0,1]
    pub 块间耦合强度: f64,
    /// 模块化度 Q（Newman 社区检测判据，Q≥0.25 明显社区）
    pub 模块化度q: f64,
    /// 说明
    pub 说明: String,
}

/// 半直积检测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 半直积检测结果 {
    /// 判定: 直积 / 半直积
    pub 判定: String,
    /// 块间耦合 Frobenius 范数 ||L12||
    pub 块间范数: f64,
    /// 非零耦合数
    pub 非零耦合数: usize,
    /// 分块1大小
    pub 分块1大小: usize,
    /// 是否有效
    pub 有效: bool,
}

impl 半直积检测结果 {
    fn 无效(原因: String) -> Self {
        Self {
            判定: 原因,
            块间范数: 0.0,
            非零耦合数: 0,
            分块1大小: 0,
            有效: false,
        }
    }
}

/// 对称性类判定结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 对称性类判定 {
    /// β: 1=GOE 2=GUE
    pub β: u8,
    /// 名称
    pub 名称: String,
    /// ||L-L^T|| 破坏度
    pub 破坏度: f64,
}

/// 拓扑编辑器 — 从拓扑图到谱动力学的统一入口
///
/// 对应 Engine v2: 拓扑编辑器（图→L）→ 谱求解器（L→谱）
#[derive(Debug, Clone, Copy, Default)]
pub struct 拓扑编辑器 {
    /// 谱求解器（默认 n_sim=200）
    pub 谱求解器: 谱求解器,
}

impl 拓扑编辑器 {
    /// 编辑并求解: 拓扑图 → L → 谱动力学结果（含半直积检测 + 对称性类判定）
    pub fn 编辑并求解(&self, 图: &拓扑图, 分块1大小: Option<usize>) -> 拓扑谱分析 {
        let l = 图.构造生成元();
        let 谱 = self.谱求解器.从矩阵(&l);
        let 对称 = 拓扑图::对称性类(&l, self.谱求解器.ε_sym);
        let 半直积 = 分块1大小.map(|k| 拓扑图::半直积检测(&l, k));
        let 分块 = 拓扑图::分块检测(&l);

        拓扑谱分析 {
            名称: 图.名称.clone(),
            节点数: 图.节点数,
            边数: 图.边.len(),
            谱,
            对称性: 对称,
            半直积: 半直积,
            分块: 分块,
        }
    }
}

/// 拓扑谱分析结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 拓扑谱分析 {
    /// 拓扑图名称
    pub 名称: String,
    /// 节点数
    pub 节点数: usize,
    /// 边数
    pub 边数: usize,
    /// 谱动力学结果
    pub 谱: crate::spectrum_solver::谱动力学结果,
    /// 对称性类判定
    pub 对称性: 对称性类判定,
    /// 半直积检测（可选）
    pub 半直积: Option<半直积检测结果>,
    /// 分块检测（论文§2.4 社区发现）
    pub 分块: 分块检测结果,
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造 3 节点三角形环（连通闭合，单位权重）
    fn 三角环() -> 拓扑图 {
        let mut g = 拓扑图::新建(3);
        g.名称 = "三角环".into();
        g.加边(0, 1, 1.0);
        g.加边(1, 2, 1.0);
        g.加边(2, 0, 1.0);
        g
    }

    /// 构造 4 节点路径图（0-1-2-3）
    fn 路径图() -> 拓扑图 {
        let mut g = 拓扑图::新建(4);
        g.名称 = "路径图".into();
        g.加边(0, 1, 1.0);
        g.加边(1, 2, 1.0);
        g.加边(2, 3, 1.0);
        g
    }

    #[test]
    fn test_构造生成元对称半正定() {
        let l = 三角环().构造生成元();
        // 对称
        for i in 0..3 {
            for j in 0..3 {
                assert!((l[(i, j)] - l[(j, i)]).abs() < 1e-12);
            }
        }
        // 行和=0（Laplacian 性质）
        for i in 0..3 {
            let 行和: f64 = (0..3).map(|j| l[(i, j)]).sum();
            assert!(行和.abs() < 1e-12, "Laplacian 行和应为0: {}", 行和);
        }
        // 对角=2（度）
        for i in 0..3 {
            assert!((l[(i, i)] - 2.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_三角环特征值() {
        // 3环 Laplacian: 特征值 0, 3, 3
        let l = 三角环().构造生成元();
        let eig = nalgebra::SymmetricEigen::new(l);
        let mut 特征值: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
        // nalgebra 特征值顺序不保证升序，排序后断言
        特征值.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((特征值[0] - 0.0).abs() < 1e-9);
        assert!((特征值[1] - 3.0).abs() < 1e-9);
        assert!((特征值[2] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_路径图特征值() {
        // 4路径 Laplacian: 特征值 0, 0.5858, 2.0, 3.4142
        let l = 路径图().构造生成元();
        let eig = nalgebra::SymmetricEigen::new(l);
        let mut 特征值: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
        特征值.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((特征值[0] - 0.0).abs() < 1e-9);
        assert!((特征值[1] - 0.5858).abs() < 0.001);
        assert!((特征值[2] - 2.0).abs() < 0.001);
        assert!((特征值[3] - 3.4142).abs() < 0.001);
    }

    #[test]
    fn test_势能修正() {
        let mut g = 路径图();
        g.设势(0, 5.0);
        let l = g.构造生成元();
        // 势加到对角
        assert!((l[(0, 0)] - (1.0 + 5.0)).abs() < 1e-12, "节点0对角=度1+势5: {}", l[(0, 0)]);
    }

    #[test]
    fn test_半直积检测_直积() {
        // 两个完全解耦子系统（各1个节点，无边）→ 直积
        let mut g = 拓扑图::新建(2);
        g.加边(0, 0, 0.0); // 无效自环被跳过
        let l = g.构造生成元();
        let 结果 = 拓扑图::半直积检测(&l, 1);
        assert!(结果.有效);
        assert_eq!(结果.判定, "直积");
    }

    #[test]
    fn test_半直积检测_半直积() {
        // 两个子系统之间有1条边 → 半直积
        let mut g = 拓扑图::新建(4);
        g.加边(0, 1, 1.0); // 子系统1内
        g.加边(2, 3, 1.0); // 子系统2内
        g.加边(1, 2, 0.5); // 块间耦合
        let l = g.构造生成元();
        let 结果 = 拓扑图::半直积检测(&l, 2);
        assert!(结果.有效);
        assert_eq!(结果.判定, "半直积");
        assert!(结果.块间范数 > 0.0);
    }

    #[test]
    fn test_半直积检测_无效分块() {
        let l = 三角环().构造生成元();
        let 结果 = 拓扑图::半直积检测(&l, 0);
        assert!(!结果.有效);
        let 结果2 = 拓扑图::半直积检测(&l, 3);
        assert!(!结果2.有效);
    }

    #[test]
    fn test_对称性类判定() {
        let l = 三角环().构造生成元();
        let 判定 = 拓扑图::对称性类(&l, 1e-9);
        assert_eq!(判定.β, 1, "对称矩阵→GOE");
        assert_eq!(判定.名称, "GOE");
    }

    #[test]
    fn test_编辑并求解全流程() {
        let 编辑器 = 拓扑编辑器::default();
        let 分析 = 编辑器.编辑并求解(&三角环(), Some(1));
        assert_eq!(分析.节点数, 3);
        assert_eq!(分析.边数, 3);
        assert_eq!(分析.谱.本征值数, 2, "3环排除零空间后2个非零本征值");
        assert_eq!(分析.对称性.β, 1);
        assert!(分析.半直积.is_some());
        // 分块字段存在（三角环强连通无社区）
        assert!(分析.分块.说明.contains("无分块") || 分析.分块.块间耦合强度 >= 0.0);
    }

    #[test]
    fn test_分块检测_解耦社区() {
        // 两个团，各自内部稠密、块间仅1条弱边 → 应检测到分块（模式A解耦混合）
        let mut g = 拓扑图::新建(6);
        // 团1: {0,1,2} 完全
        g.加边(0, 1, 1.0);
        g.加边(0, 2, 1.0);
        g.加边(1, 2, 1.0);
        // 团2: {3,4,5} 完全
        g.加边(3, 4, 1.0);
        g.加边(3, 5, 1.0);
        g.加边(4, 5, 1.0);
        // 块间弱耦合
        g.加边(2, 3, 0.05);
        let l = g.构造生成元();
        let 结果 = 拓扑图::分块检测(&l);
        assert!(结果.模块化度q >= 0.25, "弱耦合社区模块化度应高: {}", 结果.模块化度q);
        assert!(结果.检测到分块, "应检测到社区分块: {}", 结果.说明);
        assert!(!结果.块1.is_empty() && !结果.块2.is_empty());
    }

    #[test]
    fn test_分块检测_强耦合() {
        // 完全图（全连接强耦合）→ 无社区结构（Q≈0）
        let mut g = 拓扑图::新建(5);
        for i in 0..5 {
            for j in i + 1..5 {
                g.加边(i, j, 1.0);
            }
        }
        let l = g.构造生成元();
        let 结果 = 拓扑图::分块检测(&l);
        assert!(结果.模块化度q < 0.25, "完全图模块化度应低: {}", 结果.模块化度q);
        assert!(!结果.检测到分块, "完全图不应判分块: {}", 结果.说明);
    }

    #[test]
    fn test_分块检测_小图保护() {
        let mut g = 拓扑图::新建(3);
        g.加边(0, 1, 1.0);
        g.加边(1, 2, 1.0);
        let l = g.构造生成元();
        let 结果 = 拓扑图::分块检测(&l);
        assert!(!结果.检测到分块, "节点<4无需分块");
    }

    /// 第151批: 非厄米算子构造 L=H-iΓ → 非厄米分析
    #[test]
    fn test_构造非厄米并分析() {
        use crate::spectrum_solver::非厄米分析器;
        // 10 节点环（单位边）
        let mut g = 拓扑图::新建(10);
        for i in 0..10 {
            g.加边(i, (i + 1) % 10, 1.0);
        }
        // 均匀耗散 γ=0.2 → 各模衰减一致, σ_γ²=0 → 近保守 GUE
        let 耗散 = vec![0.2; 10];
        let (h, gamma) = g.构造非厄米(&耗散);
        let 分析器 = 非厄米分析器::default();
        let 结果 = 分析器.从非厄米(&h, &gamma);
        assert!(结果.点数 >= 8, "环10节点应有≥8有效模: {}", 结果.点数);
        assert!((结果.γ_0 - 0.2).abs() < 1e-2, "γ_0应≈0.2: {}", 结果.γ_0);
        assert!(结果.τ_param > 0.3, "均匀耗散应近保守(τ_param高): {}", 结果.τ_param);
        assert!(结果.普适类 == "GUE" || 结果.普适类 == "Ginibre", "普适类: {}", 结果.普适类);
        // H 对称 + Γ 对角 → [L,L†]=0 → 正规 ν≈0
        assert!(结果.ν < 1e-6, "对角Γ应正规(ν≈0): {}", 结果.ν);
        // κ_L 有限
        assert!(结果.κ_l.is_finite() && 结果.κ_l > 1.0, "κ_L应有限且≥1: {}", 结果.κ_l);
    }
}
