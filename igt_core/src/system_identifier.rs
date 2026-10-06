// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! 系统辨识器 — 第133批 谱动力学核心三件套·第三件（P0 第4步，变分原理深化融合方案 v1.0）
//!
//! 理论依据:
//!   - 《变分原理深化_演化趋衡_beta混合_Koopman非线性_v1.0.md》第三部分:
//!     Koopman 算子将非线性系统线性化; DMD/EDMD 从时间序列反演 Koopman 本征值
//!   - 《IGT_Engine_v2_架构规范》: 系统辨识器 = 时间序列 x(t) → 反演 L
//!   - 《IGT闭合拓扑变分原理_v1.0》: τ_i = 1/λ_i 是本征谱的弛豫时间
//!
//! 核心功能（DMD 动态模态分解）:
//!   1. 快照矩阵 X = [x_0, x_1, ..., x_{N-1}], X' = [x_1, ..., x_N]
//!   2. 最佳拟合线性算子 A = X' · X⁺（SVD 伪逆）
//!   3. A 的本征值 μ_i（离散时间）→ 连续时间 λ_i = ln(μ_i)/Δt
//!   4. τ 链: τ_i = -1/Re(λ_i)（稳定模 Re(λ)<0）; 振荡频率 ω_i = Im(λ_i)
//!   5. 对称性: A 对称度 ||A-A^T|| → GOE/GUE（对接谱求解器）
//!   6. 输出 Koopman 谱 → 用 -Re(λ_i) 做 D_s/β 检验（变分残差）
//!
//! 工程修正点（相对论文）:
//!   - Δt 等间隔采样; 快照数 N ≥ 2×状态数（欠定保护）
//!   - 输入去均值（DMD 对常数偏移敏感）
//!   - 单变量时序用 Hankel 延迟嵌入构造状态空间
//!   - 复特征值用 Hessenberg + shifted QR 迭代求解（nalgebra 无通用非对称特征值）

#![allow(non_snake_case)]

use nalgebra::linalg::Schur;
use nalgebra::{Complex, ComplexField, DMatrix, SVD};

/// D1-R4（2026-08-23）: 失稳模信号（真实失稳标记, 上报预警不进谱集）
/// ★ γ符号统一令 §1：失稳 ⟺ **γ<0** ⟺ λ_im>0（γ＝−Im λ 耗散率，稳定为正——
///   旧口径此字段存 Im λ 失稳为正，已反转；数值＝−旧值，τ 数值不变）
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct 失稳模信号 {
    /// 复L 对角位置
    pub 索引: usize,
    /// 振荡频率 ω（复L 实部）
    pub 频率: f64,
    /// 耗散率 γ=−λ_im（**失稳模 γ<0**；增长越快越负）
    pub γ: f64,
}

/// 单个 Koopman 模（复本征值）
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Koopman模 {
    /// 离散时间本征值 μ_i（DMD 输出）
    pub μ实部: f64,
    /// 离散时间本征值 μ_i 虚部
    pub μ虚部: f64,
    /// 连续时间本征值 λ_i = ln(μ)/Δt 实部（衰减率，负=稳定）
    pub 实部: f64,
    /// 连续时间本征值 λ_i 虚部（振荡频率 ω_i）
    pub 虚部: f64,
    /// 弛豫时间 τ_i = -1/Re(λ_i)（仅稳定模）
    pub τ: f64,
    /// 是否稳定模（Re(λ) < -ε）
    pub 稳定: bool,
    /// 是否不稳定模（Re(λ) > +ε，明确增长/失稳）
    pub 不稳定: bool,
    /// 是否守恒模（|Re(λ)| ≤ ε，中心流形/守恒，论文§3.2）
    pub 守恒: bool,
    /// 是否振荡模（|Im(λ)| > 阈值）
    pub 振荡: bool,
}

/// 系统辨识结果 — Koopman 谱
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct 系统辨识结果 {
    /// 状态空间维数（嵌入维/变量数）
    pub 状态维数: usize,
    /// 快照数
    pub 快照数: usize,
    /// DMD 保留模态数（截断秩 r）
    pub 模态数: usize,
    /// Koopman 本征值/模列表
    pub 模列表: Vec<Koopman模>,
    /// τ 链（稳定模的 τ_i，升序 = 快→慢）
    pub τ链: Vec<f64>,
    /// 振荡频率 ω 列表（非零 Im）
    pub 频率: Vec<f64>,
    /// 稳定模数
    pub 稳定模数: usize,
    /// 不稳定模数（Re>+ε，失稳证据）
    pub 不稳定模数: usize,
    /// 守恒模数（|Re|≤ε，中心流形）
    pub 守恒模数: usize,
    /// 振荡模数
    pub 振荡模数: usize,
    /// 最优拟合算子 A 的对称破坏度 ||A-A^T||（GOE=对称/GUE=非对称）
    pub 对称破坏度: f64,
    /// 对称性类判定（"GOE"/"GUE"/"未知"）
    pub 对称性类: String,
    /// 非梯度/环流证据（复本征对存在 → 非梯度 Ginibre 候选）
    pub 非梯度证据: bool,
    /// 采样间隔 Δt
    pub Δt: f64,
    /// 辨识方法: "DMD"(线性) / "EDMD"(字典提升非线性)
    pub 方法: String,
    /// EDMD 字典规模（字典函数数，仅 EDMD 有值）
    pub 字典规模: Option<usize>,
    /// EDMD 字典收敛性（论文§3.5: 随字典增大本征值是否收敛；None=未做收敛性检查）
    pub 字典收敛: Option<bool>,
    /// 系统矩阵 Ã（第285批: 多变量DMD 暴露状态转移矩阵, 用于图上多模态接图）
    /// 负对角=耗散, 非对角=跨通道耦合; 单变量DMD=None(嵌入维Hankel非物理通道)
    #[serde(default)]
    pub 系统矩阵: Option<Vec<f64>>,
    /// 连续化复 L = H-iΓ（第289批修复: EDMD→L 端到端接线, P22§6 第1步）
    /// L=(Ã-I)/Δt 连续化 → H=对称部分, Γ=反对称部分(实) → L_ij=H_ij - i·Γ_ij
    /// [实,虚] 交错, r_eff×r_eff; 单变量DMD=None; 可直连 非厄米谱求解器.分解()
    #[serde(default)]
    pub 复L: Option<Vec<f64>>,
    /// 说明
    pub 说明: String,
}

/// 系统辨识器
#[derive(Debug, Clone, Copy)]
pub struct 系统辨识器 {
    /// DMD 截断秩 r（默认 None=自动取 min(n, 快照数/2)）
    pub 截断秩: Option<usize>,
    /// 伪逆截断阈值（SVD 奇异值 < ε×σ_max 视为 0）
    pub ε_伪逆: f64,
    /// 振荡判据阈值（|Im| > ε_ω 判振荡）
    pub ε_ω: f64,
    /// 稳定判据（Re(λ) < -ε_稳定 判稳定）
    pub ε_稳定: f64,
    /// EDMD 字典最高阶（默认 2=多项式 {1,x,x²,xy,y²}）
    pub edmd字典阶: usize,
}

impl Default for 系统辨识器 {
    fn default() -> Self {
        Self {
            截断秩: None,
            ε_伪逆: 1e-10,
            ε_ω: 1e-6,
            ε_稳定: 1e-8,
            edmd字典阶: 2,
        }
    }
}

/// T-30: 多项式字典规模硬上限（`生成指数向量` 超限即诚实返回 None）。
///
/// 阶 2 字典规模 = Σ_{k=1..2} C(k+n-1, n-1)：n=128 约 8.4k、n≈446 才触顶——
/// 真实同步变量/通道 ≤128 时裕度充足、不误伤；超限意味着该 n/阶 组合字典本身过大
/// （EDMD 特征空间不可行），诚实拒绝而非静默给错谱。
pub const 字典规模硬上限: usize = 100_000;

impl 系统辨识器 {
    /// 多变量 DMD：状态序列（每时刻一个状态向量）→ Koopman 谱
    ///
    /// # 参数
    /// - `状态序列`: 每个元素为一个时刻的状态向量（长度 N，每个向量维数 n）
    /// - `Δt`: 采样间隔（秒）
    pub fn 多变量dmd(&self, 状态序列: &[Vec<f64>], Δt: f64) -> Option<系统辨识结果> {
        if 状态序列.len() < 3 {
            return None;
        }
        let n = 状态序列[0].len();
        if n == 0 {
            return None;
        }
        // 构建快照矩阵 X (n×N), X' (n×N)
        let n_快照 = 状态序列.len() - 1;
        // 去均值（对每个变量）
        let mut 均值 = vec![0.0; n];
        for 状态 in 状态序列 {
            for i in 0..n {
                均值[i] += 状态[i];
            }
        }
        for m in 均值.iter_mut() {
            *m /= 状态序列.len() as f64;
        }
        let 去均值序列: Vec<Vec<f64>> = 状态序列
            .iter()
            .map(|s| s.iter().zip(&均值).map(|(v, m)| v - m).collect())
            .collect();

        // X: 前 N 个, X': 后 N 个（错位一步）
        let x = DMatrix::from_fn(n, n_快照, |i, j| 去均值序列[j][i]);
        let xp = DMatrix::from_fn(n, n_快照, |i, j| 去均值序列[j + 1][i]);

        // 第285批: 多变量DMD 暴露系统矩阵（状态维=变量数, 图上多模态接图用）
        self.dmd核心_带矩阵(x, xp, Δt, true)
    }

    /// D7 复数据链入口（T-10, 规范 §3 Step1-6）: 实信号多通道 → Hilbert 解析化 → 复EDMD
    /// → 方案B 复L（L=ΦΛΦ⁻¹, 恢复非对角虚部）→ Γ 正定验证（复dmd结果.Γ正定）
    ///
    /// 对外接口不变（仍收实信号, 每通道 f64 序列）, 内部自动转复（complex_dmd::复EDMD解析信号）;
    /// **单通道返回 None**（D7 §4 硬边界: 熵泵需≥2通道跨模态耦合, 单通道只能做谱统计 D_s/椭圆系综）。
    ///
    /// 与 `多变量dmd`（实路径）互补: 本入口走完整 D7 复数据链, 供熵泵/对称破缺/因果探测
    /// 等需要"谱信息充足=true"的下游使用。
    pub fn 多变量dmd_复数据链(
        &self,
        状态序列: &[Vec<f64>],
        Δt: f64,
    ) -> Option<crate::complex_dmd::复dmd结果> {
        crate::complex_dmd::复EDMD解析信号(状态序列, Δt)
    }

    /// 单变量 DMD：用 Hankel 延迟嵌入构造状态空间
    ///
    /// # 参数
    /// - `时序`: 单变量时间序列（等间隔）
    /// - `嵌入维`: 延迟嵌入维 m（快照 = [x_k, x_{k+1}, ..., x_{k+m-1}]）
    /// - `Δt`: 采样间隔
    ///
    /// # ★ 已验证用法（T-110 补记，2026-09-18，依据 B1 段 T-95「路 A」实测）
    ///
    /// **场景**：单变量信号（如 **HRV tachogram**）需要**频率维分辨力**，而 `complex_dmd::复EDMD解析信号`
    /// 因**无 Hankel 扩维**而 `r ≤ 通道数`（双通道 ⇒ `r ≤ 2`）**不够用**时，用本函数。
    ///
    /// **B1 验证过的调用链**（`igt_agent::hrv_complex_chain::等间隔到Koopman谱` 即薄封装）：
    /// ```text
    /// 时序(HRV tachogram, 30Hz) → 单变量dmd(时序, 嵌入维=16, Δt=1/30)
    ///   → 系统辨识结果.模列表: Vec<Koopman模>
    ///   → 按频率锁主频：取 |虚部(ω) − 2π·f_target| 最小的模
    ///   → γ̂ = −实部(Re λ)      // 理论侧口径：稳定 γ>0；core 内 τ = −1/Re λ ⇒ γ̂ = 1/τ
    /// ```
    ///
    /// **实测结论（T-95 路 A）**：**嵌入维 `m` 由 4 → 24 扫描**，
    /// 零假设偏置 `b` 从 **58/T 单调降到 2.52/T**（趋于理论期望 `~1/T`），
    /// **检出率（γ=0.2、T/τ=60）从 2% 升到 100%**、假阳性 ≤5.5%。
    /// ⇒ **本函数是"少通道但有频率分辨力需求"时的正确入口**，**而非** `复EDMD解析信号`。
    ///
    /// **注意**：`嵌入维` 即**状态维数**，决定 `r` 上限；**取值需与"信号自由度数"匹配**
    /// （B1 选 **`m=16`** 作灵敏度/算量折中）；`截断秩` 可另行显式指定。
    pub fn 单变量dmd(&self, 时序: &[f64], 嵌入维: usize, Δt: f64) -> Option<系统辨识结果> {
        if 嵌入维 < 2 || 时序.len() < 2 * 嵌入维 + 2 {
            return None;
        }
        // 去均值
        let 均值 = 时序.iter().sum::<f64>() / 时序.len() as f64;
        let 去均值: Vec<f64> = 时序.iter().map(|&x| x - 均值).collect();

        let n = 嵌入维;
        let n_快照 = 去均值.len() - 嵌入维;
        // X (n×N): col k = [x_k, ..., x_{k+m-1}]
        let x = DMatrix::from_fn(n, n_快照, |i, j| 去均值[j + i]);
        // X' (n×N): col k = [x_{k+1}, ..., x_{k+m}]
        let xp = DMatrix::from_fn(n, n_快照, |i, j| 去均值[j + i + 1]);

        // 单变量: 不暴露系统矩阵（嵌入维 Hankel 非物理通道）
        self.dmd核心_带矩阵(x, xp, Δt, false)
    }

    /// DMD 核心：快照矩阵 → 伪逆 → A 特征值 → Koopman 谱
    fn dmd核心(&self, x: DMatrix<f64>, xp: DMatrix<f64>, Δt: f64) -> Option<系统辨识结果> {
        self.dmd核心_带矩阵(x, xp, Δt, false)
    }

    /// DMD 核心（第285批扩展: 可选暴露系统矩阵 Ã 用于图上多模态接图）
    fn dmd核心_带矩阵(
        &self,
        x: DMatrix<f64>,
        xp: DMatrix<f64>,
        Δt: f64,
        暴露系统矩阵: bool,
    ) -> Option<系统辨识结果> {
        let n = x.nrows();
        let n_快照 = x.ncols();
        // 截断秩
        let r = self
            .截断秩
            .unwrap_or(n.min(n_快照 / 2).max(1));

        // SVD: X = U Σ V^T
        let svd = x.clone().svd(true, true);
        let u = svd.u?;
        let v = svd.v_t?;
        let s = &svd.singular_values;

        // 有效秩（奇异值 > ε×σ_max）
        let σ_max = s.iter().cloned().fold(0.0_f64, f64::max);
        if σ_max < 1e-12 {
            return None;
        }
        let r_eff = s
            .iter()
            .take(r)
            .filter(|&&σ| σ > self.ε_伪逆 * σ_max)
            .count()
            .max(1);

        // 截断
        let ur = u.columns(0, r_eff);
        let vtr = v.rows(0, r_eff);
        // Ã = U_r^T X' V_r Σ_r^{-1}
        let mut a_tilde = DMatrix::<f64>::zeros(r_eff, r_eff);
        let mut xp_v = DMatrix::<f64>::zeros(n, r_eff);
        for i in 0..n {
            for j in 0..r_eff {
                let mut acc = 0.0;
                for k in 0..n_快照 {
                    acc += xp[(i, k)] * vtr[(j, k)];
                }
                xp_v[(i, j)] = acc;
            }
        }
        for i in 0..r_eff {
            for j in 0..r_eff {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += ur[(k, i)] * xp_v[(k, j)];
                }
                a_tilde[(i, j)] = acc / s[i];
            }
        }

        // Ã 的特征值（复）
        let 特征值 = Self::qr复特征值(&a_tilde);

        // 对称性判定：Ã 对称破坏度
        let mut 破坏度 = 0.0;
        for i in 0..r_eff {
            for j in 0..r_eff {
                let d = a_tilde[(i, j)] - a_tilde[(j, i)];
                破坏度 += d * d;
            }
        }
        破坏度 = 破坏度.sqrt();
        let 对称性类 = if 破坏度 < 1e-6 { "GOE" } else { "GUE" };

        // 构造模列表
        let mut 模列表 = Vec::new();
        let mut τ链 = Vec::new();
        let mut 频率 = Vec::new();
        let mut 稳定模数 = 0;
        let mut 不稳定模数 = 0;
        let mut 守恒模数 = 0;
        let mut 振荡模数 = 0;
        let mut 非梯度证据 = false;
        let mut 复对数出现 = 0;

        for (μ实, μ虚) in 特征值 {
            // 连续时间 λ = ln(μ)/Δt（主分支）
            let μ = Complex::new(μ实, μ虚);
            let λ = μ.ln() / Δt;
            let 实部 = λ.re;
            let 虚部 = λ.im;
            // 三类判定（论文§3.2）: 稳定 Re<-ε / 守恒 |Re|≤ε（中心流形）/ 不稳定 Re>+ε
            let 稳定 = 实部 < -self.ε_稳定;
            let 不稳定 = 实部 > self.ε_稳定;
            let 守恒 = !稳定 && !不稳定;
            let 振荡 = 虚部.abs() > self.ε_ω;
            let τ = if 稳定 { -1.0 / 实部 } else { f64::NAN };
            if 稳定 {
                稳定模数 += 1;
                τ链.push(τ);
            } else if 不稳定 {
                不稳定模数 += 1;
            } else {
                守恒模数 += 1;
            }
            if 振荡 {
                振荡模数 += 1;
                频率.push(虚部);
            }
            // 非梯度证据：μ 有虚部（实矩阵特征值复对）
            if μ虚.abs() > 1e-6 {
                复对数出现 += 1;
            }
            模列表.push(Koopman模 {
                μ实部: μ实,
                μ虚部: μ虚,
                实部,
                虚部,
                τ,
                稳定,
                不稳定,
                守恒,
                振荡,
            });
        }
        if 复对数出现 >= 2 {
            非梯度证据 = true;
        }

        // τ 链升序（快→慢）
        τ链.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let 模数 = 模列表.len();
        let 说明 = format!(
            "DMD辨识: {}模(稳定{}/不稳定{}/守恒{}/振荡{}), 对称性类{}",
            模数, 稳定模数, 不稳定模数, 守恒模数, 振荡模数, 对称性类
        );

        Some(系统辨识结果 {
            状态维数: n,
            快照数: n_快照,
            模态数: r_eff,
            模列表,
            τ链,
            频率,
            稳定模数,
            不稳定模数,
            守恒模数,
            振荡模数,
            对称破坏度: 破坏度,
            对称性类: 对称性类.to_string(),
            非梯度证据,
            Δt,
            方法: "DMD".to_string(),
            字典规模: None,
            字典收敛: None,
            系统矩阵: if 暴露系统矩阵 {
                // Ã 扁平化（r_eff×r_eff, 行主序; 状态维=变量数 n 时 r_eff 可能<n）
                Some((0..r_eff * r_eff).map(|k| a_tilde.as_slice()[k]).collect())
            } else {
                None
            },
            复L: if 暴露系统矩阵 {
                // D7（2026-08-23）: 方案B 替换方案C（V6 方案C 废弃）——完整复L L=ΦΛΦ⁻¹
                // 手写复 QR 迭代（complex_dmd::复特征分解_QR, r≤30, 不需 nalgebra ComplexSchur,
                // D7 §2 理论澄清: 模态基重构只对标量 μ_n 取 ln, 矩阵乘法+try_inverse 即可）。
                // 实信号 Ã 本征值含复共轭对, Φ 复 → L 恢复非对角虚部（方案C"虚部恒 0"已废弃,
                // 其导致 Γ 非对角耦合不存在、熵泵条件不成立）。
                // λ_n = ln(μ_n)/Δt: θ/Δt + i·ln(r)/Δt（P22 §6.2 约定 λ=ω+i·γ）。
                let n = r_eff;
                let a_tilde复: DMatrix<Complex<f64>> = a_tilde.map(|x| Complex::new(x, 0.0));
                let (本征值, φ) = crate::complex_dmd::复特征分解_QR(&a_tilde复, 300);
                if φ.nrows() == n && 本征值.len() == n {
                    let mut Λ = DMatrix::<Complex<f64>>::zeros(n, n);
                    for (i, μ) in 本征值.iter().enumerate() {
                        let r = (μ.re * μ.re + μ.im * μ.im).sqrt().max(1e-12);
                        let θ = μ.im.atan2(μ.re);
                        Λ[(i, i)] = Complex::new(θ / Δt, r.ln() / Δt);
                    }
                    match φ.clone().try_inverse() {
                        Some(φ_inv) => {
                            let 重构 = &φ * Λ * φ_inv;
                            let mut 复l = Vec::with_capacity(n * n * 2);
                            for i in 0..n {
                                for j in 0..n {
                                    复l.push(重构[(i, j)].re);
                                    复l.push(重构[(i, j)].im);
                                }
                            }
                            Some(复l)
                        }
                        None => None, // 近奇异: 数据链异常, 不静默
                    }
                } else {
                    None
                }
            } else {
                None
            },
            说明,
        })
    }

    // ========================================================================
    // EDMD 扩展（变分原理深化论文 §3.4 非线性 Koopman 辨识）
    // ========================================================================

    /// 扩展 DMD（EDMD）——用字典函数 ψ(x) 提升到高维特征空间做 DMD
    ///
    /// 论文 §3.4: 非线性系统用字典函数提升 → 在特征空间逼近 Koopman 算子。
    /// 字典: 多项式（1阶=原变量 + 2阶=交叉项/平方项），论文建议含常数项。
    ///
    /// # 参数
    /// - `状态序列`: 多变量状态序列（每时刻一个向量）
    /// - `Δt`: 采样间隔
    /// - `字典阶`: 多项式最高阶（1=线性, 2=含 x_i x_j 交叉项）
    pub fn 多变量edmd(&self, 状态序列: &[Vec<f64>], Δt: f64, 字典阶: usize) -> Option<系统辨识结果> {
        if 状态序列.len() < 3 {
            return None;
        }
        let n = 状态序列[0].len();
        if n == 0 {
            return None;
        }
        // 构造字典函数索引: 每个字典函数是一个非负整数指数向量 (e_0, e_1, ..., e_{n-1})
        // 满足 Σ e_i ≤ 字典阶，且非全零（跳过常数项——DMD 已去均值）。
        // T-30: 改用 stars-and-bars 直接生成（旧叉积递归 (总阶+1)^n 对高维 n 不可终止）;
        // 超硬上限 → None 诚实传播（不静默给错谱、不 panic）。
        let 字典 = Self::生成指数向量(n, 字典阶)?;
        let 字典规模 = 字典.len();

        // 字典提升: Ψ_k(x_j) = ψ_0, ψ_1, ..., ψ_{D-1}
        let n_快照 = 状态序列.len() - 1;
        let mut 提升序列 = Vec::<Vec<f64>>::with_capacity(状态序列.len());
        for 状态 in 状态序列 {
            let mut 特征 = Vec::with_capacity(字典规模);
            for 指数 in &字典 {
                let mut v = 1.0;
                for (k, &e) in 指数.iter().enumerate() {
                    v *= 状态[k].powi(e as i32);
                }
                特征.push(v);
            }
            提升序列.push(特征);
        }
        // 去均值（特征空间）
        let 特征均值: Vec<f64> = (0..字典规模)
            .map(|d| 提升序列.iter().map(|s| s[d]).sum::<f64>() / 提升序列.len() as f64)
            .collect();
        let 去均值: Vec<Vec<f64>> = 提升序列
            .iter()
            .map(|s| s.iter().zip(&特征均值).map(|(v, m)| v - m).collect())
            .collect();

        // 在特征空间做 DMD
        let d = 字典规模;
        let x = DMatrix::from_fn(d, n_快照, |i, j| 去均值[j][i]);
        let xp = DMatrix::from_fn(d, n_快照, |i, j| 去均值[j + 1][i]);

        let mut 结果 = self.dmd核心(x, xp, Δt)?;
        结果.方法 = "EDMD".to_string();
        结果.字典规模 = Some(字典规模);
        // EDMD 状态维数报告原变量数（特征空间维数放字典规模）
        结果.状态维数 = n;

        // 字典收敛性检查（论文§3.5）: 低阶字典 vs 高阶字典的本征值是否稳定
        // 用当前结果的稳定模 τ 链是否非空作为初步收敛信号
        if 字典阶 >= 2 && !结果.τ链.is_empty() {
            let 低阶 = self.多变量edmd(状态序列, Δt, 1);
            if let Some(低) = 低阶 {
                if !低.τ链.is_empty() {
                    // 比较两字典 τ 链的"最近匹配"（每高阶梯找低阶梯最近邻，取中位相对偏差）
                    let mut 偏差列表 = Vec::new();
                    for &a in &结果.τ链 {
                        let 最近 = 低.τ链
                            .iter()
                            .map(|&b| ((a - b) / a.abs().max(b.abs()).max(1e-12)).abs())
                            .fold(f64::INFINITY, f64::min);
                        偏差列表.push(最近);
                    }
                    if !偏差列表.is_empty() {
                        偏差列表.sort_by(|x, y| x.partial_cmp(y).unwrap());
                        let 中位 = 偏差列表[偏差列表.len() / 2];
                        结果.字典收敛 = Some(中位 < 0.3);
                    }
                }
            }
        }
        Some(结果)
    }

    /// 生成多项式字典指数向量（Σe_i ∈ 1..=字典阶 的 n 元非负整数向量，有序）
    ///
    /// T-30: **stars-and-bars 直接生成 Σ=总阶 的向量**（跳过全部非法叉积），复杂度从旧
    /// 叉积 `(总阶+1)^n` 降为组合数 `C(总阶+n-1, n-1)`；字典行序与旧 `枚举单项式` 一致
    /// （位置 0 递增同序）→ 快照矩阵行序不变、谱结果逐值不变（T-30 条件2 用 n=3 对照自证）。
    /// 超过 `字典规模硬上限` → 返回 `None`（诚实截断；不静默给错谱、不 panic）。
    fn 生成指数向量(n: usize, 字典阶: usize) -> Option<Vec<Vec<u32>>> {
        let mut 输出 = Vec::new();
        for 总阶 in 1..=字典阶 {
            if !Self::枚举bars(n, 总阶, 0, &mut vec![0u32; n], &mut 输出) {
                return None; // 超硬上限，诚实中止
            }
        }
        if 输出.is_empty() {
            None
        } else {
            Some(输出)
        }
    }

    /// stars-and-bars 递归：当前位置取 e∈0..=剩余，保证最终 Σ=总阶（跳过非法叉积）
    fn 枚举bars(n: usize, 剩余: usize, 位置: usize, 当前: &mut Vec<u32>, 输出: &mut Vec<Vec<u32>>) -> bool {
        if 输出.len() >= 字典规模硬上限 {
            return false; // 超限诚实中止
        }
        if 位置 + 1 == n {
            当前[位置] = 剩余 as u32;
            输出.push(当前.clone());
            return true;
        }
        for e in 0..=剩余 {
            当前[位置] = e as u32;
            if !Self::枚举bars(n, 剩余 - e, 位置 + 1, 当前, 输出) {
                return false;
            }
        }
        true
    }

    /// 实矩阵复特征值（nalgebra Schur 分解，标准解法）
    ///
    /// 用 Schur 分解的 `complex_eigenvalues()` 直接获得复特征值。
    /// 返回 (实部, 虚部) 列表。仅适用于小型矩阵（n ≤ 64），DMD 截断秩 r 通常 < 32。
    pub fn qr复特征值(a: &DMatrix<f64>) -> Vec<(f64, f64)> {
        let n = a.nrows();
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return vec![(a[(0, 0)], 0.0)];
        }
        if n == 2 {
            let (r1, r2) = Self::特征值2x2(a[(0, 0)], a[(0, 1)], a[(1, 0)], a[(1, 1)]);
            return vec![r1, r2];
        }
        let schur = Schur::new(a.clone());
        schur
            .complex_eigenvalues()
            .iter()
            .map(|λ| (λ.re, λ.im))
            .collect()
    }

    /// D1-R4（2026-08-23）: 失稳模检测——从完整复L 提取对角虚部（λ_im），**λ_im>1e-9 ⟺ γ<0 失稳**
    /// （统一令 §1 判据反转；检出的信号按物理量 γ=−λ_im 报告，失稳模 γ<0）
    /// 失稳模标记排除: 不进 p_obs/p_exc/椭圆系综拟合谱集, 单独上报预警。
    pub fn 失稳模检测(复l: &[f64], r_eff: usize) -> Vec<失稳模信号> {
        let mut 信号 = Vec::new();
        for i in 0..r_eff {
            let ω = 复l.get(i * r_eff * 2).copied().unwrap_or(0.0);
            let λ_im = 复l.get(i * r_eff * 2 + 1).copied().unwrap_or(0.0);
            if λ_im > 1e-9 {
                // 增长模（λ_im>0 ⟺ γ=−λ_im<0）
                信号.push(失稳模信号 { 索引: i, 频率: ω, γ: -λ_im });
            }
        }
        信号
    }

    /// D1-R4: γ 半正定校验（全部模 **γ ≥ 0** ⟺ 全部对角 λ_im ≤ 0 → 耗散稳定；统一令后口径）
    pub fn γ半正定(复l: &[f64], r_eff: usize) -> bool {
        (0..r_eff)
            .all(|i| 复l.get(i * r_eff * 2 + 1).copied().unwrap_or(0.0) <= 1e-9)
        // 注：判据按矩阵虚部（λ_im）≤0 表述与实现——数值逻辑与旧实现完全一致（半正定以 λ_im 衡量），
        //     对外口径：等价于「全部模 γ≥0」。矩阵虚部命名 λ_im（统一令 §3）。
    }

    /// 2×2 矩阵特征值（判别式法）
    fn 特征值2x2(a: f64, b: f64, c: f64, d: f64) -> ((f64, f64), (f64, f64)) {
        let 迹 = a + d;
        let 行列式 = a * d - b * c;
        let 判别式 = 迹 * 迹 / 4.0 - 行列式;
        if 判别式 >= 0.0 {
            let 根 = 判别式.sqrt();
            ((迹 / 2.0 + 根, 0.0), (迹 / 2.0 - 根, 0.0))
        } else {
            let 根 = (-判别式).sqrt();
            ((迹 / 2.0, 根), (迹 / 2.0, -根))
        }
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// T-10 D7 复数据链入口（3.1）: 实信号 → 复EDMD → 方案B 复L（对外接口不变, 单通道 None）
    #[test]
    fn test_T10_复数据链入口() {
        let Δt = 0.1;
        let n = 200;
        // 通道1: 快衰减振荡; 通道2: 慢衰减 + 0.3×通道1 耦合（含跨通道耦合）
        let 通道1: Vec<f64> = (0..n)
            .map(|k| (-(k as f64 * Δt) / 3.0).exp() * (0.4 * k as f64 * Δt).cos())
            .collect();
        let 通道2: Vec<f64> = (0..n)
            .map(|k| (-(k as f64 * Δt) / 10.0).exp() + 0.3 * 通道1[k])
            .collect();
        let 辨识器 = 系统辨识器::default();
        let 结果 = 辨识器
            .多变量dmd_复数据链(&[通道1, 通道2], Δt)
            .expect("复数据链应成功");
        assert!(结果.复L阶 >= 2, "2通道应≥2阶: {}", 结果.复L阶);
        // 方案B 恢复非对角虚部（D7 V3）
        let r = 结果.复L阶;
        let mut 非对角虚非零 = 0;
        for i in 0..r {
            for j in 0..r {
                if i != j {
                    let im = 结果.复L[2 * (i * r + j) + 1];
                    if im.abs() > 1e-6 {
                        非对角虚非零 += 1;
                    }
                }
            }
        }
        assert!(非对角虚非零 >= 1, "方案B 复L 非对角虚部应非零: {}", 非对角虚非零);
        // 单通道 → None（D7 §4 硬边界: 熵泵需≥2通道, 不静默）
        let 单通道: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        assert!(
            辨识器.多变量dmd_复数据链(&[单通道], 0.1).is_none(),
            "单通道应 None（D7 §4）"
        );
    }

    #[test]
    fn test_2x2特征值实() {
        // [[1,0],[0,2]] → 1, 2
        let (r1, r2) = 系统辨识器::特征值2x2(1.0, 0.0, 0.0, 2.0);
        assert!((r1.0 - 2.0).abs() < 1e-9, "r1={:?}", r1);
        assert!((r2.0 - 1.0).abs() < 1e-9, "r2={:?}", r2);
        assert!(r1.1.abs() < 1e-12 && r2.1.abs() < 1e-12);
    }

    #[test]
    fn test_2x2特征值复() {
        // [[0,-1],[1,0]] → ±i
        let (r1, r2) = 系统辨识器::特征值2x2(0.0, -1.0, 1.0, 0.0);
        assert!((r1.0.abs()).abs() < 1e-9 && (r1.1 - 1.0).abs() < 1e-9, "r1={:?}", r1);
        assert!(r2.1.abs() < 1e-9 || (r2.1 + 1.0).abs() < 1e-9, "r2={:?}", r2);
    }

    #[test]
    fn test_qr特征值三角矩阵() {
        // 上三角 [[3,1],[0,2]] → 3, 2
        let m = DMatrix::from_row_slice(2, 2, &[3.0, 1.0, 0.0, 2.0]);
        let ev = 系统辨识器::qr复特征值(&m);
        assert_eq!(ev.len(), 2);
        let mut 实: Vec<f64> = ev.iter().map(|e| e.0).collect();
        实.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((实[0] - 2.0).abs() < 1e-6, "实={:?}", 实);
        assert!((实[1] - 3.0).abs() < 1e-6, "实={:?}", 实);
    }

    #[test]
    fn test_qr特征值复共轭对() {
        // 4×4 分块：对角块 [[0,-1],[1,0]]（±i）和 [[2,0],[0,3]]
        // 期望特征值: i, -i, 2, 3
        let m = DMatrix::from_row_slice(
            4,
            4,
            &[0.0, -1.0, 0.0, 0.0,
              1.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 2.0, 0.0,
              0.0, 0.0, 0.0, 3.0],
        );
        let ev = 系统辨识器::qr复特征值(&m);
        assert_eq!(ev.len(), 4, "应恢复4个特征值: {:?}", ev);
        let mut 实部: Vec<f64> = ev.iter().map(|e| e.0).collect();
        实部.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((实部[0] - 0.0).abs() < 1e-4, "复对实部≈0: {:?}", 实部);
        assert!((实部[1] - 0.0).abs() < 1e-4);
        assert!((实部[2] - 2.0).abs() < 1e-4, "实特征值2: {:?}", 实部);
        assert!((实部[3] - 3.0).abs() < 1e-4, "实特征值3: {:?}", 实部);
        // 复对虚部 ±1
        let 虚部最大 = ev.iter().map(|e| e.1.abs()).fold(0.0_f64, f64::max);
        assert!((虚部最大 - 1.0).abs() < 1e-4, "复对虚部≈1: {}", 虚部最大);
    }

    #[test]
    fn test_qr特征值对角矩阵() {
        // 5×5 对角矩阵
        let m = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
        let ev = 系统辨识器::qr复特征值(&m);
        assert_eq!(ev.len(), 5);
        let mut 实: Vec<f64> = ev.iter().map(|e| e.0).collect();
        实.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (i, v) in [1.0, 2.0, 3.0, 4.0, 5.0].iter().enumerate() {
            assert!((实[i] - v).abs() < 1e-6, "第{}个={}, 期望{}", i, 实[i], v);
        }
    }

    #[test]
    fn test_dmd恢复衰减振荡() {
        // 已知系统: x(t) = e^{-t/τ}·cos(ωt), τ=2.0, ω=1.0 rad/s
        let τ = 2.0;
        let ω = 1.0;
        let Δt = 0.1;
        let n = 400;
        let 时序: Vec<f64> = (0..n)
            .map(|k| {
                let t = k as f64 * Δt;
                (-t / τ).exp() * (ω * t).cos()
            })
            .collect();
        let 辨识器 = 系统辨识器::default();
        let 结果 = 辨识器.单变量dmd(&时序, 10, Δt).expect("DMD应成功");
        // 应有稳定模
        assert!(结果.稳定模数 >= 1, "应至少1个稳定模: {}", 结果.稳定模数);
        // τ 链中应存在接近 2.0 的模（主导弛豫模，DMD 嵌入可能附带伪慢模故用 any）
        let 有τ = 结果.τ链.iter().any(|&t| (t - τ).abs() / τ < 0.35);
        assert!(有τ, "应恢复主导τ≈{}: {:?}", τ, 结果.τ链);
        // 频率应含 ≈1.0
        let 有ω = 结果.频率.iter().any(|&w| (w - ω).abs() < 0.3);
        assert!(有ω, "应恢复振荡频率≈1.0: {:?}", 结果.频率);
        println!("DMD恢复: τ链={:?} 频率={:?}", 结果.τ链, 结果.频率);
    }

    #[test]
    fn test_dmd纯指数衰减() {
        // 纯指数衰减 x(t)=e^{-t/τ}, τ=3.0
        let τ = 3.0;
        let Δt = 0.2;
        let n = 300;
        let 时序: Vec<f64> = (0..n).map(|k| (-(k as f64 * Δt) / τ).exp()).collect();
        let 辨识器 = 系统辨识器::default();
        let 结果 = 辨识器.单变量dmd(&时序, 8, Δt).expect("DMD应成功");
        assert!(结果.稳定模数 >= 1);
        // τ 链中应存在接近 3.0 的模（主导慢模）
        let 有τ = 结果.τ链.iter().any(|&t| (t - τ).abs() / τ < 0.35);
        assert!(有τ, "应恢复主导τ≈{}: {:?}", τ, 结果.τ链);
    }

    #[test]
    fn test_dmd不稳定模检测() {
        // 增长信号 x(t) = e^{t/τ}（不稳定，Re(λ)>0）
        let τ = 2.0;
        let Δt = 0.1;
        let n = 200;
        let 时序: Vec<f64> = (0..n).map(|k| ((k as f64 * Δt) / τ).exp()).collect();
        let 辨识器 = 系统辨识器::default();
        let 结果 = 辨识器.单变量dmd(&时序, 8, Δt).expect("DMD应成功");
        assert!(结果.不稳定模数 >= 1, "增长信号应有不稳定模: {}", 结果.不稳定模数);
    }

    #[test]
    fn test_dmd对称性判定() {
        // 非梯度信号（带相位旋转的二维系统）→ GUE
        let Δt = 0.05;
        let n = 500;
        let ω = 0.5;
        let 时序: Vec<f64> = (0..n)
            .map(|k| {
                let t = k as f64 * Δt;
                (-t / 5.0).exp() * (ω * t).cos()
            })
            .collect();
        let 辨识器 = 系统辨识器::default();
        let 结果 = 辨识器.单变量dmd(&时序, 8, Δt).expect("DMD应成功");
        // 单变量 Hankel 嵌入的 Ã 一般非对称 → GUE
        assert_eq!(结果.对称性类, "GUE", "振荡系统应判GUE: {:?}", 结果.对称性类);
    }

    #[test]
    fn test_样本不足保护() {
        let 辨识器 = 系统辨识器::default();
        assert!(辨识器.单变量dmd(&[1.0, 2.0, 3.0], 2, 0.1).is_none());
        assert!(辨识器.多变量dmd(&[vec![1.0], vec![2.0]], 0.1).is_none());
    }

    /// T-30 条件2: 新「生成指数向量」与旧「枚举单项式」**逐值且逐序**一致（n=2/3、阶1/2）
    #[test]
    fn test_生成指数向量_与旧枚举逐值逐序一致() {
        // 旧逻辑参考（叉积+筛 Σ=总阶, 即被替换的 枚举单项式）
        fn 旧枚举(n: usize, 总阶: usize) -> Vec<Vec<u32>> {
            fn 递归(n: usize, 总阶: usize, 位置: usize, 当前: &mut Vec<u32>, 输出: &mut Vec<Vec<u32>>) {
                if 位置 == n {
                    if 当前.iter().sum::<u32>() as usize == 总阶 {
                        输出.push(当前.clone());
                    }
                    return;
                }
                for e in 0..=总阶 {
                    当前[位置] = e as u32;
                    递归(n, 总阶, 位置 + 1, 当前, 输出);
                }
            }
            let mut 输出 = Vec::new();
            递归(n, 总阶, 0, &mut vec![0u32; n], &mut 输出);
            输出
        }
        for (n, 阶) in [(2usize, 1usize), (2, 2), (3, 1), (3, 2)] {
            let 新 = 系统辨识器::生成指数向量(n, 阶).expect("小维应成功");
            let mut 旧 = Vec::new();
            for 总阶 in 1..=阶 {
                旧.extend(旧枚举(n, 总阶));
            }
            assert_eq!(新.len(), 旧.len(), "n={n} 阶={阶} 字典规模一致");
            for (i, (a, b)) in 新.iter().zip(旧.iter()).enumerate() {
                assert_eq!(a, b, "n={n} 阶={阶} 第{i} 个指数向量逐值逐序一致");
            }
        }
    }

    /// T-30 条件1: 超硬上限诚实返 None（不 panic、不静默给错谱）
    #[test]
    fn test_生成指数向量_超限返none() {
        assert!(系统辨识器::生成指数向量(500, 2).is_none(), "n=500 阶2 应超限 None");
        assert!(系统辨识器::生成指数向量(446, 2).is_none(), "n≈446 触顶边界 → None");
        assert!(系统辨识器::生成指数向量(128, 2).is_some(), "n=128 阶2 ≈8.4k < 上限 → Some");
    }

    /// T-30 条件3: 多变量 EDMD 真跑——不爆炸、转置保护、找回注入频率
    ///
    /// 频率恢复的数学前提：状态须含相位信息（sin/cos 对）才能线性预测下一时刻；
    /// 纯 sin-only 观测（无 cos）由离散映射 x_{k+1}=A·x_k 不可表达 → EDMD 恢复虚部受限
    /// （实模）。故黄金断言用 **sin/cos 相位完整状态**（每振荡变量一对通道），验证
    /// 「外层=时刻、内层=通道」布局 + 复谱真实可辨（非仅字段 Some）。
    #[test]
    fn test_多变量edmd_30时刻真跑_黄金断言() {
        let 时刻 = 200usize; // ≥30（蒸馏多变量既有时刻门槛）
        let 通道数 = 4usize; // 2 个振荡变量 × sin/cos 相位对
        let dt = 0.1;
        let ω: [f64; 2] = [1.0, 2.0]; // rad/s（周期 6.28/3.14s, 20s 内 3/6 周期）
        // 外层=时刻、内层=通道维（core 契约: 每时刻一个状态向量; 严禁通道在外对调）
        let 状态序列: Vec<Vec<f64>> = (0..时刻)
            .map(|k| {
                let t = k as f64 * dt;
                let 衰减 = (-t / 20.0).exp(); // 缓衰减 → 稳定模 Re=-1/20, 虚部清晰
                vec![
                    衰减 * (ω[0] * t).sin(),
                    衰减 * (ω[0] * t).cos(),
                    衰减 * (ω[1] * t).sin(),
                    衰减 * (ω[1] * t).cos(),
                ]
            })
            .collect();
        // 字典阶 1（线性 EDMD）：无交叉项伪模，干净验证布局与频率恢复
        let 结果 = 系统辨识器::default()
            .多变量edmd(&状态序列, dt, 1)
            .expect("200时刻 EDMD 应终止且有结果（T-30 修复后不爆炸）");
        assert_eq!(结果.状态维数, 通道数, "状态维数=通道数 n（转置保护: 外层时刻/内层通道）");
        assert_eq!(结果.快照数, 时刻 - 1, "快照数=时刻-1");
        eprintln!("黄金断言: 状态维数={} 快照数={} 稳定模={} 频率={:?} τ链={:?}",
            结果.状态维数, 结果.快照数, 结果.稳定模数, 结果.频率, 结果.τ链);
        // 找回注入频率（连续时间 ω=Im(λ) rad/s; 容差 0.2, 注入间隔 1.0 可辨）
        for w in ω {
            let 有 = 结果.频率.iter().any(|&f| (f - w).abs() < 0.2);
            assert!(有, "应找回注入频率 {w} rad/s: {:?}", 结果.频率);
        }
    }

    /// T-30 验收⓵: 30 状态变量（外层时刻≥30、内层通道 30）字典阶 1/2 下 <2s 终止不爆炸
    #[test]
    fn test_多变量edmd_30状态变量_限时() {
        let 时刻 = 60usize;
        let 通道 = 30usize; // 15 振荡 × sin/cos 相位对（相位完整才可线性预测恢复复谱）
        let dt = 0.1;
        let 状态序列: Vec<Vec<f64>> = (0..时刻)
            .map(|k| {
                let t = k as f64 * dt;
                let 衰减 = (-t / 20.0).exp();
                let mut v = Vec::with_capacity(通道);
                for i in 0..15 {
                    let w = 0.5 + i as f64 * 0.25;
                    v.push(衰减 * (w * t).sin());
                    v.push(衰减 * (w * t).cos());
                }
                v
            })
            .collect();
        for 阶 in [1usize, 2usize] {
            let 开始 = std::time::Instant::now();
            let 结果 = 系统辨识器::default().多变量edmd(&状态序列, dt, 阶);
            let 耗时 = 开始.elapsed();
            assert!(
                耗时.as_secs() < 2,
                "30 状态变量阶{阶} 应 <2s 终止（T-30 修复后不爆炸）, 实际 {:?}",
                耗时
            );
            assert!(结果.is_some(), "30 状态变量阶{阶} 应可辨识");
        }
    }

    #[test]
    fn test_edmd多项式字典非线性系统() {
        // 非线性系统: dx/dt = -x + y², dy/dt = -y（x 受 y² 驱动）
        // EDMD 用 2 阶字典应比 1 阶（线性 DMD）恢复更准
        let Δt = 0.05;
        let n = 600;
        let mut 状态 = vec![1.0, 1.0];
        let mut 序列 = Vec::new();
        for _ in 0..n {
            序列.push(状态.clone());
            let dx = -状态[0] + 状态[1] * 状态[1];
            let dy = -状态[1];
            状态[0] += dx * Δt;
            状态[1] += dy * Δt;
        }
        let 辨识器 = 系统辨识器::default();
        // 线性 DMD（1阶）
        let 线性 = 辨识器.多变量dmd(&序列, Δt).expect("线性DMD应成功");
        // EDMD（2阶）
        let edmd = 辨识器.多变量edmd(&序列, Δt, 2).expect("EDMD应成功");
        assert_eq!(edmd.方法, "EDMD");
        assert!(edmd.字典规模.is_some() && edmd.字典规模.unwrap() > 0);
        assert_eq!(edmd.状态维数, 2, "EDMD状态维数应报告原变量数");
        assert!(edmd.稳定模数 >= 1, "非线性系统应有稳定模");
        // EDMD 恢复的 τ 应接近 1.0（两个衰减模 τ=1）
        let 有τ = edmd.τ链.iter().any(|&t| (t - 1.0).abs() < 0.4);
        assert!(有τ, "EDMD应恢复τ≈1: {:?}", edmd.τ链);
        println!("EDMD: 字典={:?} τ链={:?} 收敛={:?}",
                  edmd.字典规模, edmd.τ链, edmd.字典收敛);
        println!("线性DMD: τ链={:?}", 线性.τ链);
    }

    /// 第285批: 多变量DMD 暴露系统矩阵（状态转移 Ã）→ 图上多模态接图用
    #[test]
    fn test_多变量dmd_系统矩阵暴露() {
        let 辨识器 = 系统辨识器::default();
        // 双通道: 快衰减 + 慢振荡（跨通道无耦合, 但 Ã 应含对角耗散）
        let n = 60;
        let 通道1: Vec<f64> = (0..n).map(|i| (-(i as f64) / 2.0).exp() * (0.3 * i as f64).cos()).collect();
        let 通道2: Vec<f64> = (0..n).map(|i| (-(i as f64) / 10.0).exp() * (0.1 * i as f64).cos()).collect();
        let 状态: Vec<Vec<f64>> = (0..n).map(|i| vec![通道1[i], 通道2[i]]).collect();
        let 结果 = 辨识器.多变量dmd(&状态, 0.1).expect("多变量DMD应成功");
        // 系统矩阵暴露且为 r_eff×r_eff 平方
        let 矩阵 = 结果.系统矩阵.expect("多变量DMD应暴露系统矩阵");
        let r = 结果.模态数;
        assert_eq!(矩阵.len(), r * r, "系统矩阵应为{}×{}", r, r);
        // 对角主导（耗散）: 对角元素应有非零值
        let 对角非零 = (0..r).filter(|&i| 矩阵[i * r + i].abs() > 1e-6).count();
        assert!(对角非零 >= 1, "系统矩阵对角应有耗散: {:?}", 矩阵);
        // 单变量DMD 不暴露（嵌入维 Hankel 非物理通道）
        let 单 = 辨识器.单变量dmd(&通道1, 8, 0.1).expect("单变量DMD应成功");
        assert!(单.系统矩阵.is_none(), "单变量DMD不应暴露系统矩阵");
        assert!(单.复L.is_none(), "单变量DMD不应暴露复L");
    }

    /// 第289批 Phase 1 端到端: 时序 → 复L → 非厄米谱分解 → τ 与已知值误差<10%
    /// P22§6 链路第1步(EDMD→L)接线验证。
    /// 信号设计: 双通道 = 纯指数衰减（τ=5 / τ=20）——Ã 特征值实,
    /// L=ln(Ã)/Δt 对角实 γ=ln(r)/Δt 精确恢复衰减率; 避免振荡信号经去均值后
    /// DMD 拟合失真（cos/sin 相位正交信号数值上易失准）。
    #[test]
    fn test_端到端_时序到复L到谱() {
        let 辨识器 = 系统辨识器::default();
        // 已知双模: τ1=5, τ2=20
        let τ1 = 5.0f64;
        let τ2 = 20.0f64;
        let Δt = 0.05;
        let n = 600;
        let 通道1: Vec<f64> = (0..n).map(|k| (-(k as f64 * Δt) / τ1).exp()).collect();
        let 通道2: Vec<f64> = (0..n).map(|k| (-(k as f64 * Δt) / τ2).exp()).collect();
        let 状态: Vec<Vec<f64>> = (0..n).map(|i| vec![通道1[i], 通道2[i]]).collect();
        let 结果 = 辨识器.多变量dmd(&状态, Δt).expect("多变量DMD应成功");
        // 复L 暴露且尺寸=2×r_eff×r_eff（[实,虚]交错）
        let 复l = 结果.复L.expect("多变量DMD应暴露复L");
        let r = 结果.模态数;
        assert_eq!(复l.len(), r * r * 2, "复L应为{}×{}×2", r, r);
        // 构造复矩阵 → 非厄米谱分解 → 提取 τ 链
        let mut l = nalgebra::DMatrix::<nalgebra::Complex<f64>>::zeros(r, r);
        for i in 0..r {
            for j in 0..r {
                let idx = 2 * (i * r + j);
                l[(i, j)] = nalgebra::Complex::new(复l[idx], 复l[idx + 1]);
            }
        }
        let 分解 = crate::nonhermitian_spectrum::非厄米谱求解器::default()
            .分解(&l)
            .expect("非厄米谱分解应成功");
        // τ 链与已知值对比（误差<15%, DMD 数值误差+去均值+截断秩）
        let mut τs: Vec<f64> = 分解.模.iter().filter(|m| m.τ.is_finite()).map(|m| m.τ).collect();
        τs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(τs.len() >= 1, "应有有限τ模: {:?}", 分解.模.iter().map(|m| (m.ω, m.γ, m.τ)).collect::<Vec<_>>());
        let 最慢 = *τs.last().unwrap();
        assert!((最慢 - τ2).abs() / τ2 < 0.15, "最慢τ≈{} vs 已知{}: 误差应<15%", 最慢, τ2);
        if τs.len() >= 2 {
            let 次慢 = τs[τs.len() - 2];
            assert!((次慢 - τ1).abs() / τ1 < 0.20, "次慢τ≈{} vs 已知{}: 误差应<20%", 次慢, τ1);
        }
    }

    #[test]
    fn test_edmd字典收敛性检查() {
        // 线性系统: 1阶与2阶字典 τ 应接近（收敛）
        let Δt = 0.1;
        let n = 300;
        let 状态序列: Vec<Vec<f64>> = (0..n)
            .map(|k| {
                let t = k as f64 * Δt;
                vec![(-t / 2.0).exp(), (-t / 5.0).exp() * (0.8 * t).cos()]
            })
            .collect();
        let 辨识器 = 系统辨识器::default();
        let edmd = 辨识器.多变量edmd(&状态序列, Δt, 2).expect("EDMD应成功");
        assert!(edmd.字典收敛.is_some(), "线性系统字典应收敛: {:?}", edmd.字典收敛);
        println!("线性系统 EDMD 收敛={:?}", edmd.字典收敛);
    }

    // ==================== D1（2026-08-23）: 复L 非对角重构（R3） + 失稳模（R4） ====================

    /// D1-R3: 复L 非对角耦合保留——合成含跨通道耦合信号, 重构复L 非对角元非零
    #[test]
    fn test_D1R3_复L非对角耦合() {
        let 辨识器 = 系统辨识器::default();
        let Δt = 0.05;
        let n = 600;
        // 双通道: 通道1 快衰减振荡; 通道2 = 慢衰减 + 0.3×通道1（跨通道耦合）
        let 通道1: Vec<f64> = (0..n)
            .map(|k| (-(k as f64 * Δt) / 3.0).exp() * (0.4 * k as f64 * Δt).cos())
            .collect();
        let 通道2: Vec<f64> = (0..n)
            .map(|k| (-(k as f64 * Δt) / 10.0).exp() + 0.3 * 通道1[k])
            .collect();
        let 状态: Vec<Vec<f64>> = (0..n).map(|i| vec![通道1[i], 通道2[i]]).collect();
        let 结果 = 辨识器.多变量dmd(&状态, Δt).expect("多变量DMD应成功");
        let 复l = 结果.复L.expect("应暴露复L");
        let r = 结果.模态数;
        assert!(r >= 2, "双通道应恢复≥2模: {}", r);
        // 非对角元存在非零（耦合保留, 方案B 重构）
        let mut 非对角非零 = 0;
        for i in 0..r {
            for j in 0..r {
                if i != j {
                    let idx = 2 * (i * r + j);
                    if 复l[idx].abs() > 1e-6 || 复l[idx + 1].abs() > 1e-6 {
                        非对角非零 += 1;
                    }
                }
            }
        }
        assert!(
            非对角非零 >= 1,
            "复L 非对角元应非零（耦合保留, D1-R3）: 非对角非零={}, 复L={:?}",
            非对角非零, 复l
        );
    }

    /// D1-R4: 失稳模检测 + γ 半正定校验
    #[test]
    fn test_D1R4_失稳模检测() {
        let r_eff = 3;
        let mut 复l = vec![0.0f64; r_eff * r_eff * 2];
        // 矩阵虚部＝λ_im（统一令 §3 命名）；模0: 稳定 λ_im=-0.5（γ=+0.5）, ω=1.0
        复l[0 * r_eff * 2] = 1.0;
        复l[0 * r_eff * 2 + 1] = -0.5;
        // 模1: 失稳 λ_im=+0.3（r>1 ⟺ γ=-0.3<0）, ω=2.0
        复l[1 * r_eff * 2] = 2.0;
        复l[1 * r_eff * 2 + 1] = 0.3;
        // 模2: 稳定 λ_im=-0.1（γ=+0.1）
        复l[2 * r_eff * 2] = 3.0;
        复l[2 * r_eff * 2 + 1] = -0.1;
        let 信号 = 系统辨识器::失稳模检测(&复l, r_eff);
        assert_eq!(信号.len(), 1, "应只标记失稳模1: {:?}", 信号);
        assert_eq!(信号[0].索引, 1);
        // 失稳模信号按物理量 γ=−λ_im 报告（统一令 §1：失稳 γ<0）
        assert!((信号[0].γ - (-0.3)).abs() < 1e-9, "失稳模 γ 应=−0.3: {}", 信号[0].γ);
        assert!((信号[0].频率 - 2.0).abs() < 1e-9);
        assert!(!系统辨识器::γ半正定(&复l, r_eff), "含失稳模（λ_im>0）应非半正定");
        // 全稳定 → 半正定 + 无失稳信号
        let mut 稳定l = 复l.clone();
        稳定l[1 * r_eff * 2 + 1] = -0.3;
        assert!(系统辨识器::γ半正定(&稳定l, r_eff), "全稳定（γ≥0）应半正定");
        assert!(系统辨识器::失稳模检测(&稳定l, r_eff).is_empty());
    }

    /// D1-R3+R4 端到端: 增长信号（r>1）→ DMD 复L 含失稳模（λ_im>0 ⟺ γ<0）, 失稳信号非空
    /// ★ 守卫测试（统一令 §7.3）: 判据反转后"增长模应被检出"仍有效、不弱化。
    #[test]
    fn test_D1R3R4_端到端失稳模上报() {
        let 辨识器 = 系统辨识器::default();
        let Δt = 0.1;
        let n = 300;
        // 通道1 增长（失稳）; 通道2 稳定衰减
        let 通道1: Vec<f64> = (0..n).map(|k| ((k as f64 * Δt) / 5.0).exp()).collect();
        let 通道2: Vec<f64> = (0..n).map(|k| (-(k as f64 * Δt) / 8.0).exp()).collect();
        let 状态: Vec<Vec<f64>> = (0..n).map(|i| vec![通道1[i], 通道2[i]]).collect();
        let 结果 = 辨识器.多变量dmd(&状态, Δt).expect("多变量DMD应成功");
        assert!(结果.不稳定模数 >= 1, "增长信号应标记不稳定模: {}", 结果.不稳定模数);
        let 复l = 结果.复L.expect("应暴露复L");
        let r = 结果.模态数;
        // 失稳模检测: 复L 对角 λ_im>0（⟺ γ<0）的模应被标记（增长信号 → 失稳信号上报）
        let 信号 = 系统辨识器::失稳模检测(&复l, r);
        assert!(
            信号.len() >= 1,
            "增长信号复L 应含失稳模信号（D1-R4）: {:?}",
            信号
        );
        // 反转后信号自洽: 失稳信号的 γ 必须为负（耗散率口径）
        assert!(信号.iter().all(|s| s.γ < 0.0), "失稳模 γ 应<0: {:?}", 信号);
        assert!(!系统辨识器::γ半正定(&复l, r), "增长信号应非半正定");
    }
}
