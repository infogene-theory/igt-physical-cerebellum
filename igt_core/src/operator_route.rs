// SPDX-License-Identifier: BSL-1.0
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! 算子分层路由（T-25: D16《个人Agent算子分层路由与数据准入规范》v1.0 正式 §4.2/§5.2/§6/§8/§12）
//!
//! 类型与 route() **只在本模块定义一次**（拓扑同一律: 个人/企业是同一 igt_core 阶梯的
//! 两 τ_obs 尺度投影, 禁在 igt_agent / engine / Flutter Dart 侧平行另写）。
//! 纯数据结构 + 纯函数, 零 I/O（纯 std、无外部依赖、确定性: 同输入必同输出）。
//!
//! 三层=逐级放松近似的嵌套投影链（P41 推论）: A=π_topo(C) 丢时间维、B=π_real(C) 正规极限
//! 丢虚轴、C 完整。"按需升级"第一性根据=用满足结论的最弱假设层; 停在 B 不是精度低而可能是
//! 方向错（伪平稳）。ceiling 由 Δt/T/σ/N 四维量**算**、不按设备名查表; route 不采信调用方
//! 自报 ceiling（一律由 infer_tier_ceiling 内算）。

use serde::{Deserialize, Serialize};

/// 算子精度层（嵌套投影链 A⊂B⊂C; rank A<B<C 便于取 min）
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OperatorTier {
    /// A 群论层: π_topo(C) 丢时间维（拓扑层读数）
    A_Group,
    /// B 实投影谱层（正规极限）: π_real(C) 丢虚轴（GUE/GOE 仅作实特征值统计参照）
    B_RealSpectrum,
    /// C 非厄米层: 完整复谱（含非对角/非正规/近 EP）
    C_NonHermitian,
}

impl OperatorTier {
    /// 层序 rank（A=0 < B=1 < C=2, 便于取 min/比较）
    pub fn rank(self) -> u8 {
        match self {
            OperatorTier::A_Group => 0,
            OperatorTier::B_RealSpectrum => 1,
            OperatorTier::C_NonHermitian => 2,
        }
    }
}

/// 数据准入标签（D16 §4.2/§5.2, 字段不增删; Δt/T/σ/N 四维量已折叠入既有字段）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAdmissionLabel {
    /// 数据准入等级 D0-D3（0=无数据 / 1=低 / 2=中 / 3=高）
    pub data_class: u8,
    /// 是否有 RR（心拍间隔）序列
    pub rr_available: bool,
    /// 通道数 N（N=1 数学上不可辨识非对角/非正规/熵泵）
    pub channel_count: usize,
    /// 信号质量 σ（<1% 即 quality≥0.7）
    pub quality: f64,
    /// 窗口/最长τ比 T/τ_max（≥3 满足 T≥3τ_max; None=未知）
    pub window_tau_max_ratio: Option<f64>,
    /// 奈奎斯特满足（Δt≤τ_min/2）
    pub nyquist_ok: bool,
    /// 黑盒数据源（无物理语义 → 封顶 B）
    pub blackbox: bool,
    /// 来源模态（HRV/PPG/运动/对话/... 标注用）
    pub source_modality: String,
}

/// 降级原因（8 个全; 优先级 Blackbox>NoRR>SingleChannel>WindowLT3Tau>NyquistFail>
/// QualityLow>BaselineNotReady>ObserverLevelLocked）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DegradeReason {
    /// 黑盒数据源 → 封顶 B
    Blackbox,
    /// 无 RR 序列 → A
    NoRR,
    /// 单通道 N=1 → 不可辨识非对角 → 封顶 B
    SingleChannel,
    /// 窗口不足 3τ（T<3τ_max）→ 封顶 B
    WindowLT3Tau,
    /// 奈奎斯特不满足（Δt>τ_min/2）→ 封顶 B
    NyquistFail,
    /// 信号质量低（σ≥1% → quality<0.7）→ 封顶 B
    QualityLow,
    /// 冷启动基线未就绪 → 结论性请求不解锁 C
    BaselineNotReady,
    /// 观测者等级 O 层不足 → 按 O 层解锁
    ObserverLevelLocked,
}

/// 观测者等级 O → 解锁层（§6.1: O0-1→A / O2→B / O3→C）
fn o层(observer_o: u8) -> OperatorTier {
    match observer_o {
        0 | 1 => OperatorTier::A_Group,
        2 => OperatorTier::B_RealSpectrum,
        _ => OperatorTier::C_NonHermitian,
    }
}

/// ① 推断算子精度层封顶（D16 §5.2 + 补强三/四: ceiling 由 Δt/T/σ/N 四维量算, 不按设备名查表）
///
/// - D0 或 无 RR → A
/// - blackbox 或 D1 → 封顶 B
/// - D2 → 封顶 B（D2 不达 C 门槛; N=1 亦不可辨识）
/// - 仅 D3 且 N≥2 且 nyquist_ok（Δt≤τ_min/2）且 window_tau_max_ratio≥3（T≥3τ_max）
///   且 quality≥0.7（σ<1%）→ C; 任一不满足降回 B
pub fn infer_tier_ceiling(adm: &DataAdmissionLabel) -> OperatorTier {
    if adm.data_class == 0 || !adm.rr_available {
        return OperatorTier::A_Group;
    }
    if adm.blackbox {
        return OperatorTier::B_RealSpectrum;
    }
    if adm.data_class < 3 {
        return OperatorTier::B_RealSpectrum; // D1/D2 封顶 B
    }
    // D3 四门槛:
    if adm.channel_count < 2 {
        return OperatorTier::B_RealSpectrum; // N=1 不可辨识非对角
    }
    if !adm.nyquist_ok {
        return OperatorTier::B_RealSpectrum;
    }
    if adm.window_tau_max_ratio.map_or(true, |r| r < 3.0) {
        return OperatorTier::B_RealSpectrum; // T<3τ_max
    }
    if adm.quality < 0.7 {
        return OperatorTier::B_RealSpectrum; // σ≥1%
    }
    OperatorTier::C_NonHermitian
}

/// ② 路由（D16 §6/§7.5/§8）: actual=min(req, ceiling, O层); actual<req → 最主要降级原因
///
/// - ceiling 由 infer_tier_ceiling 内算（**不采信调用方自报**）
/// - O 解锁层: O0-1→A / O2→B / O3→C（§6.1）
/// - 冷启动 baseline_ready=false: 结论性请求被 BaselineNotReady 门控、不解锁 C（§7.5）
/// - reason 优先级: Blackbox>NoRR>SingleChannel>WindowLT3Tau>NyquistFail>QualityLow>
///   BaselineNotReady>ObserverLevelLocked; actual==req → None
/// - 确定性: 纯函数、无 IO/时间/随机, 同输入必同输出
pub fn route(
    req: OperatorTier,
    adm: &DataAdmissionLabel,
    observer_o: u8,
    baseline_ready: bool,
) -> (OperatorTier, Option<DegradeReason>) {
    let ceiling = infer_tier_ceiling(adm);
    let o层 = o层(observer_o);
    // 冷启动门控: baseline 未就绪时不了解锁 C
    let o层_eff = if !baseline_ready {
        o层.min(OperatorTier::B_RealSpectrum)
    } else {
        o层
    };
    let actual = req.min(ceiling).min(o层_eff);
    let reason = 降级原因(req, adm, actual, o层, baseline_ready);
    (actual, reason)
}

/// 按优先级判定最主要降级原因（actual<req 时调用; T-25ⓑ: actual 由 route 传入, 不再重复计算 min）
fn 降级原因(
    req: OperatorTier,
    adm: &DataAdmissionLabel,
    actual: OperatorTier,
    o层: OperatorTier,
    baseline_ready: bool,
) -> Option<DegradeReason> {
    if actual == req {
        return None; // 无降级
    }
    // 优先级: Blackbox > NoRR > SingleChannel > WindowLT3Tau > NyquistFail >
    //         QualityLow > BaselineNotReady > ObserverLevelLocked
    if adm.blackbox {
        return Some(DegradeReason::Blackbox);
    }
    if !adm.rr_available {
        return Some(DegradeReason::NoRR);
    }
    if adm.channel_count < 2 {
        return Some(DegradeReason::SingleChannel);
    }
    if adm.window_tau_max_ratio.map_or(true, |r| r < 3.0) {
        return Some(DegradeReason::WindowLT3Tau);
    }
    if !adm.nyquist_ok {
        return Some(DegradeReason::NyquistFail);
    }
    if adm.quality < 0.7 {
        return Some(DegradeReason::QualityLow);
    }
    if !baseline_ready && req > OperatorTier::B_RealSpectrum {
        return Some(DegradeReason::BaselineNotReady);
    }
    if o层 < req {
        return Some(DegradeReason::ObserverLevelLocked);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// D3 全门槛满足的 label（供测试复用）
    fn d3_全过() -> DataAdmissionLabel {
        DataAdmissionLabel {
            data_class: 3,
            rr_available: true,
            channel_count: 2,
            quality: 0.9,
            window_tau_max_ratio: Some(5.0),
            nyquist_ok: true,
            blackbox: false,
            source_modality: "test".to_string(),
        }
    }

    // ==================== V1-V5（D16 §12 本单必做） ====================

    #[test]
    fn v1_d0请c_降a_无rr() {
        // D0 → ceiling=A; 请 C → (A, Some(NoRR))
        let adm = DataAdmissionLabel {
            data_class: 0,
            rr_available: false,
            ..d3_全过()
        };
        let (t, r) = route(OperatorTier::C_NonHermitian, &adm, 3, true);
        assert_eq!(t, OperatorTier::A_Group);
        assert_eq!(r, Some(DegradeReason::NoRR));
    }

    #[test]
    fn v2_d1黑盒_封顶b_reason黑盒() {
        // D1 + blackbox → 封顶 B; 请 C → (B, Some(Blackbox))
        let adm = DataAdmissionLabel {
            data_class: 1,
            blackbox: true,
            ..d3_全过()
        };
        let (t, r) = route(OperatorTier::C_NonHermitian, &adm, 3, true);
        assert_eq!(t, OperatorTier::B_RealSpectrum);
        assert_eq!(r, Some(DegradeReason::Blackbox));
    }

    #[test]
    fn v3_baseline未就绪_不解锁c() {
        // baseline_ready=false（纵 D3/O3 全满足）→ BaselineNotReady、actual 不解锁 C
        let adm = d3_全过();
        let (t, r) = route(OperatorTier::C_NonHermitian, &adm, 3, false);
        assert_eq!(t, OperatorTier::B_RealSpectrum, "冷启动不解锁 C");
        assert_eq!(r, Some(DegradeReason::BaselineNotReady));
    }

    #[test]
    fn v4_确定性_rank_投影_升c被拒() {
        // 确定性: 同输入两次结果恒等
        let adm = d3_全过();
        let a1 = route(OperatorTier::C_NonHermitian, &adm, 3, true);
        let a2 = route(OperatorTier::C_NonHermitian, &adm, 3, true);
        assert_eq!(a1, a2, "同输入必同输出（确定性）");

        // rank: C>B>A 便于向下投影
        assert!(OperatorTier::A_Group.rank() < OperatorTier::B_RealSpectrum.rank());
        assert!(OperatorTier::B_RealSpectrum.rank() < OperatorTier::C_NonHermitian.rank());

        // 可向下投影: min(A,B)=A / min(C,B)=B
        assert_eq!(
            OperatorTier::A_Group.min(OperatorTier::B_RealSpectrum),
            OperatorTier::A_Group
        );
        assert_eq!(
            OperatorTier::C_NonHermitian.min(OperatorTier::B_RealSpectrum),
            OperatorTier::B_RealSpectrum
        );

        // B/A 升 C 被拒: req=C 但 ceiling=B（N=1）→ actual=B
        let adm_b = DataAdmissionLabel {
            channel_count: 1, // N=1 → 封顶 B
            ..d3_全过()
        };
        let (t, _) = route(OperatorTier::C_NonHermitian, &adm_b, 3, true);
        assert_eq!(t, OperatorTier::B_RealSpectrum, "B 层无法升 C");
    }

    #[test]
    fn v5_o2锁b_observer_locked_与_o3全过到c() {
        let adm = d3_全过();
        // O2+D3 足窗 → 锁到 B、ObserverLevelLocked
        let (t, r) = route(OperatorTier::C_NonHermitian, &adm, 2, true);
        assert_eq!(t, OperatorTier::B_RealSpectrum);
        assert_eq!(r, Some(DegradeReason::ObserverLevelLocked));
        // O3+D3+四门槛全过+baseline_ready → (C, None)
        let (t2, r2) = route(OperatorTier::C_NonHermitian, &adm, 3, true);
        assert_eq!(t2, OperatorTier::C_NonHermitian);
        assert_eq!(r2, None);
    }

    // ==================== 四维边界（固化补强三/四） ====================

    #[test]
    fn 边界_n1封b() {
        let adm = DataAdmissionLabel {
            channel_count: 1,
            ..d3_全过()
        };
        assert_eq!(infer_tier_ceiling(&adm), OperatorTier::B_RealSpectrum);
    }

    #[test]
    fn 边界_window小于3tau封b() {
        let adm = DataAdmissionLabel {
            window_tau_max_ratio: Some(2.5),
            ..d3_全过()
        };
        assert_eq!(infer_tier_ceiling(&adm), OperatorTier::B_RealSpectrum);
    }

    #[test]
    fn 边界_quality低门槛封b() {
        let adm = DataAdmissionLabel {
            quality: 0.6,
            ..d3_全过()
        };
        assert_eq!(infer_tier_ceiling(&adm), OperatorTier::B_RealSpectrum);
    }

    #[test]
    fn 边界_全过到c() {
        assert_eq!(infer_tier_ceiling(&d3_全过()), OperatorTier::C_NonHermitian);
    }

    #[test]
    fn 边界_d2封b_即便n大于等于2() {
        // D2 且 N≥2 → 仍封顶 B（D2 不达 C 门槛）
        let adm = DataAdmissionLabel {
            data_class: 2,
            ..d3_全过()
        };
        assert_eq!(infer_tier_ceiling(&adm), OperatorTier::B_RealSpectrum);
    }

    // ==================== route 层 reason 锁定（PM T-25 补测: 8 个 DegradeReason 全锁定） ====================

    #[test]
    fn reason_window不足3tau() {
        // 请 C、O3、baseline=true, 仅窗口不足 → (B, Some(WindowLT3Tau))
        let adm = DataAdmissionLabel {
            window_tau_max_ratio: Some(2.5),
            ..d3_全过()
        };
        let (t, r) = route(OperatorTier::C_NonHermitian, &adm, 3, true);
        assert_eq!(t, OperatorTier::B_RealSpectrum);
        assert_eq!(r, Some(DegradeReason::WindowLT3Tau));
    }

    #[test]
    fn reason_奈奎斯特不满足() {
        let adm = DataAdmissionLabel {
            nyquist_ok: false,
            ..d3_全过()
        };
        let (t, r) = route(OperatorTier::C_NonHermitian, &adm, 3, true);
        assert_eq!(t, OperatorTier::B_RealSpectrum);
        assert_eq!(r, Some(DegradeReason::NyquistFail));
    }

    #[test]
    fn reason_质量低() {
        let adm = DataAdmissionLabel {
            quality: 0.6,
            ..d3_全过()
        };
        let (t, r) = route(OperatorTier::C_NonHermitian, &adm, 3, true);
        assert_eq!(t, OperatorTier::B_RealSpectrum);
        assert_eq!(r, Some(DegradeReason::QualityLow));
    }

    #[test]
    fn reason_单通道() {
        // SingleChannel 优先级高于 Window/Nyquist/Quality, 在 d3_全过 基础上仅改 N=1
        let adm = DataAdmissionLabel {
            channel_count: 1,
            ..d3_全过()
        };
        let (t, r) = route(OperatorTier::C_NonHermitian, &adm, 3, true);
        assert_eq!(t, OperatorTier::B_RealSpectrum);
        assert_eq!(r, Some(DegradeReason::SingleChannel));
    }

    // ==================== 序列化（frb 兼容） ====================

    #[test]
    fn test_序列化往返() {
        let adm = d3_全过();
        let json = serde_json::to_string(&adm).unwrap();
        let back: DataAdmissionLabel = serde_json::from_str(&json).unwrap();
        assert_eq!(back.data_class, 3);
        assert_eq!(back.source_modality, "test");
        let t = OperatorTier::C_NonHermitian;
        let j2 = serde_json::to_string(&t).unwrap();
        let b2: OperatorTier = serde_json::from_str(&j2).unwrap();
        assert_eq!(b2, OperatorTier::C_NonHermitian);
    }
}
