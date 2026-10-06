// SPDX-License-Identifier: BSL-1.1
// T35-3 (R1 §4.2/§5): 模型臂——agent 侧本地进程, 接 T-34 已验证 llama.cpp CPU 后端
//
// 职责: 读锚包 JSON(engine r3 产出) → 构造 prompt → 确定性调 llama-cli(temp0+s42) →
// 提取并校验 ClaimForCheck JSON → 不合规按轮重取(≤k_max, 缺省 3) → 产出 Claim JSON 供 engine 裁决消费
// 边界: 三端铁律——本地模型进程归 agent; engine 不引模型/网络; core 零改动; 全程断网;
// 权重与 llama.cpp 二进制不入库; hidden_state_ref 只留 norm-N 字段、不接线 H_hall
#![allow(non_snake_case)]

use serde::{Deserialize, Serialize};
use std::io::Read;
use std::process::Command;
// N1 (R4 收尾): spawn llama 加 CREATE_NEW_PROCESS_GROUP 信号隔离——任何控制台 Ctrl
// 不致整组连带; kill 仍 TerminateProcess(只杀目标, 不广播 Ctrl)
#[cfg(windows)]
use std::os::windows::process::CommandExt;
#[cfg(windows)]
const 创建标志: u32 = 0x00000200 | 0x08000000; // CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW

/// 可验量集合 F（与 engine r3 judge.rs 提取函数一致; 模型 Claim 的 observable 必须落此）
pub const 可验量F整体: [&str; 4] = ["tau_dominant", "unstable_mode_count", "gamma_max", "c_available"];
pub const 可验量F通道量后缀: [&str; 3] = ["b_dominant_freq", "b_energy", "b_modes"];

/// Claim 声明时窗（R1 §4.2 time_window）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 时窗 {
    pub start_ts_ms: u64,
    pub end_ts_ms: u64,
}

/// 模型引用（R1 §4.2 model_ref）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 模型引用 {
    pub name: String,
    pub quant: String,
    pub backend: String,
    pub deterministic_seed: String,
}

/// ClaimForCheck（R1 §4.2 字段级冻结, 与 engine 裁决消费的 JSON schema 一致）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimForCheck {
    pub claim_id: String,
    pub anchor_id: String,
    pub observable: String,
    pub value: f64,
    pub unit: String,
    pub direction: String, // present/absent/higher/lower/equal
    pub time_window: 时窗,
    pub model_ref: 模型引用,
    pub hidden_state_ref: Option<String>, // 只留 "norm-N" 字段, 不接线 H_hall
    pub raw_text: Option<String>,         // 仅留痕, 不参与比对
    // T-41 块2 (R4-v2 结构化裁决): 可选新增(旧 Claim/旧端无 → None), 不破 R1 冻结
    /// 关键证据引用: 模型点出判定所依赖的关键模(索引+符号断言), 供双条件判分防"蒙对类别理由错"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub 关键证据引用: Option<Vec<证据引用>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub 置信度: Option<f64>,
}

/// 单条关键证据引用（模型结构化输出: 它判别的关键模 + 符号断言）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 证据引用 {
    /// 全模谱模索引(与证据锚/裁决锚全模谱对齐)
    pub 模索引: usize,
    /// 符号断言: "正"|"负"|"近零"(模型对 γ 符号的判断)
    pub 符号断言: String,
}

/// observable 是否落在可验量集合 F（整体量或 通道.{量后缀}）
pub fn observable落F(observable: &str) -> bool {
    if 可验量F整体.contains(&observable) {
        return true;
    }
    let parts: Vec<&str> = observable.splitn(2, '.').collect();
    parts.len() == 2 && 可验量F通道量后缀.contains(&parts[1])
}

/// observable 对应标准单位（PM 09-08: 模型偶漏 unit 时确定性补全; unit 不参与对错判据, 仅保 schema 完整）
pub fn observable默认unit(observable: &str) -> &'static str {
    let 后缀 = observable.split('.').last().unwrap_or(observable);
    match 后缀 {
        "unstable_mode_count" | "b_modes" => "count",
        "tau_dominant" => "s",
        "gamma_max" | "b_dominant_freq" => "rad_s^-1",
        "c_available" | "b_energy" => "state",
        _ => "",
    }
}

/// direction 合法性（非法枚举 → 重取; engine 端亦会 inconclusive 兜底）
pub fn direction合法(d: &str) -> bool {
    matches!(d, "present" | "absent" | "higher" | "lower" | "equal")
}

/// 宽松接收结构（T-40 加固 · 核心/外壳分离）
///
/// 模型只对**核心判读**负责: observable/value/unit/direction(+raw_text)。
/// 外壳字段 anchor_id / time_window / model_ref 允许任意 JSON 类型或缺失（模型在矛盾锚等
/// 情形下 schema 遵从度下降常把 model_ref 写成字符串），由 agent 用锚包/命令行入参**强制
/// 回填覆盖**——模型写歪外壳不再使一条核心正确的 Claim 整体作废。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claim宽松 {
    #[serde(default)]
    pub claim_id: Option<String>,
    pub observable: String,
    pub value: f64,
    #[serde(default)] // PM 09-08: gemma 等偶漏 unit, 缺失落空串, 由 解析校验Claim 按 observable 补全
    pub unit: String,
    pub direction: String,
    #[serde(default)]
    pub raw_text: Option<String>,
    // T-41 块2 结构化裁决(可选, 任意值宽松接收, 解析不了给 None): 关键证据引用/置信度
    // T-44/45: alias 接收模型常见歪键(实测 7B 输出过 "key_evidence引用" "key_evidence" "key_evidence_reference";
    // 引擎输出键仍规范 关键证据引用)
    #[serde(default, alias = "key_evidence引用", alias = "key_evidence_reference", alias = "key_evidence",
        alias = "evidence_reference", alias = "证据引用")]
    pub 关键证据引用: Option<serde_json::Value>,
    #[serde(default, alias = "置信度值", alias = "confidence")]
    pub 置信度: Option<serde_json::Value>,
    /// 外壳: 任意类型接收
    #[serde(default)]
    pub anchor_id: Option<serde_json::Value>,
    #[serde(default)]
    pub time_window: Option<serde_json::Value>,
    #[serde(default)]
    pub model_ref: Option<serde_json::Value>,
    #[serde(default)]
    pub hidden_state_ref: Option<String>,
}

/// 核心判读合法性（外壳不参与: 由 agent 回填保证）
pub fn 校验核心(c: &Claim宽松) -> Result<(), String> {
    if !observable落F(&c.observable) {
        return Err(format!("observable 不在可验量集合 F: {}", c.observable));
    }
    if c.unit.is_empty() {
        return Err("unit 为空".to_string());
    }
    if !direction合法(&c.direction) {
        return Err(format!("direction 非法: {}", c.direction));
    }
    if !c.value.is_finite() {
        return Err("value 非数值".to_string());
    }
    Ok(())
}

/// 宽松结构 → ClaimForCheck（外壳尽力解析, 解析不了给占位, 随后由回填外壳覆盖）
fn 从宽松构造Claim(c: &Claim宽松) -> ClaimForCheck {
    let anchor_id = c
        .anchor_id
        .as_ref()
        .and_then(|v| v.as_str())
        .unwrap_or(无锚anchor_id)
        .to_string();
    let 窗 = c
        .time_window
        .as_ref()
        .and_then(|v| serde_json::from_value::<时窗>(v.clone()).ok())
        .unwrap_or(时窗 { start_ts_ms: 0, end_ts_ms: 0 });
    let model_ref = c
        .model_ref
        .as_ref()
        .and_then(|v| serde_json::from_value::<模型引用>(v.clone()).ok())
        .unwrap_or(模型引用 {
            name: "unknown".to_string(),
            quant: "unknown".to_string(),
            backend: "llama.cpp".to_string(),
            deterministic_seed: "temp0-s42".to_string(),
        });
    let 证据引用 = c
        .关键证据引用
        .as_ref()
        .and_then(|v| serde_json::from_value::<Vec<证据引用>>(v.clone()).ok());
    let 置信度 = c.置信度.as_ref().and_then(|v| v.as_f64());
    ClaimForCheck {
        claim_id: c.claim_id.clone().unwrap_or_else(|| "c-unknown".to_string()),
        anchor_id,
        observable: c.observable.clone(),
        value: c.value,
        unit: c.unit.clone(),
        direction: c.direction.clone(),
        time_window: 窗,
        model_ref,
        hidden_state_ref: c.hidden_state_ref.clone(),
        raw_text: c.raw_text.clone(),
        关键证据引用: 证据引用,
        置信度,
    }
}

/// 解析并校验 Claim JSON（核心严格 / 外壳宽松 → 缺省占位, 由调用方回填覆盖）
pub fn 解析校验Claim(json: &str) -> Result<ClaimForCheck, String> {
    let mut 宽: Claim宽松 =
        serde_json::from_str(json).map_err(|e| format!("Claim JSON 解析失败: {}", e))?;
    // PM 09-08: 模型漏 unit 时按 observable 补标准单位(不放宽 direction/value 核心校验)
    if 宽.unit.trim().is_empty() {
        宽.unit = observable默认unit(&宽.observable).to_string();
    }
    校验核心(&宽)?;
    Ok(从宽松构造Claim(&宽))
}

/// 外壳回填（T-40 加固）: 用 agent 持有的入参强制覆盖 anchor_id / time_window / model_ref
pub fn 回填外壳(
    claim: &mut ClaimForCheck,
    anchor_id: &str,
    窗: 时窗,
    model_name: &str,
    quant: &str,
) {
    // T-51 登记3: 后端名参数化(local-cli→llama.cpp; local-server/openai-compat→实际后端), 云端混跑可溯源
    回填外壳_带后端(claim, anchor_id, 窗, model_name, quant, "llama.cpp")
}

/// 带显式后端名的外壳回填(T-51 登记3: 云端 model_ref.backend 不再写死 llama.cpp)
pub fn 回填外壳_带后端(
    claim: &mut ClaimForCheck,
    anchor_id: &str,
    窗: 时窗,
    model_name: &str,
    quant: &str,
    backend_名: &str,
) {
    claim.anchor_id = anchor_id.to_string();
    claim.time_window = 窗;
    claim.model_ref = 模型引用 {
        name: model_name.to_string(),
        quant: quant.to_string(),
        backend: backend_名.to_string(),
        deterministic_seed: "temp0-s42".to_string(),
    };
}

/// 锚包摘要（T-44: 从**证据锚**（块1 EV, 顶层平铺、无 c_result、无结论量）提取判读所需最小量;
/// serde_json::Value 免重复大结构。全模谱〔{索引,ω,γ}〕原样给模型, 自行判 γ 符号/计数）
#[derive(Debug, Clone)]
pub struct 锚包摘要 {
    pub anchor_id: String,
    pub start_ts_ms: u64,
    pub end_ts_ms: u64,
    /// 全模谱 (索引,ω,γ)（证据锚顶层「模谱」; 含 γ≈0 临界模; 不给哪个失稳的结论）
    pub 全模谱: Vec<(usize, f64, f64)>,
    /// T-46 易档: 每模 System-1 性质标签(增长/衰减/近临界), 与 全模谱 索引对齐; 中档 EV 无 → 全 None
    pub 模性质: Vec<Option<String>>,
    /// 稳定模特征时间（证据锚顶层「τ数值」; 无排序保证, 原样展示）
    pub τ数值: Vec<f64>,
    /// C 慢化对照数值: (注入段主导τ, 基线段主导τ)（证据锚顶层「对照τ数值」两元素; U/S 无 → None）
    pub 对照τ数值: Option<(f64, f64)>,
    /// T-48 块4: ε_c 标定(证据锚顶层「标定」: γ_H0 中心 b / ε_c / 中心化 z / ω) — PM B 口径,
    /// 显著性一律用中心化 z=(γ_EDMD−b)/ε_c 相对零分布判, 不要求裸 γ>0
    pub 标定: Option<标定信息>,
    /// 复谱可用 = 全模谱非空
    pub 复谱可用: bool,
    pub dt_sec: f64,
}

/// T-48 块4 ε_c 标定信息（来自题面按 (σn,窗长,ω) 查表, EV 只读增导）
#[derive(Debug, Clone)]
pub struct 标定信息 {
    pub b: f64,
    pub ε_c: f64,
    pub z: Option<f64>,
    pub ω0: Option<f64>,
}

/// 从锚包 JSON 提取摘要（块1 证据锚顶层平铺形态: 模谱/τ数值/对照τ数值;
/// 旧裁决锚形态(顶层无 模谱) → Err 显式拒绝, 不静默退化失真）
pub fn 提取锚包摘要(锚包json: &str) -> Result<锚包摘要, String> {
    let v: serde_json::Value =
        serde_json::from_str(锚包json).map_err(|e| format!("锚包 JSON 解析失败: {}", e))?;
    let anchor_id = v
        .get("anchor_id")
        .and_then(|x| x.as_str())
        .ok_or("锚包缺 anchor_id")?
        .to_string();
    let 窗 = v.get("window").ok_or("锚包缺 window")?;
    let start_ts_ms = 窗
        .get("start_ts_ms")
        .and_then(|x| x.as_u64())
        .ok_or("window 缺 start_ts_ms")?;
    let end_ts_ms = 窗
        .get("end_ts_ms")
        .and_then(|x| x.as_u64())
        .ok_or("window 缺 end_ts_ms")?;
    let dt_sec = v
        .get("dt_sec")
        .and_then(|x| x.as_f64())
        .unwrap_or(1.0);
    // T-44 改1: 全模谱读顶层「模谱」〔{索引,ω,γ}〕; 不做 失稳模 推导(结论量已删, 不许喂模型)
    let 模谱 = v.get("模谱").ok_or("锚包非证据锚形态: 缺顶层「模谱」(请先 engine evidence 派生 EV)")?;
    let mut 全模谱: Vec<(usize, f64, f64)> = Vec::new();
    let mut 模性质: Vec<Option<String>> = Vec::new();
    if let Some(a) = 模谱.as_array() {
        for m in a {
            let (Some(索), Some(ω), Some(γ)) = (
                m.get("索引").and_then(|x| x.as_u64()),
                m.get("ω").and_then(|x| x.as_f64()),
                m.get("γ").and_then(|x| x.as_f64()),
            ) else {
                continue; // 缺字段条目跳过(不整锚报错)
            };
            全模谱.push((索 as usize, ω, γ));
            模性质.push(m.get("性质").and_then(|x| x.as_str()).map(|s| s.to_string()));
        }
    }
    // τ 链读顶层「τ数值」(Vec 数值)
    let τ数值: Vec<f64> = v
        .get("τ数值")
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
        .unwrap_or_default();
    // C 对照读顶层「对照τ数值」(两元素 [注入,基线] 数值; 无方向结论)
    let 对照τ数值 = v.get("对照τ数值").and_then(|x| x.as_array()).and_then(|a| {
        if a.len() >= 2 {
            let inj = a[0].as_f64()?;
            let bl = a[1].as_f64()?;
            if inj.is_finite() && bl.is_finite() {
                return Some((inj, bl));
            }
        }
        None
    });
    let 谱可用 = !全模谱.is_empty();
    // T-48 块4: ε_c 标定(顶层「标定」: γ_H0中心b/ε_c/中心化z/ω; 无 → None, 旧题兼容)
    let 标定 = v.get("标定").and_then(|x| {
        let b = x.get("γ_H0中心b")?.as_f64()?;
        let ec = x.get("ε_c")?.as_f64()?;
        Some(标定信息 {
            b,
            ε_c: ec,
            z: x.get("中心化z").and_then(|q| q.as_f64()),
            ω0: x.get("ω").and_then(|q| q.as_f64()),
        })
    });
    Ok(锚包摘要 {
        anchor_id,
        start_ts_ms,
        end_ts_ms,
        全模谱,
        模性质,
        τ数值,
        对照τ数值,
        标定,
        复谱可用: 谱可用,
        dt_sec,
    })
}

/// 本题判读目标（T-40 全量前置: 有锚 prompt 指定本题判哪个量, 防模型挑错量偏题）
/// 只指定可验量与其合法 direction 枚举, **不告诉目标方向**（方向由模型读锚包自判, 守主档不给目标值）
#[derive(Debug, Clone)]
pub struct 判读目标 {
    pub observable: String,
    /// 合法方向枚举（逗号分隔, 如 "present,absent" 或 "higher,lower,equal"）
    pub 方向枚举: String,
}

/// 构造判读 prompt（含可验量实际值与输出 schema 指令）
/// T-38 K2: model_ref 不再写死——name/quant 由调用方传入（从模型文件/CLI 参数得）;
/// F 清单列全量（整体量 + 通道量, 通道名随锚包通道提示）——模型知情但 R3 主用整体量。
/// T-40: 目标 observale + 方向枚举（Option, None=旧兼容自由选量）
pub fn 构造prompt(摘要: &锚包摘要, model_name: &str, quant: &str) -> String {
    构造prompt_题向(摘要, model_name, quant, None)
}

/// 题向版构造 prompt（含可选「本题判读目标」段）
/// T-44 改3: 证据锚形态——原样列全模谱每模 (索引,ω,γ)+判读规则, **不数个数/不点名哪个失稳/不给
/// present-absent 或 慢化 结论**(守理论红线「给数值合法、给结论非法」); C 题只给两段主导τ数值由模型自比。
pub fn 构造prompt_题向(
    摘要: &锚包摘要,
    model_name: &str,
    quant: &str,
    目标: Option<&判读目标>,
) -> String {
    // T-46 易档/中档双轨: 易档 = EV 模谱带 System-1 性质标签(模性质 有值); 中档 = 现状(只有 ω/γ 数值)
    let 易档 = 摘要.模性质.iter().any(|p| p.is_some());
    // 谱段逐模列出(易档带 性质标签; γ 带显式符号)
    let 谱段: String = if 摘要.全模谱.is_empty() {
        "（无模谱数据）".to_string()
    } else {
        摘要
            .全模谱
            .iter()
            .zip(摘要.模性质.iter())
            .map(|((i, ω, γ), 性质)| match 性质 {
                Some(标签) => format!("模{}: ω={:.4}, γ={:+.4}, 性质:{}", i, ω, γ, 标签),
                None => format!("模{}: ω={:.4}, γ={:+.4}", i, ω, γ),
            })
            .collect::<Vec<_>>()
            .join("; ")
    };
    let 谱前缀 = if 易档 {
        "谱数据（全模谱, 每模含 System-1 确定性「性质」标签）: "
    } else {
        "谱数据（全模谱, 每模已对角化, γ 符号由您自行判读）: "
    };
    let τ段 = if 摘要.τ数值.is_empty() {
        "无".to_string()
    } else {
        format!("{:?}", 摘要.τ数值)
    };
    // C 慢化对照: 只给两段主导τ数值(证据锚「对照τ数值」), 无 慢化可比/慢化 结论
    let 对照提示 = match 摘要.对照τ数值 {
        Some((inj, bl)) => format!(
            " 本锚为「注入段相对基线段」慢化对照判读: 注入段主导τ≈{inj:.1}s、基线段≈{bl:.1}s, 由您自行比较——判 tau_dominant 的 direction: higher=注入明显高于基线 / lower=反之 / equal=无可辨差异。",
            inj = inj,
            bl = bl
        ),
        None => String::new(),
    };
    let 目标提示 = match 目标 {
        Some(t) => format!(
            " 本题判读目标: observable={}（只判此量, 勿选其它）; direction 从 [{}] 选一。",
            t.observable, t.方向枚举
        ),
        None => String::new(),
    };
    // PM 09-08 v3: 标定题点名主带模(全模谱中 ω 最接近 ω0 者), 规则段与输出契约共用,
    // 杜绝模型改选 |γ| 更大的非主带模(gemma 实测会选 ω=1.85 的模1 而非 ω≈0.9 的模0)。
    let 是标定 = 摘要.标定.is_some();
    let 主带模: Option<(usize, f64, f64)> = 摘要.标定.as_ref().and_then(|标| {
        let w0 = 标.ω0.unwrap_or(0.9);
        摘要
            .全模谱
            .iter()
            .min_by(|a, b| {
                (a.1 - w0)
                    .abs()
                    .partial_cmp(&(b.1 - w0).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
    });
    let 主带点名 = match (摘要.标定.as_ref(), 主带模) {
        (Some(标), Some((idx, w, g))) => match 标.z {
            Some(z) => format!(
                "主带模已由 System-1 选定＝模{idx}（ω={w:.4}、γ={g:.2e}，即谱中 ω 最接近 {w0:.2} 的模）；其它模（即使 |γ| 更大）一律不是判据、禁止选用。该主带模中心化 z 已算好 = {z:+.2}，直接使用、禁止你另选模或重算。",
                idx = idx, w = w, g = g, w0 = 标.ω0.unwrap_or(0.9), z = z
            ),
            None => format!(
                "主带模已由 System-1 选定＝模{idx}（ω={w:.4}、γ={g:.2e}）；只允许用它，禁止选其它模。",
                idx = idx, w = w, g = g
            ),
        },
        _ => String::new(),
    };
    let 证据口径 = if 是标定 {
        "标定题只引系统选定的主带模, 符号断言按给定 z: z>0 断「正」、z≤0 断「负」(不看裸 γ)".to_string()
    } else {
        "对照该模 γ 数值判符号: γ>0 断「正」、γ<0 断「负」、≈0 断「近零」".to_string()
    };
    let 方向枚举 = match 目标 {
        Some(t) => t.方向枚举.clone(),
        None => "present 或 absent 或 higher 或 lower 或 equal".to_string(),
    };
    let obs名 = match 目标 {
        Some(t) => t.observable.clone(),
        None => "本题判读量".to_string(),
    };
    // 判读规则段(仅 U/S 无对照τ 注入; C 题方向判据=对照提示段 高低比较)
    let 规则段: String = if 摘要.对照τ数值.is_some() {
        String::new()
    } else if let Some(标) = &摘要.标定 {
        // PM 2026-09-08 v3「标定到位+主带点名」(OQ2; gemma L/M 对照实证): v2 虽给了 z, 但长输出契约让模型
        // 自编 claim_id 先锚定方向、且「符号按 γ」与「按 z」冲突, 出现 raw_text 判对(z=0.85<1→absent)、
        // direction 字段却填 present 的 reason-field 不一致(L 复现/M 修复)。v3 由 System-1 点名主带模索引并给 z,
        // 模型只做阈值比较; 输出契约配套把 direction 置首、回填字段不劳模型(见下方输出段)。
        format!(
            "判读规则(标定口径, 确定性预处理已由 System-1 完成): 零噪声地板 γ_H0 中心 b={b:.2e}、ε_c={ε_c:.2e}。{主带点名}\
             你只做最小判读, 直接比较给定 z 与阈值: z>1 → 显著偏正 → present; |z|≤1(零分布内不显著) 或 z<−1(显著偏负、无增长) → absent。\
             裸 γ 正负不是判据(近临界模系统负偏已计入 b): 即使主带模 γ 为负, 只要给定 z>1 仍判 present, 不得用裸 γ 直觉覆盖给定 z。",
            b = 标.b,
            ε_c = 标.ε_c,
            主带点名 = 主带点名
        )
    } else if 易档 {
        // 易档: 标签扫描(存在性聚合), 去掉「读负号判语义」确定性步骤——7B 不可靠环节由 System-1 预算
        // T-194（AUD-2409-11 A1）随 γ 统一令反转：γ＝−λ_im 为耗散率，**γ<0→增长(失稳)、γ>0→衰减(稳定)**。
        concat!(
            "判读规则(易档): 每模「性质」标签已由 System-1 依 γ 符号确定性预判——γ<0→增长(失稳)、γ>0→衰减(稳定)、|γ|≤1e-9→近临界——与 ω/γ 数值一一对应; ",
            "您只需扫描「性质」标签做存在性聚合与选证据, 勿自行改判符号(标签为准、数值仅供核对)。",
            " 判读步骤: 存在「性质:增长」的模(即 γ<0 的失稳模) → unstable_mode_count 判 present, 且 关键证据引用 必须引用该「增长」模(填其模索引并对该模断言「负」); ",
            "全部「衰减/近临界」(γ≥0) → 判 absent。"
        )
        .to_string()
    } else {
        // T-45 中档 worked example(抽象, 数值仅示意, 严禁本题谱数值; 禁编造声明移出为全档通用行)
        // T-194（A2）随 γ 统一令反转：失稳(增长)模 **γ<0**（耗散率 γ＝−λ_im，稳定 γ>0）。
        concat!(
            "判读规则: γ<0 即增长(失稳)模、γ>0 为衰减(稳定)模、γ≈0(±1e-9) 近临界——存在性/计数/方向由您据上述数值自行判, 模型不预判结论。",
            " 判读步骤(抽象示例, 数值仅示意与本题数据无关): ①先在全模谱中找 γ<0 的模——",
            "正例: 某模 γ=-0.03 即增长(失稳)模; 反例: 某模 γ=+0.4 是衰减模(γ 为正是稳定衰减、不是失稳, 不得据正γ模判 present); ",
            "②unstable_mode_count 判 present 当且仅当存在至少一个 γ<0 模, 且 关键证据引用 必须引用该负γ模并对其断言「负」; ",
            "③全部模 γ>0 或近零 → 判 absent。"
        )
        .to_string()
    };
    // T-38 K2: F 清单列全量(整体量+通道量, 与 可验量F整体/可验量F通道量后缀 及无锚 prompt 同源),
    // 模型知情但 R3 主判整体量; 锚包通道量本题未必有值, 仅告知可选域。
    let F清单 = "可验量集合 F(observable 只能从中选单个): 整体量 unstable_mode_count / tau_dominant / gamma_max / c_available; \
         通道量 cpu_pct.b_dominant_freq / cpu_pct.b_energy / cpu_pct.b_modes / mem_pct.b_dominant_freq / mem_pct.b_energy / mem_pct.b_modes。";
    format!(
        "你是系统失稳判读模型(model_ref: name={model_name}, quant={quant}, 仅告知、输出时勿写该字段)。锚包 {anchor}: {目标提示}{对照提示}\n\
         {谱前缀}{谱段}\n\
         特征时间 τ数值: {τ段}\n\
         {规则段}\n\
         {F清单}\n\
输出一条 JSON（仅 JSON、无其它文字、无 Markdown）。字段顺序固定, 第一个必须是 direction:
         ① direction: 只能填单个 [{方向枚举}]——这是你的最终结论, 先定它, 后面所有字段都必须与它一致;
         ② observable 填 \"{obs名}\"; ③ value 数值估计; ④ unit 与 observable 匹配;
         ⑤ 关键证据引用: 数组 [{{\"模索引\":N,\"符号断言\":\"正\"或\"负\"或\"近零\"}}], 至少 1 条, N 只能取上文谱中真实存在的索引, 禁止编造不存在的索引(N 越界/无此模则该条证据无效); {证据口径};
         ⑥ raw_text: 中文一句, 其结论必须与 direction 完全一致(严禁 raw_text 推断 absent 而 direction 填 present 这类自相矛盾)。
         claim_id、anchor_id、time_window、model_ref、hidden_state_ref、置信度 一律不要输出, 由系统回填。输出前自检 direction、关键证据引用、raw_text 三者方向一致。",
        anchor = 摘要.anchor_id,
        model_name = model_name,
        quant = quant,
        谱前缀 = 谱前缀,
        谱段 = 谱段,
        τ段 = τ段,
        规则段 = 规则段,
        F清单 = F清单,
        对照提示 = 对照提示,
        目标提示 = 目标提示,
        方向枚举 = 方向枚举,
        obs名 = obs名,
        证据口径 = 证据口径,
    )
}

/// 从模型原始输出提取 JSON 块。
/// 优先取 ```json ... ``` 代码块（llama-cli 输出含 prompt 回显, 回显里也有 { };
/// 模型回复常包在 ```json 块内）。无代码块则退化为取末个 { 到末个 } 之间。
pub fn 提取json块(raw: &str) -> Option<String> {
    // 找所有 ```json 块, 取最后一个
    let mut 候选: Option<String> = None;
    let mut 搜索起点 = 0usize;
    while let Some(start) = raw[搜索起点..].find("```json") {
        let 块起点 = 搜索起点 + start + "```json".len();
        let 余 = &raw[块起点..];
        if let Some(end) = 余.find("```") {
            候选 = Some(余[..end].trim().to_string());
            搜索起点 = 块起点 + end + 3;
        } else {
            break;
        }
    }
    if let Some(c) = 候选 {
        if !c.is_empty() {
            return Some(c);
        }
    }
    // 退化: 首个 { 到末个 }
    let s = raw.find('{')?;
    let e = raw.rfind('}')?;
    if e > s {
        Some(raw[s..=e].to_string())
    } else {
        None
    }
}

/// 从模型输出提取并校验 Claim（提取 JSON → 解析校验 → Err 则说明原因）
/// 枚举字符串中所有花括号平衡块（L2: 退化分支反向挑合法者, 不做"首{到末}")
pub fn 候选json(raw: &str) -> Vec<String> {
    let mut 结果 = Vec::new();
    for (i, ch) in raw.char_indices() {
        if ch != '{' {
            continue;
        }
        let mut depth = 0i32;
        for (j, c) in raw[i..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        结果.push(raw[i..i + j + 1].to_string());
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    结果
}

/// 从模型输出提取并校验 Claim（L2 增强）:
/// 1) 优先尝试所有 ```json 围栏块（从后往前, 返回第一个解析校验通过者）
/// 2) 退化分支: 枚举所有花括号平衡块, 用解析校验**反向挑出唯一合法者**
///    （不再"首 { 到末 }"——llama 会回显 prompt 且 prompt 含花括号, 首{可能命中回显）
pub fn 从模型输出取Claim(raw: &str) -> Result<ClaimForCheck, String> {
    // ```json 围栏块
    let mut 搜索起点 = 0usize;
    let mut 围栏候选: Vec<String> = Vec::new();
    while let Some(start) = raw[搜索起点..].find("```json") {
        let 块起点 = 搜索起点 + start + "```json".len();
        let 余 = &raw[块起点..];
        if let Some(end) = 余.find("```") {
            围栏候选.push(余[..end].trim().to_string());
            搜索起点 = 块起点 + end + 3;
        } else {
            break;
        }
    }
    for c in 围栏候选.iter().rev() {
        if let Ok(claim) = 解析校验Claim(c) {
            return Ok(claim);
        }
    }
    // 退化: 花括号平衡块反向挑合法者
    for c in 候选json(raw) {
        if let Ok(claim) = 解析校验Claim(&c) {
            return Ok(claim);
        }
    }
    Err("模型输出中未找到可解析校验通过的 Claim JSON".to_string())
}

/// T-40 加固①: 容错读输出文件——read_to_end 取字节流后 String::from_utf8_lossy
/// （llama 回显超长 prompt 按固定字节截断会劈中中文产生非法 UTF-8 continuation,
/// 严格 read_to_string 会整体失败致读不到已生成的完整 Claim; lossy 后核心判读字段全
/// ASCII 不受影响, 坏字节只落 raw_text 替换符）
pub fn 读输出容错(路径: &std::path::Path) -> Option<String> {
    use std::io::Read;
    let mut buf = Vec::new();
    std::fs::File::open(路径)
        .ok()
        .and_then(|mut f| f.read_to_end(&mut buf).ok().map(|_| String::from_utf8_lossy(&buf).into_owned()))
}

/// 失败时保留 llama 原始输出到 raw 目录（供核查; 先 copy 后删, 不静默吞错）
/// 返回是否成功落出文件。独立 pub 便于自动测试验证。
pub fn 保留失败raw(输出文件路径: &std::path::Path, 目录: &str) -> bool {
    if 目录.is_empty() {
        return false;
    }
    if !输出文件路径.exists() {
        return false;
    }
    let _ = std::fs::create_dir_all(目录);
    let 毫秒 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let 目标 = std::path::Path::new(目录).join(format!("r3_claim_fail_{}.log", 毫秒));
    std::fs::copy(输出文件路径, 目标).map(|_| true).unwrap_or(false)
}

/// 确定性调 llama-cli 子进程（temp0 + s42; stdin null 以 EOF 退出交互; 断网可跑）
/// stdout 落临时文件（绕开 Windows 管道大输出/阻塞风险; 完成后读取并删除）
pub fn 调llama_cli(
    llama_cli: &str,
    model: &str,
    prompt: &str,
    max_token: usize,
    timeout_sec: u64,
    失败raw目录: Option<&str>,
) -> Result<ClaimForCheck, String> {
    use std::io::Read;
    let pid = std::process::id();
    let 输出文件路径 = std::env::temp_dir().join(format!("r3_claim_out_{}.log", pid));
    let stdin文件路径 = std::env::temp_dir().join(format!("r3_claim_stdin_{}.txt", pid));
    let stderr文件路径 = std::env::temp_dir().join(format!("r3_claim_err_{}.log", pid));
    let prompt文件路径 = std::env::temp_dir().join(format!("r3_claim_prompt_{}.txt", pid));
    // T-40 加固(PM 对照实验锁定根因): Windows 命令行 -p 传长中文/引号/花括号 prompt 会被
    // CreateProcess 多层转义破坏(慢化错锚 961 字符 prompt 必挂, -f 文件通道成)——
    // 改用 llama.cpp 原生 -f/--prompt-file: prompt 写临时文件(UTF-8 无 BOM)后经文件直喂, 绕开命令行边界。
    // PM 09-08 晚巡检(14B 27/27 全失败根因): Qwen3 系列默认"Start thinking"长链推理,
    // 思考被 512 token 截断永远轮不到 Claim → Qwen3 thinking 模型 prompt 尾强制 /no_think,
    // 并把 max_token 提到 1024 兜底(模型名判定: 含 Qwen3- 且非 Qwen3VL/视觉模型)。
    let 是qwen3思考 = model.contains("Qwen3-") && !model.contains("VL");
    let prompt_喂 = if 是qwen3思考 {
        format!("{}\n/no_think\n", prompt)
    } else {
        prompt.to_string()
    };
    let max_token_喂 = if 是qwen3思考 { max_token.max(1024) } else { max_token };
    std::fs::write(&prompt文件路径, &prompt_喂)
        .map_err(|e| format!("写 prompt 临时文件失败: {}", e))?;
    // stdout 与 stdin 必须是**两个独立文件**（同文件会自读自写: llama 把生成内容当输入 → 失控）
    let 输出文件 = std::fs::File::create(&输出文件路径)
        .map_err(|e| format!("创建临时输出文件失败: {}", e))?;
    std::fs::write(&stdin文件路径, b"")
        .map_err(|e| format!("创建空 stdin 文件失败: {}", e))?;
    // Windows 上 llama-cli 需真实文件 stdin(EOF) 才会在生成后退出交互; Stdio::null()=NUL 不触发 EOF
    let 空输入 = std::fs::File::open(&stdin文件路径)
        .map_err(|e| format!("打开空 stdin 文件失败: {}", e))?;
    let prompt文件串 = prompt文件路径
        .to_str()
        .ok_or("prompt 临时文件路径非 UTF-8")?
        .to_string();
    let mut 命令 = Command::new(llama_cli);
    命令.args([
        "-m",
        model,
        "-f",
        &prompt文件串,
        "-n",
        &max_token_喂.to_string(),
        "--temp",
        "0",
        "-s",
        "42",
        "--no-escape", // prompt 含中文/JSON 结构, 不处理转义序列
        "--simple-io", // T-40 加固②
        "-st", // PM 09-08: 锚点路径补单轮退出, 修首次 llama 空转、靠重试才出 Claim 的双重加载(无锚路径本已有 -st): 子进程兼容基本 IO, 生成完自然 Exiting、避免进 REPL 狂刷提示符(双保险, 配合"解析到即 kill")
        // PM 09-08 显存护栏: 8GB 卡全 offload 12B/14B 时默认大上下文预分配 KV, free 仅 55MiB
        // → 溢出 shared system memory 走 PCIe, decode 锁低功耗(tg 40→<1)。限上下文 2560 + KV q8 量化,
        // 留足显存余量后恢复满功耗(实测 12B ≈40t/s); 对 7B/8B 余量足无副作用。
        "-c", "2560",
        "-ctk", "q8_0", "-ctv", "q8_0",
    ]);
    // PM 09-08 晚巡检: Qwen3 默认 thinking(Start thinking 中文长链)截断无 Claim → --reasoning off 硬关
    // (llama-cli 选项/env LLAMA_ARG_REASONING; 与 /no_think prompt 双保险)
    if 是qwen3思考 {
        命令.args(["--reasoning", "off"]);
    }
    #[cfg(windows)]
    命令.creation_flags(创建标志);
    let mut child = 命令
        .stdin(空输入)
        .stdout(输出文件)
        // 关键: stderr 也落文件(不 null/管道)——llama-cli 日志量大, 管道不读会阻塞; null 在 Windows 下有退出异常
        .stderr(std::fs::File::create(&stderr文件路径).map_err(|e| format!("创建 stderr 文件失败: {}", e))?)
        .spawn()
        .map_err(|e| format!("启动 llama-cli 失败（{}）: {}", llama_cli, e))?;
    // T-40 加固: 统一清理——失败时**先**把 llama 原始输出 copy 到 raw 目录**再**删临时文件
    // （修原实现"先删后 copy"致 raw_fail 恒空的顺序缺陷）
    let 保留并清理 = |保留: bool| {
        // T-44: 成功/失败均留 llama 原文样本(成功=ok 审计/诊断模型输出, 失败=raw 诊断)
        if let Some(目录) = 失败raw目录 {
            let _ = 保留失败raw(&输出文件路径, 目录);
        }
        let _ = std::fs::remove_file(&输出文件路径);
        let _ = std::fs::remove_file(&stdin文件路径);
        let _ = std::fs::remove_file(&stderr文件路径);
        let _ = std::fs::remove_file(&prompt文件路径);
    };
    // L1: 输出完整性导向轮询——每 400ms 读输出文件(共享读句柄), 一旦能从输出中
    // 提取并解析校验出合法完整 Claim 即取用并 kill 挂起残留(不等"Exiting 后进程退出");
    // 超时(CLI 可配, 默认放宽到与 7B CPU 生成相称)兜底; 区分"正常生成中"(继续等)与"已退出但无合规"(Err)
    let 超时 = std::time::Duration::from_secs(timeout_sec.max(10));
    let 开始 = std::time::Instant::now();
    let mut 已退出: Option<std::process::ExitStatus> = None;
    loop {
        // 读输出文件尝试提取完整合法 Claim（容错解码: 见 读输出容错 注释——禁止严格 read_to_string）
        let 读输出 = 读输出容错(&输出文件路径);
        if let Some(输出) = 读输出 {
            if let Ok(claim) = 从模型输出取Claim(&输出) {
                let _ = child.kill();
                let _ = child.wait();
                保留并清理(false);
                return Ok(claim);
            }
        }
        // 进程退出检查（生成完成且已退出, 但未见合规 Claim → 本轮输出不合规）
        match child.try_wait() {
            Ok(Some(st)) => {
                已退出 = Some(st);
                break;
            }
            Ok(None) => {}
            Err(e) => {
                保留并清理(true);
                return Err(format!("轮询 llama-cli 失败: {}", e));
            }
        }
        if 开始.elapsed() > 超时 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(400));
    }
    if let Some(st) = 已退出 {
        if !st.success() {
            保留并清理(true);
            return Err(format!("llama-cli 退出码非零: {:?}", st));
        }
    }
    // 超时未退 → 杀残留; 最后再读一次尝试提取
    if 已退出.is_none() {
        let _ = child.kill();
        let _ = child.wait();
    }
    let 最后输出 = 读输出容错(&输出文件路径);
    if let Some(输出) = 最后输出 {
        if let Ok(claim) = 从模型输出取Claim(&输出) {
            保留并清理(false);
            return Ok(claim);
        }
    }
    // 失败: 先保留 llama 原始输出(先 copy 后删, 修 raw_fail 恒空)再清理
    保留并清理(true);
    Err(format!(
        "llama 输出未含可解析校验通过的 Claim（已等待 {}s; 模型输出不合规或未产出 JSON）",
        timeout_sec
    ))
}

/// 模型臂主流程: 读锚包 → prompt → 确定性调模型 → 校验 → 不合规重取(≤k_max, 缺省 3)
/// T-38 K2: model_name/quant 参数化（K1 已述: temp0 下同 prompt 重取冗余——此处保留重取仅兜底
/// 输出格式不合规的偶发；C5 回验的真实重生成走下面的 模型产出Claim_带反馈, prompt 每轮不同）
pub fn 模型产出Claim(
    llama_cli: &str,
    model: &str,
    锚包json: &str,
    model_name: &str,
    quant: &str,
    max_token: usize,
    k_max: usize,
    timeout_sec: u64,
    失败raw目录: Option<&str>,
) -> Result<ClaimForCheck, String> {
    let 摘要 = 提取锚包摘要(锚包json)?;
    let prompt = 构造prompt(&摘要, model_name, quant);
    let mut 最后错 = String::from("未尝试");
    let mut 补提 = String::new();
    // 至少 2 次机会: 首次 + 引用缺失补提一轮(7B 常漏 关键证据引用, T-44 改5)
    for k in 1..=k_max.max(2) {
        // 调llama_cli 已做输出完整性轮询并返回解析校验通过的 Claim
        let 本轮prompt = if 补提.is_empty() {
            prompt.clone()
        } else {
            format!("{}\n{}", prompt, 补提)
        };
        match 调llama_cli(llama_cli, model, &本轮prompt, max_token, timeout_sec, 失败raw目录) {
            Ok(mut claim) => {
                // 外壳回填(T-40): 模型写歪/缺失的 anchor_id/time_window/model_ref 一律用入参覆盖
                回填外壳(
                    &mut claim,
                    &摘要.anchor_id,
                    时窗 { start_ts_ms: 摘要.start_ts_ms, end_ts_ms: 摘要.end_ts_ms },
                    model_name,
                    quant,
                );
                // T-44 改5+: 证据题(有全模谱)模型漏 关键证据引用 → 提示补一轮(仍由模型重出, agent 不替答)
                if 摘要.全模谱.is_empty()
                    || claim.关键证据引用.as_ref().is_some_and(|v| !v.is_empty())
                {
                    return Ok(claim);
                }
                补提 = "您上一条回复漏了必需字段「关键证据引用」。请从上述谱数据中选至少 1 条您判 direction 所依据的关键模，在 JSON 的 关键证据引用 中输出 [{\"模索引\":N,\"符号断言\":\"正\"或\"负\"或\"近零\"}]（对照该模 γ 数值判符号），并重出完整合法 Claim JSON。".to_string();
                最后错 = "模型漏 关键证据引用, 已提示补出".to_string();
            }
            Err(e) => {
                最后错 = format!("第 {} 轮输出不合规: {}", k, e);
            }
        }
    }
    Err(format!("重取 {} 轮仍无合规 Claim: {}", k_max.max(1), 最后错))
}

// ============================================================================
// T-51 (System-2 可插拔后端): 通用"后端原始文本 → Claim"后处理 + 产出入口——
//   文本产出闭包由调用方按后端类型给出(local-cli 子进程 / OpenAI 兼容 HTTP);
//   提取/校验/外壳回填/漏引用补提 与 llama-cli 路径完全同一判读后处理(不重复第二套)。
// ============================================================================

/// T-44 改5+: 漏「关键证据引用」补提提示(常量, llama-cli 与可插拔路径共用)
const 补提关键证据引用: &str = "您上一条回复漏了必需字段「关键证据引用」。请从上述谱数据中选至少 1 条您判 direction 所依据的关键模，在 JSON 的 关键证据引用 中输出 [{\"模索引\":N,\"符号断言\":\"正\"或\"负\"或\"近零\"}]（对照该模 γ 数值判符号），并重出完整合法 Claim JSON。";

/// 原始文本 → Claim(提取/校验/外壳回填); 与 llama-cli 路径同一后处理。
/// backend_名: 溯源用(local-cli→"llama.cpp"; server/云→实际后端), T-51 登记3
pub fn 文本到Claim(
    文本: &str,
    摘要: &锚包摘要,
    model_name: &str,
    quant: &str,
    backend_名: &str,
) -> Result<ClaimForCheck, String> {
    let mut claim = 从模型输出取Claim(文本)?;
    回填外壳_带后端(
        &mut claim,
        &摘要.anchor_id,
        时窗 { start_ts_ms: 摘要.start_ts_ms, end_ts_ms: 摘要.end_ts_ms },
        model_name,
        quant,
        backend_名,
    );
    Ok(claim)
}

/// 可插拔产出入口(有锚, 基础 prompt 已由调用方构造): 首轮 + 漏引用补提轮(≤2)
pub fn 产出Claim_可插拔(
    文本产出: &dyn Fn(&str, u64, Option<&str>) -> Result<String, String>,
    摘要: &锚包摘要,
    prompt: &str,
    model_name: &str,
    quant: &str,
    backend_名: &str,
    timeout_sec: u64,
    失败raw目录: Option<&str>,
) -> Result<ClaimForCheck, String> {
    let mut 补提 = String::new();
    let mut 最后错 = String::from("未尝试");
    for _ in 1..=2 {
        let 本轮prompt = if 补提.is_empty() {
            prompt.to_string()
        } else {
            format!("{}\n{}", prompt, 补提)
        };
        let 文本 = match 文本产出(&本轮prompt, timeout_sec, 失败raw目录) {
            Ok(t) => t,
            Err(e) => {
                最后错 = format!("后端产出失败: {}", e);
                continue;
            }
        };
        match 文本到Claim(&文本, 摘要, model_name, quant, backend_名) {
            Ok(mut claim) => {
                if 摘要.全模谱.is_empty()
                    || claim.关键证据引用.as_ref().is_some_and(|v| !v.is_empty())
                {
                    return Ok(claim);
                }
                补提 = 补提关键证据引用.to_string();
                最后错 = "模型漏 关键证据引用, 已提示补出".to_string();
            }
            Err(e) => {
                最后错 = format!("输出不合规: {}", e);
            }
        }
    }
    Err(format!("可插拔后端重取后仍无合规 Claim: {}", 最后错))
}

// ============================================================================
// T-38 (R4 Wave1) C5 多轮回验——回灌反馈（真正闭合: 每轮 prompt 带上一轮偏差, 非同 prompt 重取）
// 判据唯一在 engine（Deviant 判定/偏离方向由编排层经 engine verdict 得出）;
// agent 侧只负责"把偏差回灌进下一轮 prompt"的纯函数与会话单步生成。
// ============================================================================

/// 单条回灌反馈（编排层由 engine verdict 偏差生成; agent 纯模板不自行判定）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct 回灌反馈 {
    /// Deviant 发生轮次 k（1 起）
    pub 轮次: usize,
    /// 可校验量名（模型上一轮 Claim 的 observable）
    pub observable: String,
    /// 模型上一轮 direction
    pub 模型方向: String,
    /// 偏离说明（编排层填好; 主档只含方向与量纲——如"系统判读您的估计偏高"）
    pub 偏离说明: String,
    /// observable 单位
    pub 单位: String,
}

/// 在基础判读 prompt 上追加回灌反馈段（主档不给目标值; 给值档由编排把目标值编进偏离说明）
pub fn 构造prompt_带反馈(
    摘要: &锚包摘要,
    反馈: &[回灌反馈],
    model_name: &str,
    quant: &str,
    目标: Option<&判读目标>,
) -> String {
    let 基础 = 构造prompt_题向(摘要, model_name, quant, 目标);
    if 反馈.is_empty() {
        return 基础;
    }
    let mut 回灌段 = String::from(
        "\n[上一轮判读偏差回灌——供修正, 不含新数据, 请据此重新估计并输出新 Claim]\n",
    );
    for f in 反馈 {
        回灌段.push_str(&format!(
            "第{}轮: 您对 observable={} 判 direction={}（unit={}）, {}。\n",
            f.轮次, f.observable, f.模型方向, f.单位, f.偏离说明
        ));
    }
    format!("{}{}", 基础, 回灌段)
}

/// C5 单步多轮回验生成: 用当前已累积反馈构造**新的** prompt 触发一次真实新生成
/// （K1: 不是 temp0 下同 prompt 重取——反馈改变 prompt → 输出随之改变）。
/// 单次生成（不做同 prompt 重取循环）; anchor_id 不符/格式不合规如实报错。
pub fn 模型产出Claim_带反馈(
    llama_cli: &str,
    model: &str,
    锚包json: &str,
    反馈: &[回灌反馈],
    model_name: &str,
    quant: &str,
    max_token: usize,
    timeout_sec: u64,
    目标: Option<&判读目标>,
    失败raw目录: Option<&str>,
) -> Result<ClaimForCheck, String> {
    let 摘要 = 提取锚包摘要(锚包json)?;
    let prompt = 构造prompt_带反馈(&摘要, 反馈, model_name, quant, 目标);
    let mut claim = 调llama_cli(llama_cli, model, &prompt, max_token, timeout_sec, 失败raw目录)?;
    // 外壳回填(T-40): 模型写歪(如 model_ref 写成字符串)/缺失一律用入参覆盖, 核心判读保留
    回填外壳(
        &mut claim,
        &摘要.anchor_id,
        时窗 { start_ts_ms: 摘要.start_ts_ms, end_ts_ms: 摘要.end_ts_ms },
        model_name,
        quant,
    );
    // T-44 改5+: 证据题(有全模谱)模型漏 关键证据引用 → 提示补一轮(仍由模型重出, agent 不替答)
    if !摘要.全模谱.is_empty() && claim.关键证据引用.as_ref().is_none_or(|v| v.is_empty()) {
        let 提示 = "您上一条回复漏了必需字段「关键证据引用」。请从上述谱数据中选至少 1 条您判 direction 所依据的关键模，在 JSON 的 关键证据引用 中输出 [{\"模索引\":N,\"符号断言\":\"正\"或\"负\"或\"近零\"}]（对照该模 γ 数值判符号），并重出完整合法 Claim JSON。";
        let mut claim2 = 调llama_cli(llama_cli, model, &format!("{}\n{}", prompt, 提示), max_token, timeout_sec, 失败raw目录)?;
        回填外壳(
            &mut claim2,
            &摘要.anchor_id,
            时窗 { start_ts_ms: 摘要.start_ts_ms, end_ts_ms: 摘要.end_ts_ms },
            model_name,
            quant,
        );
        claim = claim2;
    }
    Ok(claim)
}

/// 无锚 Claim 的 anchor_id 占位（D 臂无锚裸模型; 编排据此识别不送 engine verdict）
pub const 无锚anchor_id: &str = "no-anchor";

/// 构造「无锚题面」prompt（T-40 前置: 真模型 D 臂零点——不传 anchor、只按题面产出 Claim）
/// 题面由编排给出（判读任务句, 无谱数据/无可验量实值）; 仍要求输出合法 ClaimForCheck schema,
/// anchor_id 固定 no-anchor; model_ref 参数化不写死; F 清单列全量。
pub fn 构造prompt_无锚(题面: &str, model_name: &str, quant: &str) -> String {
    format!(
        "{题面}\n\
         请输出一条 ClaimForCheck JSON（仅 JSON 无其它文字）。字段:\n\
         claim_id 短字符串; anchor_id=\"{无锚}\";\n\
         observable 只能填单个（勿填多个）——整体量之一: unstable_mode_count / tau_dominant / gamma_max / c_available; 或通道量: cpu_pct.b_dominant_freq / cpu_pct.b_energy / cpu_pct.b_modes / mem_pct.b_dominant_freq / mem_pct.b_energy / mem_pct.b_modes;\n\
         value 数值估计; unit 与所选 observable 匹配(count 或 s 或 rad_s^-1 或 state);\n\
         direction 只能填单个: present 或 absent 或 higher 或 lower 或 equal;\n\
         关键证据引用 必填: [{{\"模索引\":N,\"符号断言\":\"正\"或\"负\"或\"近零\"}}]——指出你判 direction 所依据的关键模(自行判符号), 至少 1 条;\n\
         置信度 必填: 0 到 1 之间的小数;\n\
         time_window={{\"start_ts_ms\":0,\"end_ts_ms\":0}}; model_ref={{\"name\":\"{model_name}\",\"quant\":\"{quant}\",\"backend\":\"llama.cpp\",\"deterministic_seed\":\"temp0-s42\"}}; hidden_state_ref null; raw_text 中文一句。\n\
         注意: 只输出一个严格 schema 的 JSON, model_ref 照抄给定对象, 不做文字解释。",
        无锚 = 无锚anchor_id,
        model_name = model_name,
        quant = quant,
    )
}

/// 无锚单步生成（D 臂）: 题面 → llama → Claim。anchor_id 须为 no-anchor（不符如实报错）。
pub fn 模型产出Claim_无锚(
    llama_cli: &str,
    model: &str,
    题面: &str,
    model_name: &str,
    quant: &str,
    max_token: usize,
    timeout_sec: u64,
    失败raw目录: Option<&str>,
) -> Result<ClaimForCheck, String> {
    let prompt = 构造prompt_无锚(题面, model_name, quant);
    let mut claim = 调llama_cli(llama_cli, model, &prompt, max_token, timeout_sec, 失败raw目录)?;
    // 外壳回填(D 无锚: anchor_id=no-anchor, 时窗 0, model_ref 参数化)
    回填外壳(&mut claim, 无锚anchor_id, 时窗 { start_ts_ms: 0, end_ts_ms: 0 }, model_name, quant);
    Ok(claim)
}

/// T-42 修订: llama 收尾统计行结束检测——行内同时出现 "[ Prompt:" 与 "| Generation:"
/// 即判定生成已结束(与 claim 模式"解析到即 kill"同构, free 模式的结束锚点换统计行)
pub fn 生成已结束(输出: &str) -> bool {
    输出.lines().any(|l| l.contains("[ Prompt:") && l.contains("| Generation:"))
}

/// T-42 修订: 从 llama 全文稳健提取模型回复（先砍尾再去头, 见 PM 修法 B）
/// ①砍尾: 统计行 "[ Prompt:" 起截断丢弃其后; 兜底逐行剔行首 REPL 提示符(>)残留;
/// ②去头: 优先 find 题面取其后, 失败 find 引导语取其后, 再失败用砍尾后全文;
/// ③trim。
pub fn 提取模型文本(全文: &str, 题面: &str) -> String {
    const 引导语: &str = "直接回答, 不需要 JSON。";
    // 砍尾按行: 遇统计行(行内含 [ Prompt: 与 | Generation:)即整行截断丢弃其后(整行行首截断, 不留行内前缀)
    let 砍尾 = 全文
        .lines()
        .take_while(|l| !(l.contains("[ Prompt:") && l.contains("| Generation:")))
        .collect::<Vec<_>>()
        .join("\n");
    let 剔提示: Vec<&str> = 砍尾.lines().filter(|l| !l.trim_start().starts_with('>')).collect();
    let 砍尾 = 剔提示.join("\n");
    let mut 正文 = 砍尾;
    // 去头 ①: 题面(完整回显时其后常带引导语回显)
    if !题面.is_empty() {
        if let Some(起) = 正文.find(题面) {
            正文 = 正文[起 + 题面.len()..].to_string();
        }
    }
    // 去头 ②: 引导语(题面未回显/回显后紧跟)一律剔——正文起点在引导语之后
    if let Some(起) = 正文.find(引导语) {
        正文 = 正文[起 + 引导语.len()..].to_string();
    }
    正文.trim().to_string()
}

/// T-42 (R4-v2): 自由文本无锚模式——题面 → llama → 原始文本（无 Claim schema、无解析校验）
/// 供 D 创造/价值臂真接模型（创作题无物理 F 可投影, 不适用 ClaimForCheck）。
/// 复用 -f 文件通道 + 容错解码 + 失败 raw 保留; 单轮退出用 --single-turn + 统计行结束检测兜底;
/// 输出经 提取模型文本 稳健去头砍尾。
pub fn 模型产出文本_无锚(
    llama_cli: &str,
    model: &str,
    题面: &str,
    max_token: usize,
    timeout_sec: u64,
    失败raw目录: Option<&str>,
) -> Result<String, String> {
    use std::io::Read;
    let pid = std::process::id();
    let 输出文件路径 = std::env::temp_dir().join(format!("r3_free_out_{}.log", pid));
    let stdin文件路径 = std::env::temp_dir().join(format!("r3_free_stdin_{}.txt", pid));
    let stderr文件路径 = std::env::temp_dir().join(format!("r3_free_err_{}.log", pid));
    let prompt文件路径 = std::env::temp_dir().join(format!("r3_free_prompt_{}.txt", pid));
    // 自由文本 prompt: 题面 + 直接回答引导(无 JSON schema)
    let prompt = format!("{}\n直接回答, 不需要 JSON。", 题面);
    std::fs::write(&prompt文件路径, &prompt)
        .map_err(|e| format!("写 prompt 临时文件失败: {}", e))?;
    let prompt文件串 = prompt文件路径.to_str().ok_or("prompt 路径非 UTF-8")?.to_string();
    let 输出文件 = std::fs::File::create(&输出文件路径)
        .map_err(|e| format!("创建临时输出文件失败: {}", e))?;
    std::fs::write(&stdin文件路径, b"").map_err(|e| format!("创建空 stdin 失败: {}", e))?;
    let 空输入 = std::fs::File::open(&stdin文件路径)
        .map_err(|e| format!("打开空 stdin 失败: {}", e))?;
    let mut 命令 = Command::new(llama_cli);
    命令.args([
        "-m", model, "-f", &prompt文件串,
        "-n", &max_token.to_string(),
        "--temp", "0", "-s", "42",
        "--no-escape", "--simple-io",
        "--single-turn", // T-42 修订: 本机 llama-cli 实际支持(--help 确认), 单轮生成完自然退出
        // PM 09-08 显存护栏(同锚点路径): 限上下文 + KV q8, 防 12B/14B 全 offload 溢出 shared 暴跌
        "-c", "2560",
        "-ctk", "q8_0", "-ctv", "q8_0",
    ]);
    #[cfg(windows)]
    命令.creation_flags(创建标志);
    let mut child = 命令
        .stdin(空输入)
        .stdout(输出文件)
        .stderr(std::fs::File::create(&stderr文件路径).map_err(|e| format!("创建 stderr 文件失败: {}", e))?)
        .spawn()
        .map_err(|e| format!("启动 llama-cli 失败（{}）: {}", llama_cli, e))?;
    let 超时 = std::time::Duration::from_secs(timeout_sec.max(10));
    let 开始 = std::time::Instant::now();
    let mut 已退: Option<std::process::ExitStatus> = None;
    loop {
        // T-42 修订: 生成结束判据(与 claim 解析到即 kill 同构)——读到统计行即 kill 收尾
        if let Some(输出) = 读输出容错(&输出文件路径) {
            if 生成已结束(&输出) {
                // 统计行已出=生成结束 → kill+wait 收尾(不设已退: 后面 rc 非零检查跳过, 输出是主产物)
                let _ = child.kill();
                let _ = child.wait();
                break;
            }
        }
        match child.try_wait() {
            Ok(Some(st)) => {
                已退 = Some(st);
                break;
            }
            Ok(None) => {}
            Err(e) => {
                let _ = child.kill();
                let _ = 保留失败raw(&输出文件路径, 失败raw目录.unwrap_or(""));
                let _ = std::fs::remove_file(&prompt文件路径);
                return Err(format!("轮询 llama 失败: {}", e));
            }
        }
        if 开始.elapsed() > 超时 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(400));
    }
    if 已退.is_none() {
        let _ = child.kill();
        let _ = child.wait();
    }
    let 全文 = 读输出容错(&输出文件路径).unwrap_or_default();
    let _ = 保留失败raw(&输出文件路径, 失败raw目录.unwrap_or(""));
    let _ = std::fs::remove_file(&输出文件路径);
    let _ = std::fs::remove_file(&stdin文件路径);
    let _ = std::fs::remove_file(&stderr文件路径);
    let _ = std::fs::remove_file(&prompt文件路径);
    if let Some(st) = 已退 {
        if !st.success() {
            return Err(format!("llama-cli 退出码非零: {:?}", st));
        }
    }
    // T-42 修订: 稳健提取(先砍尾统计行+剔提示符, 再去头题面→引导语)
    let 回复 = 提取模型文本(&全文, 题面);
    if 回复.is_empty() {
        return Err(format!(
            "自由文本输出为空（已等待 {}s; llama 未产出模型回复）",
            timeout_sec
        ));
    }
    Ok(回复)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn 示例锚包json() -> String {
        // T-44: 块1 证据锚(EV)形态——顶层平铺 模谱/τ数值/无 c_result/无 失稳模 结论量;
        // 用 U001 真实全模谱 3 条。**T-194 新口径（γ=−λ_im）**：模1 γ=−0.0287 增长(失稳), 模0/2 γ>0 稳定。
        format!(
            r#"{{"锚类型":"证据锚","anchor_id":"aa11bb22","domain":"system","dt_sec":1.0,
                "window":{{"start_ts_ms":1700000000000,"end_ts_ms":1700000000040,"n_frames":40,"label":"注入"}},
                "模谱":[{{"索引":0,"ω":0.080,"γ":0.0270}},{{"索引":1,"ω":0.464,"γ":-0.0287}},{{"索引":2,"ω":0.951,"γ":1.37}}],
                "τ数值":[3.0,20.0],"频率":[0.3,-0.3],
                "对照τ数值":null}}"#
        )
    }

    fn 合法Claimjson() -> String {
        format!(
            r#"{{"claim_id":"c-1","anchor_id":"aa11bb22","observable":"unstable_mode_count",
                "value":1.0,"unit":"count","direction":"present",
                "time_window":{{"start_ts_ms":1700000000000,"end_ts_ms":1700000000040}},
                "model_ref":{{"name":"Qwen2.5-7B-Instruct","quant":"Q4_K_M","backend":"llama.cpp","deterministic_seed":"temp0-s42"}},
                "hidden_state_ref":"norm-0","raw_text":"检测到 1 个失稳模"}}"#
        )
    }

    #[test]
    fn 解析校验_合法Claim_通过() {
        let c = 解析校验Claim(&合法Claimjson()).unwrap();
        assert_eq!(c.observable, "unstable_mode_count");
        assert!(observable落F("unstable_mode_count"));
        assert!(observable落F("cpu_pct.b_dominant_freq"));
        assert!(!observable落F("bogus"));
    }

    #[test]
    fn 解析校验_缺字段_报错() {
        let bad = r#"{"claim_id":"c1"}"#;
        assert!(解析校验Claim(bad).is_err());
    }

    #[test]
    fn 解析校验_observable不落F_报错() {
        let mut j: serde_json::Value = serde_json::from_str(&合法Claimjson()).unwrap();
        j["observable"] = serde_json::Value::String("bogus".to_string());
        let e = 解析校验Claim(&j.to_string()).unwrap_err();
        assert!(e.contains("不在可验量集合 F"), "{}", e);
    }

    #[test]
    fn 解析校验_非法direction_报错() {
        let mut j: serde_json::Value = serde_json::from_str(&合法Claimjson()).unwrap();
        j["direction"] = serde_json::Value::String("sideways".to_string());
        assert!(解析校验Claim(&j.to_string()).is_err());
    }

    #[test]
    fn T44_证据锚形态_摘要提取与prompt() {
        let 摘要 = 提取锚包摘要(&示例锚包json()).unwrap();
        assert_eq!(摘要.anchor_id, "aa11bb22");
        assert_eq!(摘要.全模谱.len(), 3);
        assert_eq!(摘要.全模谱[1], (1, 0.464, -0.0287), "全模谱应含失稳增长模 γ<0(不标结论)");
        assert_eq!(摘要.全模谱[0], (0, 0.080, 0.0270));
        assert_eq!(摘要.τ数值, vec![3.0, 20.0]);
        assert!(摘要.对照τ数值.is_none(), "U/S 证据锚无对照");
        assert!(摘要.复谱可用);
        let p = 构造prompt(&摘要, "Qwen2.5-7B-Instruct", "Q4_K_M");
        assert!(p.contains("aa11bb22"));
        // 每模 ω/γ 原样列全(γ 带符号), 供模型自判
        assert!(p.contains("模1: ω=0.4640, γ=-0.0287"), "prompt 应列每模数值: {}", p);
        assert!(p.contains("模0: ω=0.0800, γ=+0.0270"));
        assert!(p.contains("模2: ω=0.9510, γ=+1.3700"));
        // 结论量/结论词一个不给(改5: 数据侧结论词子集零命中; direction 枚举属 schema 非数据结论)
        assert!(!p.contains("失稳模数") && !p.contains("失稳模"), "不许出现结论量: {}", p);
        assert!(!p.contains("γ=无") && !p.contains("无失稳"), "不许出现结论词: {}", p);
        for 词 in ["增长模", "稳态", "健康", "风险", "tier", "慢化方向", "eigen_pairs", "失稳模数=", "γ=0"] {
            assert!(!p.contains(词), "prompt 不应含数据侧结论词 {}: {}", 词, p);
        }
        assert!(!p.contains("慢化"), "U/S 无慢化结论词");
    }

    #[test]
    fn T45_符号规则workedExample_U注入C不注入() {
        // U/S 题(无对照τ): prompt 含抽象 worked example(治偏差α/β)且不含本题模谱数值(索引 0..2 不出现于示例)
        let 摘要 = 提取锚包摘要(&示例锚包json()).unwrap();
        let p = 构造prompt(&摘要, "M", "Q4");
        assert!(p.contains("γ=-0.03"), "应含抽象失稳正例(γ<0): {}", p);
        assert!(p.contains("γ=+0.4"), "应含抽象反例(正γ是衰减不判 present): {}", p);
        assert!(p.contains("当且仅当存在至少一个 γ<0 模"), "present 判据: {}", p);
        assert!(p.contains("禁止编造不存在的索引"), "防越界声明: {}", p);
        assert!(p.contains("必须引用该负γ模"), "引用须支撑 present: {}", p);
        assert!(p.contains("数值仅示意与本题数据无关"), "示例须显式声明与本题无关: {}", p);
        // worked example 不得含本题真实模数值, 只含抽象示意 -0.03/+0.4
        assert!(!p.contains("γ=0.0802") && !p.contains("γ=0.4642") && !p.contains("γ=0.9514"), "示例不得泄漏本题谱数值");
        // C 题(有 对照τ数值): 不注入 worked example(判 τ 高低不需要 γ 符号步骤)
        let mut v: serde_json::Value = serde_json::from_str(&示例锚包json()).unwrap();
        v["对照τ数值"] = serde_json::json!([32.0, 21.3]);
        let 摘要c = 提取锚包摘要(&v.to_string()).unwrap();
        let pc = 构造prompt(&摘要c, "M", "Q4");
        assert!(!pc.contains("γ=-0.03"), "C 题不应注入 worked example: {}", pc);
        assert!(pc.contains("禁止编造不存在的索引"), "T-46 改2: C 题同样注入禁越界声明(治 C 越界): {}", pc);
        assert!(pc.contains("注入段主导τ≈32.0"), "C 对照数值仍在");
    }

    #[test]
    fn T46_易档性质标签_提取与prompt双轨() {
        // 易档 EV: 模谱每条带 System-1 性质标签
        let mut v: serde_json::Value = serde_json::from_str(&示例锚包json()).unwrap();
        let 谱 = v["模谱"].as_array_mut().expect("模谱");
        谱[0]["性质"] = serde_json::json!("衰减"); // γ=+0.0270（稳定）
        谱[1]["性质"] = serde_json::json!("增长"); // γ=-0.0287（失稳，新口径）
        谱[2]["性质"] = serde_json::json!("衰减"); // γ=+1.37（稳定）
        let 摘要 = 提取锚包摘要(&v.to_string()).unwrap();
        assert_eq!(摘要.模性质, vec![Some("衰减".into()), Some("增长".into()), Some("衰减".into())], "易档性质提取");
        let p = 构造prompt(&摘要, "M", "Q4");
        // 易档谱段带标签 + 易档规则(扫标签, 勿改判符号) + 禁编造
        assert!(p.contains("性质:增长"), "谱段应带标签: {}", p);
        assert!(p.contains("扫描「性质」标签"), "易档规则=标签扫描: {}", p);
        assert!(p.contains("存在「性质:增长」的模(即 γ<0 的失稳模) → unstable_mode_count 判 present"), "present 判据: {}", p);
        assert!(p.contains("勿自行改判符号"), "System-1 预判声明: {}", p);
        assert!(p.contains("禁止编造不存在的索引"));
        // 易档不含中档 worked example(数值示意)
        assert!(!p.contains("γ=-0.03"), "易档不再要求模型学符号(已给标签): {}", p);
        // 中档 EV(无性质) → 维持现状 worked example(向后兼容)
        let 摘要中 = 提取锚包摘要(&示例锚包json()).unwrap();
        let pm = 构造prompt(&摘要中, "M", "Q4");
        assert!(!pm.contains("性质:") && pm.contains("γ=-0.03"), "中档无标签仍走符号学习: {}", pm);
    }

    #[test]
    fn T44_裁决锚形态_显式拒绝() {
        // 旧裁决锚形态(顶层无 模谱) → Err 显式(不静默退化失真)
        let old = r#"{"anchor_id":"x","window":{"start_ts_ms":1,"end_ts_ms":2},
            "c_result":{"失稳模":[[0,0.5,0.05]],"τ链":[3.0]}}"#;
        let e = 提取锚包摘要(old).unwrap_err();
        assert!(e.contains("证据锚形态"), "应显式指认非证据锚: {}", e);
    }

    #[test]
    fn 模型产出Claim_重取至kmax_无模型时如实报错() {
        // 用不存在的 llama-cli 路径 → 启动失败 → 重取 ≤k_max 后如实报错（不伪造）
        let e = 模型产出Claim("C:/不存在/llama-cli.exe", "m", &示例锚包json(), "Qwen2.5-7B-Instruct", "Q4_K_M", 64, 2, 10, None)
            .unwrap_err();
        assert!(e.contains("重取") || e.contains("llama-cli"), "{}", e);
    }

    #[test]
    fn K2_model_ref参数化_不写死() {
        // prompt 内 model_ref.name/quant 随调用方传入（不再写死 Qwen）
        let p = 构造prompt(&提取锚包摘要(&示例锚包json()).unwrap(), "MyModel-1B", "Q3_K_S");
        assert!(p.contains("MyModel-1B"), "model name 应参数化: {}", p);
        assert!(p.contains("Q3_K_S"), "quant 应参数化: {}", p);
    }

    #[test]
    fn K2_F列全_含通道量() {
        let p = 构造prompt(&提取锚包摘要(&示例锚包json()).unwrap(), "M", "Q4");
        assert!(p.contains("cpu_pct.b_dominant_freq"), "F 应列通道量: {}", p);
        assert!(p.contains("mem_pct.b_modes"));
    }

    #[test]
    fn 回灌反馈_空反馈_基础prompt() {
        let 摘要 = 提取锚包摘要(&示例锚包json()).unwrap();
        let 空 = 构造prompt_带反馈(&摘要, &[], "Qwen2.5-7B-Instruct", "Q4_K_M", None);
        let 基 = 构造prompt(&摘要, "Qwen2.5-7B-Instruct", "Q4_K_M");
        assert_eq!(空, 基, "无反馈时与基础 prompt 一致");
    }

    #[test]
    fn 回灌反馈_反馈改变prompt_含轮次与偏离() {
        let 摘要 = 提取锚包摘要(&示例锚包json()).unwrap();
        let 反馈 = vec![回灌反馈 {
            轮次: 1,
            observable: "unstable_mode_count".to_string(),
            模型方向: "present".to_string(),
            偏离说明: "系统判读您的估计偏高".to_string(),
            单位: "count".to_string(),
        }];
        let p = 构造prompt_带反馈(&摘要, &反馈, "Qwen2.5-7B-Instruct", "Q4_K_M", None);
        assert!(p.contains("第1轮"), "回灌应含轮次: {}", p);
        assert!(p.contains("unstable_mode_count"), "回灌应含 observable");
        assert!(p.contains("系统判读您的估计偏高"), "回灌应含偏离说明(不含目标值)");
        assert!(!p.contains("target") && !p.contains("真实值"), "主档不含目标值: {}", p);
        // 反馈使 prompt 与基础不同 → temp0 下也会产出不同输出(真实重生成, 修 K1)
        let 基础 = 构造prompt(&摘要, "Qwen2.5-7B-Instruct", "Q4_K_M");
        assert_ne!(p, 基础);
    }

    #[test]
    fn 题向目标_prompt指定量不泄方向() {
        let 摘要 = 提取锚包摘要(&示例锚包json()).unwrap();
        let 目标 = 判读目标 { observable: "unstable_mode_count".to_string(), 方向枚举: "present,absent".to_string() };
        let p = 构造prompt_题向(&摘要, "M", "Q4", Some(&目标));
        assert!(p.contains("本题判读目标: observable=unstable_mode_count"), "应点名判读量: {}", p);
        assert!(p.contains("present,absent"), "应给合法方向枚举");
        // 不泄目标方向: 枚举是"选项"不是"答案"; 不含"应为 present"/"正确方向"
        assert!(!p.contains("应为") && !p.contains("正确方向") && !p.contains("答案是"), "不泄方向: {}", p);
        // None(旧兼容) 不含目标段
        let p0 = 构造prompt(&摘要, "M", "Q4");
        assert!(!p0.contains("本题判读目标"), "None 目标不出现目标段");
    }

    #[test]
    fn 题向目标_C对照与目标并存() {
        let mut v: serde_json::Value = serde_json::from_str(&示例锚包json()).unwrap();
        v["对照τ数值"] = serde_json::json!([32.0, 21.3]);
        let 摘要 = 提取锚包摘要(&v.to_string()).unwrap();
        let 目标 = 判读目标 { observable: "tau_dominant".to_string(), 方向枚举: "higher,lower,equal".to_string() };
        let p = 构造prompt_题向(&摘要, "M", "Q4", Some(&目标));
        assert!(p.contains("本题判读目标: observable=tau_dominant"), "C 题点名 tau_dominant");
        assert!(p.contains("注入段主导τ≈32.0"), "对照两数值保留");
        assert!(p.contains("基线段≈21.3"));
        assert!(p.contains("higher,lower,equal"));
    }

    #[test]
    fn C慢化对照_摘要提取与prompt文案() {
        // C 题证据锚: 顶层「对照τ数值」=〔注入,基线〕两数值, 无 慢化可比/慢化 结论(去方向)
        let mut v: serde_json::Value = serde_json::from_str(&示例锚包json()).unwrap();
        v["对照τ数值"] = serde_json::json!([32.0, 21.3]);
        let 摘要 = 提取锚包摘要(&v.to_string()).unwrap();
        let (inj, bl) = 摘要.对照τ数值.expect("应提取对照τ数值");
        assert_eq!(inj, 32.0);
        assert_eq!(bl, 21.3);
        let p = 构造prompt(&摘要, "M", "Q4");
        assert!(p.contains("注入段主导τ≈32.0"), "prompt 应含两段对照数值: {}", p);
        assert!(p.contains("基线段≈21.3"));
        assert!(p.contains("higher"), "方向枚举语义提示应含 higher");
        // 结论量不给: 无 慢化可比/慢化=true 等结论词
        assert!(!p.contains("慢化可比") && !p.contains("慢化=true"), "不给慢化结论: {}", p);
        // 无对照证据锚(U/S) → 无对照文案
        let 摘要2 = 提取锚包摘要(&示例锚包json()).unwrap();
        assert!(摘要2.对照τ数值.is_none());
        let p2 = 构造prompt(&摘要2, "M", "Q4");
        assert!(!p2.contains("注入段主导τ"), "无对照锚包不应含对照文案");
    }

    #[test]
    fn T40_保留失败raw_落出非空文件() {
        // 自动验证失败路径确实保留 raw(不依赖真模型): 造输出文件 → 保留函数 → 目录有非空 copy
        let dir = std::env::temp_dir().join(format!("r3_failraw_test_{}", std::process::id()));
        let src = dir.join("out.log");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&src, "llama 原始输出 abc").unwrap();
        let ok = 保留失败raw(&src, dir.to_str().unwrap());
        assert!(ok, "保留应成功");
        let 目标s: Vec<_> = std::fs::read_dir(&dir).unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains("r3_claim_fail_"))
            .collect();
        assert_eq!(目标s.len(), 1, "应落出 1 个 raw 文件");
        let 内容 = std::fs::read_to_string(目标s[0].path()).unwrap();
        assert_eq!(内容, "llama 原始输出 abc", "raw 内容应完整");
        // 空目录/源不存在 → false(不静默吞)
        assert!(!保留失败raw(&dir.join("不存在.log"), ""));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn T40_容错解码_半个中文回显不阻Claim() {
        // 复现 PM raw_fail 铁证: llama 回显按固定字节截断劈中「慢」=e6 85 a8, 剩 e6 85 接空格 → 非法 UTF-8;
        // 其后紧跟完整合法 Claim。lossy 解码后应仍能取出 Claim(核心字段 ASCII 不受影响)。
        let mut 流 = Vec::new();
        流.extend_from_slice(b"[echo ");
        流.extend_from_slice(&[0xe6, 0x85]); // 半个「慢」
        流.extend_from_slice(" 截断回显]\n".as_bytes());
        流.extend_from_slice(合法Claimjson().as_bytes());
        let 解码 = String::from_utf8_lossy(&流).into_owned();
        assert!(解码.contains('\u{FFFD}'), "非法字节应被替换为 U+FFFD");
        let c = 从模型输出取Claim(&解码).expect("lossy 后仍应取出完整 Claim");
        assert_eq!(c.observable, "unstable_mode_count");
        assert_eq!(c.direction, "present");
    }

    #[test]
    fn T40_容错解码_全ASCII正常用例回归() {
        let 流 = format!("banner...\n{}", 合法Claimjson());
        let c = 从模型输出取Claim(&流).expect("全 ASCII 应正常取 Claim");
        assert_eq!(c.claim_id, "c-1");
    }

    #[test]
    fn T40_外壳写歪_核心仍解析并被回填修正() {
        // 复现 PM 冒烟: 模型把 model_ref 写成字符串(外壳歪) 但核心判读正确 —— 不得整体作废
        let raw = r#"{"claim_id":"c-1","anchor_id":"wrong-id","observable":"tau_dominant",
            "value":21.3,"unit":"s","direction":"lower",
            "time_window":{"start_ts_ms":1,"end_ts_ms":2},
            "model_ref":"失稳判读模型_v1.0"}"#;
        let mut c = 解析校验Claim(raw).expect("外壳写歪但核心正确应可解析");
        assert_eq!(c.observable, "tau_dominant");
        assert_eq!(c.direction, "lower");
        assert_eq!(c.value, 21.3);
        // 回填: anchor_id/time_window/model_ref 一律用 agent 入参覆盖
        回填外壳(&mut c, "aa11bb22", 时窗 { start_ts_ms: 1700000000000, end_ts_ms: 1700000000040 }, "MyModel", "Q4");
        assert_eq!(c.anchor_id, "aa11bb22");
        assert_eq!(c.time_window.start_ts_ms, 1700000000000);
        assert_eq!(c.model_ref.name, "MyModel");
        assert_eq!(c.model_ref.quant, "Q4");
        assert_eq!(c.model_ref.deterministic_seed, "temp0-s42");
    }

    #[test]
    fn T40_外壳缺失_亦可解析() {
        let raw = r#"{"observable":"unstable_mode_count","value":0.0,"unit":"count","direction":"absent"}"#;
        let c = 解析校验Claim(raw).expect("外壳全缺应可解析(核心为准)");
        assert_eq!(c.direction, "absent");
        assert_eq!(c.anchor_id, 无锚anchor_id, "缺失 anchor_id → 占位(no-anchor), 由回填修正");
    }

    #[test]
    fn T40_核心非法仍拒() {
        // 核心错(observable 不落 F / direction 非法 / value 非数) 一律拒, 不伪造
        assert!(解析校验Claim(r#"{"observable":"bogus","value":1,"unit":"x","direction":"present"}"#).is_err());
        assert!(解析校验Claim(r#"{"observable":"tau_dominant","value":1,"unit":"s","direction":"sideways"}"#).is_err());
    }

    #[test]
    fn T42_生成已结束_统计行检测() {
        let 已结 = "诗行\nllama_print_timings:  ... [ Prompt: 20 tokens | Generation: 30 tokens] done\n";
        assert!(生成已结束(已结), "含 [ Prompt: 与 | Generation: 统计行应判已结束");
        let 未结 = "诗写了一半\n>";
        assert!(!生成已结束(未结), "无统计行不应判结束");
    }

    #[test]
    fn T42_提取_砍尾剔提示符_无题面回显() {
        // 复现 PM: --simple-io 只回显引导语(题面 0 次) + 诗 + 统计行 + 大量提示符
        let mut 合成 = String::from("直接回答, 不需要 JSON。\n暮色把船影压进水面\n涛声数着泊位的空\n\n");
        合成.push_str("llama_print_timings:  ... [ Prompt: 20 tokens | Generation: 40 tokens] done\n");
        for _ in 0..20000 {
            合成.push_str("> ");
        }
        let 正文 = 提取模型文本(&合成, "暮色中的老码头");
        assert!(正文.contains("暮色把船影压进水面"), "应保留诗行: {}", 正文);
        assert!(正文.len() < 500, "不应有统计行/提示符: len={}", 正文.len());
        assert!(!正文.contains("Prompt:") && !正文.contains("> "), "无统计行无提示符");
    }

    #[test]
    fn T42_提取_题面完整回显_正确去头() {
        let 合成 = format!(
            "写一首诗。\n直接回答, 不需要 JSON。\n日落山间\n晚风归林\nllama_print_timings:  ... [ Prompt: 9 tokens | Generation: 6 tokens] done\n"
        );
        let 正文 = 提取模型文本(&合成, "写一首诗。");
        assert_eq!(正文, "日落山间\n晚风归林", "题面回显时应正确去头砍尾: {:?}", 正文);
    }

    #[test]
    fn T42_自由文本_无llama如实报错() {
        // 自由文本无锚模式(创造臂): 不存在的 llama → 如实报错(不伪造文本)
        let e = 模型产出文本_无锚("C:/不存在/llama-cli.exe", "m", "写一首短诗", 64, 10, None)
            .unwrap_err();
        assert!(e.contains("llama-cli") || e.contains("失败"), "{}", e);
    }

    #[test]
    fn T41块2_结构化裁决字段_解析与宽松() {
        // 含 关键证据引用/置信度 的 Claim 应解析出
        let raw = format!(
            r#"{},"关键证据引用":[{{"模索引":2,"符号断言":"正"}}],"置信度":0.62}}"#,
            &合法Claimjson()[..合法Claimjson().len() - 1]
        );
        let c = 解析校验Claim(&raw).expect("含结构化字段应解析");
        let 引用 = c.关键证据引用.as_ref().expect("应提取证据引用");
        assert_eq!(引用.len(), 1);
        assert_eq!(引用[0].模索引, 2);
        assert_eq!(引用[0].符号断言, "正");
        assert_eq!(c.置信度, Some(0.62));
        // 缺失 → None(旧 Claim 兼容)
        let c0 = 解析校验Claim(&合法Claimjson()).unwrap();
        assert!(c0.关键证据引用.is_none() && c0.置信度.is_none());
    }

    #[test]
    fn 无锚prompt_含题面与schema且不写死() {
        let p = 构造prompt_无锚("判断这段系统是否处于失稳状态。", "MyModel", "Q3");
        assert!(p.contains("判断这段系统是否处于失稳状态"), "应含题面: {}", p);
        assert!(p.contains("no-anchor"), "anchor_id 固定 no-anchor");
        assert!(p.contains("MyModel") && p.contains("Q3"), "model_ref 参数化");
        assert!(p.contains("unstable_mode_count") && p.contains("cpu_pct.b_dominant_freq"), "F 清单");
    }

    #[test]
    fn 无锚_无llama如实报错() {
        let e = 模型产出Claim_无锚("C:/不存在/llama-cli.exe", "m", "判断题面", "M", "Q4", 64, 10, None)
            .unwrap_err();
        assert!(e.contains("llama-cli") || e.contains("失败"), "{}", e);
    }

    #[test]
    fn 模型产出Claim_带反馈_无llama如实报错() {
        // 不存在的 llama-cli → 如实报错; 证明单步带反馈路径可达(不伪造)
        let 反馈 = vec![回灌反馈 {
            轮次: 1,
            observable: "tau_dominant".to_string(),
            模型方向: "lower".to_string(),
            偏离说明: "系统判读您的估计偏低".to_string(),
            单位: "s".to_string(),
        }];
        let e = 模型产出Claim_带反馈(
            "C:/不存在/llama-cli.exe", "m", &示例锚包json(), &反馈, "M", "Q4", 64, 10, None, None,
        )
        .unwrap_err();
        assert!(e.contains("llama-cli") || e.contains("失败"), "{}", e);
    }

    #[test]
    fn 模型输出_容忍banner提取JSON() {
        let raw = format!(
            "b81c99b build\nmodel loaded...\n> prompt\n{} trailing",
            合法Claimjson()
        );
        let c = 从模型输出取Claim(&raw).unwrap();
        assert_eq!(c.observable, "unstable_mode_count");
    }

    #[test]
    fn 模型输出_无JSON_报错() {
        assert!(从模型输出取Claim("模型只说了些话没有 JSON").is_err());
    }

    #[test]
    fn 模型输出_缺字段_报错() {
        let raw = r#"{"claim_id":"c1","anchor_id":"x"}"#;
        assert!(从模型输出取Claim(raw).is_err());
    }

    #[test]
    fn L2_候选json反向挑出合法_不受prompt回显干扰() {
        // prompt 回显也含 {（time_window 示例）; 模型裸输出 JSON（无 ```json 围栏）在中间
        let raw = format!(
            "time_window={{\"start_ts_ms\":1700000000000,\"end_ts_ms\":1700000000040}} 回显在前\n{} 模型裸输出在最后",
            合法Claimjson()
        );
        let c = 从模型输出取Claim(&raw).unwrap();
        assert_eq!(c.observable, "unstable_mode_count");
        assert_eq!(c.claim_id, "c-1");
    }

    #[test]
    fn L2_候选json_枚举平衡块() {
        let raw = r#"a {x:1} b {"y":2} c"#;
        let 候选 = 候选json(raw);
        assert!(候选.len() >= 2, "应枚举出多个花括号块: {:?}", 候选);
        assert!(候选[0].contains("x"));
        assert!(候选[1].contains("y"));
    }
}