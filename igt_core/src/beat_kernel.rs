// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! beat 域 — 拍频纯计算（本体层）+ 个体节律比对 trait 骨架
//!
//! ## 来源与处置（MIGRATION-FINAL 直迁总攻）
//! - **Step0-1（本批）**: engine `beat_library/beat_computer.rs`（299 行）+ 其依赖的纯数据
//!   类型（types.rs）与只读 trait（traits.rs 的 `AtomTableProvider`）**下沉本模块**——
//!   **逐字搬迁、不改算法、不改命名**（Step2 engine 只改 use 路径便于核对）。
//!   engine 原位文件由 Step2 删除（本批不动，中间态允许重复）。
//! - **R1.3 骨架**: `BeatCompute`（个体节律比对 trait）保留——R3.1/Step1 agent 节律库实现。
//!
//! 未下沉（属采集/消费侧，Step1 迁 agent）: `BeatSource`/`BeatConsumer`/`GeneProvider`/
//! `TemporalProvider` 四个读写侧 trait、以及 beat_library 其余 13 文件（含 cache/persistence）。
//!
//! ## 核心公式（搬自 engine，未改）
//! - **耦合强度**: K_AB = (f_A · f_B) / (|f_A − f_B| + ε)
//! - **熵产生**: σ = Σ σ_i，σ_i = 1/K_AB_i
//! - **弛豫时间**: τ = 1/σ
//! - ε 防除零常数（默认 1e-10）；K_AB 越大→耦合越强→σ 越小→τ 越小→系统越活跃
//!
//! 理论锚点: 元衍拓扑学 V8.0 / drift_hash_root_v8.0；P22 §10.5 复谱距离同族。
//! core 五类约束（ARCH-1 v2.1 §5）: 无 fs/net/时间/随机/异步——本模块全纯计算。

#![allow(non_snake_case)]

use serde::{Deserialize, Serialize};

// ============================================================================
// 拍频纯数据类型（下沉自 engine beat_library/types.rs，逐字搬迁不改结构）
// ============================================================================

/// 频率原子 — 文本中单个字符或词组的频率表示
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeatAtom {
    pub token: String,
    pub freq_text: f64,
    pub freq_beat: f64,
    pub layer: AtomLayer,
    pub entropy_contribution: f64,
}

impl BeatAtom {
    /// 创建新原子
    pub fn new(token: String, freq_text: f64, layer: AtomLayer) -> Self {
        Self {
            token,
            freq_text,
            freq_beat: 0.0,
            layer,
            entropy_contribution: 0.0,
        }
    }

    /// 设置拍频
    pub fn with_beat(mut self, beat: f64) -> Self {
        self.freq_beat = beat;
        self
    }

    /// 设置熵贡献
    pub fn with_entropy(mut self, entropy: f64) -> Self {
        self.entropy_contribution = entropy;
        self
    }
}

/// 原子层级枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AtomLayer {
    L0,
    L1,
    L2,
}

impl AtomLayer {
    pub fn as_str(&self) -> &str {
        match self {
            AtomLayer::L0 => "L0",
            AtomLayer::L1 => "L1",
            AtomLayer::L2 => "L2",
        }
    }
}

/// 轻量拍频对 — 邻接表中存储的拍频关系
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeatPair {
    pub atom_a_hash: String,
    pub atom_b_hash: String,
    pub beat_value: f64,
    pub coupling: f64,
    pub timestamp: f64,
}

/// 拍频记录 — 两个频率原子之间的拍频关系
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeatRecord {
    /// 拍频记录唯一标识
    pub id: String,
    /// 来源标识 (文档/段落/句子哈希)
    pub source_id: String,
    /// 参与拍频的原子列表
    pub atoms: Vec<BeatAtom>,
    /// 弛豫时间 τ (越小越活跃)
    pub tau: f64,
    /// 熵产生 σ
    pub sigma: f64,
    /// 耦合强度 K_AB
    pub k_ab: f64,
    /// 时间戳
    pub timestamp: f64,
}

impl BeatRecord {
    pub fn new(
        id: String,
        source_id: String,
        atoms: Vec<BeatAtom>,
        k_ab: f64,
        tau: f64,
        sigma: f64,
        timestamp: f64,
    ) -> Self {
        Self {
            id,
            source_id,
            atoms,
            k_ab,
            tau,
            sigma,
            timestamp,
        }
    }
}

/// 频率分桶 — 按频率范围组织拍频记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyBucket {
    pub range_start: f64,
    pub range_end: f64,
    pub beats: Vec<BeatRecord>,
}

impl FrequencyBucket {
    pub fn new(range_start: f64, range_end: f64) -> Self {
        Self {
            range_start,
            range_end,
            beats: Vec::new(),
        }
    }

    /// 判断频率是否在此桶范围内
    pub fn contains(&self, freq: f64) -> bool {
        freq >= self.range_start && freq < self.range_end
    }

    pub fn insert(&mut self, beat: BeatRecord) {
        self.beats.push(beat);
    }

    pub fn len(&self) -> usize {
        self.beats.len()
    }

    pub fn is_empty(&self) -> bool {
        self.beats.is_empty()
    }
}

/// 拍频查询请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeatQuery {
    pub freq_min: f64,
    pub freq_max: f64,
    pub layer: Option<AtomLayer>,
    pub limit: usize,
}

impl BeatQuery {
    pub fn new(freq_min: f64, freq_max: f64, limit: usize) -> Self {
        Self {
            freq_min,
            freq_max,
            layer: None,
            limit,
        }
    }

    pub fn with_layer(mut self, layer: AtomLayer) -> Self {
        self.layer = Some(layer);
        self
    }
}

impl Default for BeatQuery {
    fn default() -> Self {
        Self {
            freq_min: 0.0,
            freq_max: f64::MAX,
            layer: None,
            limit: 100,
        }
    }
}

// ============================================================================
// 只读 trait（下沉自 engine beat_library/traits.rs 的 AtomTableProvider）
// ============================================================================

/// L0 原子表只读接口（只读 `&self`，L0 核心数据不可被拍频库修改）
pub trait AtomTableProvider {
    /// 查询指定 token 的文本频率
    fn freq_of(&self, token: &str) -> Option<f64>;
}

// ============================================================================
// 拍频计算引擎（下沉自 engine beat_library/beat_computer.rs，逐字搬迁不改算法）
// ============================================================================

/// 拍频计算引擎
///
/// 负责计算两个频率原子之间的耦合强度 K_AB、多原子共振记录、弛豫时间 τ 和熵产生 σ。
pub struct BeatComputer {
    /// L0 原子表只读访问接口
    atom_provider: Box<dyn AtomTableProvider>,
    /// 防除零常数
    epsilon: f64,
}

impl BeatComputer {
    /// 创建拍频计算引擎
    pub fn new(atom_provider: Box<dyn AtomTableProvider>, epsilon: f64) -> Self {
        Self {
            atom_provider,
            epsilon,
        }
    }

    /// 使用默认 epsilon 值创建引擎
    pub fn with_default_epsilon(atom_provider: Box<dyn AtomTableProvider>) -> Self {
        Self::new(atom_provider, 1e-10)
    }

    // ===== 核心计算 =====

    /// 计算两个 token 之间的耦合强度 K_AB
    ///
    /// 公式: K_AB = (f_A * f_B) / (|f_A - f_B| + ε)
    pub fn compute_beat(&self, token_a: &str, token_b: &str) -> Option<f64> {
        let f_a = self.atom_provider.freq_of(token_a)?;
        let f_b = self.atom_provider.freq_of(token_b)?;

        let diff = (f_a - f_b).abs();
        let k_ab = (f_a * f_b) / (diff + self.epsilon);

        Some(k_ab)
    }

    /// 计算多原子共振记录
    ///
    /// 对 token 列表中所有相邻对计算 K_AB，汇总为单条 BeatRecord。
    pub fn compute_resonance(&self, tokens: Vec<String>) -> BeatRecord {
        if tokens.is_empty() {
            return BeatRecord::new(
                "empty".to_string(),
                "none".to_string(),
                Vec::new(),
                0.0,
                f64::MAX,
                0.0,
                0.0,
            );
        }

        let mut atoms: Vec<BeatAtom> = Vec::with_capacity(tokens.len());
        let mut total_sigma: f64 = 0.0;
        let mut avg_k_ab: f64 = 0.0;
        let mut pair_count: usize = 0;

        // 创建原子
        for token in &tokens {
            let freq = self.atom_provider.freq_of(token).unwrap_or(0.0);
            let layer = if token.chars().count() == 1 {
                AtomLayer::L0
            } else {
                AtomLayer::L1
            };
            atoms.push(BeatAtom {
                token: token.clone(),
                freq_text: freq,
                freq_beat: 0.0,
                layer,
                entropy_contribution: 0.0,
            });
        }

        // 计算相邻对的 K_AB
        for i in 0..tokens.len().saturating_sub(1) {
            if let Some(k_ab) = self.compute_beat(&tokens[i], &tokens[i + 1]) {
                let sigma_i = if k_ab > 0.0 { 1.0 / k_ab } else { 0.0 };
                total_sigma += sigma_i;
                avg_k_ab += k_ab;
                pair_count += 1;

                // 更新原子的熵贡献
                atoms[i].entropy_contribution += sigma_i / 2.0;
                atoms[i + 1].entropy_contribution += sigma_i / 2.0;

                // 更新拍频
                let beat = (atoms[i].freq_text - atoms[i + 1].freq_text).abs();
                atoms[i].freq_beat = beat;
                atoms[i + 1].freq_beat = beat;
            }
        }

        if pair_count > 0 {
            avg_k_ab /= pair_count as f64;
        }

        let tau = self.compute_tau(total_sigma);

        let id = if tokens.len() >= 2 {
            format!("beat_{}_{}", tokens[0], tokens[tokens.len() - 1])
        } else if tokens.len() == 1 {
            format!("beat_{}", tokens[0])
        } else {
            "beat_empty".to_string()
        };

        BeatRecord::new(id, "resonance".to_string(), atoms, avg_k_ab, tau, total_sigma, 0.0)
    }

    // ===== 弛豫时间 & 熵计算 =====

    /// 根据熵产生 σ 计算弛豫时间 τ（τ = 1/σ；σ ≤ 0 返回 f64::MAX）
    pub fn compute_tau(&self, sigma: f64) -> f64 {
        if sigma > 0.0 {
            1.0 / sigma
        } else {
            f64::MAX
        }
    }

    /// 计算拍频记录列表的总熵产生 σ（σ_i = 1/K_AB_i）
    pub fn compute_sigma(&self, records: &[BeatRecord]) -> f64 {
        records
            .iter()
            .map(|r| if r.k_ab > 0.0 { 1.0 / r.k_ab } else { 0.0 })
            .sum()
    }
}

// ============================================================================
// R1.3 个体节律比对骨架（R3.1/Step1 agent 节律库实现，本批无实现）
// ============================================================================

/// 节律样本（纯数据: 时间戳 + 值序列, 无来源绑定——采集层在 agent）
#[derive(Debug, Clone, PartialEq)]
pub struct 节律样本 {
    /// 采样时间点（秒, 可非等间隔）
    pub 时间: Vec<f64>,
    /// 对应观测值
    pub 值: Vec<f64>,
}

/// 节律谱（纯计算产物）——ω/γ 两维独立输出
#[derive(Debug, Clone, PartialEq)]
pub struct 节律谱 {
    /// 主导 τ = 1/γ（耗散维度, 秒）
    pub 主导τ: f64,
    /// 主频 ω = Re(λ)（振荡维度, 弧度/秒; **非 2π/τ**）
    pub 主频ω: f64,
    /// 幅值谱（按频率分箱, 降序）
    pub 幅值谱: Vec<f64>,
}

/// 节律比对结果
#[derive(Debug, Clone, PartialEq)]
pub struct 节律比对结果 {
    /// 相似度 ∈[0,1]
    pub 相似度: f64,
    /// 是否判定为同一节律（阈值由调用方给定, core 不内置业务阈值）
    pub 匹配: bool,
}

/// 节律比对纯计算 trait（R1.3 空骨架——本批无实现, R3.1/Step1 落地）
pub trait BeatCompute {
    /// 节律样本 → 节律谱（主导τ/主频ω/幅值谱）
    fn 节律τ谱(样本: &节律样本) -> Option<节律谱>;
    /// 两节律谱相似度 ∈[0,1]（复谱距离 + 相位对齐）
    fn 相似度(a: &节律谱, b: &节律谱) -> f64;
    /// 按给定阈值比对两节律
    fn 比对(a: &节律谱, b: &节律谱, 阈值: f64) -> Option<节律比对结果>;
}

// ============================================================================
// 测试（随搬迁一同搬运——数值断言锁死算法未变）
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // ===== 拍频类型测试（搬自 engine types.rs）=====

    #[test]
    fn test_beat_atom_creation() {
        let atom = BeatAtom::new("测".to_string(), 77.7, AtomLayer::L0);
        assert_eq!(atom.token, "测");
        assert_eq!(atom.freq_text, 77.7);
        assert_eq!(atom.freq_beat, 0.0);
        assert_eq!(atom.layer, AtomLayer::L0);
    }

    #[test]
    fn test_beat_atom_with_beat() {
        let atom = BeatAtom::new("试".to_string(), 88.8, AtomLayer::L0)
            .with_beat(11.1)
            .with_entropy(0.5);
        assert_eq!(atom.freq_beat, 11.1);
        assert_eq!(atom.entropy_contribution, 0.5);
    }

    #[test]
    fn test_atom_layer_as_str() {
        assert_eq!(AtomLayer::L0.as_str(), "L0");
        assert_eq!(AtomLayer::L1.as_str(), "L1");
        assert_eq!(AtomLayer::L2.as_str(), "L2");
    }

    #[test]
    fn test_beat_record_creation() {
        let atoms = vec![
            BeatAtom::new("A".to_string(), 100.0, AtomLayer::L0),
            BeatAtom::new("B".to_string(), 120.0, AtomLayer::L0),
        ];
        let record = BeatRecord::new(
            "rec-1".to_string(),
            "src-1".to_string(),
            atoms.clone(),
            2400.0,
            0.000416,
            24000.0,
            1000.0,
        );
        assert_eq!(record.id, "rec-1");
        assert_eq!(record.source_id, "src-1");
        assert_eq!(record.atoms.len(), 2);
        assert_eq!(record.k_ab, 2400.0);
    }

    #[test]
    fn test_frequency_bucket_contains() {
        let bucket = FrequencyBucket::new(100.0, 200.0);
        assert!(bucket.contains(150.0));
        assert!(!bucket.contains(50.0));
        assert!(!bucket.contains(200.0));
        assert!(bucket.contains(100.0));
    }

    #[test]
    fn test_beat_query_default() {
        let query: BeatQuery = Default::default();
        assert_eq!(query.limit, 100);
        assert!(query.layer.is_none());
    }

    #[test]
    fn test_beat_query_with_layer() {
        let query = BeatQuery::new(10.0, 100.0, 50).with_layer(AtomLayer::L1);
        assert_eq!(query.freq_min, 10.0);
        assert_eq!(query.freq_max, 100.0);
        assert_eq!(query.limit, 50);
        assert_eq!(query.layer, Some(AtomLayer::L1));
    }

    #[test]
    fn test_serialize_roundtrip() {
        let atom = BeatAtom::new("测".to_string(), 77.7, AtomLayer::L0);
        let json = serde_json::to_string(&atom).unwrap();
        let restored: BeatAtom = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.token, atom.token);
        assert_eq!(restored.freq_text, atom.freq_text);
        assert_eq!(restored.layer, atom.layer);
    }

    // ===== 拍频计算测试（搬自 engine beat_computer.rs）=====

    /// 测试用固定频率映射的 Mock 原子表
    struct TestAtomTable {
        freq_map: HashMap<String, f64>,
    }

    impl TestAtomTable {
        fn new(map: HashMap<String, f64>) -> Self {
            Self { freq_map: map }
        }
    }

    impl AtomTableProvider for TestAtomTable {
        fn freq_of(&self, token: &str) -> Option<f64> {
            self.freq_map.get(token).copied()
        }
    }

    /// 创建标准测试引擎
    fn make_engine() -> BeatComputer {
        let mut map = HashMap::new();
        map.insert("A".to_string(), 100.0);
        map.insert("B".to_string(), 120.0);
        map.insert("C".to_string(), 200.0);
        map.insert("D".to_string(), 50.0);
        map.insert("E".to_string(), 100.0); // 与 A 同频
        BeatComputer::new(Box::new(TestAtomTable::new(map)), 1e-10)
    }

    #[test]
    fn test_compute_beat_formula() {
        let engine = make_engine();
        // K_AB = (100*120) / (|100-120| + 1e-10) ≈ 600
        let k_ab = engine.compute_beat("A", "B").unwrap();
        let expected = (100.0 * 120.0) / (20.0 + 1e-10);
        assert!((k_ab - expected).abs() < 1e-6, "K_AB={}, expected={}", k_ab, expected);
        // K_AB = (200*50) / (150 + ε) ≈ 66.667
        let k_ab2 = engine.compute_beat("C", "D").unwrap();
        let expected2 = (200.0 * 50.0) / (150.0 + 1e-10);
        assert!((k_ab2 - expected2).abs() < 1e-6);
    }

    #[test]
    fn test_compute_beat_zero_diff() {
        let engine = make_engine();
        // 同频 → K_AB = 10000/ε → 极大值
        let k_ab = engine.compute_beat("A", "E").unwrap();
        let expected = (100.0 * 100.0) / 1e-10;
        assert!((k_ab - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn test_compute_beat_missing_token() {
        let engine = make_engine();
        assert!(engine.compute_beat("A", "Z").is_none());
        assert!(engine.compute_beat("Z", "A").is_none());
        assert!(engine.compute_beat("Z", "X").is_none());
    }

    #[test]
    fn test_compute_resonance_multiple() {
        let engine = make_engine();
        let tokens = vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
        ];
        let record = engine.compute_resonance(tokens);
        assert_eq!(record.atoms.len(), 4);
        assert!(record.k_ab > 0.0, "avg_k_ab should be positive");
        assert!(record.sigma > 0.0, "sigma should be positive");
        assert!(record.tau > 0.0 && record.tau < f64::MAX, "tau={}", record.tau);
        assert_eq!(record.source_id, "resonance");
    }

    #[test]
    fn test_compute_resonance_empty() {
        let engine = make_engine();
        let record = engine.compute_resonance(Vec::new());
        assert!(record.atoms.is_empty());
        assert_eq!(record.id, "empty");
        assert_eq!(record.tau, f64::MAX);
    }

    #[test]
    fn test_compute_resonance_single() {
        let engine = make_engine();
        let record = engine.compute_resonance(vec!["A".to_string()]);
        assert_eq!(record.atoms.len(), 1);
        assert_eq!(record.atoms[0].token, "A");
        assert_eq!(record.tau, f64::MAX); // sigma=0 → tau=∞
    }

    #[test]
    fn test_compute_sigma_and_tau() {
        let engine = make_engine();
        let atom_a = BeatAtom::new("x".to_string(), 100.0, AtomLayer::L0);
        let records = vec![
            BeatRecord::new("1".to_string(), "s1".to_string(), vec![atom_a.clone()], 600.0, 0.0, 0.0, 0.0),
            BeatRecord::new("2".to_string(), "s2".to_string(), vec![atom_a.clone()], 300.0, 0.0, 0.0, 0.0),
            BeatRecord::new("3".to_string(), "s3".to_string(), vec![atom_a], 66.667, 0.0, 0.0, 0.0),
        ];
        let sigma = engine.compute_sigma(&records);
        assert!(sigma > 0.01 && sigma < 0.03, "sigma={}", sigma);
        let tau = engine.compute_tau(sigma);
        assert!(tau > 30.0 && tau < 70.0, "tau={}", tau);
    }

    #[test]
    fn test_compute_sigma_zero_k_ab() {
        let engine = make_engine();
        let atom = BeatAtom::new("x".to_string(), 100.0, AtomLayer::L0);
        let records = vec![BeatRecord::new(
            "1".to_string(),
            "s1".to_string(),
            vec![atom],
            0.0,
            0.0,
            0.0,
            0.0,
        )];
        assert_eq!(engine.compute_sigma(&records), 0.0);
    }

    #[test]
    fn test_compute_sigma_empty_slice() {
        let engine = make_engine();
        assert_eq!(engine.compute_sigma(&[]), 0.0);
    }

    #[test]
    fn test_compute_tau_zero_sigma() {
        let engine = make_engine();
        assert_eq!(engine.compute_tau(0.0), f64::MAX);
        assert_eq!(engine.compute_tau(-1.0), f64::MAX);
    }

    // ===== R1.3 骨架占位（实现落 R3.1/Step1 后替换）=====

    #[test]
    fn 测试_R13_骨架在位() {
        let 样本 = 节律样本 { 时间: vec![0.0, 1.0, 2.0], 值: vec![0.0, 1.0, 0.0] };
        assert_eq!(样本.时间.len(), 样本.值.len(), "样本等长");
        let 谱 = 节律谱 { 主导τ: 10.0, 主频ω: 0.5, 幅值谱: vec![1.0, 0.5] };
        assert!(谱.主导τ > 0.0 && 谱.主频ω.is_finite());
        let 结果 = 节律比对结果 { 相似度: 0.9, 匹配: true };
        assert!(结果.相似度 >= 0.0 && 结果.相似度 <= 1.0);
        // trait 约束编译链验证（全关联函数 trait 非 object-safe, 用泛型约束）
        fn _约束验证<T: BeatCompute>(样本: &节律样本) -> Option<节律谱> {
            T::节律τ谱(样本)
        }
        let _ = _约束验证::<未实现占位>;
    }

    /// 占位实现者（仅供编译链验证; R3.1/Step1 落地后由真实实现替换）
    struct 未实现占位;
    impl BeatCompute for 未实现占位 {
        fn 节律τ谱(_样本: &节律样本) -> Option<节律谱> {
            None // 骨架期: 未实现（Step1 agent 节律库落地）
        }
        fn 相似度(_a: &节律谱, _b: &节律谱) -> f64 {
            0.0
        }
        fn 比对(_a: &节律谱, _b: &节律谱, _阈值: f64) -> Option<节律比对结果> {
            None
        }
    }
}
