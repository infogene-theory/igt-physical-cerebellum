// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! IGT τ谱动力学纯计算核心（第210批隔离）
//!
//! 从原 `igt-core` 单体内隔离出的**独立纯计算 crate**——只含谱动力学数学核心，
//! 零数据库/网络/UI 依赖，独立编译独立测试（<30s）。
//!
//! 理论依据:
//!   - 《IGT Agent _ Engine 新开发计划》: igt_core 应为核心计算库，与平台/UI解耦
//!   - 变分原理深化方案: 谱动力学三件套 + Lyapunov + 演化模拟 + 逆元构建 + τ分层
//!
//! 模块划分（迁移自原 engine/src/l1/，内部零 crate:: 依赖，可干净隔离）:
//!   - spectrum_solver: 谱求解器（L→本征值/τ链/D_s/β/p值/η + 非厄米/Ginibre）
//!   - topology_editor: 拓扑编辑器（图节点+κ边→L=D-A+V + 分块β检测）
//!   - system_identifier: 系统辨识器（DMD/EDMD→Koopman谱→τ链/ω）
//!   - lyapunov: Lyapunov指数（Rosenstein轨道邻域法→λ_max + ε_wave）
//!   - evolution_simulator: 演化模拟器（Metropolis变异-选择→H定理 + 温度扫描）
//!   - inverse_builder: 逆元构建器（联合耗散算子+模-反模配对→逆元形成度）
//!   - tau_layered_classifier: τ分层分类器（五维τ指纹+自动分层准入 v2.0）
//!   - tau_resonance: τ共振（共振融合/共振通道，供分层分类器测试使用）

// 中文命名 + 希腊字母(κ/τ) 项目特色
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(elided_lifetimes_in_paths)]
#![allow(mismatched_lifetime_syntaxes)]
#![allow(dead_code)]
#![allow(unused_must_use)]
#![allow(private_interfaces)]
#![allow(unreachable_patterns)]
#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(unused_mut)]
#![allow(unused_parens)]
#![allow(unexpected_cfgs)]
#![allow(unused_macros)]
#![allow(unused_imports)]
#![allow(unused_comparisons)]
#![allow(confusable_idents)]
#![allow(mixed_script_confusables)]

pub mod d9_dataprep; // D9(2026-08-25): 数据接入与预处理层（TimeSeries + 质量门槛 + 保τ）
pub mod spectrum_solver;
pub mod topology_editor;
pub mod system_identifier;
pub mod lyapunov;
pub mod evolution_simulator;
pub mod inverse_builder;
pub mod tau_layered_classifier;
pub mod tau_resonance;
// ===== 第214批: CORE六域扩充 =====
pub mod group_theory; // separation域: 群公理校验/代数结构(纯std)
pub mod relaxation_spectrum; // acquisition域: 时序→弛豫频谱(STFT)
pub mod tau_spectrum; // spectrum域: 本征τ谱/τ分量/τ间距统计
pub mod storage; // storage域: τ谱数据结构+序列化(Agent/Engine共享)
pub mod multiscale_health; // prediction域: 五层权重引擎+生存τ+失稳判别(第229批, docs/理论扩展)
pub mod disturbance_attribution; // prediction域: 扰动归因κ共振耦合+滞后对齐+贡献分解(第232批)
pub mod nonhermitian_spectrum; // 第267批 P22§6: 非厄米复谱+双正交归一化+椭圆系综D_s^(2D)+谱分离SW
pub mod baseline_ellipse; // MIGRATION-FINAL 归一: 个人基准椭圆统计（P22§13 椭圆雷达, flutter 双实现归一第一例）
pub mod health_fusion; // MIGRATION-FINAL 归一: 双/三判据融合预警（P22§13+P19, flutter 双实现归一第三例）
pub mod observer_level; // MIGRATION-FINAL 归一: 观察者等级能力代理评分（P22 自指深度近似, flutter 双实现归一第六例）
pub mod body_scale_monitor; // MIGRATION-FINAL 归一: D_s 三区/临界慢化/健康度 H（P22 §7.1/7.2/7.5, flutter 双实现归一第七例）
pub mod perception_status; // MIGRATION-FINAL 归一: 多模态感知覆盖度/分级/补数据建议（第354批, flutter 双实现归一第八例）
pub mod perception_8ch; // MIGRATION-FINAL 归一: 感知 8 通道窗口组织/快慢τ/非对角耦合摘要（第378批 M2, flutter 双实现归一第九例）
pub mod gene_diagnosis; // MIGRATION-FINAL 归一: 基因驱动诊断（第307批 P22 §10.7, flutter 双实现归一第十例）
pub mod warning_scanner; // MIGRATION-FINAL 归一: 九类生活预警扫描（第304批, 对齐 agent warning.rs, flutter 双实现归一第十一例）
// T-182 例9: guide_recommendation 无算量内核（scene→rhythm 决策 + 措辞）⇒ 整体迁 agent
//           `igt_agent::inverse_advice::推荐引导`（理论侧163拆分线 §2.1#2），core 模块删除。
pub mod tau_fet_ledger; // MIGRATION-FINAL 归一: tau-FET 贡献账本规则（batch 393 D12, 对齐 engine tau_fet.rs, flutter 双实现归一第十三例）
// T-182 例11: gene_guide_link 无算量内核（硬编码标签→节奏推荐映射表）⇒ 整体迁 agent
//            `igt_agent::inverse_advice::{标签节奏id,推荐节奏}`（理论侧163拆分线 §2.1#2），core 模块删除。
pub mod operator_route; // T-25 (D16 v1.0 正式 §4.2/§5.2/§6/§8/§12): 算子分层路由 core 骨架（OperatorTier/DataAdmissionLabel/DegradeReason/route, 拓扑同一律唯一定义）
pub mod spectral_equivalence; // MIGRATION-FINAL 归一: 谱等价标量三元判定（P22§10.5, flutter 双实现归一第四例）
pub mod complex_spectrum_match; // 第268批 P22§10.5: 参与度基因匹配(复谱距离+子空间夹角+参与度余弦+谱等价)
pub mod inverse_recommender; // 第270批 P22§8: 逆元建议器(推荐干预点→造波方案→模拟验证→InterventionPlan)
pub mod band_projection; // 第282批 P22v10.1§2.6: 三层耗散联通引擎(三层带宽投影+跨层自能Σ+造波回归验证)
pub mod entropy_pump_core; // D3(2026-08-23): 熵泵机制core化(严格定义 L=H-iΓ 耗散相消干涉, 依赖D1复L非对角+D2观测者带宽)
pub mod analytical_signal; // D7(2026-08-23): Hilbert解析信号变换(复EDMD前置, 频域实现)
pub mod complex_dmd; // D7(2026-08-23): 复EDMD数据链(手写QR迭代r≤30 + 方案B模态基重构 L=ΦΛΦ⁻¹ 恢复非对角虚部)
pub mod symmetry_breaking; // D5(2026-08-23): 对称破缺检测(谱隙/τ分裂/戈德斯通/层级化, 依赖D1本征值准确)
pub mod causal_probe; // D6(2026-08-23): 造波因果探测(κ_ij=w_i†δL v_j完整复数 + 方向判据 + 安全约束ε_wave<0.3)
pub mod steady_engine; // D5/D6§3.3(2026-08-23): 统一稳态引擎(谱分离+对称破缺+熵泵+因果探测聚合, 共享谱分解+观测者带宽)
pub mod temperature_field; // D8(2026-08-24, P29/P30): 温度场时空模块(区域聚合+区域主导τ+时空判据+异常类型)
pub mod d13_acceptance; // D13(2026-08-24): 合成数据验证流水线+性能基准(P22a 对齐, 本征值恢复<0.05/双正交<10^-10)
pub mod ffi; // FFI域(第220批): C ABI 导出(移动端/脚本语言嵌入, cdylib)
// ===== R1.3（DISPATCH-R1R2 / CODE-AUDIT-001）: R3 迁端的 core 侧接口预留（空骨架, 零实现, 无 IO）=====
pub mod human_machine; // 人机域: trait CrossReferenceProjector（T_AB 投影/带宽窗/三子保真度 F）← engine l1/human_machine_bridge.rs（R3.3 实现）
pub mod kappa_channel; // κ通道域: trait CollectiveCoupling（L_offdiag/κ_ij 纯计算）← engine l4 organ 共振/resonance_coupling（R3 实现）
pub mod beat_kernel; // beat域: trait BeatCompute（节律比对纯计算）← engine beat_library/beat_computer.rs（R3.1 实现）
// ===== Step1-④（MIGRATION-FINAL）: 器官层共享纯类型（execution 迁 agent 前置）=====
pub mod organs_types; // ActionVector/ActionDomain/复向量分量/SensoryInput/封装状态/KappaMatrixSnapshot/WorldState ← engine organs/层纯结构（供 engine organs 与 agent execution 共用）
// ===== Step0-3d（MIGRATION-FINAL）: Engine 自观测快照纯数据层 =====
pub mod self_observation; // κ矩阵/brain状态/意识涌现/perception/熵债 五观测 + engine自观测快照 + 预警等级判定 ← engine l1/self_observation/snapshot.rs（纯数据/纯判定）
// ===== Step0-3c-2a（MIGRATION-FINAL）: perception 谱计算层（拆提交第2批: 纯计算）=====
pub mod perception_spectrum; // 计算τ谱/简化功率谱/检测峰值/计算熵值水平/识别拓扑特征/计算熵流路径/τ谱诊断 ← engine l1/perception/tau_spectrum.rs（纯计算）
// ===== Step0-3c-1（MIGRATION-FINAL）: perception 纯类型层（拆提交第1批: 纯类型/枚举/纯方法）=====
pub mod perception_types; // CsiFrame/CsiComplex/Esp32NodeType/TauSpectrum/EntropyLevel/EntropyFlowPath/拓扑类型/TopologyFeature/TauDiagnosis/BaselineTauAnchor ← engine l1/perception/{csi_data_format,tau_spectrum}.rs
// ===== Step0-3b-2（MIGRATION-FINAL）: 干预验证纯统计下沉（依赖 shared_types::Sbi8Result）=====
pub mod intervention_validator; // 干预验证器（基线/后测对比、Cohen's d、p 值、统计功效、批量对比）← engine l1/intervention_validator.rs
// ===== Step0-3a（MIGRATION-FINAL）: 造波安全控制层下沉（依赖 core lyapunov，无新增依赖）=====
pub mod wave_safety; // 造波安全控制/非线性安全控制（Lyapunov ε_wave 闭环）← engine l1/wave_safety.rs
// ===== Step0-2a（MIGRATION-FINAL）: 共享纯类型层（engine/agent 共用，消除跨端重复定义）=====
pub mod shared_types; // OrganId(←engine l1/organ_router.rs) + FrequencyPayload/BeatAtom/ReferenceSystemType/RelaxationPayload(←engine l1/igt_message.rs)
// ===== 第430批 ARCH-1 §2.2 纯净性迁移批2（engine→core, 零crate依赖干净隔离） =====
pub mod riemann_spectrum; // §19.13.72 黎曼谱偏离度 D_s (GUE KS 检验, L3/L4 谱分析层)
pub mod unified_entropy_debt; // §19.13.73 统一量纲 D_total (综合熵债, L7 健康评估层; 依赖 group_theory/riemann_spectrum 已在 core)
// ===== 第429批 ARCH-1 §5.3 纯净性迁移: 纯计算 P 定理模块（engine→core, 零crate内部依赖可干净隔离） =====
// 注: dissipation/field_decoupling 依赖 l1::three_field_dynamics（保留在 engine）暂不迁移
pub mod triple_lock; // P3 三重锁定定理(结构-功能-演化锁定)
pub mod chirality; // P4a 手性对称性破缺定理(系统演化方向)
pub mod emergence; // P9 自指观测者涌现定理(复杂度阈值→涌现)
pub mod control; // 控制容量分析(自我控制能力)
pub mod duality; // 二象性平衡(系统二象性分析)
pub mod inertia; // 惯性守恒(惯性向量分析)
// ===== T-124（PM 61）: 指针仪表 L1 确定性谱链 → core 纯函数（**零 IO / 零新增依赖**）=====
// 来源: T-123 Python 原型(scripts/meter_cv_detect.py)的 L1 部分; **不移植 L0**(Hough 圆/指针检测留在 Python)。
// ★ 边缘算子如实标注: Python 用 OpenCV Canny, 此处为**自实现 Sobel+NMS** ⇒ 覆盖率/圆度**不完全等价**;
//   刻度/指针两路**不用边缘** ⇒ 与 Python 逐字对齐, 供读数对账。
pub mod meter_reading; // 读数确定式(下限+(上限−下限)×((指针角−刻度起) mod 360)/张角) + 角度工具 + 量程
pub mod meter_body_gate; // 本体门控5量(刻度周期性分/峰数/圆边覆盖率/圆度/指针置信) + 刻度判据/径向连续暗条(纯计算)
// ===== T-126（PM 65）: 波形图 L1 确定性谱链 → core 纯函数（**零 IO / 零新增依赖**）=====
// 来源: T-125 Python 原型(scripts/waveform_cv_detect.py)的 L1 部分; **不移植 L0**(Hough 轴检出/曲线抽取/每列连通段数留在 Python)。
// ★ 边界如实: 门控中"每列连通段数中位"需**原始像素掩膜** ⇒ **不在此模块**, 由 L0 判定后作为外部结论传入。
pub mod waveform_l1; // 波形L1(峰峰值/频率(峰距中位→周期像素→周期数)/趋势(斜率+符号, 图像与物理两个都给)/对称性) + 门控(轴存在/列覆盖/起伏/段数中位由外部给)
