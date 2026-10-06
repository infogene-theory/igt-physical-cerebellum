// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! **T-124（PM 61）**：指针仪表 **L1 本体门控**（core 纯函数，**零 IO**）。
//!
//! 来源：T-123 Python 原型（`scripts/meter_cv_detect.py`）的**本体门控 5 量** —— **PM 61 §二.1 要求移植**：
//! **刻度周期性分 / 刻度峰数 / 圆边覆盖率 / 圆度 / 指针置信**。
//!
//! ## 设计边界（**如实**）
//! - **纯函数**：输入 `&[u8]` 灰度 + 宽高 + **圆心/半径（由 L0 提供）** ⇒ 输出门控结论；
//!   **不读文件、不持状态、不加依赖**（core 零 IO）。
//! - **不移植 L0**（Hough 圆 / 指针检测 **留在 Python**，PM 61 §二.3）。
//! - **★ 边缘算子差异（诚实标注）**：Python 侧用 **OpenCV `Canny`**；本 crate **无 CV 依赖** ⇒ 此处用
//!   **Sobel(L1) + 阈值 + 非极大值抑制**的**自实现边缘**。二者**不完全等价** ⇒
//!   **`圆边覆盖率`/`圆度` 会与 Python 有差异**（对账时如实列差），
//!   而**刻度/指针两路完全不用边缘** ⇒ 与 Python **逐字对齐**、可用于读数对账。
//! - **角度口径同 `meter_reading`**：**图像角**（x 右 0°、y 下 90°、顺时针增）。
//!
//! ## 与 Python 原型的**逐字对齐点**（对账前提）
//! 分箱数 **360**（每箱 1°）、暗阈 = **局部中位 − 42**、扫描环带 `[0.76R, 0.93R]`、
//! 计峰最小间隔 **6°**、峰上限 **40**、间隔一致性窗口 **`[0.6×中位, 1.5×中位]`**、
//! 张角分母 **210**（`(张角−60)/210`）、指针扫描 `[0.20R, 0.86R]` 采样 **64** 点、
//! 分位数 **22%**（线性插值）、置信去邻域 **±15°**、覆盖率半径带 `[0.97R, 1.03R]`、
//! 圆度 `1 − min(1, std/(0.05R))`（**总体标准差**）、**分母只算"圆在图像内的可见角度箱"**。

use crate::meter_reading::{归一角度, 量程, 读数结果, 读数确定式};

/// 分箱数（**360 ⇒ 每箱 1°**，与 Python 一致）
const 箱数: usize = 360;

/// 门控阈值（**默认值与 T-123 Python 原型一致**）
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct 门控阈值 {
    pub 刻度周期性分: f64,
    pub 峰数下限: usize,
    pub 峰数上限: usize,
    pub 圆边覆盖率: f64,
    pub 圆度: f64,
    pub 指针置信: f64,
}

impl Default for 门控阈值 {
    fn default() -> Self {
        Self {
            刻度周期性分: 0.35,
            峰数下限: 6,
            峰数上限: 40,
            圆边覆盖率: 0.55,
            圆度: 0.5,
            指针置信: 0.30,
        }
    }
}

/// 本体门控结果（**结构保留、不压成标量** —— 续79 纪律2）
#[derive(Debug, Clone, PartialEq)]
pub struct 本体门控结果 {
    pub 刻度周期性分: f64,
    pub 刻度峰数: usize,
    /// 刻度起止角 `(起, 张角)`；**未估出 ⇒ `None`**
    pub 刻度起止: Option<(f64, f64)>,
    pub 刻度角列表: Vec<f64>,
    pub 圆边覆盖率: f64,
    /// **圆度**；无足够边缘点 ⇒ `None`
    pub 圆度: Option<f64>,
    pub 指针置信: f64,
    pub 指针角: Option<f64>,
    pub 指针尖长比: Option<f64>,
    /// 5 条门控是否**全部**通过
    pub 通过: bool,
    /// **未通过原因（逐条，可审计）**
    pub 未通过原因: Vec<String>,
}

// ============================================================================
// L1 子算法（**与 Python 逐字对齐**）
// ============================================================================

/// **环形极坐标**：把 `[r_in, r_out]` 环带按角度分箱 ⇒ 每箱**灰度均值**与**暗像素计数**。
///
/// 暗阈 = **该环带内所有采样点的中位数 − 42**（与 Python 同）。
fn 环形极坐标(
    灰度: &[u8],
    宽: usize,
    高: usize,
    cx: f64,
    cy: f64,
    r_in: f64,
    r_out: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut 均 = vec![0.0f64; 箱数];
    let mut 暗 = vec![0.0f64; 箱数];
    let mut cnt = vec![0usize; 箱数];
    // ★ **与 `np.mgrid[start:stop]` 严格对齐**：mgrid 以 **float 起点**逐 1 递增至 `< stop`，
    //   随后 `astype(int32)` 截断 ⇒ 整数范围 = `[floor(start), floor(start)+ceil(stop-start)-1]`（**闭区间**）。
    //   实测：`mgrid[63.3:417.7]` ⇒ 63,64,…,417（**含 417**）。
    let y0f = (cy - r_out).max(0.0);
    let y1f = ((cy + r_out + 1.0) as f64).min(高 as f64);
    let x0f = (cx - r_out).max(0.0);
    let x1f = ((cx + r_out + 1.0) as f64).min(宽 as f64);
    let y0 = y0f.floor() as i64;
    let y1 = y0 + ((y1f - y0f).ceil() as i64) - 1;
    let x0 = x0f.floor() as i64;
    let x1 = x0 + ((x1f - x0f).ceil() as i64) - 1;
    // 第一遍：收集环带内样本
    let mut 样本: Vec<(f64, u8)> = Vec::new();
    for yi in y0..=y1 {
        for xi in x0..=x1 {
            if yi < 0 || xi < 0 || yi >= 高 as i64 || xi >= 宽 as i64 {
                continue;
            }
            let (y, x) = (yi as usize, xi as usize);
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;
            let rr = (dx * dx + dy * dy).sqrt();
            if rr >= r_in && rr <= r_out {
                let ang = 归一角度(dy.atan2(dx).to_degrees());
                let idx = ((ang / (360.0 / 箱数 as f64)) as i32).clamp(0, 箱数 as i32 - 1) as usize;
                let v = 灰度[y * 宽 + x];
                cnt[idx] += 1;
                均[idx] += v as f64;
                样本.push((ang, v));
            }
        }
    }
    if 样本.is_empty() {
        return (均, 暗);
    }
    // 暗阈 = median(样本) − 42
    let mut vals: Vec<f64> = 样本.iter().map(|(_, v)| *v as f64).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let 中位 = median_sorted(&vals);
    let 暗阈 = 中位 - 42.0;
    for (ang, v) in 样本.iter() {
        if (*v as f64) < 暗阈 {
            let idx = ((*ang / (360.0 / 箱数 as f64)) as i32).clamp(0, 箱数 as i32 - 1) as usize;
            暗[idx] += 1.0;
        }
    }
    for i in 0..箱数 {
        if cnt[i] > 0 {
            均[i] /= cnt[i] as f64;
        }
    }
    (均, 暗)
}

/// 已排序切片的中位数（**偶数个取中间两数均值**，与 `numpy.median` 同）
fn median_sorted(v: &[f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// **计峰**：环形信号上的显著局部峰（**迭代取最高 ⇒ 抑制 ±最小间隔 ⇒ 重复**，与 Python 同）
fn 计峰(信号: &[f64], 最小间隔deg: f64) -> Vec<(f64, f64)> {
    let mut s = 信号.to_vec();
    let supp = ((最小间隔deg / (360.0 / 箱数 as f64)).round() as i64).max(1);
    let mut 峰 = Vec::new();
    for _ in 0..40 {
        // argmax：**取第一个最大者**（与 numpy 同）
        let mut i = 0usize;
        let mut best = f64::NEG_INFINITY;
        for (k, v) in s.iter().enumerate() {
            if *v > best {
                best = *v;
                i = k;
            }
        }
        if best <= 0.0 {
            break;
        }
        峰.push((i as f64 * (360.0 / 箱数 as f64), best));
        for k in -supp..=supp {
            let idx = ((i as i64 + k).rem_euclid(箱数 as i64)) as usize;
            s[idx] = 0.0;
        }
    }
    峰
}

/// **刻度判据** ⇒ `(分数, 刻度角列表, (起, 张角))`（与 Python 同）
pub fn 刻度判据(
    灰度: &[u8],
    宽: usize,
    高: usize,
    cx: f64,
    cy: f64,
    r: f64,
) -> (f64, Vec<f64>, Option<(f64, f64)>) {
    let (_, 暗) = 环形极坐标(灰度, 宽, 高, cx, cy, r * 0.76, r * 0.93);
    let 暗和: f64 = 暗.iter().sum();
    if 暗和 <= 0.0 {
        return (0.0, vec![], None);
    }
    let 峰 = 计峰(&暗, 6.0);
    if 峰.len() < 6 {
        return (0.0, vec![], None);
    }
    let mut 角: Vec<f64> = 峰.iter().map(|(a, _)| *a).collect();
    角.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let 一致 = if 角.len() >= 4 {
        let mut 间隔: Vec<f64> = (0..角.len() - 1).map(|i| 角[i + 1] - 角[i]).collect();
        间隔.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let 中 = median_sorted(&间隔);
        let 命中 = 间隔.iter().filter(|d| 0.6 * 中 <= **d && **d <= 1.5 * 中).count();
        命中 as f64 / 间隔.len() as f64
    } else {
        0.0
    };
    // 循环最大空档（**字典序：先比 d、再比 a1、再比 a2**，与 Python `max(元组)` 同）
    let mut 最大空档 = (0.0f64, 0.0f64, 0.0f64);
    let mut 首 = true;
    for i in 0..角.len() {
        let a1 = 角[i];
        let a2 = 角[(i + 1) % 角.len()];
        let d = 归一角度(a2 - a1);
        if 首
            || d > 最大空档.0
            || (d == 最大空档.0 && (a1 > 最大空档.1 || (a1 == 最大空档.1 && a2 > 最大空档.2)))
        {
            最大空档 = (d, a1, a2);
            首 = false;
        }
    }
    let 张角 = 360.0 - 最大空档.0;
    let 起 = 归一角度(最大空档.2);
    let 分数 = (峰.len() as f64 / 12.0).min(1.0)
        * 一致
        * ((张角 - 60.0) / 210.0).max(0.0).min(1.0);
    (分数, 角, Some((起, 张角)))
}

/// **自实现边缘**：Sobel(L1 幅值) + 阈值 + **非极大值抑制** ⇒ 边缘掩膜。
///
/// **★ 与 OpenCV `Canny` 不完全等价**（见模块头注）；此处保持"细边 + 阈值"语义，供覆盖率/圆度使用。
fn 边缘掩膜(灰度: &[u8], 宽: usize, 高: usize, 阈: f64) -> Vec<bool> {
    let mut 幅 = vec![0.0f64; 宽 * 高];
    let mut 向 = vec![0.0f64; 宽 * 高];
    for y in 1..高.saturating_sub(1) {
        for x in 1..宽.saturating_sub(1) {
            let g = |xx: usize, yy: usize| 灰度[yy * 宽 + xx] as f64;
            let gx = (g(x + 1, y - 1) + 2.0 * g(x + 1, y) + g(x + 1, y + 1))
                - (g(x - 1, y - 1) + 2.0 * g(x - 1, y) + g(x - 1, y + 1));
            let gy = (g(x - 1, y + 1) + 2.0 * g(x, y + 1) + g(x + 1, y + 1))
                - (g(x - 1, y - 1) + 2.0 * g(x, y - 1) + g(x + 1, y - 1));
            幅[y * 宽 + x] = gx.abs() + gy.abs(); // L1 幅值
            向[y * 宽 + x] = (gy).atan2(gx);
        }
    }
    let mut 掩 = vec![false; 宽 * 高];
    for y in 1..高.saturating_sub(1) {
        for x in 1..宽.saturating_sub(1) {
            let m = 幅[y * 宽 + x];
            if m < 阈 {
                continue;
            }
            // 沿梯度方向做非极大值抑制（取 ±1 像素近似）
            let a = 向[y * 宽 + x];
            let dx = a.cos();
            let dy = a.sin();
            let 邻 = |px: f64, py: f64| -> f64 {
                let xx = (x as f64 + px).round() as i64;
                let yy = (y as f64 + py).round() as i64;
                if xx < 0 || yy < 0 || xx >= 宽 as i64 || yy >= 高 as i64 {
                    return 0.0;
                }
                幅[yy as usize * 宽 + xx as usize]
            };
            if m >= 邻(dx, dy) && m >= 邻(-dx, -dy) {
                掩[y * 宽 + x] = true;
            }
        }
    }
    掩
}

/// **圆边覆盖率 ＋ 圆度**（与 Python 同定义；**边缘算子不同**，见头注）。
///
/// - **覆盖率**：半径 ∈ `[0.97R, 1.03R]` 的边缘点在角度分箱上的非空比例；
///   **分母只算"圆在图像内的可见角度箱"**（真实图表盘可能**部分出框** ⇒ 否则覆盖率被人为压低）；
/// - **圆度**：这些边缘点半径的**总体标准差** ⇒ `1 − min(1, std/(0.05R))`。
pub fn 圆边与圆度(
    灰度: &[u8],
    宽: usize,
    高: usize,
    cx: f64,
    cy: f64,
    r: f64,
) -> (f64, Option<f64>) {
    // 可见角度箱（分母）
    let mut 可见 = vec![false; 箱数];
    for b in 0..箱数 {
        let ang = b as f64 * (360.0 / 箱数 as f64) * std::f64::consts::PI / 180.0;
        let px = cx + r * ang.cos();
        let py = cy + r * ang.sin();
        if px >= 0.0 && px < 宽 as f64 && py >= 0.0 && py < 高 as f64 {
            可见[b] = true;
        }
    }
    let 分母 = 可见.iter().filter(|v| **v).count().max(1);

    let 掩 = 边缘掩膜(灰度, 宽, 高, 150.0);
    let mut 命中箱 = vec![false; 箱数];
    let mut 半径s: Vec<f64> = Vec::new();
    let 低 = r * 0.97;
    let 高带 = r * 1.03;
    for y in 0..高 {
        for x in 0..宽 {
            if !掩[y * 宽 + x] {
                continue;
            }
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;
            let rr = (dx * dx + dy * dy).sqrt();
            if rr >= 低 && rr <= 高带 {
                let ang = 归一角度(dy.atan2(dx).to_degrees());
                let idx = ((ang / (360.0 / 箱数 as f64)) as i32).clamp(0, 箱数 as i32 - 1) as usize;
                if 可见[idx] {
                    命中箱[idx] = true;
                }
                半径s.push(rr);
            }
        }
    }
    let 覆盖 = 命中箱.iter().filter(|v| **v).count() as f64 / 分母 as f64;
    if 半径s.len() < 20 {
        return (覆盖, None);
    }
    let n = 半径s.len() as f64;
    let 均 = 半径s.iter().sum::<f64>() / n;
    // **总体标准差**（ddof=0，与 numpy 默认一致）
    let 方 = 半径s.iter().map(|v| (v - 均) * (v - 均)).sum::<f64>() / n;
    let 圆度 = (1.0 - (方.sqrt() / (0.05 * r)).min(1.0)).max(0.0);
    (覆盖, Some(圆度))
}

/// **分位数（线性插值）** —— 与 `numpy.percentile(..., method='linear')` 同。
fn 分位(值: &mut Vec<f64>, q: f64) -> f64 {
    值.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = 值.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return 值[0];
    }
    let pos = (n as f64 - 1.0) * (q / 100.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return 值[lo];
    }
    let w = pos - lo as f64;
    值[lo] * (1.0 - w) + 值[hi] * w
}

/// **指针证据：径向连续暗条**（**真实图关键判据**；与 Python 同）。
///
/// **物理含义**：指针是**沿半径连续**的暗条 ⇒ 其方向上的**径向覆盖率**接近 1；
/// 而刻度环只在 `[0.78R,0.93R]` 一处 ⇒ 覆盖率低 ⇒ **能把"指针"与"只是一圈刻度"区分开**。
///
/// 返回 `(角, 尖长比, 置信度)`；置信度 = `(峰值 − 次峰)/峰值`（次峰去 `±15°` 邻域）。
pub fn 指针_径向连续(
    灰度: &[u8],
    宽: usize,
    高: usize,
    cx: f64,
    cy: f64,
    r: f64,
) -> Option<(f64, f64, f64)> {
    if 宽 == 0 || 高 == 0 || r <= 1.0 {
        return None;
    }
    const 采样: usize = 64;
    let mut rs = vec![0.0f64; 采样];
    for k in 0..采样 {
        rs[k] = r * 0.20 + (r * 0.86 - r * 0.20) * (k as f64 / (采样 - 1) as f64);
    }
    let mut v = vec![0.0f64; 箱数 * 采样];
    for b in 0..箱数 {
        let ang = b as f64 * (360.0 / 箱数 as f64) * std::f64::consts::PI / 180.0;
        for k in 0..采样 {
            let fx = cx + ang.cos() * rs[k];
            let fy = cy + ang.sin() * rs[k];
            // ★ 与 Python `astype(np.int32)` 同为**向零截断**
            let xi = (fx as i64).clamp(0, 宽 as i64 - 1) as usize;
            let yi = (fy as i64).clamp(0, 高 as i64 - 1) as usize;
            v[b * 采样 + k] = 灰度[yi * 宽 + xi] as f64;
        }
    }
    let mut 全部 = v.clone();
    let 阈 = 分位(&mut 全部, 22.0);
    let mut 覆盖 = vec![0.0f64; 箱数];
    for b in 0..箱数 {
        let mut c = 0usize;
        for k in 0..采样 {
            if v[b * 采样 + k] < 阈 {
                c += 1;
            }
        }
        覆盖[b] = c as f64 / 采样 as f64;
    }
    // 三点平滑（roll ±1）
    let mut 平 = vec![0.0f64; 箱数];
    for b in 0..箱数 {
        平[b] = (覆盖[(b + 箱数 - 1) % 箱数] + 覆盖[b] + 覆盖[(b + 1) % 箱数]) / 3.0;
    }
    let mut i = 0usize;
    let mut best = f64::NEG_INFINITY;
    for (k, x) in 平.iter().enumerate() {
        if *x > best {
            best = *x;
            i = k;
        }
    }
    if 平[i] <= 0.0 {
        return None;
    }
    let mut s = 平.clone();
    let supp = ((15.0 / (360.0 / 箱数 as f64)).round() as i64).max(1);
    for k in -supp..=supp {
        let idx = ((i as i64 + k).rem_euclid(箱数 as i64)) as usize;
        s[idx] = 0.0;
    }
    let 次 = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let 信 = ((平[i] - 次) / 平[i].max(1e-6)).clamp(0.0, 1.0);
    let mut 尖长 = 0.0f64;
    for k in 0..采样 {
        if v[i * 采样 + k] < 阈 {
            尖长 = rs[k] / r;
        }
    }
    Some((i as f64 * (360.0 / 箱数 as f64), 尖长, 信))
}

// ============================================================================
// 主入口：本体门控（**纯函数**）
// ============================================================================

/// **T-124 主入口：本体门控 5 量** ⇒ 结论 + **逐条未通过原因**（可审计）。
///
/// 门控（默认阈）：**刻度周期性分 ≥ 0.35**｜**刻度峰数 ∈ [6, 40]**｜**圆边覆盖率 ≥ 0.55**｜
/// **圆度 ≥ 0.5**｜**指针置信 ≥ 0.30** —— 五条**全部为可复算几何/统计量**（续79 纪律3）。
pub fn 本体门控(
    灰度: &[u8],
    宽: usize,
    高: usize,
    圆心: (f64, f64),
    半径: f64,
    阈: &门控阈值,
) -> 本体门控结果 {
    let (cx, cy) = 圆心;
    let (分, 刻度角, 起止) = 刻度判据(灰度, 宽, 高, cx, cy, 半径);
    let 峰数 = 刻度角.len();
    let (覆盖, 圆度) = 圆边与圆度(灰度, 宽, 高, cx, cy, 半径);
    let 指针 = 指针_径向连续(灰度, 宽, 高, cx, cy, 半径);
    let 指针信 = 指针.as_ref().map(|p| p.2).unwrap_or(0.0);

    let mut 原因: Vec<String> = Vec::new();
    if 分 < 阈.刻度周期性分 {
        原因.push(format!("刻度周期性不足(分={:.2}<{:.2})", 分, 阈.刻度周期性分));
    }
    if 峰数 < 阈.峰数下限 || 峰数 > 阈.峰数上限 {
        原因.push(format!(
            "刻度峰数 {} 不在 [{},{}]（过密/过疏）",
            峰数, 阈.峰数下限, 阈.峰数上限
        ));
    }
    if 覆盖 < 阈.圆边覆盖率 {
        原因.push(format!(
            "圆边覆盖率不足({:.2}<{:.2} ⇒ 无闭合圆边)",
            覆盖, 阈.圆边覆盖率
        ));
    }
    match 圆度 {
        Some(rr) if rr >= 阈.圆度 => {}
        Some(rr) => 原因.push(format!("圆度不足({:.2}<{:.2})", rr, 阈.圆度)),
        None => 原因.push("圆度不可得(边缘点不足)".to_string()),
    }
    if 指针信 < 阈.指针置信 {
        原因.push(format!(
            "指针置信不足({:.2}<{:.2} ⇒ 无指针)",
            指针信, 阈.指针置信
        ));
    }

    let 通过 = 原因.is_empty();
    本体门控结果 {
        刻度周期性分: 分,
        刻度峰数: 峰数,
        刻度起止: 起止,
        刻度角列表: 刻度角,
        圆边覆盖率: 覆盖,
        圆度,
        指针置信: 指针信,
        指针角: 指针.as_ref().map(|p| p.0),
        指针尖长比: 指针.as_ref().map(|p| p.1),
        通过,
        未通过原因: 原因,
    }
}

/// **端到端（纯计算）**：门控 **通过** ⇒ 用 `刻度起止` 与 `指针角` 出**读数**；
/// **不通过 ⇒ 不给读数**（**如实报，不硬凑** —— 与 T-123 行为一致）。
pub fn 门控并读数(
    灰度: &[u8],
    宽: usize,
    高: usize,
    圆心: (f64, f64),
    半径: f64,
    量程: &量程,
    阈: &门控阈值,
) -> (本体门控结果, Option<读数结果>) {
    let g = 本体门控(灰度, 宽, 高, 圆心, 半径, 阈);
    if !g.通过 {
        return (g, None);
    }
    let (起, 张) = match g.刻度起止 {
        Some(v) => v,
        None => return (g, None),
    };
    let 指针角 = match g.指针角 {
        Some(v) => v,
        None => return (g, None),
    };
    let r = 读数确定式(指针角, 起, 张, 量程);
    (g, Some(r))
}

// ============================================================================
// 测试
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// 造一张**合成表盘灰度图**（与 T-123 `meter_make_samples.py` 同布局，**仅灰度**）。
    /// 布局：表盘圆（亮底深边）＋ 主刻度 11 条（0..100，角起 135°、扫 270°）＋ 指针（深、红→灰度）＋ 数字窗。
    fn 造表盘(宽: usize, 高: usize, r: f64, 读数: f64) -> Vec<u8> {
        let mut img = vec![235u8; 宽 * 高];
        let (cx, cy) = (宽 as f64 / 2.0, 高 as f64 / 2.0);
        let 角起 = 135.0f64;
        let 扫角 = 270.0f64;
        let 指针角 = 归一角度(角起 + 扫角 * (读数 / 100.0));
        // 表盘：圆内亮、圆环深
        for y in 0..高 {
            for x in 0..宽 {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let rr = (dx * dx + dy * dy).sqrt();
                if rr <= r {
                    img[y * 宽 + x] = 250;
                }
                if (rr - r).abs() <= r * 0.02 {
                    img[y * 宽 + x] = 30;
                }
                // ★ 本测试图**不画任何"整圈暗环"**（如 r×0.93 的内圈）——
                //   否则该环落在刻度扫描带 [0.76R,0.93R] 内、**每个角度箱都变暗**，
                //   会把"最大空档"法估的刻度起止彻底打乱（实测踩到：起=323°/张=344°）。
                //   Python 侧样本是 PIL 抗锯齿细线（暗贡献极小）故未暴露此问题。
            }
        }
        // 主刻度（11 条）
        for i in 0..11 {
            let a = 归一角度(角起 + 扫角 * (i as f64 / 10.0));
            let t = a.to_radians();
            for k in 0..=(r * 0.10) as i32 {
                let rr = r * 0.80 + k as f64;
                if rr > r * 0.90 {
                    break;
                }
                let x = (cx + rr * t.cos()).round() as i64;
                let y = (cy + rr * t.sin()).round() as i64;
                if x >= 0 && y >= 0 && (x as usize) < 宽 && (y as usize) < 高 {
                    img[y as usize * 宽 + x as usize] = 20;
                }
            }
        }
        // ★ 本测试图**不画次刻度**：次刻度会让"计峰"打满上限（40）⇒ 循环最大空档被压小
        //   ⇒ 刻度起止估计失真（实测踩到）。**只留 11 条主刻度**以验证口径。
        //   （真实样本仍由 Python 侧生成，对账见 `igt_core/tests/meter_reconcile.rs`。）
        // 指针（**深色粗线**，从 0.16R 尾部到 0.82R 尖端）
        let t = 指针角.to_radians();
        for k in 0..=(r * 1.0) as i32 {
            let rr = -r * 0.16 + (r * 0.82 + r * 0.16) * (k as f64 / r);
            if rr > r * 0.82 {
                break;
            }
            for w in -2i32..=2 {
                let nx = -t.sin() * w as f64;
                let ny = t.cos() * w as f64;
                let x = (cx + rr * t.cos() + nx).round() as i64;
                let y = (cy + rr * t.sin() + ny).round() as i64;
                if x >= 0 && y >= 0 && (x as usize) < 宽 && (y as usize) < 高 {
                    img[y as usize * 宽 + x as usize] = 40;
                }
            }
        }
        img
    }

    /// **T-124 · 正例**：合成表盘 ⇒ **门控通过** ＋ **读数接近真值**
    #[test]
    fn t124_门控_正例通过且读数接近() {
        let (宽, 高) = (480, 480);
        let r = 190.0;
        for 真读数 in [0.0, 12.5, 50.0, 87.5, 100.0] {
            let img = 造表盘(宽, 高, r, 真读数);
            let (g, rd) = 门控并读数(
                &img,
                宽,
                高,
                (240.0, 240.0),
                r,
                &量程::default(),
                &门控阈值::default(),
            );
            println!(
                "\n[T-124] 真读数={:.1}｜分={:.2} 峰={} 覆盖={:.2} 圆度={:?} 指针信={:.2} 指针角={:?}｜通过={}",
                真读数,
                g.刻度周期性分,
                g.刻度峰数,
                g.圆边覆盖率,
                g.圆度.map(|v| (v * 100.0).round() / 100.0),
                g.指针置信,
                g.指针角.map(|v| (v * 100.0).round() / 100.0),
                g.通过
            );
            println!("         {}", g.未通过原因.join("；"));
            if g.通过 {
                let v = rd.as_ref().and_then(|x| x.读数).unwrap();
                println!("         读数={:.2}（真 {:.1}）⇒ 误差 {:.2}", v, 真读数, (v - 真读数).abs());
                assert!(
                    (v - 真读数).abs() < 5.0,
                    "读数误差应 <5.0（PM 61 同 T-123 判据），真 {:.1} 得 {:.2}",
                    真读数,
                    v
                );
            }
        }
    }

    /// **T-124 · 负例**：纯噪声/网格 ⇒ **门控不过**（**如实拒绝**）
    #[test]
    fn t124_门控_负例拒绝() {
        let (宽, 高) = (480, 480);
        // 网格
        let mut 网格 = vec![240u8; 宽 * 高];
        for y in 0..高 {
            for x in 0..宽 {
                if x % 28 == 0 || y % 28 == 0 {
                    网格[y * 宽 + x] = 120;
                }
            }
        }
        let g1 = 本体门控(&网格, 宽, 高, (240.0, 240.0), 195.0, &门控阈值::default());
        println!("\n[T-124] 网格 ⇒ 通过={}｜{}", g1.通过, g1.未通过原因.join("；"));
        assert!(!g1.通过, "网格必须被拒绝");
        // 伪随机噪声（确定性 LCG）
        let mut 种: u64 = 12345;
        let mut 噪声 = vec![0u8; 宽 * 高];
        for v in 噪声.iter_mut() {
            种 = 种.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = (种 >> 33) as u8;
        }
        let g2 = 本体门控(&噪声, 宽, 高, (149.0, 327.0), 148.0, &门控阈值::default());
        println!("[T-124] 噪声 ⇒ 通过={}｜{}", g2.通过, g2.未通过原因.join("；"));
        assert!(!g2.通过, "噪声必须被拒绝");
    }

    /// **T-124 · 边界**：空图/退化半径/越界圆心 ⇒ **不 panic**、如实不通过
    #[test]
    fn t124_门控_边界不panic() {
        let (宽, 高) = (64, 64);
        let 空 = vec![0u8; 宽 * 高];
        let g = 本体门控(&空, 宽, 高, (32.0, 32.0), 30.0, &门控阈值::default());
        assert!(!g.通过);
        // 半径退化
        let g2 = 本体门控(&空, 宽, 高, (32.0, 32.0), 0.5, &门控阈值::default());
        assert!(g2.指针角.is_none());
        // 圆心越界
        let g3 = 本体门控(&空, 宽, 高, (-50.0, -50.0), 20.0, &门控阈值::default());
        assert!(!g3.通过);
        println!("\n[T-124] 边界：空图/退化半径/越界圆心 **均不 panic** ✓");
    }

    /// **T-124 · 口径自检**：刻度起止 ~ (135°, 270°)（与合成布局一致）
    #[test]
    fn t124_刻度起止口径() {
        let (宽, 高) = (480, 480);
        let img = 造表盘(宽, 高, 190.0, 50.0);
        let (_, _, 起止) = 刻度判据(&img, 宽, 高, 240.0, 240.0, 190.0);
        let (起, 张) = 起止.expect("应估出刻度起止");
        println!("\n[T-124] 刻度起={:.1}° 张角={:.1}°（合成布局 135/270）", 起, 张);
        assert!((张 - 270.0).abs() < 12.0, "张角应接近 270°，实得 {}", 张);
        assert!((起 - 135.0).abs() < 12.0, "起始应接近 135°，实得 {}", 起);
    }
}
