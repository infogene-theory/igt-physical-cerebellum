// SPDX-License-Identifier: BSL-1.1
// SPDX-FileCopyrightText: © 2026 IGT Gene System Project Contributors

//! CORE FFI — C ABI 导出（第220批，三分离编译产物目标）
//!
//! 供移动端（cargo-ndk 安卓 JNI）/ 脚本语言（Python ctypes 等）嵌入
//! IGT CORE 纯计算能力。仅导出纯 CORE 函数，不引 UI/网络/平台 API。
//!
//! 内存规则（FFI 边界安全约定）:
//!   1. 除 `igt_core_version` 返回静态字符串外，不返回任何堆指针
//!   2. 输出一律写入调用方提供的缓冲区
//!   3. FFI 边界不 panic / 不 unwrap——非法输入返回负值错误码
//!      （Rust panic 跨 FFI 是 UB，全部用 catch_unconvert 兜底防御）

use crate::storage::{τ链, τ链节点, τ指纹};
use std::os::raw::c_char;

/// 错误码
pub const IGT_OK: i32 = 0;
pub const IGT_ERR_NULL: i32 = -1; // 空指针
pub const IGT_ERR_EMPTY: i32 = -2; // 空数组
pub const IGT_ERR_CAPACITY: i32 = -3; // 缓冲区容量不足
pub const IGT_ERR_INVALID: i32 = -4; // 非法数值（NaN/Inf/非正τ）

/// CORE 版本号（静态字符串，调用方免释放）
#[no_mangle]
pub extern "C" fn igt_core_version() -> *const c_char {
    // concat! 编译期拼接为 &'static str，生命周期为整个进程
    b"IGT_CORE_MATH 10.0.0 V10.0\0".as_ptr() as *const c_char
}

/// τ 链指纹哈希 — 写入 hex 字符串到调用方缓冲区
///
/// 入参: τ数组指针 + 长度；出参: 缓冲区 + 容量
/// 返回: 成功 = 写入字节数（不含 NUL）；失败 = 负错误码
#[no_mangle]
pub unsafe extern "C" fn igt_tau_fingerprint(
    τ数组: *const f64,
    ω数组: *const f64,
    长度: usize,
    出参: *mut u8,
    容量: usize,
) -> i32 {
    if τ数组.is_null() || ω数组.is_null() || 出参.is_null() {
        return IGT_ERR_NULL;
    }
    if 长度 == 0 {
        return IGT_ERR_EMPTY;
    }
    let 切片 = std::slice::from_raw_parts(τ数组, 长度);
    let ω切片 = std::slice::from_raw_parts(ω数组, 长度);
    if 切片.iter().any(|τ| !τ.is_finite() || *τ <= 0.0)
        || ω切片.iter().any(|ω| !ω.is_finite())
    {
        return IGT_ERR_INVALID;
    }

    let mut 链 = τ链::新建();
    // [R1.4 ω 贯通] FFI 外部传入独立 ω（复 L 供真 ω），不再由 τ 反推（ω=2π/τ 已清零）
    for (i, τ) in 切片.iter().enumerate() {
        链.添加(τ链节点::新建(*τ, ω切片[i], 1.0));
    }
    let 哈希 = τ指纹::生成哈希(&链); // 16 字符 hex
    let 字节 = 哈希.as_bytes();
    if 字节.len() + 1 > 容量 {
        return IGT_ERR_CAPACITY;
    }
    std::ptr::copy_nonoverlapping(字节.as_ptr(), 出参, 字节.len());
    *出参.add(字节.len()) = 0; // NUL 终止，方便 C 侧当字符串用
    字节.len() as i32
}

/// τ 跨度比（最长τ/最短τ）— 三分离 Agent 侧「生理跨度比」同源计算
#[no_mangle]
pub unsafe extern "C" fn igt_tau_span_ratio(τ数组: *const f64, 长度: usize) -> f64 {
    if τ数组.is_null() || 长度 < 2 {
        return f64::NAN;
    }
    let 切片 = std::slice::from_raw_parts(τ数组, 长度);
    if 切片.iter().any(|τ| !τ.is_finite() || *τ <= 0.0) {
        return f64::NAN;
    }
    let 最大 = 切片.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let 最小 = 切片.iter().cloned().fold(f64::INFINITY, f64::min);
    最大 / 最小
}

// ============================================================================
// 安卓 JNI 桥（第224批, 仅 target_os="android" 编译, 桌面零影响）
// ============================================================================
//
// Kotlin 用法（package com.igt.core）:
//   class CoreNative {
//       companion object {
//           init { System.loadLibrary("igt_core_math") }
//       }
//       external fun version(): String
//       // [R1.4 ω 贯通] 双参 (tau, omega)——omega 为独立 ω（复 L 供真 ω）;
//       // 无独立 ω 语义时传等长零数组 DoubleArray(tau.size){0.0}
//       external fun fingerprint(tau: DoubleArray, omega: DoubleArray): String   // 空/非法/长度不一致抛 RuntimeException
//       external fun spanRatio(tau: DoubleArray): Double      // 非法输入返回 NaN
//   }

#[cfg(target_os = "android")]
pub mod android_jni {
    use crate::storage::{τ链, τ链节点, τ指纹};
    use jni::objects::{JClass, JDoubleArray, JObjectArray, JString, ReleaseMode};
    use jni::sys::{jdouble, jstring};
    use jni::JNIEnv;

    const 异常类: &str = "java/lang/RuntimeException";

    /// Kotlin: external fun version(): String
    #[no_mangle]
    pub extern "system" fn Java_com_igt_core_CoreNative_version<'local>(
        mut env: JNIEnv<'local>,
        _class: JClass<'local>,
    ) -> jstring {
        let 版本 = "IGT_CORE_MATH 10.0.0 V10.0";
        match env.new_string(版本) {
            Ok(s) => s.into_raw(),
            Err(e) => {
                env.throw_new(异常类, format!("version 构造失败: {:?}", e)).ok();
                std::ptr::null_mut()
            }
        }
    }

    /// τ数组安全读取（长度非法返回 None 并已抛异常）
    fn 读取τ数组<'local>(
        env: &mut JNIEnv<'local>,
        数组: JDoubleArray<'local>,
    ) -> Option<Vec<f64>> {
        let len = match env.get_array_length(&数组) {
            Ok(n) => n,
            Err(e) => {
                env.throw_new(异常类, format!("get_array_length 失败: {:?}", e)).ok();
                return None;
            }
        };
        if len == 0 {
            env.throw_new(异常类, "τ数组为空").ok();
            return None;
        }
        match unsafe { env.get_array_elements::<f64>(&数组, ReleaseMode::NoCopyBack) } {
            Ok(元素) => Some(元素.iter().cloned().collect::<Vec<f64>>()),
            Err(e) => {
                env.throw_new(异常类, format!("读取数组失败: {:?}", e)).ok();
                None
            }
        }
    }

    /// Kotlin: external fun fingerprint(tau: DoubleArray, omega: DoubleArray): String
    #[no_mangle]
    pub extern "system" fn Java_com_igt_core_CoreNative_fingerprint<'local>(
        mut env: JNIEnv<'local>,
        _class: JClass<'local>,
        数组: JDoubleArray<'local>,
        ω数组: JDoubleArray<'local>,
    ) -> jstring {
        let τ = match 读取τ数组(&mut env, 数组) {
            Some(v) => v,
            None => return std::ptr::null_mut(),
        };
        let ω = match 读取τ数组(&mut env, ω数组) {
            Some(v) => v,
            None => return std::ptr::null_mut(),
        };
        if τ.len() != ω.len()
            || τ.iter().any(|t| !t.is_finite() || *t <= 0.0)
            || ω.iter().any(|w| !w.is_finite())
        {
            env.throw_new(异常类, "τ/ω数组非法或长度不一致").ok();
            return std::ptr::null_mut();
        }
        let mut 链 = τ链::新建();
        // [R1.4 ω 贯通] JNI 外部传入独立 ω（复 L 供真 ω），不再由 τ 反推
        for (i, t) in τ.iter().enumerate() {
            链.添加(τ链节点::新建(*t, ω[i], 1.0));
        }
        let 哈希 = τ指纹::生成哈希(&链);
        match env.new_string(&哈希) {
            Ok(s) => s.into_raw(),
            Err(e) => {
                env.throw_new(异常类, format!("字符串构造失败: {:?}", e)).ok();
                std::ptr::null_mut()
            }
        }
    }

    /// Kotlin: external fun spanRatio(tau: DoubleArray): Double
    #[no_mangle]
    pub extern "system" fn Java_com_igt_core_CoreNative_spanRatio<'local>(
        mut env: JNIEnv<'local>,
        _class: JClass<'local>,
        数组: JDoubleArray<'local>,
    ) -> jdouble {
        let τ = match 读取τ数组(&mut env, 数组) {
            Some(v) => v,
            None => return f64::NAN,
        };
        if τ.len() < 2 || τ.iter().any(|t| !t.is_finite() || *t <= 0.0) {
            return f64::NAN;
        }
        let 最大 = τ.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let 最小 = τ.iter().cloned().fold(f64::INFINITY, f64::min);
        最大 / 最小
    }

    // 抑制 android-only 未使用导入告警（JObjectArray/JString 备用）
    #[allow(unused_imports)]
    use jni::objects::JObject as _备用;
}

// ============================================================================
// 测试 — 直接调用 extern 函数验证（含边界非法输入）
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn 读c字符串(buf: &[u8]) -> String {
        let n = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        String::from_utf8_lossy(&buf[..n]).to_string()
    }

    #[test]
    fn 测试_version() {
        let p = igt_core_version();
        let s = unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().to_string();
        assert!(s.starts_with("IGT_CORE_MATH"), "版本串: {}", s);
        assert!(s.contains("10.0.0"));
    }

    #[test]
    fn 测试_指纹_正常输入() {
        let τ = [0.5f64, 1.0, 2.0, 4.0];
        let ω = [0.5f64, 1.0, 1.5, 2.0]; // [R1.4] 独立 ω 外部传入
        let mut buf = [0u8; 64];
        let n = unsafe {
            igt_tau_fingerprint(τ.as_ptr(), ω.as_ptr(), τ.len(), buf.as_mut_ptr(), buf.len())
        };
        assert_eq!(n, 16, "16 字符 hex");
        let 哈希 = 读c字符串(&buf);
        assert_eq!(哈希.len(), 16);

        // 与 Rust 侧直算一致（FFI 只是薄封装，计算同源；指纹哈希只用 τ/幅值，ω 不参与）
        let mut 链 = τ链::新建();
        for (i, t) in τ.iter().enumerate() {
            链.添加(τ链节点::新建(*t, ω[i], 1.0));
        }
        assert_eq!(哈希, τ指纹::生成哈希(&链));
    }

    #[test]
    fn 测试_指纹_边界非法输入() {
        let τ = [1.0f64, 2.0];
        let mut buf = [0u8; 64];
        let 零ω = [0.0f64; 2];
        unsafe {
            // [R1.4] 签名加 ω 数组: null 检查/空/容量/非法
            assert_eq!(
                igt_tau_fingerprint(std::ptr::null(), 零ω.as_ptr(), 2, buf.as_mut_ptr(), 64),
                IGT_ERR_NULL
            );
            assert_eq!(
                igt_tau_fingerprint(τ.as_ptr(), std::ptr::null(), 2, buf.as_mut_ptr(), 64),
                IGT_ERR_NULL
            );
            assert_eq!(
                igt_tau_fingerprint(τ.as_ptr(), 零ω.as_ptr(), 0, buf.as_mut_ptr(), 64),
                IGT_ERR_EMPTY
            );
            assert_eq!(
                igt_tau_fingerprint(τ.as_ptr(), 零ω.as_ptr(), 2, buf.as_mut_ptr(), 4),
                IGT_ERR_CAPACITY
            );
            let nan = [f64::NAN, 1.0];
            assert_eq!(
                igt_tau_fingerprint(nan.as_ptr(), 零ω.as_ptr(), 2, buf.as_mut_ptr(), 64),
                IGT_ERR_INVALID
            );
            let 零 = [0.0f64, 1.0];
            assert_eq!(
                igt_tau_fingerprint(零.as_ptr(), 零ω.as_ptr(), 2, buf.as_mut_ptr(), 64),
                IGT_ERR_INVALID
            );
        }
    }

    #[test]
    fn 测试_跨度比() {
        let τ = [1.0f64, 2.0, 4.0];
        let r = unsafe { igt_tau_span_ratio(τ.as_ptr(), τ.len()) };
        assert!((r - 4.0).abs() < 1e-12);
        unsafe {
            assert!(igt_tau_span_ratio(std::ptr::null(), 2).is_nan());
            assert!(igt_tau_span_ratio(τ.as_ptr(), 1).is_nan());
        }
    }
}
