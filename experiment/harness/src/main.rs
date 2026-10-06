#![allow(non_snake_case)]
//! IGT 物理小脑 R4 模型臂（精简 local-cli 版）
//!
//! 与项目原始 `igt_agent/src/bin/r3_claim.rs` 的 local-cli 分支同一逻辑，
//! 但去掉了 local-server / openai-compat / 多模态图片等需要 reqwest 网络依赖的路径，
//! 只保留：读锚包 → 构造 prompt → 以子进程方式调用 llama.cpp 的 llama-cli →
//! 提取并校验 ClaimForCheck（支持反馈闭环与无锚零点）。
mod r3_claim;

use r3_claim::{
    判读目标, 回灌反馈, 模型产出Claim, 模型产出Claim_带反馈, 模型产出Claim_无锚, 模型产出文本_无锚,
    ClaimForCheck,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut 锚包路径 = String::new();
    let mut 输出路径 = String::new();
    let mut llama_cli = String::new();
    let mut model = String::new();
    let mut model_name = String::from("Qwen2.5-7B-Instruct");
    let mut quant = String::from("Q4_K_M");
    let mut feedback_json = String::new();
    let mut 题面 = String::new(); // --prompt-only：无锚题面（D 臂零点）
    let mut 自由题面 = String::new(); // --prompt-only-free：自由文本无锚（创造臂）
    let mut target_obs = String::new();
    let mut 方向枚举 = String::new();
    let mut raw_fail_dir = String::new();
    let mut max_token = 512usize;
    let mut k_max = 3usize;
    let mut timeout_sec = 120u64;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--anchor" => { 锚包路径 = args.get(i + 1).cloned().unwrap_or_default(); i += 2; }
            "--out" => { 输出路径 = args.get(i + 1).cloned().unwrap_or_default(); i += 2; }
            "--llama-cli" => { llama_cli = args.get(i + 1).cloned().unwrap_or_default(); i += 2; }
            "--model" => { model = args.get(i + 1).cloned().unwrap_or_default(); i += 2; }
            "--model-name" => { model_name = args.get(i + 1).cloned().unwrap_or_default(); i += 2; }
            "--quant" => { quant = args.get(i + 1).cloned().unwrap_or_default(); i += 2; }
            "--feedback" => { feedback_json = args.get(i + 1).cloned().unwrap_or_default(); i += 2; }
            "--prompt-only" => { 题面 = args.get(i + 1).cloned().unwrap_or_default(); i += 2; }
            "--prompt-only-free" => { 自由题面 = args.get(i + 1).cloned().unwrap_or_default(); i += 2; }
            "--target-observable" => { target_obs = args.get(i + 1).cloned().unwrap_or_default(); i += 2; }
            "--direction-enum" => { 方向枚举 = args.get(i + 1).cloned().unwrap_or_default(); i += 2; }
            "--raw-fail-dir" => { raw_fail_dir = args.get(i + 1).cloned().unwrap_or_default(); i += 2; }
            "--max-token" => { max_token = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(512); i += 2; }
            "--k-max" => { k_max = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(3); i += 2; }
            "--timeout" => { timeout_sec = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(120); i += 2; }
            _ => i += 1,
        }
    }

    if llama_cli.is_empty() || model.is_empty() {
        eprintln!(
            "用法: r3-claim (--anchor 锚包.json | --prompt-only 无锚题面 | --prompt-only-free 自由题面) \
             --out out.json --llama-cli llama-cli.exe --model model.gguf \
             [--feedback feedback.json] [--target-observable .. --direction-enum ..] \
             [--model-name ..] [--quant ..] [--max-token N] [--timeout N]"
        );
        std::process::exit(2);
    }
    let raw目录: Option<&str> = if raw_fail_dir.is_empty() { None } else { Some(&raw_fail_dir) };

    // ---- 自由文本无锚（创造臂）----
    if !自由题面.is_empty() {
        match 模型产出文本_无锚(&llama_cli, &model, &自由题面, max_token, timeout_sec, raw目录) {
            Ok(文本) => 落盘文本(&输出路径, &文本),
            Err(e) => { eprintln!("模型臂失败: {}", e); std::process::exit(1); }
        }
        return;
    }

    // ---- 无锚题面（D 臂零点）----
    if !题面.is_empty() {
        match 模型产出Claim_无锚(
            &llama_cli, &model, &题面, &model_name, &quant, max_token, timeout_sec, raw目录,
        ) {
            Ok(claim) => 落盘claim(&输出路径, &claim),
            Err(e) => { eprintln!("模型臂失败: {}", e); std::process::exit(1); }
        }
        return;
    }

    // ---- 锚包模式 ----
    if 锚包路径.is_empty() {
        eprintln!("锚包模式需 --anchor；或用 --prompt-only / --prompt-only-free");
        std::process::exit(2);
    }
    let 锚包json = match std::fs::read_to_string(&锚包路径) {
        Ok(s) => s,
        Err(e) => { eprintln!("读锚包失败: {}", e); std::process::exit(1); }
    };
    let 目标: Option<判读目标> = if !target_obs.is_empty() && !方向枚举.is_empty() {
        Some(判读目标 { observable: target_obs.clone(), 方向枚举: 方向枚举.clone() })
    } else {
        None
    };
    let 反馈: Vec<回灌反馈> = if feedback_json.is_empty() {
        Vec::new()
    } else {
        match serde_json::from_str(&feedback_json) {
            Ok(f) => f,
            Err(e) => { eprintln!("feedback JSON 解析失败: {}", e); std::process::exit(1); }
        }
    };
    let result = if !反馈.is_empty() || 目标.is_some() {
        模型产出Claim_带反馈(
            &llama_cli, &model, &锚包json, &反馈, &model_name, &quant,
            max_token, timeout_sec, 目标.as_ref(), raw目录,
        )
    } else {
        模型产出Claim(
            &llama_cli, &model, &锚包json, &model_name, &quant,
            max_token, k_max, timeout_sec, raw目录,
        )
    };
    match result {
        Ok(claim) => 落盘claim(&输出路径, &claim),
        Err(e) => { eprintln!("模型臂失败: {}", e); std::process::exit(1); }
    }
}

fn 落盘文本(输出路径: &str, 文本: &str) {
    if 输出路径.is_empty() {
        println!("{}", 文本);
    } else {
        std::fs::write(输出路径, 文本).expect("写文本失败");
        println!("已写入 {}", 输出路径);
    }
}

fn 落盘claim(输出路径: &str, claim: &ClaimForCheck) {
    let json = serde_json::to_string_pretty(claim).unwrap_or_default();
    if 输出路径.is_empty() {
        println!("{}", json);
    } else {
        std::fs::write(输出路径, json).expect("写 claim 文件失败");
        println!("Claim 已写入 {}", 输出路径);
    }
}
