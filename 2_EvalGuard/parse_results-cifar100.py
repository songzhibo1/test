import os
import json
import pandas as pd
from pathlib import Path
import argparse
import re

# 针对 CIFAR-100 的结果路径
RESULTS_DIR = Path("results/surrogate_ft/cifar100")

def get_experiment_group(tag):
    """CIFAR-100 标签分类逻辑"""
    tag = tag.lower()
    if "_rw_sweep" in tag: return "Exp2_RWSweep"
    if "_d_sweep" in tag: return "Exp3_DeltaLogitSweep"
    if "_fp_hard" in tag: return "Exp4_FalsePositive"
    if tag.startswith("c100_"): return "Exp1_MainTable"
    return "Unknown"

def format_p(p):
    if p is None: return "N/A"
    return "0.0e+00" if p == 0.0 else f"{p:.1e}"

def format_shift(s):
    return f"{s:.4f}" if s is not None else "N/A"

def format_acc(a):
    return f"{a:.1%}" if isinstance(a, float) else "N/A"

def extract_metrics(json_path, design_mode):
    """提取 0%, 1%, 5%, 10% 的全量指标"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. 基础信息
    t_arch = data.get("teacher", {}).get("name", "Unknown")
    s_arch = data.get("student", {}).get("architecture", "Unknown")
    arch_pair = f"{t_arch} -> {s_arch}"
    trigger_mode = data.get("trigger_mode", "N/A").replace("_trigger", "").upper()
    t_acc = data.get("teacher", {}).get("accuracy", "N/A")  # 原始教师精度

    filename = json_path.name
    tag = filename.split("__tag_")[-1].replace(".json", "") if "__tag_" in filename else "N/A"
    exp_group = get_experiment_group(tag)

    # 2. 提取全量参数
    watermark = data.get("watermark_config", {})
    baseline = data.get("distillation_baseline", {})
    ft_conf = data.get("ft_config", {})

    record_common = {
        "Experiment": exp_group, "Tag": tag, "Trigger": trigger_mode, "Arch_Pair": arch_pair,
        "Dist_Ep": baseline.get("dist_epochs", "N/A"), "FT_Ep": ft_conf.get("ft_epochs", "N/A"),
        "Temp": baseline.get("temperature", "N/A"), "V_Temp": watermark.get("verify_temperature", "N/A"),
        "RW": watermark.get("r_w", "N/A"), "Delta": watermark.get("delta_logit", "N/A"),
        "Beta": watermark.get("beta", "N/A"), "Trig_Size": baseline.get("n_triggers_used", "N/A"),
        "T_Acc": format_acc(t_acc),
        "S_Acc(FT0%)": format_acc(baseline.get("surrogate_accuracy")),
    }

    # 排序辅助
    sort_T = int(re.search(r'_T(\d+)', tag).group(1)) if re.search(r'_T(\d+)', tag) else 0
    sort_param = float(re.search(r'sweep_([\d\.]+)', tag).group(1)) if re.search(r'sweep_([\d\.]+)', tag) else 0.0

    # 3. 🌟 修复 Design 提取键名：替换为真实的 EvalGuard 键
    b_all_designs = baseline.get("all_designs", {})
    if not b_all_designs:
        designs_to_extract = ["default"]
    elif design_mode == "all":
        # 修复：写入真正的计算设计名称
        designs_to_extract = ["mean_rest", "single_ctrl", "suspect_top1"]
    else:
        designs_to_extract = [design_mode]

    records = []
    for d_name in designs_to_extract:
        b_target = b_all_designs.get(d_name, baseline) if d_name != "default" else baseline
        
        # 3. 提取微调各阶段结果
        ft_steps = {0.01: "FT1%", 0.05: "FT5%", 0.1: "FT10%"}
        ft_results = {k: {} for k in ft_steps.keys()}
        for ft in data.get("ft_results", []):
            frac = ft.get("ft_fraction")
            if frac in ft_results:
                f_all_designs = ft.get("all_designs", {})
                f_target = f_all_designs.get(d_name, ft) if d_name != "default" else ft
                ft_results[frac] = {
                    "acc": format_acc(ft.get("accuracy")),
                    "shift": format_shift(f_target.get("confidence_shift")),
                    "p": format_p(f_target.get("p_value")),
                    "ver": f_target.get("verified", "N/A")
                }

        row = record_common.copy()
        row.update({
            "Design": d_name, "_sort_T": sort_T, "_sort_param": sort_param,
            # FT 0%
            "FT0%_Shift": format_shift(b_target.get("confidence_shift")),
            "FT0%_P": format_p(b_target.get("p_value")),
            "FT0%_Ver": b_target.get("verified", "N/A"),
            # FT 1%
            "FT1%_Acc": ft_results[0.01].get("acc", "N/A"),
            "FT1%_Shift": ft_results[0.01].get("shift", "N/A"),
            "FT1%_P": ft_results[0.01].get("p", "N/A"),
            "FT1%_Ver": ft_results[0.01].get("ver", "N/A"),
            # FT 5%
            "FT5%_Acc": ft_results[0.05].get("acc", "N/A"),
            "FT5%_Shift": ft_results[0.05].get("shift", "N/A"),
            "FT5%_P": ft_results[0.05].get("p", "N/A"),
            "FT5%_Ver": ft_results[0.05].get("ver", "N/A"),
            # FT 10%
            "FT10%_Acc": ft_results[0.1].get("acc", "N/A"),
            "FT10%_Shift": ft_results[0.1].get("shift", "N/A"),
            "FT10%_P": ft_results[0.1].get("p", "N/A"),
            "FT10%_Ver": ft_results[0.1].get("ver", "N/A"),
        })
        records.append(row)
    return records

def main():
    parser = argparse.ArgumentParser()
    # 🌟 修复 argparse 的选择列表
    parser.add_argument("--design", type=str, default="mean_rest", choices=["mean_rest", "single_ctrl", "suspect_top1", "all"])
    args = parser.parse_args()

    if not RESULTS_DIR.exists(): return
    json_files = list(RESULTS_DIR.rglob("*.json"))
    records = []
    for jf in json_files:
        try: records.extend(extract_metrics(jf, args.design))
        except Exception as e: print(f"⚠️ Error {jf.name}: {e}")

    if not records: return
    df = pd.DataFrame(records)
    
    # 多级排序
    df = df.sort_values(by=["Experiment", "Trigger", "Arch_Pair", "_sort_T", "_sort_param", "Design"])

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 10000)

    for exp_name, exp_group in df.groupby("Experiment", sort=False):
        print(f"\n{'█'*60} {exp_name} {'█'*60}")
        p = exp_group.iloc[0]
        print(f" ⚙️ [PARAMS] Dist_Ep: {p['Dist_Ep']} | FT_Ep: {p['FT_Ep']} | RW: {p['RW']} | Delta: {p['Delta']} | Beta: {p['Beta']} | V_Temp: {p['V_Temp']}")
        print("─"*350)

        for trig_name, trig_group in exp_group.groupby("Trigger", sort=False):
            print(f"\n▼ 触发器模式: 【 {trig_name} 】")
            for arch_pair, arch_group in trig_group.groupby("Arch_Pair", sort=False):
                print(f"  ↳ 模型对: [ {arch_pair} ]")
                
                # 🌟 终端展示全景：包含 T_Acc，以及各阶段的 Acc, Shift, P, Ver
                disp = [
                    "Tag", "Temp", "T_Acc", "S_Acc(FT0%)", 
                    "FT0%_Shift", "FT0%_P", "FT0%_Ver", 
                    "FT1%_Acc", "FT1%_Shift", "FT1%_P", "FT1%_Ver", 
                    "FT5%_Acc", "FT5%_Shift", "FT5%_P", "FT5%_Ver", 
                    "FT10%_Acc", "FT10%_Shift", "FT10%_P", "FT10%_Ver"
                ]
                if args.design == "all": disp.insert(1, "Design")
                print(arch_group[disp].to_string(index=False))
                print()

    # 🌟 导出 CSV
    output_df = df.drop(columns=['_sort_T', '_sort_param'])
    output_csv = RESULTS_DIR / f"summary_results_c100_{args.design}_ULTIMATE.csv"
    output_df.to_csv(output_csv, index=False)
    print(f"✅ CIFAR-100 结果已完美修复 Design 键名，并包含全量精度及 P值/Ver 验证指标。CSV 路径: {output_csv}")

if __name__ == "__main__":
    main()