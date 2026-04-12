import os
import json
import pandas as pd
from pathlib import Path
import argparse
import re

# 定义结果存放的根目录
RESULTS_DIR = Path("results/surrogate_ft/cifar10")

def get_experiment_group(tag):
    """修复版：使用模糊匹配，确保 trig_sweep 等实验不再被标记为 Unknown"""
    tag = tag.lower()
    if "c10_" in tag: return "Exp1_MainTable"
    if "rw_sweep" in tag: return "Exp2_RWSweep"
    if "d_sweep" in tag: return "Exp3_DeltaLogitSweep"
    if "beta_sweep" in tag: return "Exp4_BetaSweep"
    if "nq_sweep" in tag: return "Exp5_QuerySweep"
    if "trig_sweep" in tag: return "Exp6_TriggerSizeSweep"
    if "seed_" in tag: return "Exp7_MultiSeed"
    if "fp_hard" in tag: return "Exp8_FalsePositive"
    if "xarch_" in tag: return "Exp9_CrossArch"
    return "Unknown"

def format_p(p):
    if p is None: return "N/A"
    return "0.0e+00" if p == 0.0 else f"{p:.1e}"

def format_shift(s):
    return f"{s:.4f}" if s is not None else "N/A"

def format_acc(a):
    return f"{a:.1%}" if isinstance(a, float) else "N/A"

def extract_metrics(json_path, design_mode):
    """从单个 JSON 文件中提取核心指标"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. 提取基础标识信息
    teacher = data.get("teacher", {}).get("name", "N/A")
    student = data.get("student", {}).get("architecture", "N/A")
    arch_pair = f"{teacher} -> {student}"
    trigger_mode = data.get("trigger_mode", "N/A").replace("_trigger", "").upper()
    t_acc = data.get("teacher", {}).get("accuracy", "N/A")

    filename = json_path.name
    tag = filename.split("__tag_")[-1].replace(".json", "") if "__tag_" in filename else "N/A"
    exp_group = get_experiment_group(tag)

    # 2. 提取全局超参数（含 Beta, V_Temp, Epochs）
    watermark = data.get("watermark_config", {})
    baseline = data.get("distillation_baseline", {})
    ft_conf = data.get("ft_config", {})

    record_common = {
        "Experiment": exp_group,
        "Tag": tag,
        "Trigger": trigger_mode,
        "Arch_Pair": arch_pair,
        "Dist_Ep": baseline.get("dist_epochs", "N/A"),
        "FT_Ep": ft_conf.get("ft_epochs", "N/A"),
        "Temp": baseline.get("temperature", "N/A"),
        "V_Temp": watermark.get("verify_temperature", "N/A"),
        "RW": watermark.get("r_w", "N/A"),
        "Delta": watermark.get("delta_logit", "N/A"),
        "Beta": watermark.get("beta", "N/A"),
        "Trig_Size": baseline.get("n_triggers_used", "N/A"),
        "T_Acc": format_acc(t_acc),
        "S_Acc(FT0%)": format_acc(baseline.get("surrogate_accuracy")),
    }

    # 用于数字排序的辅助键
    sort_T = 0
    match_t = re.search(r'_T(\d+)', tag)
    if match_t: sort_T = int(match_t.group(1))
    sort_param = 0.0
    match_param = re.search(r'sweep_([\d\.]+)', tag)
    if match_param: sort_param = float(match_param.group(1))

    # 3. 🌟 修复 Bug：确保存取真实存在的 Design 键值
    b_all_designs = baseline.get("all_designs", {})
    if not b_all_designs:
        designs_to_extract = ["default_hard"]
    elif design_mode == "all":
        # 👉 替换为了真实的 EvalGuard 键名
        designs_to_extract = ["mean_rest", "single_ctrl", "suspect_top1"]
    else:
        designs_to_extract = [design_mode]

    records = []
    for d_name in designs_to_extract:
        # 获取目标 baseline 设计
        b_target = b_all_designs.get(d_name, baseline) if d_name != "default_hard" else baseline
        
        # 提取各个微调阶段 (0.01, 0.05, 0.1) 的 Acc/Shift/P/Ver
        ft_steps = {0.01: "FT1%", 0.05: "FT5%", 0.1: "FT10%"}
        ft_results = {k: {} for k in ft_steps.keys()}
        for ft in data.get("ft_results", []):
            frac = ft.get("ft_fraction")
            if frac in ft_results:
                f_all_designs = ft.get("all_designs", {})
                f_target = f_all_designs.get(d_name, ft) if d_name != "default_hard" else ft
                ft_results[frac] = {
                    "acc": format_acc(ft.get("accuracy")),
                    "shift": format_shift(f_target.get("confidence_shift")),
                    "p": format_p(f_target.get("p_value")),
                    "ver": f_target.get("verified", "N/A")
                }

        row = record_common.copy()
        row.update({
            "Design": d_name,
            "_sort_T": sort_T, "_sort_param": sort_param,
            
            # FT 0% 指标
            "FT0%_Shift": format_shift(b_target.get("confidence_shift")),
            "FT0%_P": format_p(b_target.get("p_value")),
            "FT0%_Ver": b_target.get("verified", "N/A"),
            
            # FT 1% 指标
            "FT1%_Acc": ft_results[0.01].get("acc", "N/A"),
            "FT1%_Shift": ft_results[0.01].get("shift", "N/A"),
            "FT1%_P": ft_results[0.01].get("p", "N/A"),
            "FT1%_Ver": ft_results[0.01].get("ver", "N/A"),  # 👈 补齐验证结果
            
            # FT 5% 指标
            "FT5%_Acc": ft_results[0.05].get("acc", "N/A"),
            "FT5%_Shift": ft_results[0.05].get("shift", "N/A"),
            "FT5%_P": ft_results[0.05].get("p", "N/A"),
            "FT5%_Ver": ft_results[0.05].get("ver", "N/A"),  # 👈 补齐验证结果
            
            # FT 10% 指标
            "FT10%_Acc": ft_results[0.1].get("acc", "N/A"),
            "FT10%_Shift": ft_results[0.1].get("shift", "N/A"),
            "FT10%_P": ft_results[0.1].get("p", "N/A"),
            "FT10%_Ver": ft_results[0.1].get("ver", "N/A"),
        })
        records.append(row)
    return records

def main():
    parser = argparse.ArgumentParser()
    # 🌟 修复 Bug：将 choices 更新为真实的键名
    parser.add_argument("--design", type=str, default="mean_rest", 
                        choices=["mean_rest", "single_ctrl", "suspect_top1", "all"])
    args = parser.parse_args()

    if not RESULTS_DIR.exists(): return
    json_files = list(RESULTS_DIR.rglob("*.json"))
    records = []
    for jf in json_files:
        try:
            records.extend(extract_metrics(jf, args.design))
        except Exception as e:
            print(f"⚠️ 解析错误 {jf.name}: {e}")

    if not records: return
    df = pd.DataFrame(records)
    
    # 按照 实验 > 触发器 > 模型对 > 温度 > 参数 排序
    df = df.sort_values(by=["Experiment", "Trigger", "Arch_Pair", "_sort_T", "_sort_param", "Design"])

    # 终端全屏展示优化
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 10000)
    pd.set_option('display.max_colwidth', None)

    for exp_name, exp_group in df.groupby("Experiment", sort=False):
        print(f"\n{'█'*60} {exp_name} {'█'*60}")
        
        # 打印各组实验的公共参数表头
        c_dist = exp_group['Dist_Ep'].mode()[0]
        c_ft = exp_group['FT_Ep'].mode()[0]
        c_rw = exp_group['RW'].mode()[0]
        c_vtemp = exp_group['V_Temp'].mode()[0]
        print(f" ⚙️ [PARAMS] Dist_Ep: {c_dist} | FT_Ep: {c_ft} | RW: {c_rw} | V_Temp: {c_vtemp}")
        print("─"*280)

        for trig_name, trig_group in exp_group.groupby("Trigger", sort=False):
            print(f"\n▼ 触发器模式: 【 {trig_name} 】")
            
            for arch_pair, arch_group in trig_group.groupby("Arch_Pair", sort=False):
                print(f"  ↳ 模型对: [ {arch_pair} ]")
                
                # 🌟 终端全量打印列表：涵盖 T_Acc，以及所有阶段的 P 和 Ver
                disp = [
                    "Tag", "Temp", "Delta", "Beta", "T_Acc", "S_Acc(FT0%)", 
                    "FT0%_Shift", "FT0%_P", "FT0%_Ver",
                    "FT1%_Acc", "FT1%_Shift", "FT1%_P", "FT1%_Ver",
                    "FT5%_Acc", "FT5%_Shift", "FT5%_P", "FT5%_Ver",
                    "FT10%_Acc", "FT10%_Shift", "FT10%_P", "FT10%_Ver"
                ]
                if args.design == "all": disp.insert(1, "Design")
                
                print(arch_group[disp].to_string(index=False))
                print()

    # 4. 保存 CSV
    output_df = df.drop(columns=['_sort_T', '_sort_param'])
    output_csv = RESULTS_DIR / f"summary_results_c10_{args.design}_ULTIMATE.csv"
    output_df.to_csv(output_csv, index=False)
    print(f"\n✅ 终端已全量打印。CSV 已同步保存所有精度、各种 Shift 指标及 Beta 参数。")
    print(f"📍 文件地址: {output_csv}")

if __name__ == "__main__":
    main()