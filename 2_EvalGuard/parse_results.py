import glob
import json
import os
import csv

def main():
    search_path = "results/surrogate_ft/cifar10/**/*.json"
    json_files = sorted(glob.glob(search_path, recursive=True))

    if not json_files:
        print("没有找到 JSON 文件，请检查路径。")
        return

    # 表头新增了 Frac (微调比例)
    headers = ["Tag", "Mode", "T", "rw", "d", "Frac", "Acc", "Design", "Shift", "P-value", "V"]
    rows = []

    for path in json_files:
        filename = os.path.basename(path)
        with open(path, 'r', encoding='utf-8') as f:
            try: 
                d = json.load(f)
            except: 
                continue
        
        trig_mode = d.get("trigger_mode", "?")
        wc = d.get("watermark_config", {})
        rw, d_logit = wc.get("r_w", "-"), wc.get("delta_logit", "-")
        temp = d.get("distillation_baseline", {}).get("temperature", "-")
        
        if "_tag_" in filename:
            tag = filename.split("_tag_")[-1].replace(".json", "")
        else:
            tag = filename[:20]

        # 核心改进：遍历所有的 ft_results (0%, 1%, 5%, 10%)
        ft_results = d.get("ft_results", [])
        for ft_item in ft_results:
            frac = f"{ft_item.get('ft_fraction', 0.0)*100:.0f}%"
            acc = f"{ft_item.get('accuracy', 0.0)*100:.2f}%"
            
            all_designs = ft_item.get("all_designs", {})
            for design in ["single_ctrl", "mean_rest"]:
                res = all_designs.get(design, {})
                if not res: continue
                
                shift = res.get("confidence_shift", 0.0)
                pval = res.get("p_value", 1.0)
                verif = "YES" if res.get("verified", False) else "NO"
                
                # 【修复点】在这里提前把浮点数格式化为漂亮的字符串
                shift_str = f"{shift:+8.4f}"
                pval_str = f"{pval:8.1e}"
                
                # 全部转成纯文本格式存入列表
                rows.append([
                    str(tag), str(trig_mode), str(temp), str(rw), str(d_logit), 
                    frac, acc, design, shift_str, pval_str, verif
                ])

    # 打印 ASCII 表格 (全部使用字符串占位符 's')
    fmt_str = "{:<22s} | {:<12s} | {:<3s} | {:<4s} | {:<3s} | {:<5s} | {:<7s} | {:<12s} | {:>9s} | {:>9s} | {:<3s}"
    header_line = fmt_str.format(*headers)
    sep = "-" * len(header_line)
    
    print("\n" + header_line + "\n" + sep)
    for r in rows:
        print(fmt_str.format(*r))
    print(sep)

    # 存为 CSV
    csv_filename = "full_experiment_trajectory.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"\n[完成] 完整攻击轨迹已存至: {csv_filename}")

if __name__ == "__main__":
    main()