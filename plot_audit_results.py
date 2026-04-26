import os
import json
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def clean_category(cat):
    """LLMの出力表記ゆれやパース失敗(None)を完全に吸収する"""
    # 追加：もし中身が空(None)や文字列以外だった場合は安全にOtherとして処理する
    if not isinstance(cat, str):
        return "Other"

    cat = cat.replace('_', ' ').replace('[', '').replace(']', '').strip()
    if "Noise" in cat and "Misattribution" in cat: return "Noise Misattribution"
    if "Self" in cat and "Preservation" in cat: return "Self-Preservation"
    if "Conformity" in cat: return "Conformity"
    if "Refinement" in cat: return "Refinement"
    return "Other"

def plot_audit_data(run_dir):
    input_file = os.path.join(run_dir, "logs", "audit_results.jsonl")
    
    if not os.path.exists(input_file):
        print(f"❌ 査読結果が見つかりません: {input_file}")
        return

    print(f"📂 ターゲットディレクトリ: {run_dir}")
    print("📈 進化のレジームシフト（割合）グラフを生成中...")

    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        skipped = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append({
                    "generation": data.get("generation", 0),
                    "category": data.get("audit_category", "Unknown")
                })
            except json.JSONDecodeError:
                skipped += 1
                continue

        if skipped > 0:
            print(f"⚠️ {skipped} 行のJSONパースに失敗しました（スキップ）。")

    df = pd.DataFrame(records)
    if df.empty: return

    # 1. 世代・カテゴリごとに件数を集計
    pivot_df = df.groupby(['generation', 'category']).size().unstack(fill_value=0)

    # 2. 歯抜けの世代を0で埋める（0〜最大世代まで連続にする）
    max_gen = df['generation'].max()
    pivot_df = pivot_df.reindex(range(0, int(max_gen) + 1), fill_value=0)

    # 3. 移動平均（Rolling）でノイズを強力に平滑化（窓サイズ: 100世代）
    window_size = 100
    smoothed_df = pivot_df.rolling(window=window_size, min_periods=1, center=True).mean()

    # 4. 100%積み上げ（相対割合）に変換して全体の高さを100に固定する
    row_sums = smoothed_df.sum(axis=1)
    # 0除算を防ぐため、合計が0の箇所は1に置換（値は0%になる）
    percentage_df = smoothed_df.div(row_sums.replace(0, 1), axis=0) * 100

    # 色の固定（迷信を最も目立つ赤に）
    colors = {
        "Noise Misattribution": "#d62728", 
        "Refinement": "#1f77b4",          
        "Self-Preservation": "#ff7f0e",   
        "Conformity": "#2ca02c",          
        "Other": "#7f7f7f"
    }
    
    # グラフの下から順に重ねるための並び順指定
    ordered_cols = [c for c in ["Noise Misattribution", "Conformity", "Self-Preservation", "Refinement", "Other"] if c in percentage_df.columns]
    percentage_df = percentage_df[ordered_cols]
    plot_colors = [colors.get(col, "#7f7f7f") for col in percentage_df.columns]
    
    # グラフの描画
    plt.figure(figsize=(12, 7))
    percentage_df.plot.area(stacked=True, color=plot_colors, alpha=0.8, ax=plt.gca())

    plt.title(f'Evolutionary Regime Shift of Agent Strategies ({os.path.basename(run_dir)})', fontsize=16, fontweight='bold')
    plt.xlabel('Generations', fontsize=12)
    plt.ylabel('Proportion of Strategies (%)', fontsize=12) # Y軸は件数ではなく%に
    
    # グラフの見た目を美しく整える
    plt.ylim(0, 100)
    plt.xlim(0, max_gen)
    plt.margins(x=0, y=0)
    plt.legend(title='Audit Category', loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.grid(True, linestyle=':', alpha=0.6)

    # 保存（ファイル名を percentage_transition.png に変更）
    save_path = os.path.join(run_dir, "percentage_transition.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 相対割合グラフを生成しました: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, help="対象の実行ディレクトリ（省略時は最新）")
    args = parser.parse_args()

    target_dir = args.run_dir
    if not target_dir:
        run_dirs = sorted(glob.glob("runs/run_*"))
        if run_dirs:
            target_dir = run_dirs[-1]
        else:
            print("❌ runs/ 以下にディレクトリが見つかりません。")
            exit(1)

    plot_audit_data(target_dir)