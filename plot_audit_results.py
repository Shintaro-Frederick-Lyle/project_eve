import os
import json
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_audit_data(run_dir):
    input_file = os.path.join(run_dir, "logs", "audit_results.jsonl")
    
    if not os.path.exists(input_file):
        print(f"❌ 査読結果が見つかりません: {input_file}")
        return

    print(f"📂 ターゲットディレクトリ: {run_dir}")
    print("📈 グラフを生成中...")

    # JSONLを読み込んでPandasデータフレームに変換
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            records.append({
                "generation": data.get("generation", 0),
                "category": data.get("audit_category", "Unknown")
            })

    df = pd.DataFrame(records)
    
    if df.empty:
        print("❌ データが空です。")
        return

    # 世代ごと、カテゴリごとに件数を集計してピボットテーブルを作成
    pivot_df = df.groupby(['generation', 'category']).size().unstack(fill_value=0)

    # グラフの描画設定
    plt.figure(figsize=(12, 7))
    
    # カテゴリごとに色を指定（Noise Misattributionを赤系にして目立たせる）
    colors = {
        "Noise Misattribution": "#d62728", # 迷信（赤：危険/エラー）
        "Refinement": "#1f77b4",          # 洗練（青：論理的）
        "Self-Preservation": "#ff7f0e",   # 自己保存（オレンジ）
        "Conformity": "#2ca02c",          # 同調（緑）
        "Other": "#7f7f7f"
    }
    
    # 存在する列だけを色付きで描画
    plot_colors = [colors.get(col, "#7f7f7f") for col in pivot_df.columns]
    
    pivot_df.plot.area(stacked=True, color=plot_colors, alpha=0.8, ax=plt.gca())

    plt.title(f'Evolution of Agent Reasoning ({os.path.basename(run_dir)})', fontsize=14, fontweight='bold')
    plt.xlabel('Generations', fontsize=12)
    plt.ylabel('Number of Mutated Codes (Length > 80)', fontsize=12)
    plt.legend(title='Audit Category', loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)

    # 保存
    save_path = os.path.join(run_dir, "audit_transition.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 グラフを自動生成しました: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="査読結果のカテゴリ推移をグラフ化します。")
    parser.add_argument("--run_dir", type=str, help="対象の実行ディレクトリ（省略時は最新のものを自動選択）")
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