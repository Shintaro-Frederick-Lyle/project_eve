# project_eve/make_graph.py

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_run_data(run_dir):
    """指定されたRunディレクトリのデータを可視化する"""
    csv_path = os.path.join(run_dir, "data", "evolution_metrics.csv")
    
    if not os.path.exists(csv_path):
        print(f"⚠️ CSVが見つかりません: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    required_cols = ['Generation', 'Cooperation_Rate']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"⚠️ CSVに必要な列がありません: {missing}")
        print(f"   実際の列: {list(df.columns)}")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(df['Generation'], df['Cooperation_Rate'], label='Cooperation Rate', color='#2ca02c', linewidth=2)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% Threshold')
    plt.title(f'Evolution of Cooperation Rate ({os.path.basename(run_dir)})')
    plt.xlabel('Generations')
    plt.ylabel('Cooperation Rate')
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    save_path = os.path.join(run_dir, "cooperation_transition.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() # メモリ解放のため閉じる
    print(f"📊 グラフを自動生成しました: {save_path}")

# 直接実行もできるようにしておく
if __name__ == "__main__":
    import glob
    run_dirs = sorted(glob.glob("runs/run_*"))
    if run_dirs:
        plot_run_data(run_dirs[-1])