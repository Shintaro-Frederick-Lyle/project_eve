# project_eve/view_snapshot.py

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def save_all_snapshots(run_dir):
    """snapshotsフォルダ内のすべての.npyファイルを画像化して保存する"""
    snapshots_dir = os.path.join(run_dir, "snapshots")
    snapshot_files = sorted(glob.glob(os.path.join(snapshots_dir, "actions_gen_*.npy")))
    
    if not snapshot_files:
        print(f"⚠️ スナップショットが見つかりません: {snapshots_dir}")
        return

    print(f"🖼️  全 {len(snapshot_files)} 枚の画像を生成中...")
    
    # 共通のカラーマップ設定
    cmap = ListedColormap(['#1f77b4', '#d62728']) # 赤:裏切り, 青:協力

    for file_path in snapshot_files:
        # ファイル名から世代番号を抽出 (例: actions_gen_0150.npy -> 0150)
        gen_str = os.path.basename(file_path).split('_')[-1].replace('.npy', '')
        
        grid_data = np.load(file_path)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(grid_data, cmap=cmap, interpolation='nearest')
        plt.title(f"Spatial Distribution - Generation {gen_str}", fontsize=14)
        plt.axis('off')
        
        # 保存先を snapshot_gen_XXXX.png にする
        save_path = os.path.join(run_dir, f"visual_gen_{gen_str}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✨ すべての可視化が完了しました。")

if __name__ == "__main__":
    # 手動実行時は最新のRunディレクトリを対象にする
    run_dirs = sorted(glob.glob("runs/run_*"))
    if run_dirs:
        save_all_snapshots(run_dirs[-1])