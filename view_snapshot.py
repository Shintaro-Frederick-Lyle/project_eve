# project_eve/view_snapshot.py

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
import io

def save_all_snapshots(run_dir):
    """
    snapshotsフォルダ内のnpyから、行動(赤青)の進化GIFと最初・最後のPNGを生成する。
    中間ファイルをディスクに書き出さない高速統合版。
    """
    snapshots_dir = os.path.join(run_dir, "snapshots")
    # actions_gen_*.npy を取得
    snapshot_files = sorted(glob.glob(os.path.join(snapshots_dir, "actions_gen_*.npy")))
    
    if not snapshot_files:
        print(f"⚠️ スナップショットが見つかりません: {snapshots_dir}")
        return

    print(f"📊 {len(snapshot_files)} 枚のフレームから行動進化(赤青)を解析中...")
    
    # visualsフォルダの作成
    visuals_dir = os.path.join(run_dir, "visuals")
    os.makedirs(visuals_dir, exist_ok=True)

    # カラーマップ設定 (0: 青-協力, 1: 赤-裏切り)
    cmap = ListedColormap(['#1f77b4', '#d62728'])

    frames = []
    
    for i, file_path in enumerate(snapshot_files):
        # 世代番号の抽出
        try:
            grid_data = np.load(file_path)
        except Exception as e:
            print(f"  [Warning] スキップ: {file_path} ({e})")
            continue
        gen_str = os.path.basename(file_path).split('_')[-1].replace('.npy', '')
        
        # 描画 (Matplotlibを使用してラベル付きの画像を生成)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(grid_data, cmap=cmap, interpolation='nearest')
        ax.set_title(f"Behavior Distribution - Gen {gen_str}", fontsize=12)
        ax.axis('off')
        
        # --- メモリ上でPNGに変換 ---
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img.copy()) # copy()しないとメモリ解放で消えるため注意
        
        # 最初(Gen 0)と最後(Final)だけは高画質PNGとして保存
        if i == 0:
            plt.savefig(os.path.join(visuals_dir, f"behavior_gen_0000.png"), dpi=300, bbox_inches='tight')
        elif i == len(snapshot_files) - 1:
            plt.savefig(os.path.join(visuals_dir, f"behavior_gen_final.png"), dpi=300, bbox_inches='tight')
            
        plt.close(fig)
        buf.close()

    # --- GIFの生成 ---
    if frames:
        output_gif = os.path.join(visuals_dir, "behavior_evolution.gif")
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=150,
            loop=0
        )
        print(f"✨ 行動解析が完了しました！ -> {visuals_dir}")
    else:
        print("❌ 画像フレームの生成に失敗しました。")

if __name__ == "__main__":
    import sys
    # コマンドラインから実行する場合
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        run_dirs = sorted(glob.glob("runs/run_*"))
        target_dir = run_dirs[-1] if run_dirs else None

    if target_dir:
        save_all_snapshots(target_dir)
    else:
        print("⚠️ フォルダが見つかりません。")