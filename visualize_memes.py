import os
import glob
import numpy as np
from PIL import Image

def create_meme_gif(run_dir):
    """保存されたミームIDの配列から、多色アニメーションGIFを超高速に生成する"""
    snapshots_dir = os.path.join(run_dir, "snapshots")
    npy_files = sorted(glob.glob(os.path.join(snapshots_dir, "meme_ids_gen_*.npy")))

    if not npy_files:
        print(f"⚠️ {snapshots_dir} にミームIDの記録 (meme_ids_gen_*.npy) が見つかりません。")
        print("※この機能はフェーズ1実装後に行われた新しいシミュレーションでのみ動作します。")
        return

    print(f"🎨 {len(npy_files)} 枚のフレームからミーム進化GIF（万華鏡）を生成中...")

    # --- 1. 万華鏡のカラーパレットを定義 ---
    # 第0世代の4つの基本戦略には、意味を持たせた固定のイメージカラーを割り当てる
    base_colors = [
        [0, 0, 255],     # ID 0: TFT (青 - 堅牢な防壁)
        [255, 0, 0],     # ID 1: Anti-TFT (赤 - 攻撃的侵略)
        [0, 255, 255],   # ID 2: All-X (水色 - 無垢なる協力)
        [255, 165, 0]    # ID 3: All-Y (オレンジ - 狂気の裏切り)
    ]

    # ID 4以降（突然変異で生まれた新種）には、ランダムな極彩色を割り当てる
    # ※同じIDには常に同じ色がつくように乱数シードを固定
    np.random.seed(42) 
    mutant_colors = np.random.randint(50, 255, size=(2000, 3)).tolist()
    
    # パレットを統合
    color_map = np.array(base_colors + mutant_colors, dtype=np.uint8)

    # --- 2. 画像の生成 ---
    frames = []
    for npy_file in npy_files:
        # (64, 64) のID配列をロード
        grid = np.load(npy_file)
        
        # IDがパレットの数を超えないように安全処理
        safe_grid = np.clip(grid, 0, len(color_map) - 1)
        
        # IDの配列を、RGBのピクセルデータに一瞬で変換 (64, 64, 3)
        rgb_grid = color_map[safe_grid]
        
        # PIL画像に変換し、人間が見やすいように拡大 (512x512)
        # ※ピクセルアートのようなパキッとした質感を残すため、NEAREST補間を使用
        img = Image.fromarray(rgb_grid, 'RGB')
        img = img.resize((512, 512), Image.NEAREST) 
        frames.append(img)

    # --- 3. GIFおよび高画質静止画の書き出し ---
    visuals_dir = os.path.join(run_dir, "visuals")
    os.makedirs(visuals_dir, exist_ok=True)
    
    # 1. 万華鏡GIFの保存
    output_gif = os.path.join(visuals_dir, "meme_evolution.gif")
    if not frames:
        print("⚠️ 有効なフレームがありません。GIF を生成できませんでした。")
        return
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )
    
    # 2. 研究論文・資料用の高画質PNG保存（創世と終末のみ抽出）
    if frames:
        frames[0].save(os.path.join(visuals_dir, "meme_gen_0000.png"))
        frames[-1].save(os.path.join(visuals_dir, "meme_gen_final.png"))
    
    print(f"✨ 究極の観測器が完了しました！")
    print(f"📁 GIFアニメと高画質静止画(PNG)を {visuals_dir} に保存しました。")

if __name__ == "__main__":
    import sys
    # コマンドライン引数でディレクトリを指定できるようにする
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        # 引数がない場合は、最も新しい runs/run_xxx ディレクトリを自動選択
        runs = sorted(glob.glob("runs/run_*"))
        if not runs:
            print("エラー: runsディレクトリが見つかりません。")
            sys.exit(1)
        target_dir = runs[-1]
        
    create_meme_gif(target_dir)