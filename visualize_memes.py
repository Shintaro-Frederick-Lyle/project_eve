import os
import glob
import json
import numpy as np
from PIL import Image

def export_legend_html(meme_color_map, run_dir):
    """
    ミーム（コード）と色の対応表を美しいHTMLファイルとして出力する
    """
    html_path = os.path.join(run_dir, "meme_legend.html")
    
    html_content = """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="utf-8">
        <title>🧬 Meme Color Legend</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #1e1e1e; color: #e0e0e0; padding: 30px; }
            h2 { border-bottom: 2px solid #444; padding-bottom: 10px; }
            .legend-item { display: flex; align-items: center; margin-bottom: 15px; background: #2d2d2d; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
            .color-box { width: 40px; height: 40px; border-radius: 50%; margin-right: 20px; flex-shrink: 0; border: 2px solid #555; }
            .code-container { flex-grow: 1; overflow-x: auto; }
            pre { margin: 0; white-space: pre-wrap; word-wrap: break-word; font-family: 'Courier New', Courier, monospace; font-size: 14px; color: #9cdcfe; }
        </style>
    </head>
    <body>
        <h2>🧬 Evolutionary Regime: Meme Color Mapping</h2>
        <p>空間スナップショット（GIF/画像）内の各色がどの戦略（コード）を示しているかの対応表です。</p>
    """
    
    for meme, color in meme_color_map.items():
        html_content += f"""
        <div class="legend-item">
            <div class="color-box" style="background-color: {color};"></div>
            <div class="code-container">
                <pre><code>{meme}</code></pre>
            </div>
        </div>
        """
        
    html_content += """
    </body>
    </html>
    """

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"🎨 色とコードの対応表（凡例）をHTMLで出力しました: {html_path}")

def rgb_to_hex(rgb_array):
    return "#{:02x}{:02x}{:02x}".format(int(rgb_array[0]), int(rgb_array[1]), int(rgb_array[2]))

def create_meme_gif(run_dir):
    snapshots_dir = os.path.join(run_dir, "snapshots")
    npy_files = sorted(glob.glob(os.path.join(snapshots_dir, "meme_ids_gen_*.npy")))

    if not npy_files:
        print(f"⚠️ {snapshots_dir} にミームIDの記録が見つかりません。")
        return

    print(f"🎨 {len(npy_files)} 枚のフレームからミーム進化GIF（万華鏡）を生成中...")

    # --- 🌟 1. 初期戦略（Base Strategy）の数を動的に検知 🌟 ---
    first_frame = np.load(npy_files[0])
    # 第0世代に存在する最大のIDを基準に、基本戦略の数を特定
    num_base_strategies = int(np.max(first_frame)) + 1
    print(f"🔍 第0世代から {num_base_strategies} 種類の基本戦略を自動検知しました。")

    # --- 2. カラーパレットの動的生成 ---
    # 視認性の高い基本カラーを10色用意（これ以上増えてもフォールバックで対応）
    distinct_colors = [
        [0, 122, 255],   # 0: Blue (TFT等)
        [255, 59, 48],   # 1: Red
        [52, 199, 89],   # 2: Green
        [255, 149, 0],   # 3: Orange
        [175, 82, 222],  # 4: Purple
        [90, 200, 250],  # 5: Cyan
        [255, 45, 85],   # 6: Pink
        [255, 204, 0],   # 7: Yellow
        [142, 142, 147], # 8: Gray
        [162, 132, 94]   # 9: Brown
    ]
    
    base_colors = distinct_colors[:num_base_strategies]
    # もし基本戦略が10種類を超えた場合は、ランダムな色を追加して補完する
    while len(base_colors) < num_base_strategies:
        base_colors.append(np.random.randint(50, 200, size=3).tolist())

    # 突然変異用の極彩色パレット
    np.random.seed(42) 
    mutant_colors = np.random.randint(50, 255, size=(3000, 3)).tolist()
    color_map = np.array(base_colors + mutant_colors, dtype=np.uint8)

    # --- 🌟 3. 凡例（Legend）データの動的構築 🌟 ---
    meme_color_map = {}
    
    # 手動で名前をつけたい基本戦略がある場合はここに書く（なくても動きます）
    KNOWN_BASE_LABELS = {
    }

    # 基本戦略の登録
    for i in range(num_base_strategies):
        # 辞書に名前があればそれを、なければ "Initial Base Strategy X" とする
        label = KNOWN_BASE_LABELS.get(i, f"Initial Base Strategy")
        meme_color_map[f"[Base ID: {i}] {label}"] = rgb_to_hex(color_map[i])

    # 突然変異ログを読み込み、出現順にIDを割り当てる
    history_file = os.path.join(run_dir, "logs", "mutation_history.jsonl")
    if os.path.exists(history_file):
        unique_mutants = []
        seen_memes = set()
        with open(history_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    meme = data.get("meme", "").strip()
                    if meme and meme not in seen_memes:
                        seen_memes.add(meme)
                        unique_mutants.append(meme)
                except json.JSONDecodeError:
                    continue
        
        # 突然変異IDは自動検知した基本戦略の数 (num_base_strategies) からスタートする
        for i, meme in enumerate(unique_mutants):
            color_id = num_base_strategies + i
            if color_id < len(color_map):
                meme_color_map[f"[Mutant ID: {color_id}]\n{meme}"] = rgb_to_hex(color_map[color_id])
    
    export_legend_html(meme_color_map, run_dir)

    # --- 4. 画像の生成 ---
    frames = []
    for npy_file in npy_files:
        grid = np.load(npy_file)
        safe_grid = np.clip(grid, 0, len(color_map) - 1)
        rgb_grid = color_map[safe_grid]
        img = Image.fromarray(rgb_grid, 'RGB')
        img = img.resize((512, 512), Image.NEAREST) 
        frames.append(img)

    # --- 5. GIFおよび高画質静止画の書き出し ---
    visuals_dir = os.path.join(run_dir, "visuals")
    os.makedirs(visuals_dir, exist_ok=True)
    
    output_gif = os.path.join(visuals_dir, "meme_evolution.gif")
    if frames:
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0
        )
        frames[0].save(os.path.join(visuals_dir, "meme_gen_0000.png"))
        frames[-1].save(os.path.join(visuals_dir, "meme_gen_final.png"))
    
    print(f"✨ 究極の観測器が完了しました！")
    print(f"📁 GIFアニメと高画質静止画(PNG)を {visuals_dir} に保存しました。")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        runs = sorted(glob.glob("runs/run_*"))
        if not runs:
            print("エラー: runsディレクトリが見つかりません。")
            sys.exit(1)
        target_dir = runs[-1]
        
    create_meme_gif(target_dir)