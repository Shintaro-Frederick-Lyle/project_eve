# project_eve/animate_snapshots.py

import os
import glob
from PIL import Image

def create_evolution_animation(run_dir):
    """
    保存されたPNG画像を読み込み、進化の過程をGIFアニメーションとして出力する。
    """
    # 探索先ディレクトリ（必要に応じて変更してください）
    snapshots_dir = os.path.join(run_dir, "snapshots")
    
    # 画像ファイルを取得
    # 例: visual_gen_0.png, visual_gen_50.png
    files = glob.glob(os.path.join(snapshots_dir, "visual_gen_*.png"))
    
    # snapshotsフォルダに見つからない場合は、run_dir直下も探す
    if not files:
        files = glob.glob(os.path.join(run_dir, "visual_gen_*.png"))

    if not files:
        print(f"⚠️ {run_dir} 内に画像 (visual_gen_*.png) が見つかりません。")
        return

    # ファイル名から世代数を抽出して、正しい数値順にソートする関数
    # 例: "visual_gen_100.png" -> 100
    def extract_gen(filepath):
        filename = os.path.basename(filepath)
        gen_str = filename.replace("visual_gen_", "").replace(".png", "")
        return int(gen_str)

    # 1, 10, 2 などの文字列順にならないよう、数値でソート
    files.sort(key=extract_gen)

    print(f"🎥 {len(files)} 枚の画像をGIFに結合しています...")

    # 全ての画像を開く
    images = [Image.open(f) for f in files]

    # GIFとして保存
    output_path = os.path.join(run_dir, "evolution_history.gif")
    
    # 最初の画像をベースに、残りの画像をフレームとして追加
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=200,  # 1フレームの表示時間 (ミリ秒)。200ms = 1秒間に5世代
        loop=0         # 0は無限ループ
    )
    
    print(f"✨ アニメーションの生成が完了しました！ -> {output_path}")

def get_latest_run_dir(base_dir="runs"):
    """runsフォルダの中から最新の実験フォルダを自動で見つける"""
    run_dirs = glob.glob(os.path.join(base_dir, "run_*"))
    if not run_dirs:
        return None
    return sorted(run_dirs)[-1]

if __name__ == "__main__":
    latest_run = get_latest_run_dir()
    if latest_run:
        print(f"🔍 最新の実験データ ({latest_run}) を解析します。")
        create_evolution_animation(latest_run)
    else:
        print("⚠️ runs フォルダが見つからないか、実験データがありません。")