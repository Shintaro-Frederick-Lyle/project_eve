# main.py

import argparse
import asyncio
from configs.default_config import get_default_config
from core.simulator import EveSimulator

def parse_args():
    parser = argparse.ArgumentParser(description="Project Eve Simulation")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.0,
                        help="代謝係数 λ (例: 0.0 / 0.001 / 0.01)")
    parser.add_argument("--generations", type=int, default=3000)
    parser.add_argument("--grid-size",   type=int, default=64)
    return parser.parse_args()


async def main():
    args = parse_args()
    config = get_default_config()
    
    # 🌟 フルスペック設定（64x64, 3000世代）を維持
    config["environment"]["grid_size"]  = args.grid_size
    config["environment"]["generations"] = args.generations
    config["evolution"]["mutation_rate"] = 0.01
    config["evolution"]["metabolic_rate"] = args.lam

    # 🌟 修正案に基づいたλ（代謝係数）の戦略的設定
    # Run A: 0.0    (基準群: 欺瞞的コード肥大化を誘発)
    # Run B: 0.001  (スイートスポット: 肥大化を抑え、多様性が最大化する「生命の縁」を狙う)
    # Run C: 0.01   (強圧力: 意味論的圧縮が完了し、極小ロジックが支配する世界)
    
    print(f"🌌 Project Eve: Run 開始 (λ={args.lam})")
    orchestrator = EveSimulator(config)
    await orchestrator.run_evolution()

    # 3. シミュレーション終了後の自動解析
    print("\n--- Starting Post-Simulation Analysis ---")
    try:
        from make_graph import plot_run_data
        from view_snapshot import save_all_snapshots
        from visualize_memes import create_meme_gif
        
        # 今回のRunディレクトリに対して解析を実行
        target_dir = orchestrator.run_manager.run_dir
        plot_run_data(target_dir)
        save_all_snapshots(target_dir)
        create_meme_gif(target_dir)
        
        print(f"\n✨ 全ての自動解析が完了しました。")
        print(f"成果物は {target_dir} を確認してください。")
    except Exception as e:
        print(f"⚠️ 解析中にエラーが発生しました: {e}")

if __name__ == "__main__":
    asyncio.run(main())