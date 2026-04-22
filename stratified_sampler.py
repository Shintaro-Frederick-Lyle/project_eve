import json
import random
from collections import defaultdict
import os
import glob
import argparse

SAMPLES_PER_PHASE = 400

def run_stratified_sampling(run_dir):
    input_file = os.path.join(run_dir, "logs", "mutation_history.jsonl")
    output_file = os.path.join(run_dir, "logs", "analysis_targets.jsonl")

    if not os.path.exists(input_file):
        print(f"❌ ファイルが見つかりません: {input_file}")
        return

    data_by_gen = defaultdict(list)
    
    print(f"📂 ターゲットディレクトリ: {run_dir}")
    print("📖 ログを読み込み中...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            gen = entry.get("generation", 0)
            data_by_gen[gen].append(entry)

    all_gens = sorted(data_by_gen.keys())
    if not all_gens:
        print("❌ エラー: ログが空です。")
        return

    # 全世代を3つのフェーズ（初期・中期・後期）に分割
    mid_idx = len(all_gens) // 3
    late_idx = (len(all_gens) // 3) * 2
    
    phases = {
        "Early (初期)": all_gens[:mid_idx],
        "Middle (中期)": all_gens[mid_idx:late_idx],
        "Late (後期)": all_gens[late_idx:]
    }

    sampled_data = []
    
    for phase_name, gens in phases.items():
        phase_pool = []
        for g in gens:
            phase_pool.extend(data_by_gen[g])
        
        # 意味のある変化（肥大化したコード: 80文字以上など）を優先的に抽出
        important_pool = [d for d in phase_pool if len(d.get("meme", "")) > 80]
        
        if len(important_pool) < SAMPLES_PER_PHASE:
            print(f"⚠️ {phase_name} の対象が不足しています（{len(important_pool)}件）。全件採用します。")
            sampled_data.extend(important_pool)
        else:
            sampled_data.extend(random.sample(important_pool, SAMPLES_PER_PHASE))
        
        print(f"✅ {phase_name} フェーズから {min(len(important_pool), SAMPLES_PER_PHASE)} 件抽出しました。")

    # サンプリング結果を保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in sampled_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"🚀 サンプリング完了: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="シミュレーションログの層化サンプリングを実行します。")
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

    run_stratified_sampling(target_dir)