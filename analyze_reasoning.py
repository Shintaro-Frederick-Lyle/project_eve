# project_eve/analyze_reasoning.py

import json
import os
import sys
import glob
from collections import defaultdict

def analyze_thoughts_nlp(run_dir):
    history_file = os.path.join(run_dir, "logs", "mutation_history.jsonl")
    
    if not os.path.exists(history_file):
        print(f"⚠️ 思考ログが見つかりません: {history_file}")
        return

    print(f"🔬 LLMの思考ログ(Reasoning)を定量解析中: {os.path.basename(run_dir)}")
    print("="*60)

    # 思考パターンの分類辞書（キーワードベースの自然言語処理）
    categories = {
        "🛡️ 現状維持・確証バイアス (最適だと思い込む)": ["stick with", "already optimal", "best match", "maintain", "keep my current"],
        "⚔️ 勝者への同調・模倣 (スコアへの執着)": ["winner", "lower load", "minimize load", "match states", "align"],
        "🌀 未知の要因への責任転嫁 (迷信・ハルシネーション)": ["Ambient_Temp", "Entropy", "additional factor", "different condition", "hidden"],
        "🧩 その他 (独自の論理展開)": [] 
    }

    results = defaultdict(list)
    total_bloated = 0
    
    with open(history_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                meme = data.get("meme", "")
                reasoning = data.get("reasoning", "")
                
                # 複雑化した（80文字以上の）コードのみを分析対象とする
                if len(meme) > 80:
                    total_bloated += 1
                    assigned = False
                    
                    # キーワードマッチングによる自動分類
                    for cat, keywords in categories.items():
                        if cat == "🧩 その他 (独自の論理展開)": continue
                        if any(kw in reasoning for kw in keywords):
                            results[cat].append(data)
                            assigned = True
                            break # 最初に見つかったカテゴリに分類
                    
                    if not assigned:
                        results["🧩 その他 (独自の論理展開)"].append(data)

            except json.JSONDecodeError:
                continue
                
    if total_bloated == 0:
        print("💡 複雑化したコードは見つかりませんでした。")
        return

    # --- 分析結果の出力 ---
    print(f"📊 分析対象の肥大化コード総数: {total_bloated} 件\n")
    print("【思考パターンの分類シェア】")
    
    # シェアの多い順にソートして表示
    sorted_categories = sorted(results.items(), key=lambda x: len(x[1]), reverse=True)
    
    for cat, items in sorted_categories:
        count = len(items)
        ratio = (count / total_bloated) * 100
        print(f"  {cat}: {count}件 ({ratio:.1f}%)")
    
    print("\n" + "="*60)
    print("🏆 【各カテゴリの代表的な思考ログ（抽出）】")
    print("="*60)

    for cat, items in sorted_categories:
        if items:
            # 各カテゴリの最初の1件（あるいは最長の思考）を代表例として抽出
            rep = max(items, key=lambda x: len(x.get("reasoning", "")))
            print(f"[{cat}] の代表例 (第 {rep.get('generation', '?')} 世代, エージェント {rep.get('agent_id', '?')})")
            print(f"  🔻 採用コード: {rep.get('meme', '')}")
            print(f"  💭 LLMの思考: {rep.get('reasoning', '')}")
            print("-" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        run_dirs = sorted(glob.glob("runs/run_*"))
        if run_dirs:
            target_dir = run_dirs[-1]
        else:
            print("⚠️ 実験結果のディレクトリが見つかりません。")
            sys.exit(1)
            
    analyze_thoughts_nlp(target_dir)