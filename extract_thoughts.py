# project_eve/extract_thoughts.py

import json
import os
import sys
import glob

def analyze_thoughts(run_dir):
    """指定されたRunディレクトリの思考ログを抽出し、分析する"""
    history_file = os.path.join(run_dir, "logs", "mutation_history.jsonl")
    
    if not os.path.exists(history_file):
        print(f"⚠️ 思考ログが見つかりません: {history_file}")
        return

    print(f"🧠 LLMの特異な思考プロセスを抽出中: {os.path.basename(run_dir)}")
    print("="*60)

    found_interesting = False
    
    with open(history_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                gen = data.get("generation", "Unknown")
                agent_id = data.get("agent_id", "Unknown")
                meme = data.get("meme", "")
                reasoning = data.get("reasoning", "")
                
                # 抽出条件: コードが複雑化（ネスト）しているものをピックアップ
                # 基本的な "If (...) Then (X) Else (Y)" は60文字前後。80文字以上なら肥大化と判定。
                if len(meme) > 80:
                    print(f"🧬 [第 {gen} 世代] エージェント {agent_id} の複雑化")
                    print(f"  🔺 採用コード: {meme}")
                    print(f"  💭 LLMの思考: {reasoning}")
                    print("-" * 60)
                    found_interesting = True
            except json.JSONDecodeError:
                continue
                
    if not found_interesting:
        print("💡 特にコードが急激に肥大化した変異は見つかりませんでした。")

# --- コマンドラインからの直接実行サポート ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        run_dirs = sorted(glob.glob("runs/run_*"))
        if run_dirs:
            target_dir = run_dirs[-1]
            print(f"📂 対象ディレクトリを自動選択しました: {target_dir}")
        else:
            print("⚠️ 実験結果のディレクトリ (runs/run_*) が見つかりません。")
            sys.exit(1)
            
    analyze_thoughts(target_dir)