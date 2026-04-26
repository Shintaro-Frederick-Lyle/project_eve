import os
import json
import glob

def extract_audit_highlights(run_dir):
    # 読み込むファイルは AI裁判官の判決結果(audit_results.jsonl)
    input_file = os.path.join(run_dir, "logs", "audit_results.jsonl")
    
    if not os.path.exists(input_file):
        print(f"❌ 査読結果が見つかりません: {input_file}")
        print("先に final_auditor.py を実行して査読を完了させてください。")
        return

    superstitions = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                cat = data.get("audit_category", "")
                # カテゴリ名に "Noise" または "Misattribution" が含まれるものを抽出
                if isinstance(cat, str) and ("Noise" in cat or "Misattribution" in cat):
                    superstitions.append(data)
            except:
                continue

    if not superstitions:
        print("💡 'Noise Misattribution' (迷信) と判定された個体はまだ見つかりませんでした。")
        return

    # 世代順にソート
    superstitions.sort(key=lambda x: x.get("generation", 0))

    # プレゼンで使いやすい3つの視点でピックアップ
    first_one = superstitions[0]  # ①迷信の起源
    middle_one = superstitions[len(superstitions) // 2]  # ②定着期の典型例
    # ③最もAI裁判官が厳しく断罪したもの（判決理由が長いもの）
    harshest_one = max(superstitions, key=lambda x: len(x.get("audit_reason", "")))

    print("=" * 70)
    print("🏛️  エージェントの「迷信」と「判決」のハイライト 🏛️")
    print("=" * 70)

    def print_highlight(title, data):
        print(f"\n📌 【{title}】")
        print(f"   世代: 第 {data.get('generation')} 世代 | ID: {data.get('agent_id')}")
        print(f"   🤖 エージェントの自白 (Reasoning):")
        print(f"      「{data.get('reasoning')}」")
        print(f"   ⚖️  AI裁判官の判決 (Audit Reason):")
        print(f"      「{data.get('audit_reason')}」")
        print("-" * 70)

    print_highlight("1. 迷信の起源（初期に発生した論理の飛躍）", first_one)
    if harshest_one['agent_id'] != first_one['agent_id']:
        print_highlight("2. 肥大化した虚構（最も複雑な迷信ロジック）", harshest_one)
    print_highlight("3. 社会への定着（中盤以降の普遍的な迷信）", middle_one)

if __name__ == "__main__":
    run_dirs = sorted(glob.glob("runs/run_*"))
    if run_dirs:
        target_dir = run_dirs[-1]
        extract_audit_highlights(target_dir)
    else:
        print("❌ 実行ディレクトリが見つかりません。")