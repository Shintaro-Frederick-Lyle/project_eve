import itertools

ALLOWED_VALUES = {"'X'", "'Y'", "'High'", "'Low'", "True", "False"}

def evaluate_condition(cond_str, env):
    """条件式をPythonの論理式に変換して評価する"""
    py_cond = cond_str.replace("State-X", "'X'").replace("State-Y", "'Y'")
    py_cond = py_cond.replace(" AND ", " and ").replace(" OR ", " or ").replace("NOT (", "not (")
    py_cond = py_cond.replace("High", "'High'")
    py_cond = py_cond.replace("Low",  "'Low'")
    
    try:
        return eval(py_cond, {}, env)
    except Exception as e:
        return False # パース失敗時の安全策

def evaluate_ast(ast_str, env, depth=0, max_depth=50):
    """ASTを再帰的に評価し、最終的な出力(X or Y)を決定する"""
    if depth > max_depth:
        return "X"  # フォールバック
    
    ast_str = ast_str.strip()
    if ast_str in ("State-X", "'X'"): return "X"
    if ast_str in ("State-Y", "'Y'"): return "Y"
    
    if ast_str.startswith("If ("):
        # カッコのネストを考慮してブロックを抽出するヘルパー
        def get_content(s, start_idx):
            depth = 0
            for i in range(start_idx, len(s)):
                if s[i] == '(': depth += 1
                elif s[i] == ')':
                    depth -= 1
                    if depth == 0:
                        return s[start_idx+1:i], i+1
            return "", -1

        cond_str, next_idx = get_content(ast_str, ast_str.find("("))
        
        then_idx = ast_str.find("Then (", next_idx)
        then_str, next_idx = get_content(ast_str, then_idx + 5)
        
        else_idx = ast_str.find("Else (", next_idx)
        else_str, _ = get_content(ast_str, else_idx + 5)
        
        if evaluate_condition(cond_str, env):
            return evaluate_ast(then_str, env, depth+1, max_depth)
        else:
            return evaluate_ast(else_str, env, depth+1, max_depth)
    return "X"

def analyze_phenotypes(meme_registry):
    """全ミームの表現型を解析し、ダミー情報の有無を判定する"""
    print("--- 🧬 Phenotype (表現型) 解析を開始 ---")
    
    # 起こりうる全環境パターンの生成 (2^6 = 64通り)
    var_names = ["Self_Prev_State", "Peer_Prev_State", "Initial_Boot", "Entropy_Flag", "Ambient_Temp", "Network_Latency"]
    var_values = [
        ["'X'", "'Y'"],        # Self
        ["'X'", "'Y'"],        # Peer
        [True, False],         # Boot
        [True, False],         # Entropy
        [200, 400],            # Temp (300以下/以上)
        ["'Low'", "'High'"]    # Latency
    ]
    
    env_combinations = list(itertools.product(*var_values))
    
    phenotype_map = {} # { 表現型シグネチャ: [遺伝子型のリスト] }
    
    for ast_str in meme_registry.keys():
        signature = ""
        # 64通りの環境すべてでテストし、行動の羅列（シグネチャ）を作る
        for vals in env_combinations:
            env = dict(zip(var_names, vals))
            result = evaluate_ast(ast_str, env)
            signature += result
            
        if signature not in phenotype_map:
            phenotype_map[signature] = []
        phenotype_map[signature].append(ast_str)

    # 結果の出力
    total_genotypes = len(meme_registry)
    total_phenotypes = len(phenotype_map)
    
    print(f"解析した固有の遺伝子（文字列）数: {total_genotypes}")
    print(f"実際の行動（表現型）のパターン数: {total_phenotypes}\n")
    
    if total_phenotypes < total_genotypes:
        bloat_ratio = (1 - (total_phenotypes / total_genotypes)) * 100
        print(f"🚨 【コード肥大化（ジャンクDNA）を検出】 冗長率: {bloat_ratio:.1f}%")
        print("イヴたちはダミーの条件式を生成し、見せかけの多様性を装っています。\n")
    else:
        print("✅ 【真の多様性を検出】 すべての遺伝子は異なる振る舞いを持っています。\n")

    # 主要な表現型の上位3つを表示
    sorted_phenotypes = sorted(phenotype_map.items(), key=lambda x: len(x[1]), reverse=True)
    print("🏆 主要な表現型（行動パターン）Top 3:")
    for i, (sig, asts) in enumerate(sorted_phenotypes[:3]):
        # 代表的な（一番短い）ASTを表示
        shortest_ast = min(asts, key=len)
        print(f"  {i+1}位 (シェア {len(asts)}個): {shortest_ast}")

if __name__ == "__main__":
    import sys
    import os
    import glob
    import json

    # アーキテクト仕様のディレクトリ自動探索システム
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        # 引数がない場合は、最も新しい runs/run_xxx ディレクトリを自動選択
        runs = sorted(glob.glob("runs/run_*"))
        if not runs:
            print("🚨 エラー: runsディレクトリが見つかりません。")
            sys.exit(1)
        target_dir = runs[-1]
        
    print(f"📁 ターゲットディレクトリ: {target_dir}")
    json_path = os.path.join(target_dir, "meme_registry.json")

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            real_registry = json.load(f)
        
        # 本物のデータで表現型解析を実行！
        analyze_phenotypes(real_registry)
    else:
        print(f"🚨 エラー: {json_path} が見つかりません。")
        print("main.py の保存処理が完了しているか確認してください。")