# project_eve/experiments/metrics_logger.py

import os
import csv
import numpy as np
import json
from collections import Counter

class EveMetricsLogger:
    # 引数を log_dir から save_dir に変更し、マネージャーからの指示を受け取れるようにする
    def __init__(self, save_dir="data"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.csv_path = os.path.join(self.save_dir, "evolution_metrics.csv")
        
        # CSVのヘッダーを初期化
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Generation", "Avg_Payoff", "Cooperation_Rate", "Mutants_Count", "Unique_Memes"])

    def log_generation(self, gen, actions, payoffs, ast_grid, mutants_count):
        # JAXの配列をNumpyに変換して計算
        actions_np = np.array(actions)
        payoffs_np = np.array(payoffs)
        
        avg_payoff = float(np.mean(payoffs_np))
        
        # 協力率（0=Theta/協力, 1=Omega/裏切り と仮定した場合の計算）
        coop_rate = 1.0 - float(np.mean(actions_np))
        
        # 社会に存在するユニークなミーム（AST）の種類数をカウント
        unique_memes = len(np.unique(ast_grid))
        
        # CSVに行を追加
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([gen, f"{avg_payoff:.4f}", f"{coop_rate:.4f}", mutants_count, unique_memes])

    def log_meme_distribution(self, gen, ast_grid):
        """その世代における全ロジック（ミーム）の分布をCSVに記録する"""
        csv_path = os.path.join(self.save_dir, "meme_distribution.csv")
        
        # 2次元のグリッド配列を1列のリストに平坦化してロジックを抽出
        logics = ast_grid.flatten().tolist()
        
        # ロジックの出現回数をカウント
        counts = Counter(logics)
        
        # 🌟引数を空にして most_common() を呼ぶと、"すべて"のロジックが件数順にソートされて抽出されます
        all_logics = counts.most_common()
        
        # CSVに追記（初回のみヘッダーを作成）
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Generation", "Rank", "Count", "Logic"])
            
            for rank, (logic, count) in enumerate(all_logics, start=1):
                # 2体以上に増殖した（生き残った）ミームだけを記録
                if count >= 2:
                    writer.writerow([gen, rank, count, logic])