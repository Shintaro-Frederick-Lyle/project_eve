# project_eve/experiments/metrics_logger.py

import os
import csv
import numpy as np
import json
from collections import Counter

class EveMetricsLogger:
    def __init__(self, save_dir="data"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.csv_path = os.path.join(self.save_dir, "evolution_metrics.csv")
        
        # 🌟 ヘッダーに Avg_AST_Blocks を追加
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Generation", "Avg_Payoff", "Cooperation_Rate", "Mutants_Count", "Unique_Memes", "Avg_AST_Blocks"])

    # 🌟 simulator.py が送ってくるすべての引数を受け取れるように修正
    def log_generation(self, gen, actions, payoffs, ast_grid, mutants_count, new_memes_dict=None, unique_asts_count=0, avg_ast_len=0.0):
        actions_np = np.array(actions)
        payoffs_np = np.array(payoffs)
        
        avg_payoff = float(np.mean(payoffs_np))
        coop_rate = float(np.mean(actions_np))
        
        # 🌟 新しいメトリクスもCSVに書き込む
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([gen, f"{avg_payoff:.4f}", f"{coop_rate:.4f}", mutants_count, unique_asts_count, f"{avg_ast_len:.1f}"])

    def log_meme_distribution(self, gen, ast_grid):
        csv_path = os.path.join(self.save_dir, "meme_distribution.csv")
        logics = ast_grid.flatten().tolist()
        counts = Counter(logics)
        all_logics = counts.most_common()
        
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Generation", "Rank", "Count", "Logic"])
            for rank, (logic, count) in enumerate(all_logics, start=1):
                if count >= 2:
                    writer.writerow([gen, rank, count, logic])