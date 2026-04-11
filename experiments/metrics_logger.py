# project_eve/experiments/metrics_logger.py

import os
import csv
import numpy as np

class EveMetricsLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, "evolution_metrics.csv")
        
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