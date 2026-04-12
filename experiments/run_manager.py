# project_eve/experiments/run_manager.py

import os
import json
import shutil
from datetime import datetime
import numpy as np

class EveRunManager:
    """Project Eveの実験結果を自動で仕分け・保存するマネージャー"""
    
    def __init__(self, config_dict, base_dir="runs"):
        # タイムスタンプで一意のRun IDを生成
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 今回の実験専用のディレクトリ構造を作成
        self.run_dir = os.path.join(base_dir, f"run_{self.run_id}")
        self.data_dir = os.path.join(self.run_dir, "data")
        self.logs_dir = os.path.join(self.run_dir, "logs")
        self.snapshots_dir = os.path.join(self.run_dir, "snapshots")
        
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)
        
        # 実験条件（コンフィグ）をJSONとして自動保存
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
            
        print(f"📁 [RunManager] 実験ディレクトリを作成しました: {self.run_dir}")
        self.backup_source_code()

    def save_log(self, text_content, filename="evolution_log.txt"):
        """テキストをlogsディレクトリに保存"""
        path = os.path.join(self.logs_dir, filename)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(text_content + "\n")

    def append_mutations(self, gen, new_memes_dict):
        """世代ごとに発生した新規ミームだけをJSONL形式で追記保存する"""
        path = os.path.join(self.logs_dir, "mutation_history.jsonl")
        
        # 'a' (append) モードで開いて、末尾に追記していく
        with open(path, 'a', encoding='utf-8') as f:
            for agent_id, meme_str in new_memes_dict.items():
                record = {
                    "generation": gen,
                    "agent_id": int(agent_id),
                    "meme": meme_str
                }
                # 1行のJSON文字列として書き込む
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def backup_source_code(self):
        """実行時の物理法則（コード）ごと冷凍保存する"""
        code_backup_dir = os.path.join(self.run_dir, "src_backup")
        os.makedirs(code_backup_dir, exist_ok=True)
        # 実行時の main.py と grid_jax.py をバックアップ（ファイルが存在する場合のみ）
        if os.path.exists("main.py"):
            shutil.copy2("main.py", code_backup_dir)
        if os.path.exists("env/grid_jax.py"):
            os.makedirs(os.path.join(code_backup_dir, "env"), exist_ok=True)
            shutil.copy2("env/grid_jax.py", os.path.join(code_backup_dir, "env/grid_jax.py"))

    def save_snapshot(self, gen, actions_grid):
        """特定の世代のグリッドの物理状態(行動)を保存する"""
        path = os.path.join(self.snapshots_dir, f"actions_gen_{gen:04d}.npy")
        np.save(path, np.array(actions_grid))