# project_eve/experiments/run_manager.py

import os
import json
import shutil
import glob  # 【追加】ファイル検索用
from datetime import datetime
import numpy as np

class EveRunManager:
    """Project Eveの実験結果を自動で仕分け・保存するマネージャー"""
    
    def __init__(self, config_dict, base_dir="runs"):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, f"run_{self.run_id}")
        self.data_dir = os.path.join(self.run_dir, "data")
        self.logs_dir = os.path.join(self.run_dir, "logs")
        self.snapshots_dir = os.path.join(self.run_dir, "snapshots")
        
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)
        
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
            
        print(f"📁 [EveRunManager] 実験ディレクトリを作成しました: {self.run_dir}")
        self.backup_source_code()

    def append_mutations(self, gen, new_memes_dict):
        """世代ごとに発生した新規ミームと思考ログをJSONL形式で追記保存する"""
        path = os.path.join(self.logs_dir, "mutation_history.jsonl")
        
        with open(path, 'a', encoding='utf-8') as f:
            for agent_id, data in new_memes_dict.items():
                record = {
                    "generation": gen,
                    "agent_id": int(agent_id),
                    "meme": data["ast"],         
                    "reasoning": data["reasoning"]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def backup_source_code(self):
        """【強化】実行時の全ソースコード（物理法則と認知ロジック）を完全保存する"""
        code_backup_dir = os.path.join(self.run_dir, "src_backup")
        os.makedirs(code_backup_dir, exist_ok=True)

        # プロジェクトのルートディレクトリ（EveRunManagerから見た親の階層）を取得
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        # 重要なサブディレクトリとルートの.pyファイルを全てコピー対象にする
        # 仮想環境(eve_env)や実験結果(runs)は除外する
        for py_file in glob.glob(os.path.join(project_root, "**/*.py"), recursive=True):
            if any(ex in py_file for ex in ["eve_env", "runs", "__pycache__", ".jax_cache"]):
                continue
            
            # 相対パスを維持してコピー先を決定
            rel_path = os.path.relpath(py_file, project_root)
            dest_path = os.path.join(code_backup_dir, rel_path)
            
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(py_file, dest_path)
        
        print(f"✨ [EveRunManager] プロジェクト全域のスナップショットを完了しました。")

    def save_snapshot(self, gen, actions_grid, meme_ids_grid=None):
        path_actions = os.path.join(self.snapshots_dir, f"actions_gen_{gen:04d}.npy")
        np.save(path_actions, np.array(actions_grid))
        
        if meme_ids_grid is not None:
            path_ids = os.path.join(self.snapshots_dir, f"meme_ids_gen_{gen:04d}.npy")
            np.save(path_ids, np.array(meme_ids_grid))

    def save_meme_registry(self, registry_dict):
        path = os.path.join(self.run_dir, "meme_registry.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(registry_dict, f, indent=4, ensure_ascii=False)