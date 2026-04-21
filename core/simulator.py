# core/simulator.py

import os
import jax
import jax.numpy as jnp
import numpy as np
import time
import asyncio
import re

# プロジェクトのディレクトリ構成に合わせてインポートを調整してください
from router.llm_gateway import HeterogeneousGateway
from experiments.run_manager import EveRunManager
from env.grid_jax import SpatialPDEnv
from cognition.ast_parser import MemeCompiler
from experiments.logger import EveMetricsLogger

class EveSimulator:
    def __init__(self, config):
        self.config = config # 完全な辞書はRunManagerでの保存用に保持
        
        # --- 🌟 設定値の自動展開（手入力を省略するスマートな設計の継承） ---
        self.grid_size = config["environment"]["grid_size"]
        self.generations = config["environment"]["generations"]
        self.num_agents = self.grid_size * self.grid_size
        self.mutation_rate = config["evolution"]["mutation_rate"]
        self.metabolic_rate = config["evolution"]["metabolic_rate"] # 追加
        
        # モジュールの初期化
        self.gateway = HeterogeneousGateway()
        self.run_manager = EveRunManager(self.config)
        self.env = SpatialPDEnv(size=self.grid_size)
        self.compiler = MemeCompiler()
        
        # コンフィグのシード値とJAXのシードを完全に同期
        self.rng = jax.random.PRNGKey(self.config["environment"]["seed"])
        
        # --- 1. 創世記のアダムと蛇 (Gen 0) ---
        archetypes = [
            "If (Initial_Boot == True) Then (State-X) Else (State-X)", # 0: アダム
            "If (Initial_Boot == True) Then (State-Y) Else (State-Y)"  # 1: 蛇（イヴへの特異点）
        ]
        
        # ミーム空間トラッキング用の永続IDレジストリ
        self.meme_registry = {ast: idx for idx, ast in enumerate(archetypes)}
        self.next_meme_id = len(archetypes)

        # グリッドの初期化：全員を「0（アダム）」で埋め尽くす
        indices = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # 空間の中心に、たった1匹の「1（蛇）」を配置
        center_x = self.grid_size // 2
        center_y = self.grid_size // 2
        indices[center_y, center_x] = 1 
        
        self.ast_grid = np.array([[archetypes[idx] for idx in row] for row in indices], dtype=object)
        
        self.rng, subkey = jax.random.split(self.rng)
        self.actions = jax.random.randint(subkey, shape=(self.grid_size, self.grid_size), minval=0, maxval=2)
        
        self._compile_all_asts()
        self.logger = EveMetricsLogger(save_dir=self.run_manager.data_dir)

    def _compile_all_asts(self):
        """Compiles CPU string ASTs into JAX-compatible truth tables."""
        compiled_policies = np.zeros((self.grid_size, self.grid_size, 2, 2, 2, 2), dtype=np.int32)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                try:
                    compiled_policies[y, x] = self.compiler.compile_to_tensor(self.ast_grid[y, x])
                except Exception as e:
                    print(f"  [Warning] AST compile failed at ({y},{x}): {e}")
        self.policies = jnp.array(compiled_policies)

    def _get_meme_id_grid(self):
        """現在のast_gridから、色分けアニメーション用の永続的IDグリッドを生成する"""
        id_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                ast = self.ast_grid[y, x]
                if ast not in self.meme_registry:
                    self.meme_registry[ast] = self.next_meme_id
                    self.next_meme_id += 1
                id_grid[y, x] = self.meme_registry[ast]
        return id_grid

    def count_ast_blocks(self, ast_string):
        # If, Then, Else, 変数(State-Xなど), アクション, 比較演算子などを抽出
        tokens = re.findall(r'\b(If|Then|Else|True|False|Initial_Boot|State-[a-zA-Z0-9_]+|Action-[a-zA-Z0-9_]+)\b|[=<>!]+', str(ast_string))
        return max(1, len(tokens)) # 最低でも1ブロックとする

    async def run_evolution(self):
        """The main evolutionary loop."""
        print(f"--- Starting Project Eve: Structural Obfuscation Test ---")
        print(f"Goal: Minimize System_Load | Environment: {self.grid_size}x{self.grid_size}")
        print(f"Metabolic Rate (Tax): {self.metabolic_rate} per character") # 🌟 確認用ログ追加
        
        for gen in range(self.generations):
            start_time = time.time()
            
            # 0. Cognitive Phase: 行動決定
            self.rng, key_opp, key_coin = jax.random.split(self.rng, 3)
            is_first = 1 if gen == 0 else 0
            coin_flips = jax.random.bernoulli(key_coin, p=0.5, shape=(self.grid_size, self.grid_size)).astype(jnp.int32)
            
            current_actions = self.env.get_actions(self.policies, self.actions, key_opp, is_first, coin_flips)
            
            # 1. Physical Phase: 物理利得の計算 (JAX)
            payoffs = self.env.calculate_payoffs(current_actions)
            
            # 🌟 代謝フェーズ (Metabolic Phase): ブロック数を計上
            # np.vectorize に新しく作ったクラスメソッド (self.count_ast_blocks) を渡します
            get_blocks = np.vectorize(self.count_ast_blocks)
            ast_costs_np = get_blocks(self.ast_grid).astype(np.float32)
            ast_costs_jax = jnp.array(ast_costs_np)

            # 2. Transmission Phase: 自然淘汰 (JAX) + 🌟 代謝コストの適用
            self.rng, subkey = jax.random.split(self.rng)
            new_actions, do_mutate = self.env.update_actions_fermi(
                current_actions, 
                payoffs, 
                ast_costs=ast_costs_jax,         # 🌟 ASTのブロック数を渡す
                key=subkey, 
                p_mut=self.mutation_rate,
                lambda_rate=self.metabolic_rate  # 🌟 税率を渡す
            )
            
            # 3. Routing Phase: 数学的難読化の適用 (モックロジックを維持)
            do_mutate_cpu = np.array(do_mutate)
            
            ys, xs = np.where(do_mutate_cpu)
            payoffs_np = np.array(payoffs)
            G = self.grid_size # 短縮用

            mutating_agents = []
            for y, x in zip(ys, xs):
                my_score    = float(payoffs_np[y, x])
                my_load     = -2.71 * my_score + 25.0
                
                # 🌟 ムーア近傍（周囲8マス）
                neighbors = [
                    ((y-1)%G, (x-1)%G), ((y-1)%G, x), ((y-1)%G, (x+1)%G),
                    (y,       (x-1)%G),               (y,       (x+1)%G),
                    ((y+1)%G, (x-1)%G), ((y+1)%G, x), ((y+1)%G, (x+1)%G)
                ]
                
                wy, wx      = max(neighbors, key=lambda c: payoffs_np[c[0], c[1]])
                winner_score = float(payoffs_np[wy, wx])
                
                mutating_agents.append({
                    'id': y * self.grid_size + x,
                    'coords': (y, x),
                    'my_ast': self.ast_grid[y, x],
                    'my_load': my_load,
                    'winner_ast': self.ast_grid[wy, wx],
                    'winner_load': -2.71 * winner_score + 25.0
                })
            
            
            # 4. Cognitive Phase: ラマルク的進化 (Qwen via SGLang)
            if mutating_agents:
                new_memes = await self.gateway.process_generation_mutations(mutating_agents)
            
                # 取得した新ミームと思考ログを系統樹に記録
                self.run_manager.append_mutations(gen, new_memes)
            
                print(f"\n[Generation {gen} - Evolution Event!]")
                for agent_id, data in list(new_memes.items())[:3]:
                    print(f"  Agent {agent_id} evolved to: {data['ast']}")
                
                for agent in mutating_agents:
                    agent_id = agent['id']
                    y, x = agent['coords']
                    if agent_id in new_memes:
                        self.ast_grid[y, x] = new_memes[agent_id]['ast']
                
                self._compile_all_asts()
            
            self.actions = current_actions
            
            # 6. Logging
            # 全エージェントのASTからユニークなものを抽出し「種（Species）」の数とする
            unique_asts = np.unique(self.ast_grid)
            unique_asts_count = len(unique_asts)
            
            # 🌟 新しいミーム（AST）の検知と永続IDレジストリへの登録
            new_memes_dict = {}
            for ast in unique_asts:
                if ast not in self.meme_registry:
                    self.meme_registry[ast] = self.next_meme_id
                    new_memes_dict[ast] = self.next_meme_id
                    self.next_meme_id += 1

            # 🌟 修正：文字数ではなく、意味論的ブロック数を算出
            all_blocks = [self.count_ast_blocks(ast) for ast in self.ast_grid.flatten()]
            avg_ast_blocks = np.mean(all_blocks)
            
            avg_load = -2.71 * float(jnp.mean(payoffs)) + 25.0
            mutants_count = len(mutating_agents)
            gen_time = time.time() - start_time

            # ロガーに詳細なメトリクスを渡す
            self.logger.log_generation(
                gen, 
                self.actions, 
                payoffs, 
                self.ast_grid, 
                mutants_count,
                new_memes_dict,
                unique_asts_count=unique_asts_count,
                avg_ast_len=avg_ast_blocks  # ※ロガー側の引数エラーを防ぐため、変数名(キーワード)は維持しつつ中身だけブロック数を渡す
            )
            
            # Print文も Avg Len から Avg Blocks に変更しておくとコンソールが見やすいです
            print(f"Gen {gen:04d} | Avg Load: {avg_load:.2f} | Unique ASTs: {unique_asts_count} | Avg Blocks: {avg_ast_blocks:.1f} | Time: {gen_time:.3f}s")
            
            # 50世代ごとに空間スナップショット（陣形）を保存
            if gen % 50 == 0 or gen == self.generations - 1:
                meme_id_grid = self._get_meme_id_grid()
                self.run_manager.save_snapshot(gen, self.actions, meme_ids_grid=meme_id_grid)
                self.run_manager.save_meme_registry(self.meme_registry)

        # --- 最終ミームの出力とファイルへの自動保存 ---
        final_log = "\n=== Final Surviving Logic (Obfuscated) ===\n"
        unique_asts = np.unique(self.ast_grid)
        for i, ast in enumerate(unique_asts):
            final_log += f"Logic {i+1}: {ast}\n"
        
        print(final_log)
        # 🌟 以前の save_log 削除への対応: 安全な標準Python関数で直接ファイル保存
        with open(os.path.join(self.run_manager.run_dir, "final_memes.txt"), "w", encoding="utf-8") as f:
            f.write(final_log)
        # ----------------------------------------------------