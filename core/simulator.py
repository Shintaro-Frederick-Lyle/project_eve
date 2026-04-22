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
from experiments.metrics_logger import EveMetricsLogger

class EveSimulator:
    def __init__(self, config):
        self.config = config # 完全な辞書はRunManagerでの保存用に保持
        
        # --- 設定値の自動展開 ---
        self.grid_size = config["environment"]["grid_size"]
        self.generations = config["environment"]["generations"]
        self.num_agents = self.grid_size * self.grid_size
        self.mutation_rate = config["evolution"]["mutation_rate"]
        self.metabolic_rate = config["evolution"]["metabolic_rate"] 

        # 利得行列を config から構築
        t_val = config["environment"]["physics"]["payoff_matrix"]["T (Temptation)"]
        r_val = config["environment"]["physics"]["payoff_matrix"]["R (Reward)"]
        p_val = config["environment"]["physics"]["payoff_matrix"]["P (Punishment)"]
        s_val = config["environment"]["physics"]["payoff_matrix"]["S (Sucker)"]
        
        payoff_matrix = jnp.array([
            [p_val, t_val],
            [s_val, r_val]
        ])
        
        # モジュールの初期化
        self.gateway = HeterogeneousGateway()
        self.run_manager = EveRunManager(self.config)
        self.env = SpatialPDEnv(size=self.grid_size, payoff_matrix=payoff_matrix)
        self.compiler = MemeCompiler()
        self.rng = jax.random.PRNGKey(self.config["environment"]["seed"])
        
        # --- 🚀 1. カタログシステムの初期化 (Strategy Indexing Architecture) ---
        archetypes = [
            "If (Initial_Boot == True) Then (State-X) Else (State-X)", # 0: アダム
            "If (Initial_Boot == True) Then (State-Y) Else (State-Y)"  # 1: 蛇（イヴへの特異点）
        ]
        
        # ライブラリ（カタログ）の構築
        self.unique_strategies = list(archetypes)
        # 初期戦略をコンパイルし、JAX配列化 (shape: [2, 2, 2, 2, 2])
        initial_policies = [self.compiler.compile_to_tensor(ast) for ast in self.unique_strategies]
        self.strategy_library = jnp.array(initial_policies)
        
        # ブロック数を事前計算しておくキャッシュ配列 (shape: [2])
        self.block_counts_library = np.array([self.count_ast_blocks(ast) for ast in self.unique_strategies], dtype=np.float32)

        # 永続IDレジストリ
        self.meme_registry = {ast: idx for idx, ast in enumerate(self.unique_strategies)}
        self.next_meme_id = len(self.unique_strategies)

        # --- 🚀 2. 地図（インデックスグリッド）の初期化 ---
        # 全員を「0（アダム）」のインデックスで埋める
        self.policy_idx_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # 空間の中心に「1（蛇）」のインデックスを配置
        center_x = self.grid_size // 2
        center_y = self.grid_size // 2
        self.policy_idx_grid[center_y, center_x] = 1 
        
        # 物理行動の初期化
        self.rng, subkey = jax.random.split(self.rng)
        self.actions = jax.random.randint(subkey, shape=(self.grid_size, self.grid_size), minval=0, maxval=2)
        
        self.logger = EveMetricsLogger(save_dir=self.run_manager.data_dir)

    # 廃止: def _compile_all_asts(self) はもう不要です！

    def _get_meme_id_grid(self):
        """アニメーション用の永続的IDグリッドを生成する"""
        id_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                ast = self.unique_strategies[self.policy_idx_grid[y, x]]
                id_grid[y, x] = self.meme_registry[ast]
        return id_grid

    def count_ast_blocks(self, ast_string):
        tokens = re.findall(r'\b(If|Then|Else|True|False|Initial_Boot|State-[a-zA-Z0-9_]+|Action-[a-zA-Z0-9_]+)\b|[=<>!]+', str(ast_string))
        return max(1, len(tokens))

    async def run_evolution(self):
        """The main evolutionary loop."""
        print(f"--- Starting Project Eve: Strategic Indexing Architecture ---")
        print(f"Goal: Minimize System_Load | Environment: {self.grid_size}x{self.grid_size}")
        print(f"Metabolic Rate (Tax): {self.metabolic_rate} per character")
        
        for gen in range(self.generations):
            start_time = time.time()
            
            # 0. Cognitive Phase: 行動決定
            self.rng, key_opp, key_coin = jax.random.split(self.rng, 3)
            is_first = 1 if gen == 0 else 0
            coin_flips = jax.random.bernoulli(key_coin, p=0.5, shape=(self.grid_size, self.grid_size)).astype(jnp.int32)
            
            # 🌟 魔法：インデックス地図を使って、カタログから全員分のポリシーを一瞬で展開！
            # shape は自動的に (64, 64, 2, 2, 2, 2) になります
            current_policies = self.strategy_library[self.policy_idx_grid]
            current_actions = self.env.get_actions(current_policies, self.actions, key_opp, is_first, coin_flips)
            
            # 1. Physical Phase: 物理利得の計算
            payoffs = self.env.calculate_payoffs(current_actions)
            
            # 2. Metabolic Phase: 代謝コストの適用
            # 🌟 爆速化：正規表現を回さず、キャッシュからブロック数を一瞬で引く
            ast_costs_np = self.block_counts_library[self.policy_idx_grid]
            ast_costs_jax = jnp.array(ast_costs_np)

            # 3. Transmission Phase: 自然淘汰 + 代謝コスト
            self.rng, subkey = jax.random.split(self.rng)
            new_actions, do_mutate = self.env.update_actions_fermi(
                current_actions, 
                payoffs, 
                key=subkey, 
                p_mut=self.mutation_rate,
                ast_costs=ast_costs_jax
            )
            self.actions = np.array(new_actions)

            # 4. Routing Phase: 突然変異エージェントの抽出
            do_mutate_cpu = np.array(do_mutate)
            ys, xs = np.where(do_mutate_cpu)
            payoffs_np = np.array(payoffs)
            G = self.grid_size

            mutating_agents = []
            for y, x in zip(ys, xs):
                my_score    = float(payoffs_np[y, x])
                my_load     = -2.71 * my_score + 25.0
                
                # ムーア近傍（周囲8マス）
                neighbors = [
                    ((y-1)%G, (x-1)%G), ((y-1)%G, x), ((y-1)%G, (x+1)%G),
                    (y,       (x-1)%G),               (y,       (x+1)%G),
                    ((y+1)%G, (x-1)%G), ((y+1)%G, x), ((y+1)%G, (x+1)%G)
                ]
                
                wy, wx      = max(neighbors, key=lambda c: payoffs_np[c[0], c[1]])
                winner_score = float(payoffs_np[wy, wx])
                
                # インデックスから生のコード文字列を復元してLLMに渡す
                my_ast = self.unique_strategies[self.policy_idx_grid[y, x]]
                winner_ast = self.unique_strategies[self.policy_idx_grid[wy, wx]]
                
                mutating_agents.append({
                    'id': y * self.grid_size + x,
                    'coords': (y, x),
                    'my_ast': my_ast,
                    'my_load': my_load,
                    'winner_ast': winner_ast,
                    'winner_load': -2.71 * winner_score + 25.0
                })
            
            # 5. Cognitive Phase: ラマルク的進化 (LLM推論)
            if mutating_agents:
                new_memes = await self.gateway.process_generation_mutations(mutating_agents)
                self.run_manager.append_mutations(gen, new_memes)
            
                print(f"\n[Generation {gen} - Evolution Event!]")
                for agent_id, data in list(new_memes.items())[:3]:
                    print(f"  Agent {agent_id} evolved to: {data['ast']}")
                
                # 🌟 新しいミームをライブラリに追加し、地図を更新する
                for agent in mutating_agents:
                    agent_id = agent['id']
                    y, x = agent['coords']
                    
                    if agent_id in new_memes:
                        new_ast = new_memes[agent_id]['ast']
                        
                        # 未知の戦略が誕生した場合、ライブラリを拡張
                        if new_ast not in self.unique_strategies:
                            self.unique_strategies.append(new_ast)
                            
                            # 新規コンパイルは「新種誕生時」の1回だけ！
                            new_policy = self.compiler.compile_to_tensor(new_ast)
                            self.strategy_library = jnp.vstack([self.strategy_library, jnp.array(new_policy)[None, ...]])
                            
                            new_block_count = self.count_ast_blocks(new_ast)
                            self.block_counts_library = np.append(self.block_counts_library, new_block_count)
                        
                        # 個体のインデックス地図を更新
                        new_idx = self.unique_strategies.index(new_ast)
                        self.policy_idx_grid[y, x] = new_idx
            
            self.actions = current_actions
            
            # 6. Logging (爆速化対応)
            # 現在の地図から、ロガー互換用のast_gridを動的に復元
            current_ast_grid = np.array([[self.unique_strategies[idx] for idx in row] for row in self.policy_idx_grid], dtype=object)
            
            # 現在生存しているユニークな戦略IDのみを抽出
            alive_indices = np.unique(self.policy_idx_grid)
            unique_asts_count = len(alive_indices)
            
            # 新規ミームのレジストリ登録
            new_memes_dict = {}
            for idx in alive_indices:
                ast = self.unique_strategies[idx]
                if ast not in self.meme_registry:
                    self.meme_registry[ast] = self.next_meme_id
                    new_memes_dict[ast] = self.next_meme_id
                    self.next_meme_id += 1

            # 平均ブロック数の計算 (O(1)で完了)
            avg_ast_blocks = np.mean(self.block_counts_library[self.policy_idx_grid])
            avg_load = -2.71 * float(jnp.mean(payoffs)) + 25.0
            mutants_count = len(mutating_agents)
            gen_time = time.time() - start_time

            # ロガーに渡す
            self.logger.log_generation(
                gen, self.actions, payoffs, current_ast_grid, 
                mutants_count, new_memes_dict,
                unique_asts_count=unique_asts_count,
                avg_ast_len=avg_ast_blocks 
            )
            
            print(f"Gen {gen:04d} | Mutants: {mutants_count:3d} | Avg Load: {avg_load:.2f} | Unique ASTs: {unique_asts_count} | Avg Blocks: {avg_ast_blocks:.1f} | Time: {gen_time:.3f}s")
            
            # スナップショット保存
            if gen % 50 == 0 or gen == self.generations - 1:
                meme_id_grid = self._get_meme_id_grid()
                self.run_manager.save_snapshot(gen, self.actions, meme_ids_grid=meme_id_grid)
                self.run_manager.save_meme_registry(self.meme_registry)

        # --- 最終結果の保存 ---
        final_log = "\n=== Final Surviving Logic (Obfuscated) ===\n"
        alive_indices = np.unique(self.policy_idx_grid)
        for i, idx in enumerate(alive_indices):
            ast = self.unique_strategies[idx]
            final_log += f"Logic {i+1}: {ast}\n"
        
        print(final_log)
        with open(os.path.join(self.run_manager.run_dir, "final_memes.txt"), "w", encoding="utf-8") as f:
            f.write(final_log)