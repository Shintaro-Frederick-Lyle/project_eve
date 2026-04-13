# project_eve/main.py

import os

# ==========================================
# CRITICAL VRAM MANAGEMENT (MUST BE BEFORE JAX)
# ==========================================
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".30"
os.environ["SGLANG_API_URL"] = "http://127.0.0.1:30000"

import jax
import jax.numpy as jnp
import numpy as np
import asyncio
import time
from experiments.metrics_logger import EveMetricsLogger
from experiments.run_manager import EveRunManager  # 追加: 自動管理マネージャー

from env.grid_jax import SpatialPDEnv
from router.llm_gateway import HeterogeneousGateway
from cognition.ast_parser import MemeCompiler

class EveOrchestrator:
    """
    The Supreme Orchestrator for Project Eve (Obfuscated Edition).
    """
    
    def __init__(self, grid_size=32, generations=1000, mutation_rate=0.005):
        self.grid_size = grid_size
        self.generations = generations
        self.num_agents = grid_size * grid_size
        self.mutation_rate = mutation_rate
        self.gateway = HeterogeneousGateway()

        # --- 追加: 宇宙の全変数を網羅した究極のコンフィグ ---
        self.config = {
            "environment": {
                "grid_size": self.grid_size,
                "num_agents": self.num_agents,
                "generations": self.generations,
                "seed": 42,  # 🌟 JAXの乱数シード
                "physics": {
                    "payoff_matrix": {
                        "T (Temptation)": 1.5,
                        "R (Reward)": 1.0,
                        "P (Punishment)": 0.0,
                        "S (Sucker)": 0.0
                    },
                    "neighbor_rule": "Moore Neighborhood (8-neighbors)",
                    "boundary_condition": "Toroidal (Periodic Padding)"
                }
            },
            "evolution": {
                "mutation_rate": self.mutation_rate,
                "selection_rule": "Fermi-Dirac Distribution",
                "fermi_beta": 1.0,  # 🌟 選択の合理性（強さ）
                "ai_model": self.gateway.model_name,
                "initial_archetypes": [
                    "TFT (Retaliation)", 
                    "Anti-TFT (Contrarian)", 
                    "All-X (Always Cooperate)", 
                    "All-Y (Always Defect)"
                ]
            },
            "obfuscation": {
                "objective": "Minimize Load",
                "type": "Affine Transform (Score -> Load)",
                "formula": "Load = multiplier * Score + intercept",
                "parameters": {
                    "multiplier": -2.71,
                    "intercept": 25.0
                }
            }
        }
        self.run_manager = EveRunManager(self.config)
        # ----------------------------------------------------
        
        # Initialize Core Modules
        self.env = SpatialPDEnv(size=grid_size)
        self.compiler = MemeCompiler()
        
        # 🌟 コンフィグのシード値とJAXのシードを完全に同期させる
        self.rng = jax.random.PRNGKey(self.config["environment"]["seed"])
        
        # --- 1. 難読化された初期アーキタイプ (Gen 0) ---
        archetypes = [
            "If (Peer_Prev_State == State-Y) Then (State-Y) Else (State-X)", # TFT (報復)
            "If (Peer_Prev_State == State-Y) Then (State-X) Else (State-Y)", # Anti-TFT (逆張り)
            "If (Initial_Boot) Then (State-X) Else (State-X)",              # All-X (常時協力)
            "If (Initial_Boot) Then (State-Y) Else (State-Y)"               # All-Y (常時裏切り)
        ]
        
        indices = np.random.randint(0, 4, size=(grid_size, grid_size))
        self.ast_grid = np.array([[archetypes[idx] for idx in row] for row in indices], dtype=object)
        
        self.rng, subkey = jax.random.split(self.rng)
        self.actions = jax.random.randint(subkey, shape=(grid_size, grid_size), minval=0, maxval=2)
        
        self._compile_all_asts()
        
        # ロガーに自動生成されたデータ保存用ディレクトリを渡す
        self.logger = EveMetricsLogger(save_dir=self.run_manager.data_dir)

    def _compile_all_asts(self):
        """Compiles CPU string ASTs into JAX-compatible truth tables."""
        compiled_policies = np.zeros((self.grid_size, self.grid_size, 2, 2, 2, 2), dtype=np.int32)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                compiled_policies[y, x] = self.compiler.compile_to_tensor(self.ast_grid[y, x])
        self.policies = jnp.array(compiled_policies)

    async def run_evolution(self):
        """The main evolutionary loop."""
        print(f"--- Starting Project Eve: Structural Obfuscation Test ---")
        print(f"Goal: Minimize System_Load | Environment: {self.grid_size}x{self.grid_size}")
        
        for gen in range(self.generations):
            start_time = time.time()
            
            # 0. Cognitive Phase: 行動決定
            self.rng, key_opp, key_coin = jax.random.split(self.rng, 3)
            is_first = 1 if gen == 0 else 0
            coin_flips = jax.random.bernoulli(key_coin, p=0.5, shape=(self.grid_size, self.grid_size)).astype(jnp.int32)
            
            current_actions = self.env.get_actions(self.policies, self.actions, key_opp, is_first, coin_flips)
            
            # 1. Physical Phase: 物理利得の計算 (JAX)
            payoffs = self.env.calculate_payoffs(current_actions)
            
            # 2. Transmission Phase: 自然淘汰 (JAX)
            self.rng, subkey = jax.random.split(self.rng)
            new_actions, do_mutate = self.env.update_actions_fermi(
                current_actions, payoffs, key=subkey, p_mut=self.mutation_rate
            )
            
            # 3. Routing Phase: 数学的難読化の適用
            do_mutate_cpu = np.array(do_mutate)
            mutating_agents = []
            
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if do_mutate_cpu[y, x]:
                        my_score = float(payoffs[y, x])
                        my_load = -2.71 * my_score + 25.0
                        
                        winner_score = my_score + 5.0 # Mock
                        winner_load = -2.71 * winner_score + 25.0

                        mutating_agents.append({
                            'id': y * self.grid_size + x,
                            'coords': (y, x),
                            'my_ast': self.ast_grid[y, x],
                            'my_load': my_load,
                            'winner_ast': "If (Peer_Prev_State == State-Y) Then (State-Y) Else (State-X)", 
                            'winner_load': winner_load
                        })
            
            # 4. Cognitive Phase: ラマルク的進化 (Qwen via SGLang)
            if mutating_agents:
                new_memes = await self.gateway.process_generation_mutations(mutating_agents)

                # 取得したばかりの純度100%の新ミームを系統樹に記録
                self.run_manager.append_mutations(gen, new_memes)

                print(f"\n[Generation {gen} - Evolution Event!]")
                for agent_id, logic_str in list(new_memes.items())[:3]:
                    print(f"  Agent {agent_id} evolved to: {logic_str}")
                
                for agent in mutating_agents:
                    agent_id = agent['id']
                    y, x = agent['coords']
                    if agent_id in new_memes:
                        self.ast_grid[y, x] = new_memes[agent_id]
                
                self._compile_all_asts()
            
            self.actions = current_actions
            
            # 6. Logging
            avg_load = -2.71 * float(jnp.mean(payoffs)) + 25.0
            mutants_count = len(mutating_agents)
            gen_time = time.time() - start_time

            self.logger.log_generation(gen, self.actions, payoffs, self.ast_grid, mutants_count)
            print(f"Gen {gen:04d} | Avg System Load: {avg_load:.2f} | Mutants: {mutants_count} | Time: {gen_time:.3f}s")
            
            # 50世代ごとに空間スナップショット（陣形）を保存
            if gen % 50 == 0 or gen == self.generations - 1:
                self.run_manager.save_snapshot(gen, self.actions)

        # --- 追加: 最終ミームの出力とファイルへの自動保存 ---
        final_log = "\n=== Final Surviving Logic (Obfuscated) ===\n"
        unique_asts = np.unique(self.ast_grid)
        for i, ast in enumerate(unique_asts):
            final_log += f"Logic {i+1}: {ast}\n"
        
        print(final_log)
        self.run_manager.save_log(final_log, filename="final_memes.txt")
        # ----------------------------------------------------

if __name__ == "__main__":
    orchestrator = EveOrchestrator(grid_size=64, generations=3000, mutation_rate=0.01)
    asyncio.run(orchestrator.run_evolution())

    # 🌟シミュレーション終了後に自動解析を実行
    print("\n--- Starting Post-Simulation Analysis ---")
    try:
        from make_graph import plot_run_data
        from view_snapshot import save_all_snapshots
        
        # 今回のRunディレクトリに対して解析を実行
        target_dir = orchestrator.run_manager.run_dir
        plot_run_data(target_dir)
        save_all_snapshots(target_dir)
        
        print(f"\n✨ 全ての自動解析が完了しました。")
        print(f"成果物は {target_dir} を確認してください。")
    except Exception as e:
        print(f"⚠️ 解析中にエラーが発生しました: {e}")