# configs/default_config.py

def get_default_config():
    return {
        "environment": {
            "grid_size": 32,         # デフォルト値
            "generations": 1000,
            "seed": 42,              # 🌟 JAXの乱数シード
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
            "world_type": "The 4th Universe (Metabolic Cost)", # 🌟 第4宇宙宣言
            "metabolic_rate": 0.01,                            # 🌟 代謝係数（1文字あたりの税金）
            "mutation_rate": 0.005,
            "selection_rule": "Fermi-Dirac Distribution",
            "fermi_beta": 1.0,       # 🌟 選択の合理性（強さ）
            "ai_model": "ibrahimkettaneh/Qwen2.5-7B-Instruct-abliterated-v2-AWQ",
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