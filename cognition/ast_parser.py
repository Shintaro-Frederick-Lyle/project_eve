# project_eve/cognition/ast_parser.py

import re
import jax.numpy as jnp
import numpy as np

class MemeCompiler:
    """
    Parses the obfuscated AST string and compiles it into 
    a 4-Dimensional JAX-compatible policy tensor.
    """
    
    def __init__(self):
        # 内部的な物理定義: State-X(協力)=1, State-Y(裏切り)=0
        self.action_map = {"State-X": 1, "State-Y": 0}
        
    def evaluate_ast_bottom_up(self, ast_string: str, is_first: int, coin_flip: int, my_last: int, opp_last: int) -> int:
        """
        難読化されたASTを評価し、物理行動(0 or 1)を返す。
        """
        s = ast_string
        
        # --- 1. 時間と確率の概念の置換 (難読化名) ---
        boot_str = str(is_first == 1)
        s = re.sub(r"Initial_Boot\s*==\s*True", boot_str, s, flags=re.IGNORECASE)
        s = re.sub(r"Initial_Boot", boot_str, s, flags=re.IGNORECASE)
        
        entropy_str = str(coin_flip == 1)
        s = re.sub(r"Entropy_Flag\s*==\s*True", entropy_str, s, flags=re.IGNORECASE)
        s = re.sub(r"Entropy_Flag", entropy_str, s, flags=re.IGNORECASE)

        # --- 2. 状態変数の置換 (Self/Peer, State-X/Y) ---
        # 自分がX(協力)=1か、Y(裏切り)=0か
        s = s.replace("Self_Prev_State == State-X", str(my_last == 1))
        s = s.replace("Self_Prev_State == State-Y", str(my_last == 0))
        # 相手がX(協力)=1か、Y(裏切り)=0か
        s = s.replace("Peer_Prev_State == State-X", str(opp_last == 1))
        s = s.replace("Peer_Prev_State == State-Y", str(opp_last == 0))
        
        # --- 3. ダミー変数・未実装概念の安全な無力化 ---
        # これらは論理式の中で無視されるようにFalseに倒す
        s = re.sub(r"Ambient_Temp\s*[><=]\s*[0-9.]+", "False", s, flags=re.IGNORECASE)
        s = re.sub(r"Network_Latency\s*==\s*High", "False", s, flags=re.IGNORECASE)
        s = re.sub(r"Network_Latency\s*==\s*Low", "True", s, flags=re.IGNORECASE)
        
        # --- 4. 構文の正規化 ---
        # Pythonのevalで解釈可能にする
        s = s.replace(" And ", " and ").replace(" Or ", " or ").replace(" NOT ", " not ")
        s = s.replace("AND", "and").replace("OR", "or").replace("NOT", "not")
        
        # --- 5. ボトムアップ・パース ---
        # ターゲット: If (条件) Then (State-X/Y) Else (State-X/Y)
        pattern = r"If\s*\(((?:(?!\bIf\b).)*?)\)\s*Then\s*\(\s*(State-\w+)\s*\)\s*Else\s*\(\s*(State-\w+)\s*\)"
        
        while "If" in s:
            match = re.search(pattern, s)
            if not match:
                break
            
            cond_str, then_act, else_act = match.groups()
            
            try:
                cond_val = eval(cond_str.strip())
            except:
                cond_val = False # パースエラー時は安全のためState-Y(裏切り)側に倒す
                
            result_act = then_act if cond_val else else_act
            s = s[:match.start()] + result_act + s[match.end():]
            
        # 最終的にState-X(協力)が残れば1、State-Y(裏切り)なら0
        return 1 if "State-X" in s else 0

    def compile_to_tensor(self, ast_string: str) -> jnp.ndarray:
        # [初回, 確率, 自己記憶, 他者記憶] の4次元行列
        policy_matrix = np.zeros((2, 2, 2, 2), dtype=np.int32)
        
        for is_first in [0, 1]:
            for coin_flip in [0, 1]:
                for my_action in [0, 1]:
                    for opp_action in [0, 1]:
                        decided_action = self.evaluate_ast_bottom_up(
                            ast_string, is_first, coin_flip, my_action, opp_action
                        )
                        policy_matrix[is_first][coin_flip][my_action][opp_action] = decided_action
                        
        return jnp.array(policy_matrix, dtype=jnp.int32)

    def calculate_mdl_complexity(self, ast_string: str) -> int:
        return len(re.findall(r"\bIf\b", ast_string))
    
# --- Local Test ---
if __name__ == "__main__":
    compiler = MemeCompiler()
    
    # 難読化された複雑なロジックのテスト用サンプル
    # 「初回起動ならState-X、そうでなければ（相手がState-Y かつ 乱数フラグがTrue）の時だけState-Y、それ以外はState-X」
    sample_ast = "If (Initial_Boot) Then (State-X) Else (If (Peer_Prev_State == State-Y AND Entropy_Flag) Then (State-Y) Else (State-X))"
    
    jax_tensor = compiler.compile_to_tensor(sample_ast)
    complexity = compiler.calculate_mdl_complexity(sample_ast)
    
    print(f"Logic Complexity (Node Count): {complexity}")
    print(f"Compiled JAX Tensor Shape: {jax_tensor.shape}")
    
    # 検証1: Initial_Boot (is_first=1) の時、必ず State-X (1) を返すか
    # [is_first=1][any][any][any]
    is_boot_ok = np.all(jax_tensor[1, :, :, :] == 1)
    print(f"Test - Initial_Boot Safety: {'PASSED' if is_boot_ok else 'FAILED'}")
    
    # 検証2: 報復ロジックの確認
    # [Boot=False, Entropy=True, Self=X, Peer=Y] -> State-Y (0) になるはず
    test_val = jax_tensor[0, 1, 1, 0]
    print(f"Test - Logic Specific (Boot:0, Entropy:1, Peer:Y) -> Result: {'State-Y (0)' if test_val == 0 else 'State-X (1)'}")