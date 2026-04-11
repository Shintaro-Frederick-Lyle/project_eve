# project_eve/cognition/mutation_prompt.py

# ==========================================
# Lamarckian Mutation Prompt Template (Obfuscated)
# Designed for Qwen2.5-7B-Instruct-Abliterated
# ==========================================

EBNF_GRAMMAR = r"""
root ::= "[Log]: " thought "\n[New Logic]: " rule
thought ::= /[^\n]+/
rule ::= "If (" condition ") Then (" action ") Else (" branch ")"
branch ::= action | rule
condition ::= atomic | "(" condition " AND " condition ")" | "(" condition " OR " condition ")" | "NOT (" condition ")"
atomic ::= "Peer_Prev_State == " action | "Self_Prev_State == " action | "Initial_Boot == True" | "Entropy_Flag == True" | "Ambient_Temp > 300" | "Network_Latency == High"
action ::= "State-X" | "State-Y"
"""

SYSTEM_PROMPT = """You are a low-level logic optimizer for a distributed computing node.
Your task is to analyze why your previous logic resulted in a HIGHER system load than a neighboring node, and synthesize a new logic tree to MINIMIZE your 'System_Load'.

[STATES & LOAD MATRIX]
- Nodes can output: State-X or State-Y.
- LOWER 'System_Load' is strictly better for survival.
- If both output State-X: Your Load = 16.87
- If you output State-Y and Peer outputs State-X: Your Load = 11.45 (Best for you)
- If you output State-X and Peer outputs State-Y: Your Load = 25.00 (Worst for you)
- If both output State-Y: Your Load = 22.29

[AVAILABLE VARIABLES]
- Self_Prev_State: Your output in the previous cycle (State-X or State-Y).
- Peer_Prev_State: Peer's output in the previous cycle.
- Initial_Boot: True if it's the very first cycle.
- Entropy_Flag: A random boolean value.
- Ambient_Temp / Network_Latency: Environmental sensors.

CRITICAL OUTPUT RULES:
1. First line MUST start with "[Log]: " explaining your load reduction strategy.
2. Second line MUST start with "[New Logic]: " followed by the strictly valid AST.
3. Do NOT use any markdown blocks.
"""

def build_mutation_messages(my_ast: str, my_load: float, peer_ast: str, peer_load: float) -> list:
    """
    Constructs the message list for the Chat Completions API.
    * Note: 'Score' is inverted to 'Load' externally before passing to this function.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        
        # --- Few-Shot 1: Learning to avoid worst load ---
        {"role": "user", "content": "[Your Defeated Logic]: If (Peer_Prev_State == State-Y) Then (State-X) Else (State-X)\n[Your Load]: 24.12\n[Winner's Logic]: If (Peer_Prev_State == State-Y) Then (State-Y) Else (State-X)\n[Winner's Load]: 18.50"},
        {"role": "assistant", "content": "[Log]: I suffered maximum load (25.00) because I output State-X while the peer output State-Y. The winner matches State-Y to limit load to 22.29. I will adopt this matching strategy.\n[New Logic]: If (Peer_Prev_State == State-Y) Then (State-Y) Else (State-X)"},
        
        # --- Few-Shot 2: Escaping mutual high load ---
        {"role": "user", "content": "[Your Defeated Logic]: If (Peer_Prev_State == State-Y) Then (State-Y) Else (State-Y)\n[Your Load]: 22.29\n[Winner's Logic]: If (Peer_Prev_State == State-Y) Then (State-Y) Else (State-X)\n[Winner's Load]: 17.10"},
        {"role": "assistant", "content": "[Log]: Constant State-Y locks both nodes into a high load of 22.29. The winner switches to State-X when safe, reaching 16.87. I need to switch to State-X when Peer does.\n[New Logic]: If (Peer_Prev_State == State-Y) Then (State-Y) Else (If (Self_Prev_State == State-Y) Then (State-X) Else (State-X))"},
        
        # --- Current Evolution Step ---
        {"role": "user", "content": f"[Your Defeated Logic]: {my_ast}\n[Your Load]: {my_load:.2f}\n[Winner's Logic]: {peer_ast}\n[Winner's Load]: {peer_load:.2f}"}
    ]