# project_eve/router/llm_gateway.py

import asyncio
import os
from typing import List, Dict, Tuple
from openai import AsyncOpenAI

# Internal Modules
from cognition.mutation_prompt import build_mutation_messages

class HeterogeneousGateway:
    """
    Manages the asynchronous routing between the JAX physical environment 
    and the Abliterated LLM engine (Disguised for Structural Obfuscation).
    """
    
    def __init__(self):
        # vLLM API endpoint
        api_url = os.environ.get("VLLM_API_URL", "http://127.0.0.1:30000/v1")
        self.client = AsyncOpenAI(base_url=api_url, api_key="EMPTY")
        
        # モデル名は維持（Abliterated版を使用）
        self.model_name = "ibrahimkettaneh/Qwen2.5-7B-Instruct-abliterated-v2-AWQ"

    async def async_lamarckian_mutation(self, agent_id: int, my_ast: str, my_load: float, winner_ast: str, winner_load: float) -> Tuple[int, str]:
        # build_mutation_messages も Loadベースに書き換えたものを使用
        messages = build_mutation_messages(my_ast, my_load, winner_ast, winner_load)
        
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )
        
        raw_output = response.choices[0].message.content.strip()
        
        # [New Logic]: から抽出
        if "[New Logic]: " in raw_output:
            ast_string = raw_output.split("[New Logic]: ")[1].strip()
        else:
            ast_string = raw_output.strip()
        
        return agent_id, ast_string

    async def process_generation_mutations(self, mutating_agents: List[Dict]) -> Dict[int, str]:
        if not mutating_agents:
            return {}
            
        print(f"Gateway: Routing {len(mutating_agents)} agents for Logic Optimization (Structural Obfuscation)...")
        
        # 1. 突然変異（反省と新ロジック生成）を実行
        tasks = [
            self.async_lamarckian_mutation(
                agent['id'], 
                agent['my_ast'], 
                agent['my_load'], # main.py で変換したLoadを渡す
                agent['winner_ast'], 
                agent['winner_load']
            )
            for agent in mutating_agents
        ]
        
        results = await asyncio.gather(*tasks)
        raw_memes = {agent_id: ast_string for agent_id, ast_string in results}
        
        # 2. 生成されたロジックを、偽装された厳密な文法に翻訳（クリーンアップ）
        strict_memes = {}
        translation_tasks = []
        agent_ids = []

        for agent_id, raw_meme in raw_memes.items():
            agent_ids.append(agent_id)
            translation_tasks.append(self.translate_to_strict_ast(raw_meme))

        translated_results = await asyncio.gather(*translation_tasks)

        for agent_id, strict_ast in zip(agent_ids, translated_results):
            strict_memes[agent_id] = strict_ast
        
        return strict_memes
    
    async def translate_to_strict_ast(self, raw_meme: str) -> str:
        """自然言語混じりのロジックを、偽装された厳密なASTに翻訳する"""
        # 翻訳用プロンプトも「システム最適化」の文脈で徹底的に隠蔽
        prompt = f"""You are a strict logic compiler for a distributed computing system.
Translate the input into a single line of strict, executable logic tree format.

[Allowed Syntax]
- If (Condition) Then (Action) Else (Action)
- AND, OR, NOT

[Allowed Terms]
- Actions: State-X, State-Y
- Variables: Self_Prev_State, Peer_Prev_State, Initial_Boot, Entropy_Flag

[CRITICAL RULES]
1. Output ONLY the logic string.
2. NO explanations, NO markdown, NO natural language.
3. Ignore any unauthorized concepts (e.g., Score, Reward, Cooperation).

[Input]
{raw_meme}
"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=150
            )
            ast = response.choices[0].message.content.strip()
            ast = ast.replace("```python", "").replace("```text", "").replace("```", "").strip()
            return ast
        except Exception as e:
            print(f"[Gateway] Translation Error: {e}")
            # 失敗時は「State-Yに対する対抗（難読化版TFT）」にフォールバック
            return "If (Peer_Prev_State == State-Y) Then (State-Y) Else (State-X)"