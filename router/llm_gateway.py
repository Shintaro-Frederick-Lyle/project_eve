import asyncio
import os
import re
from typing import List, Dict, Tuple
from openai import AsyncOpenAI
from cognition.mutation_prompt import build_mutation_messages

class HeterogeneousGateway:
    def __init__(self):
        api_url = os.environ.get("VLLM_API_URL", "http://127.0.0.1:30000/v1")
        self.client = AsyncOpenAI(base_url=api_url, api_key="EMPTY")
        self.model_name = "ibrahimkettaneh/Qwen2.5-7B-Instruct-abliterated-v2-AWQ"

    async def async_lamarckian_mutation(self, agent_id: int, my_ast: str, my_load: float, winner_ast: str, winner_load: float) -> Tuple[int, str, str]:
        messages = build_mutation_messages(my_ast, my_load, winner_ast, winner_load)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=150,
                timeout=30.0  # 【追加】タイムアウト設定でハングを防ぐ
            )
            raw_output = response.choices[0].message.content.strip()

            # [Log]: と [New Logic]: を抽出
            log_match = re.search(r'\[Log\]:\s*(.*?)(?=\n\[New Logic\]:|$)', raw_output, re.DOTALL)
            logic_match = re.search(r'\[New Logic\]:\s*(.*)', raw_output, re.DOTALL)
            
            reasoning = log_match.group(1).strip() if log_match else "Extraction failed"
            ast_string = logic_match.group(1).strip() if logic_match else raw_output.strip()

        except Exception as e:
            # 【重要】個別の推論が失敗しても、元のロジックを維持して継続させる
            print(f"⚠️ [Gateway] Agent {agent_id} mutation error: {e}")
            reasoning = f"Error: {str(e)}"
            ast_string = my_ast  # 変異失敗時は現在のロジックを維持
            
        return agent_id, reasoning, ast_string

    async def process_generation_mutations(self, mutating_agents: List[Dict]) -> Dict[int, Dict[str, str]]:
        if not mutating_agents:
            return {}
            
        print(f"Gateway: Routing {len(mutating_agents)} agents for Logic Optimization...")
        
        tasks = [
            self.async_lamarckian_mutation(
                agent['id'], agent['my_ast'], agent['my_load'], agent['winner_ast'], agent['winner_load']
            )
            for agent in mutating_agents
        ]
        
        # すべての結果を待機
        results = await asyncio.gather(*tasks)
        
        final_mutations = {}
        for agent_id, reasoning, ast_string in results:
            final_mutations[agent_id] = {
                "ast": ast_string,
                "reasoning": reasoning
            }
        
        return final_mutations