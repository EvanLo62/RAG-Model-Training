from typing import List
import re

def get_inference_system_prompt() -> str:
    """get system prompt for generation"""
    prompt = "You are a precise QA assistant. Answer based only on the given context. If the answer is not in the context, say 'CANNOTANSWER'."
    return prompt

def get_inference_user_prompt(query : str, context_list : List[str]) -> str:
    """Create the user prompt for generation given a query and a list of context passages."""
    formatted_contexts = "\n\n".join([
    f"[{i+1}] {context}" 
    for i, context in enumerate(context_list)
    ])

    prompt = f"""{formatted_contexts}

Q: {query} 
A:"""

    return prompt

def parse_generated_answer(pred_ans: str) -> str:
    """
    Extract answer from LLM output.
    LLM 輸出包含完整的 prompt，需要提取最後的答案部分
    """
    pred_ans = pred_ans.strip()
    
    if not pred_ans:
        return "CANNOTANSWER"
    
    # ===== 策略 1: 找 "assistant" 之後、"<think>" 之後的內容 =====
    # Qwen 模型的輸出格式: assistant\n<think>\n\n</think>\n答案
    if '<think>' in pred_ans and '</think>' in pred_ans:
        # 提取 </think> 之後的內容
        parts = pred_ans.split('</think>')
        if len(parts) > 1:
            answer = parts[-1].strip()
            # 移除可能的結束標記
            answer = re.sub(r'<\|.*?\|>.*$', '', answer, flags=re.DOTALL).strip()
            if 1 < len(answer) < 500:
                return answer
    
    # ===== 策略 2: 找最後一個 "assistant" 之後的內容 =====
    if 'assistant' in pred_ans.lower():
        # 找到最後一個 "assistant"
        idx = pred_ans.lower().rfind('assistant')
        if idx != -1:
            answer = pred_ans[idx + len('assistant'):].strip()
            # 移除 <think> 標籤
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
            # 移除其他標記
            answer = re.sub(r'<\|.*?\|>.*$', '', answer, flags=re.DOTALL).strip()
            # 只取第一行
            answer = answer.split('\n')[0].strip()
            if 1 < len(answer) < 500:
                return answer
    
    # ===== 策略 3: 找最後一個 "A:" 之後的內容 =====
    if 'A:' in pred_ans or 'a:' in pred_ans:
        # 使用正則找所有 "A:"
        matches = list(re.finditer(r'\ba\s*:', pred_ans, re.IGNORECASE))
        if matches:
            # 取最後一個匹配
            last_match = matches[-1]
            remaining = pred_ans[last_match.end():].strip()
            
            # 如果後面有 "assistant"，提取 assistant 之後的部分
            if 'assistant' in remaining.lower():
                idx = remaining.lower().find('assistant')
                remaining = remaining[idx + len('assistant'):].strip()
            
            # 移除 <think> 標籤
            remaining = re.sub(r'<think>.*?</think>', '', remaining, flags=re.DOTALL).strip()
            # 只取第一行
            answer = remaining.split('\n')[0].strip()
            if 1 < len(answer) < 500:
                return answer
    
    # ===== 策略 4: 取最後一行（可能是簡短答案） =====
    lines = [line.strip() for line in pred_ans.split('\n') if line.strip()]
    if lines:
        # 取最後一行
        last_line = lines[-1]
        # 確保不是 prompt 的一部分
        if (1 < len(last_line) < 500 and 
            not any(kw in last_line.lower() for kw in ['system', 'user', 'context', '[1]', '[2]', '[3]'])):
            return last_line
    
    return "CANNOTANSWER"