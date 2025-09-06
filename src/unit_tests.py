from src.llm_workflow import LlmWorkflow


llm = LlmWorkflow()
line = "ここですか。"
prompt = f"""
Ты — литературный переводчик. Переведи на русский язык.
Японский: アンナは毎あさ七時に起きます。
Русский: Анна встает в семь утра каждое утро.

Переведи:
Японский: {line}
Русский:
"""
resp = llm.generate_response(prompt, 1024)
print(resp)
