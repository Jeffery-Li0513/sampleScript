import openai
import os

# openai.api_key = os.getenv("OPENAI_API_KEY")
# 

openai.api_key = "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"

prompt = "在一个矩阵符号的右下角有一个小于号，这表示什么意思呢"

response = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=500, temperature=0)

message = response.choices[0].text
print("答：", message)