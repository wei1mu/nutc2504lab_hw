import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# 1. 配置第一個模型：Llama (負責專業風格)
llm_llama = ChatOpenAI(
    model="Llama-3.3-70B-Instruct-NVFP4",
    openai_api_key="none", # 根據你的伺服器設定
    openai_api_base="https://ws-03.wade0426.me/v1",
    temperature=0
)

# 2. 配置第二個模型：Gemma (負責幽默風格)
llm_gemma = ChatOpenAI(
    model="google/gemma-3-27b-it",
    openai_api_key="vllm-token",
    openai_api_base="https://ws-02.wade0426.me/v1",
    temperature=0
)

# 3. 設計 Prompt
formal_prompt = ChatPromptTemplate.from_template("你是一位專業顧問，針對「{topic}」撰寫 LinkedIn 貼文（繁體中文）。")
humorous_prompt = ChatPromptTemplate.from_template("你是一位幽默小編，針對「{topic}」撰寫 Threads 貼文（繁體中文）。")

# 4. 使用 RunnableParallel 串接不同的模型
# formal 走 Llama 路線，humorous 走 Gemma 路線
chain = RunnableParallel(
    llama_version=formal_prompt | llm_llama | StrOutputParser(),
    gemma_version=humorous_prompt | llm_gemma | StrOutputParser()
)

print("--- 雙模型 AI Agent 啟動 (Llama vs Gemma) ---")
topic = input("請輸入貼文主題：")

# 批次處理並觀察兩台伺服器同時運作的總耗時
start_time = time.time()
results = chain.invoke({"topic": topic})
duration = time.time() - start_time

print(f"\n[Llama-3.3-70B 專業版]:\n{results['llama_version']}\n")
print(f"[Gemma-3-27b 幽默版]:\n{results['gemma_version']}\n")
print(f"--- 雙機平行處理總耗時: {duration:.2f} 秒 ---")