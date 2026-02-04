# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage, SystemMessage

# # 配置模型
# llm = ChatOpenAI(
#     model="google/gemma-3-27b-it",
#     openai_api_key="vllm-token",
#     openai_api_base="https://ws-02.wade0426.me/v1",
#     streaming=True
# )

# # 定義系統角色（這會強制它說中文）
# messages = [
#     SystemMessage(content="你是一個專業的助理，請一律使用繁體中文（台灣）回答問題。")
# ]

# print("AI連線成功！(輸入 exit 結束)")

# while True:
#     user_input = input("\n：")
#     if user_input.lower() == 'exit':
#         break
    
#     # 將使用者的輸入加入對話紀錄中
#     messages.append(HumanMessage(content=user_input))
    
#     print("AI：", end="", flush=True)
#     try:
#         # 傳入整個 messages 列表
#         response_content = ""
#         for chunk in llm.stream(messages):
#             print(chunk.content, end="", flush=True)
#             response_content += chunk.content
#         print()
        
#         # (選配) 如果你希望它有記憶，可以把 AI 的回覆也存進 messages
#         # messages.append(AIMessage(content=response_content))
        
#     except Exception as e:
#         print(f"\n發生錯誤：{e}")
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# 配置模型 (Temperature 設為 0)
llm = ChatOpenAI(
    model="google/gemma-3-27b-it",
    openai_api_key="vllm-token",
    openai_api_base="https://ws-02.wade0426.me/v1",
    temperature=0,
    streaming=True
)

# 設計不同的寫作風格 Prompt
formal_prompt = ChatPromptTemplate.from_template("你是一位專業的行銷顧問，請針對主題「{topic}」撰寫一篇知性、專業的 LinkedIn 貼文，使用繁體中文。")
humorous_prompt = ChatPromptTemplate.from_template("你是一位幽默的社群小編，請針對主題「{topic}」撰寫一篇有趣、充滿迷因感並吸引年輕人的 Threads 貼文，使用繁體中文。")

# 使用 RunnableParallel 建立平行鏈
# 這裡會同時啟動兩個任務
chain = RunnableParallel(
    formal=formal_prompt | llm | StrOutputParser(),
    humorous=humorous_prompt | llm | StrOutputParser()
)

print("--- 多工 AI Agent 已啟動 (LCEL Parallel) ---")
topic = input("請輸入貼文主題：")

# 流式處理 (Streaming) 
print("\n--- [1] 串流模式 (同時輸出兩種風格) ---")
# 注意：Parallel 的 stream 會以 dict 形式回傳各個分支的碎片
for chunk in chain.stream({"topic": topic}):
    for key, content in chunk.items():
        print(f"\n[{key.upper()} 風格]: {content}", end="", flush=True)
print("\n")

# 批次處理 (Batch) 並紀錄時間 
print("--- [2] 批次模式 (計算處理時間) ---")
start_time = time.time()

# 這裡一次處理一個主題，但內部是平行執行兩個 Prompt
results = chain.invoke({"topic": topic})

end_time = time.time()
duration = end_time - start_time

print(f"結果 1 (專業版):\n{results['formal']}\n")
print(f"結果 2 (幽默版):\n{results['humorous']}\n")
print(f"--- 批次處理總耗時: {duration:.2f} 秒 ---")