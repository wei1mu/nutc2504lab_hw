from openai import OpenAI

# 1. 配置 Client
client = OpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="vllm-token"
)

print("連線成功！現在可以開始對話了（輸入 exit 結束）。")

while True:
    user_input = input("\n我：")
    if user_input.lower() == 'exit':
        break
        
    try:
        # 2. 發送請求並開啟 stream=True
        response = client.chat.completions.create(
            model="google/gemma-3-27b-it",
            messages=[{"role": "user", "content": user_input}],
            stream=True  
        )
        
        print("AI：", end="", flush=True)
        
        # 3. 用迴圈把碎片 (chunks) 拼湊出來
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                # 只印出 AI 回傳的內容，不加任何額外字元
                print(chunk.choices[0].delta.content, end="", flush=True)
        
        print() 
        
    except Exception as e:
        print(f"\n發生錯誤：{e}")