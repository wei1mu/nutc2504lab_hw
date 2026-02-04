from openai import OpenAI

client = OpenAI(
    base_url = "https://ws-02.wade0426.me/v1",
    api_key = "vllm-token"
)
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "q"]:
        print("Bye!")
        break

response = client.chat.completions.create(
    modle="google/gemma-3-27b-it",
    messanges=[
        {"role": "system", "content": "你是一個繁體中文的聊天機器人，請簡潔答覆"},
        {"role": "user", "content": user_input}
    ],
    temperature=0.7,
    max_token=128
)