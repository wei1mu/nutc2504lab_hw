import random
import json
import os
from typing import Annotated, TypedDict, Union, Literal
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode

# --- 步驟二：配置區 ---
# 根據圖片需求，使用指定的 base_url 與 model
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1", 
    api_key="123",  # 圖片註明 KEY 留空，但程式需有值才能執行
    model="google/gemma-3-27b-it",
    temperature=0
)

# --- 實作：Retry 機制的天氣 API (範例) ---
@tool
def get_weather(city: str):
    """取得指定城市的當前天氣。"""
    # 模擬 Retry 機制：隨機製造失敗
    if random.random() < 0.5:
        return f"{city} 的天氣是晴天，25度。"
    else:
        raise ValueError("天氣伺服器暫時無回應，觸發 Retry 機制測試！")

# 這裡之後可以開始構建你的 StateGraph 邏輯
print("API 配置成功！模型已準備就緒。") 