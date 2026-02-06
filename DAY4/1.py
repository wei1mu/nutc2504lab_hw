import os
import base64
import requests
import operator
from typing import Annotated, List, TypedDict
from playwright.sync_api import sync_playwright
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# --- 1. 配置與模型初始化 ---
API_KEY = "你的_API_KEY" # ⬅️ 這裡一定要改！
SEARXNG_URL = "https://puli-8080.huannago.com/search"

# 定義模型
vlm_llm = ChatOpenAI(base_url="https://ws-02.wade0426.me/v1", api_key=API_KEY, model="google/gemma-3-27b-it", temperature=0)
main_llm = ChatOpenAI(base_url="https://ws-03.wade0426.me/v1", api_key=API_KEY, model="/models/gpt-oss-120b", temperature=0)

# --- 2. 狀態定義 ---
class AgentState(TypedDict):
    input: str
    knowledge_base: Annotated[list, operator.add]
    queries: List[str]
    is_sufficient: bool
    cache_hit: bool

# --- 3. 核心工具函數 ---
def search_searxng(query: str):
    print(f"🔍 [工具] 正在搜尋: {query}...")
    try:
        res = requests.get(SEARXNG_URL, params={"q": query, "format": "json"}, timeout=10).json()
        return res.get('results', [])[:1]
    except: return []

def vlm_read(url: str):
    print(f"📸 [視覺] 正在讀取網頁: {url}...")
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=30000)
            img_b64 = base64.b64encode(page.screenshot()).decode('utf-8')
            browser.close()
            msg = HumanMessage(content=[
                {"type": "text", "text": "摘要此網頁事實"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ])
            return vlm_llm.invoke([msg]).content
        except Exception as e: return f"讀取失敗: {e}"

# --- 4. LangGraph 節點 ---
def check_cache_node(state: AgentState):
    print("➡️ [節點] 檢查快取中...")
    cache = {"你好": "你好！我是查證助手。"} # 簡單快取範例
    if state["input"] in cache:
        return {"knowledge_base": [cache[state["input"]]], "cache_hit": True}
    return {"cache_hit": False}

def planner_node(state: AgentState):
    print("➡️ [節點] 決策評估中...")
    if state.get("cache_hit") or len(state["knowledge_base"]) > 0:
        return {"is_sufficient": True}
    return {"is_sufficient": False}

def query_gen_node(state: AgentState):
    print("➡️ [節點] 生成關鍵字...")
    query = main_llm.invoke(f"生成關鍵字: {state['input']}").content
    return {"queries": [query]}

def search_tool_node(state: AgentState):
    print("➡️ [節點] 執行檢索與視覺處理...")
    res = search_searxng(state["queries"][-1])
    info = vlm_read(res[0]['url']) if res else "查無資料"
    return {"knowledge_base": [info]}

# --- 5. 構建流程圖 ---
workflow = StateGraph(AgentState)
workflow.add_node("check_cache", check_cache_node)
workflow.add_node("planner", planner_node)
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("search_tool", search_tool_node)

workflow.set_entry_point("check_cache")
workflow.add_conditional_edges("check_cache", lambda x: "end" if x["cache_hit"] else "plan", {"end": END, "plan": "planner"})
workflow.add_conditional_edges("planner", lambda x: "y" if x["is_sufficient"] else "n", {"y": END, "n": "query_gen"})
workflow.add_edge("query_gen", "search_tool")
workflow.add_edge("search_tool", "planner")

app = workflow.compile()

# --- 6. 執行測試 (這一段保證有輸出！) ---
if __name__ == "__main__":
    test_input = input()
    print(f"\n🚀 啟動任務: {test_input}")
    
    # 使用 stream 確保每個步驟都印出來
    for output in app.stream({"input": test_input, "knowledge_base": [], "queries": []}):
        for node, data in output.items():
            print(f"✅ {node} 執行完成，目前知識庫筆數: {len(data.get('knowledge_base', []))}")
    
    print("\n✨ 任務圓滿結束！")