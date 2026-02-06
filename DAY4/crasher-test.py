import os
import base64
import requests
import json
import operator
from typing import Annotated, List, TypedDict, Literal
from playwright.sync_api import sync_playwright
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# --- 1. 配置與模型初始化 ---
API_KEY = "你的_API_KEY"
SEARXNG_URL = "https://puli-8080.huannago.com/search"

# VLM 視覺模型 (gemma-3-27b-it)
vlm_llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key=API_KEY,
    model="google/gemma-3-27b-it",
    temperature=0
)

# 主決策模型 (gpt-oss-120b)
main_llm = ChatOpenAI(
    base_url="https://ws-03.wade0426.me/v1",
    api_key=API_KEY,
    model="/models/gpt-oss-120b",
    temperature=0
)

# --- 2. 狀態與工具定義 ---

class AgentState(TypedDict):
    input: str
    knowledge_base: Annotated[list, operator.add]
    queries: List[str]
    is_sufficient: bool
    cache_hit: bool
    final_answer: str

def search_searxng(query: str, limit: int = 1):
    """執行 SearXNG 搜尋"""
    params = {"q": query, "format": "json", "language": "zh-TW"}
    try:
        response = requests.get(SEARXNG_URL, params=params, timeout=10)
        return [r for r in response.json().get('results', []) if 'url' in r][:limit]
    except: return []

def vlm_read_website(url: str, title: str) -> str:
    """使用 Playwright 截圖並由 VLM 摘要內容"""
    with sync_playwright() as p:
        try:
            # 針對環境問題，優先嘗試系統 Chrome
            browser = p.chromium.launch(headless=True, channel="chrome")
            page = browser.new_page(viewport={'width': 1280, 'height': 1200})
            page.goto(url, wait_until="domcontentloaded", timeout=20000)
            page.wait_for_timeout(2000)
            
            # 截圖並轉 Base64
            b64_img = base64.b64encode(page.screenshot()).decode('utf-8')
            browser.close()

            msg = HumanMessage(content=[
                {"type": "text", "text": f"請摘要此網頁核心事實：{title}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
            ])
            return vlm_llm.invoke([msg]).content
        except Exception as e:
            return f"視覺讀取失敗: {e}"

# --- 3. LangGraph 節點實作 ---

def check_cache(state: AgentState):
    """規則：必須使用快取優化"""
    print("Log: 檢查快取...")
    cache = {"1+1": "2", "誰是台積電創辦人": "張忠謀"}
    if state["input"] in cache:
        return {"knowledge_base": [cache[state["input"]]], "cache_hit": True}
    return {"cache_hit": False}

def planner(state: AgentState):
    """規則：決策節點判斷資訊是否足夠"""
    if state.get("cache_hit"): return {"is_sufficient": True}
    
    knowledge = "\n".join(state["knowledge_base"])
    prompt = f"問題：{state['input']}\n目前資訊：{knowledge}\n足以回答嗎？請回傳 Y 或 N"
    res = main_llm.invoke(prompt).content
    return {"is_sufficient": "Y" in res.upper()}

def query_gen(state: AgentState):
    """規則：生成檢索關鍵字"""
    prompt = f"為問題 '{state['input']}' 生成一個搜尋關鍵字"
    res = main_llm.invoke(prompt).content.strip().replace('"', '')
    return {"queries": [res]}

def search_tool(state: AgentState):
    """規則：檢索+視覺處理"""
    query = state["queries"][-1]
    print(f"Log: 正在檢索 {query}...")
    results = search_searxng(query)
    if not results: return {"knowledge_base": ["查無搜尋結果"]}
    
    info = vlm_read_website(results[0]['url'], results[0]['title'])
    return {"knowledge_base": [info]}

def final_answer_node(state: AgentState):
    knowledge = "\n".join(state["knowledge_base"])
    res = main_llm.invoke(f"根據：{knowledge}\n回答問題：{state['input']}").content
    return {"final_answer": res}

# --- 4. 構建與執行工作流 ---

workflow = StateGraph(AgentState)

workflow.add_node("check_cache", check_cache)
workflow.add_node("planner", planner)
workflow.add_node("query_gen", query_gen)
workflow.add_node("search_tool", search_tool)
workflow.add_node("final_answer", final_answer_node)

workflow.set_entry_point("check_cache")
workflow.add_conditional_edges("check_cache", lambda x: "end" if x["cache_hit"] else "next", {"end": "final_answer", "next": "planner"})
workflow.add_conditional_edges("planner", lambda x: "y" if x["is_sufficient"] else "n", {"y": "final_answer", "n": "query_gen"})
workflow.add_edge("query_gen", "search_tool")
workflow.add_edge("search_tool", "planner")
workflow.add_edge("final_answer", END)

app = workflow.compile()

if __name__ == "__main__":
    print("--- 啟動自動查證系統 ---")
    inputs = {"input": "輝達最新的 AI 晶片是什麼？", "knowledge_base": [], "queries": []}
    for output in app.stream(inputs):
        print(output)