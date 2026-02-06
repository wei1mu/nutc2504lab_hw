import operator
from typing import Annotated, List, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# --- 1. 模型與狀態定義 ---

# LLM 決策模型 (使用你提供的 gpt-oss-120b)
llm = ChatOpenAI(
    base_url="https://ws-03.wade0426.me/v1",
    api_key="YOUR_API_KEY",
    model="/models/gpt-oss-120b",
    temperature=0
)

class AgentState(TypedDict):
    input: str
    knowledge_base: Annotated[list, operator.add] # 使用累加模式
    queries: List[str]
    next_step: str # 用於控制條件分支
    cache_hit: bool

# 簡單的記憶體快取 (作業要求：必須使用任一優化方式)
GLOBAL_CACHE = {
    "台積電股價": "台積電（2330）目前股價約在 1000 元上下波動（模擬數據）。"
}

# --- 2. 節點函數 (Nodes) ---

def check_cache_node(state: AgentState):
    """快取檢查節點"""
    print("--- [Node] Cache Check ---")
    user_input = state["input"]
    if user_input in GLOBAL_CACHE:
        return {"knowledge_base": [f"快取命中: {GLOBAL_CACHE[user_input]}"], "cache_hit": True}
    return {"cache_hit": False}

def planner_node(state: AgentState):
    """決策節點：判斷資訊是否充足"""
    print("--- [Node] Planner ---")
    if state.get("cache_hit"):
        return {"next_step": "end"}
    
    knowledge = "\n".join(state["knowledge_base"])
    prompt = f"問題：{state['input']}\n目前資訊：{knowledge}\n\n請判斷目前資訊是否足以完整回答問題？只需回答 'Y' 或 'N'。"
    
    res = llm.invoke(prompt).content.strip()
    return {"next_step": "end" if "Y" in res.upper() else "continue"}

def query_gen_node(state: AgentState):
    """關鍵字生成節點"""
    print("--- [Node] Query Gen ---")
    prompt = f"根據問題 '{state['input']}'，請生成一個精準的搜尋關鍵字。"
    query = llm.invoke(prompt).content.strip().replace('"', '')
    return {"queries": [query]}

def search_tool_node(state: AgentState):
    """搜尋與 VLM 整合節點"""
    print("--- [Node] Search & VLM ---")
    query = state["queries"][-1]
    
    # 呼叫你寫的搜尋函數
    results = search_searxng(query, limit=1)
    if not results:
        return {"knowledge_base": ["搜尋不到結果"]}
    
    target = results[0]
    # 呼叫你寫的 VLM 閱讀函數
    vlm_info = vlm_read_website(target['url'], target['title'])
    
    return {"knowledge_base": [f"從 {target['url']} 獲得資訊：{vlm_info}"]}

def final_answer_node(state: AgentState):
    """生成最終回答"""
    print("--- [Node] Final Answer ---")
    knowledge = "\n".join(state["knowledge_base"])
    prompt = f"請根據以下資訊回答問題：{state['input']}\n\n資訊：\n{knowledge}"
    ans = llm.invoke(prompt).content
    return {"final_answer": ans}

# --- 3. 構建 LangGraph 流程圖 ---

workflow = StateGraph(AgentState)

# 新增節點
workflow.add_node("check_cache", check_cache_node)
workflow.add_node("planner", planner_node)
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("search_tool", search_tool_node)
workflow.add_node("final_answer", final_answer_node)

# 設定連線
workflow.set_entry_point("check_cache")
workflow.add_edge("check_cache", "planner")

# 決策分支
workflow.add_conditional_edges(
    "planner",
    lambda x: x["next_step"],
    {
        "end": "final_answer",
        "continue": "query_gen"
    }
)

workflow.add_edge("query_gen", "search_tool")
workflow.add_edge("search_tool", "planner") # 形成循環循環 (Loop)
workflow.add_edge("final_answer", END)

# 編譯
app = workflow.compile()