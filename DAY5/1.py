import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range

# 1. 連接 Qdrant (Docker)
client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "test_collection"

# 獲取向量的函式 (根據圖 4 & 圖 6)
def get_embeddings(texts):
    API_URL = "https://ws-04.wade0426.me/embed"
    payload = {
        "texts": texts,
        "task_description": "檢索技術文件",
        "normalize": True
    }
    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        return response.json()["embeddings"]
    else:
        raise Exception(f"API Error: {response.text}")

# --- 自動偵測維度並建立 Collection (根據圖 3 邏輯) ---
# 先拿一筆資料測試 API 實際產出的維度
sample_vector = get_embeddings(["test"])[0]
detected_size = len(sample_vector)
print(f"目前 API 輸出的維度為: {detected_size}")

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=detected_size, distance=Distance.COSINE),
)

# 2 & 3. 準備 5 個以上的 Point 並獲得向量 (根據圖 5)
texts = ["人工智慧很有趣", "深度學習的應用", "向量資料庫指南", "Docker 部署測試", "Python 程式設計", "大語言模型趨勢"]
vectors = get_embeddings(texts)

# 4. 嵌入到 VDB (根據圖 5)
points = []
for i, (txt, vec) in enumerate(zip(texts, vectors)):
    points.append(PointStruct(
        id=i + 1,
        vector=vec,
        payload={"text": txt, "year": 2024 + i} # 加入 year 方便測試圖 7 的 Filter
    ))

client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"成功嵌入 {len(points)} 個 Point 到 Qdrant。")

# 5. 召回內容 (根據圖 6 & 圖 7)
query_text = ["AI 的好處是什麼？"]
query_vector = get_embeddings(query_text)[0]

# 執行搜尋
search_result = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=3
)

print("\n--- 召回結果 ---")
for point in search_result.points:
    print(f"ID: {point.id} | Score: {point.score:.4f} | 內容: {point.payload['text']}")

# 額外加碼：帶有限制的搜尋 (根據圖 7)
print("\n--- 帶有條件(year >= 2025)的搜尋結果 ---")
filtered_result = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    query_filter=Filter(
        must=[FieldCondition(key="year", range=Range(gte=2025))]
    ),
    limit=3
)
for point in filtered_result.points:
    print(f"ID: {point.id} | Year: {point.payload['year']} | 內容: {point.payload['text']}")