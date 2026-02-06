import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range

# 1. 連接 Qdrant 並建立 Collection (參考圖 2 & 圖 3)
client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "test_collection"

# 依據助教截圖指示：size 記得跟 model 的 embedding size 一致
# 截圖標註為 4096
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
)

# 2 & 3. 使用 API 獲得向量 (參考圖 4 & 圖 6)
def get_embeddings(texts):
    API_URL = "https://ws-04.wade0426.me/embed"
    data = {
        "texts": texts,
        "task_description": "檢索技術文件",
        "normalize": True
    }
    response = requests.post(API_URL, json=data)
    if response.status_code == 200:
        return response.json()["embeddings"]
    else:
        print(f"錯誤碼: {response.status_code}")
        return None

# 準備五個以上的 Points
print("請輸入五筆（包括）以上資料")

while(1):
    n = int(input())
    if n >= 5:
        break
    else:
        print("請重新輸入")
input_texts = [input() for i in range(n)]
embeddings = get_embeddings(input_texts)

# 4. 嵌入到 VDB (參考圖 5)
points = []
for i, (txt, vec) in enumerate(zip(input_texts, embeddings)):
    points.append(PointStruct(
        id=i + 1,
        vector=vec,
        payload={"text": txt, "year": 2024 + i} # payload 包含 text 內容
    ))

client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"成功將 {len(points)} 筆資料嵌入至 Qdrant。")

# 5. 召回內容 (參考圖 6 & 圖 7)
print("\n請輸入你想搜尋的內容：")
query_text = [input()]
query_vector = get_embeddings(query_text)[0]

# 執行搜尋 (使用截圖中的 query_points 語法)
search_result = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit= n
)

print("\n--- 召回結果 ---")
for point in search_result.points:
    print(f"ID: {point.id}")
    print(f"相似度分數: {point.score:.4f}")
    print(f"內容: {point.payload['text']}")
    print("-" * 20)