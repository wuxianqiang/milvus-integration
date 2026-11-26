import os
from langchain_core.documents import Document
from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_openai import OpenAIEmbeddings

# 1. 设置 OpenAI API Key
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# 2. 准备测试文档
documents = [
    Document(page_content="LangChain makes it easy to build contextual AI applications."),
    Document(page_content="Milvus is a high-performance, cloud-native vector database."),
    Document(page_content="BM25 is a ranking function used by search engines to estimate the relevance of documents."),
]

# 3. 定义 Milvus 连接地址（这里以本地 Milvus Lite 文件为例）
URI = "http://localhost:19530"

# 4. 初始化带有稠密嵌入和 BM25 内置函数的 Milvus 向量存储
vectorstore = Milvus.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings(),
    builtin_function=BM25BuiltInFunction(),
    vector_field=["dense", "sparse"],
    connection_args={
        "uri": URI,
        "db_name": "milvus_demo"
    },
    consistency_level="Strong",
    drop_old=True,
)

# 5. 执行混合检索（同时结合语义与关键词匹配）
query = "What is Milvus database?"
results = vectorstore.similarity_search(query, k=2)

for i, doc in enumerate(results):
    print(f"Result {i + 1}: {doc.page_content}")