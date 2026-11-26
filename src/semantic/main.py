from langchain_core.documents import Document
from langchain_milvus import Milvus
import getpass
import os
from langchain_openai import OpenAIEmbeddings


documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]



if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


URI = "./milvus_example.db"

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
)

# Search by string 字符串搜索
# results = vector_store.similarity_search(
#     "How many distribution centers does Nike have in the US?"
# )

# print(results[0])

# Return scores 返回相似度打分
# results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
# doc, score = results[0]
# print(f"Score: {score}\n")
# print(doc)


# Search by vector 通过向量搜索
# embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")
# results = vector_store.similarity_search_by_vector(embedding)
# print(results[0])

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)

# search_type，当文档有很多重复的时候选择mmr，其他情况一律使用similarity
# "similarity"（默认）：基于向量之间的相似度（如余弦相似度、欧氏距离等）进行常规的相似性检索，返回最相似的 k 个文档。
# "mmr"：最大边际相关性（Maximal Marginal Relevance）。它在保证检索结果与查询内容相关的前提下，通过调整惩罚系数来最大化结果的多样性，从而避免返回大量内容高度重复的文档片段。
# "similarity_score_threshold"：基于相似度阈值的检索。它只返回相似度得分高于指定阈值的文档。