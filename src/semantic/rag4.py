import os
from langchain_core.documents import Document
from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

# 1. 设置 OpenAI API Key
load_dotenv(override=True)

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
    embedding=DashScopeEmbeddings(model="qwen3.7-text-embedding"),
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
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

def format_docs(docs: list[Document]) -> str:
    """把检索结果拼成带编号的文本，编号供模型引用。"""
    # 逐条累积成行，最后统一用换行连接
    lines = []
    # enumerate 的 start=1 让编号从 [1] 开始，符合人的习惯
    for i, d in enumerate(docs, 1):
        # 用 get 而不是下标：faq.txt 这类记录 metadata 可能不全（§3）
        source = d.metadata.get("source", "未知")
        # 编号 + 来源 + 正文，来源写进 context 是有意的，理由见 §4.4
        lines.append(f"[{i}] （来源：{source}）{d.page_content}")
    # 一条资料一行，模型解析起来最稳
    return "\n".join(lines)

# 提示模板：system 放规则和资料，human 放用户问题
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            # system 角色：约束模型行为，用户看不到这段
            "system",
            # 第一句划定信息边界：不许用训练时记住的知识
            "你是企业知识库助手。只能依据下面提供的【资料】回答问题。\n"
            # 用「规则：」起头，让模型把下面几条当成清单而不是散文
            "规则：\n"
            # 规则 1 是拒答规则，这句话的作用见 §6.3
            "1. 资料中没有相关信息时，直接回答「根据现有资料无法回答该问题」，不要编造。\n"
            # 规则 2 要求标注引用，编号来自下面 format_docs 生成的 [1] [2] [3]
            "2. 每条结论后面用 [编号] 标注依据的资料，例如 [1]。\n"
            # 规则 3 约束输出风格，防止把资料整段抄回来
            "3. 回答简洁，不要复述资料原文。\n\n"
            # {context} 是模板变量，运行时会被检索结果替换
            "【资料】\n{context}",
        ),
        # human 角色：{question} 同样是模板变量
        ("human", "{question}"),
    ]
)
# temperature=0 让生成尽量稳定，RAG 场景不需要创造性
model = init_chat_model("openai:qwen4-max", temperature=0)

chain = (
    # 嵌在序列里时，常直接写字典，不必写 RunnableParallel：字典字面量会被自动当成并行
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    # 提示模板
    | PROMPT
    # 模型
    | model
    # 取出 .content，得到纯字符串
    | StrOutputParser()
)

query = "What is Milvus database?"
print(chain.invoke(query))