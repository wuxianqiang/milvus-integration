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
from langchain.tools import tool

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
    # search_kwargs={"k": 1},
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

@tool
def search_kb(query: str) -> str:
    """在企业知识库中检索资料。query 应为一个具体问题或关键词。"""
    # 权限过滤同样在这里生效（§7）；写死成 public 是因为工具函数拿不到登录态，
    # 真实项目要用闭包或依赖注入把用户权限传进来
    docs = retriever.invoke(query)
    # 返回值必须是字符串，而且要让模型能看懂「什么都没找到」
    # 返回空字符串会让模型困惑，明确说「没有」它才知道该换个词再试
    if not docs:
        # 这句话是给模型看的，它读到之后会自己改写查询词重试（§8.3）
        return "没有检索到相关资料。"
    # 这里特意把 chunk_id 也给模型，因为系统提示词要求它标注来源
    return format_docs(docs)

# create_agent 内置了「模型决策 → 调工具 → 把结果喂回模型」的循环（第 9 章）
agent = create_agent(
    # 和固定链用同一个模型，保证对比公平
    model="openai:qwen4-max",
    # 工具列表，这里只有一个检索工具
    tools=[search_kb],
    system_prompt=(
        # 「必须先检索」这句很关键：不写模型可能直接用训练知识回答
        "你是企业知识库助手。回答任何业务问题前，必须先用 search_kb 工具检索资料。"
        # 拒答要求，作用同 §4.3
        "只依据检索到的资料回答，资料里没有就说无法回答。"
        # 让它标注 chunk_id，这样我们能核对它到底用了哪条资料
        "回答时标注来源的 chunk_id。"
    ),
)