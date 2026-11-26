from langchain_agentsearch import AgentSearchTool

search = AgentSearchTool(mode="free")
print(search.invoke({"query": "什么是机器学习", "max_results": 3}))