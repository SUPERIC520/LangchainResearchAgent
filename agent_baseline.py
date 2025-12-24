import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent

# 1. LOAD SECRETS
load_dotenv()

# 2. TOOLS
search_tool = TavilySearch(max_results=2)
tools = [search_tool]

# 3. MODEL
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0
)

# 4. AGENT
# This creates a compiled graph automatically. No more "AgentExecutor" needed.
agent_graph = create_agent(llm, tools=tools)

# 5. RUN
# The input format is slightly different: it expects a list of messages.
inputs = {"messages": [("user", "What are the latest developments in solid state batteries as of 2025? Answer with only pure text.")]}

print("\n--- AGENT THINKING ---\n")
response = agent_graph.invoke(inputs)

print("\n--- FINAL ANSWER ---\n")
# The response is the final state of the graph. The last message is the answer.
print(response["messages"][-1].content)