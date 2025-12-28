import os
from typing import TypedDict, Annotated, List, Union
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_core.tools import tool 
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
import operator

# --- LOAD SECRETS ---
load_dotenv()

# --- TOOLS ---
@tool
def save_markdown_file(content: str, filename: str):
    """
    Writes text content to a Markdown file. 
    Use this tool to save your research reports.
    
    Args:
        content: The full text content to write.
        filename: The name of the file (e.g., 'report.md').
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"File '{filename}' saved successfully."

search_tool = TavilySearch(max_results=2)


# --- MODEL ---
llm = ChatGoogleGenerativeAI(
    model="gemini-pro-latest",
    temperature=0
)


# --- AGENTS ---
researcher_tools = [
    search_tool,
    save_markdown_file
]

researcher_memory = MemorySaver()

researcher_prompt = """You are a thorough researcher. 
Do not stop until you have concrete numbers/facts. 
If you receive feedback from the Critic, strictly follow their advice to improve your search."""

researcher_agent = create_agent(
    llm, 
    tools=researcher_tools, 
    system_prompt=researcher_prompt,
    checkpointer=researcher_memory)

# --- NODES ---
def researcher_node(state):
    # This invokes the ReAct agent. It returns a dictionary with the new messages.
    return researcher_agent.invoke(state)

# Node 2: The Critic (The New Guardrail)
def critic_node(state):
    messages = state["messages"]
    last_message = messages[-1]
    
    # We ask the LLM to play the role of a QA Manager
    critique_prompt = f"""
    You are a strict Quality Assurance Manager. 
    Review the Researcher's latest response:
    
    "{last_message.content}"
    
    Check for:
    1. Is the answer detailed and backed by facts?
    2. Did they actually use the file tool if requested?
    3. Is the tone professional?
    
    If it is GOOD, respond with exactly: "APPROVED"
    If it is BAD, respond with "RETRY: " followed by specific instructions on what to fix.
    """
    
    response = llm.invoke([HumanMessage(content=critique_prompt)])
    
    # We return the critic's response as a standard AIMessage
    # We tag it so we can spot it in logs easily
    return {"messages": [AIMessage(content=response.content, name="critic")]}

# --- CONDITIONAL LOGIC (The Router) ---
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the Critic says APPROVED, we finish.
    if "APPROVED" in last_message.content:
        return END
    
    # Otherwise, loop back to the researcher to try again.
    return "researcher"

# --- GRAPH CONSTRUCTION ---
# We use a MessageState (standard list of messages)
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

builder = StateGraph(AgentState)

builder.add_node("researcher", researcher_node)
builder.add_node("critic", critic_node)

# Flow: Start -> Researcher -> Critic -> Check -> (Loop or End)
builder.add_edge(START, "researcher")
builder.add_edge("researcher", "critic")

builder.add_conditional_edges(
    "critic",
    should_continue,
    {
        "researcher": "researcher",
        END: END
    }
)

# Compile with memory and a safety limit (max 10 steps to prevent infinite loops)
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# --- RUNNER ---
def run_week3_agent(query):
    print(f"\n🚀 STARTING TASK: {query}")
    print(f"---" * 10)
    
    inputs = {"messages": [HumanMessage(content=query)]}
    config = {"configurable": {"thread_id": "week3_test_1"}, "recursion_limit": 15}

    for step in graph.stream(inputs, config):
        for key, value in step.items():
            # 'key' is the name of the node (researcher or critic)
            last_msg = value["messages"][-1]
            
            if key == "critic":
                print(f"\n👨‍⚖️ CRITIC: {last_msg.content}")
                print(f"---" * 5)
            elif key == "researcher":
                # Only print the text content to keep it clean
                if hasattr(last_msg, "content") and last_msg.content:
                    print(f"🤖 RESEARCHER: {last_msg.content[:150]}...")

# Run the Test
# We use a vague prompt to force the Critic to trigger a retry
run_week3_agent("Find info on the new battery tech.")