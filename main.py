import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Import from your new modules
from state import AgentState
from nodes import researcher_node, critic_node, generator_node, should_continue, get_text

# Load secrets once at the start
load_dotenv()

def build_graph():
    # 1. Initialize Graph
    builder = StateGraph(AgentState)

    # 2. Add Nodes
    builder.add_node("researcher", researcher_node)
    builder.add_node("critic", critic_node)
    builder.add_node("generator", generator_node)

    # 3. Add Edges
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "critic")

    # 4. Conditional Loop
    builder.add_conditional_edges(
        "critic",
        should_continue,
        {
            "researcher": "researcher",
            "generator": "generator",
        }
    )
    builder.add_edge("generator", END)

    # 5. Compile with Memory
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

def run_agent(query: str):
    graph = build_graph()
    print(f"\n🚀 STARTING TASK: {query}")
    
    inputs = {"messages": [HumanMessage(content=query)]}
    config = {"configurable": {"thread_id": "main_thread_1"}, "recursion_limit": 15}

    for step in graph.stream(inputs, config):
        for key, value in step.items():
            last_msg = value["messages"][-1]
            clean_last_msg = get_text(last_msg)
            if key == "critic":
                print(f"\n👨‍⚖️ CRITIC: {clean_last_msg}")
            elif key == "researcher":
                 # Simplify output for readability
                if hasattr(last_msg, "content") and last_msg.content:
                    print(f"🤖 RESEARCHER: {clean_last_msg}")
            elif key == "generator":
                print(f"\n📝 GENERATOR: {clean_last_msg}")
                # Optional: Check if file tool was called
                if last_msg.tool_calls:
                    print(f"   (Saving file: {last_msg.tool_calls[0]['args'].get('filename')})")

if __name__ == "__main__":
    # Test Run
    run_agent("Find info on solid state batteries and save to batteries.md")