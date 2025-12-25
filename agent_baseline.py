import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_core.tools import tool 
from langgraph.checkpoint.memory import MemorySaver

# 1. LOAD SECRETS
load_dotenv()

# 2. TOOLS
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

tools = [
    search_tool,
    save_markdown_file
]

# 3. MODEL
llm = ChatGoogleGenerativeAI(
    model="gemini-pro-latest",
    temperature=0
)

memory = MemorySaver()

# 4. DEFINE SYSTEM PROMPT
system_prompt = """You are a Senior Research Analyst.
Follow this loop:
1. Plan: Analyze the user's request.
2. Act: Use search tools to gather data.
3. Observe: Read the results.
4. Repeat: If you need more info, search again.
5. Finalize: Save the summary to a file ONLY when you have a complete answer.
"""

# 5. AGENT
# This creates a compiled graph automatically. No more "AgentExecutor" needed.
agent_graph = create_agent(
    llm, 
    tools=tools, 
    system_prompt=system_prompt,
    checkpointer=memory)

# 6. RUN WITH EXPLICIT LOGS
def run_agent_with_logs(query, thread_id="1"):
    print(f"\nExample Task: {query}")
    print("--- STARTING AGENT LOOP ---\n")
    
    # Config tells the memory which "conversation thread" this is
    config = {"configurable": {"thread_id": thread_id}}
    
    # We use .stream() instead of .invoke() to see the steps
    inputs = {"messages": [("user", query)]}
    
    # This loop prints the "Thinking..." process
    for step in agent_graph.stream(inputs, config, stream_mode="values"):
        # The 'step' contains the current state of messages
        last_message = step["messages"][-1]
        
        # If the last message is from the AI, it's a Thought or a Tool Call
        if last_message.type == "ai":
            if last_message.tool_calls:
                print(f"🛠️  PLAN: Agent decided to call tool: {last_message.tool_calls[0]['name']}")
            else:
                # CLEAN UP GOOGLE'S RAW OUTPUT
                content = last_message.content
                if isinstance(content, list):
                    # Extract just the text part from the list
                    content = " ".join([part.get("text", "") for part in content if "text" in part])
                
                print(f"🤖 THINKING: {content}")
        
        # If the last message is a ToolMessage, it's the Observation
        elif last_message.type == "tool":
            print(f"👀 OBSERVE: Tool output received ({len(last_message.content)} chars)")

    print("\n--- AGENT FINISHED ---")

# First Run
run_agent_with_logs("Find the release date of GPT-4o and write it to gpt_dates.md")

# Second Run (Testing Memory)
# We ask a follow-up question. The agent should remember the previous context.
print("\n\n--- TESTING MEMORY (New Turn) ---")
run_agent_with_logs("What date did I just ask you about?", thread_id="1")