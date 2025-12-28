from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent
from langgraph.graph import END

# Import tools
from tools import get_research_tools, get_generator_tools

# Load secrets immediately
load_dotenv()

# --- CONFIGURATION ---
gemini = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

# --- HELPER FUNCTION: CLEAN GEMINI OUTPUT ---
def get_text(message):
    """
    Extracts plain text from a message, handling cases where 
    Gemini returns a list of content parts.
    """
    content = message.content
    if isinstance(content, list):
        # Join all text parts together
        return " ".join([part.get("text", "") for part in content if "text" in part])
    return content

# --- 1. RESEARCHER NODE (Gather Data) ---
researcher_prompt = """You are a Research Specialist.
Your goal is ONLY to gather comprehensive information on the user's topic.
Do NOT write the final report. Do NOT save any files.
Just output a detailed summary of your findings for the Reviewer."""

# Note: We use get_research_tools() here
researcher_agent = create_agent(gemini, tools=get_research_tools(), system_prompt=researcher_prompt)

def researcher_node(state):
    return researcher_agent.invoke(state)

# --- 2. CRITIC NODE (Review Data) ---

def critic_node(state):
    messages = state["messages"]
    last_message = messages[-1]
    
    # 1. Clean the input
    clean_content = get_text(last_message)

    # 2. The Critic's Prompt
    critique_prompt = f"""
    You are a Content Reviewer.
    Review the Researcher's notes below:
    "{clean_content}"
    
    TASK:
    - If the content is empty or just instructions: Respond "RETRY: No content found."
    - If the content looks like a valid summary: Respond "APPROVED".
    - Do NOT be pedantic.
    - Do not approve first try; ensure quality.
    """
    
    response = gemini.invoke([HumanMessage(content=critique_prompt)])
    
    # 3. Feedback Routing
    # If we are sending it back to the Researcher, we must wrap it as a HumanMessage.
    # This forces the Researcher to treat it as a new command from a "Manager".
    final_content = response.content
    
    if "APPROVED" in final_content:
        # If approved, keep it as AI message so the Generator sees the approval
        return {"messages": [AIMessage(content=final_content, name="critic")]}
    else:
        # If rejecting, mask it as a HumanMessage so the Researcher responds to it
        return {"messages": [HumanMessage(content=final_content, name="critic")]}

# --- 3. GENERATOR NODE (Format & Save) ---
generator_prompt = """You are a Technical Writer.
Your goal is to take the research provided in the conversation and write a professional Markdown report.
1. Summarize the findings into a clean format.
2. YOU MUST Use the 'save_markdown_file' tool to save the report to the filename requested by the user.
3. If no filename is given, default to 'research_report.md'.
"""

# Note: We use get_generator_tools() here
generator_agent = create_agent(gemini, tools=get_generator_tools(), system_prompt=generator_prompt)

def generator_node(state):
    # The generator will see the whole history (User request + Research + Critic Approval)
    # It will naturally understand what to write based on that context.
    return generator_agent.invoke(state)

# --- ROUTING LOGIC ---
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    clean_content = get_text(last_message)
    
    if "APPROVED" in clean_content:
        return "generator"
    return "researcher"