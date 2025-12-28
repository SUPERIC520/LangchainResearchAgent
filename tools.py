from langchain_core.tools import tool
from langchain_tavily import TavilySearch

@tool
def save_markdown_file(content: str, filename: str):
    """
    Writes text content to a Markdown file. 
    Use this tool to save your research reports.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"File '{filename}' saved successfully."

# Helper function to get all tools easily
def get_research_tools():
    # Researcher ONLY searches. Cannot save.
    return [TavilySearch(max_results=2)]

def get_generator_tools():
    # Generator ONLY saves.
    return [save_markdown_file]