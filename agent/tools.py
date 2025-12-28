from __future__ import annotations

from typing import List, Dict, Any
import os
import json
import re
import arxiv
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def save_markdown_file(content: str, filename: str) -> str:
    """Write Markdown content to filename."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"File '{filename}' saved successfully."

# Tavily tool (uses TAVILY_API_KEY in env)
search_tool = TavilySearch(max_results=3)

@tool
def reflect_answer(question: str, draft_answer: str) -> str:
    """
    Reviewer gate for reliability.
    Returns STRICT JSON string with fields:
      - sufficient: bool
      - issues: list[str]
      - next_action: "finalize" | "search_more" | "rewrite"
      - suggested_queries: list[str]
    """
    reviewer = ChatGoogleGenerativeAI(model="gemini-pro-latest", temperature=0)

    prompt = f"""
Return ONLY valid JSON (no markdown fences, no backticks).

User question:
{question}

Draft answer:
{draft_answer}

Decide if the draft is sufficient.

Requirements:
- Must directly answer the question.
- If the task is "list papers", the answer must contain:
  (1) paper titles,
  (2) author names,
  (3) working links (arXiv URL).
- If missing required fields, mark insufficient.

JSON schema:
{{
  "sufficient": true/false,
  "issues": ["..."],
  "next_action": "finalize" | "search_more" | "rewrite",
  "suggested_queries": ["..."]
}}
"""
    r = reviewer.invoke(prompt)
    text = r.content
    if isinstance(text, list):
        text = " ".join([p.get("text", "") for p in text if isinstance(p, dict)])

    text = text.strip()
    text = re.sub(r"^```json\\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\\s*", "", text)
    text = re.sub(r"\\s*```$", "", text)

    try:
        obj = json.loads(text)
        obj.setdefault("sufficient", False)
        obj.setdefault("issues", [])
        obj.setdefault("next_action", "rewrite")
        obj.setdefault("suggested_queries", [])
        return json.dumps(obj)
    except Exception:
        return json.dumps({
            "sufficient": False,
            "issues": ["Reflection did not return valid JSON", text[:200]],
            "next_action": "rewrite",
            "suggested_queries": []
        })

@tool
def arxiv_search_papers(
    query: str,
    max_results: int = 5,
    sort: str = "relevance"
) -> str:
    """
    Search arXiv and return JSON list of papers.
    Each paper includes: title, authors, published, link, summary.
    """
    sort_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "submitted": arxiv.SortCriterion.SubmittedDate,
        "last_updated": arxiv.SortCriterion.LastUpdatedDate,
    }
    sort_by = sort_map.get(sort.lower(), arxiv.SortCriterion.Relevance)

    search = arxiv.Search(query=query, max_results=max_results, sort_by=sort_by)

    papers: List[Dict[str, Any]] = []
    for r in search.results():
        papers.append({
            "title": r.title,
            "authors": [a.name for a in r.authors],
            "published": str(r.published.date()) if r.published else None,
            "link": r.entry_id,
            "summary": (r.summary or "")[:600]
        })

    return json.dumps(papers, ensure_ascii=False)

@tool
def local_read_text_file(path: str) -> str:
    """Read a local UTF-8 text file by path."""
    if not os.path.exists(path):
        return f"File not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

@tool
def quick_python(expr: str) -> str:
    """Evaluate a simple python expression. Example: '2+2'."""
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Python error: {e}"

def get_tools():
    return [
        arxiv_search_papers,
        search_tool,
        save_markdown_file,
        reflect_answer,
        local_read_text_file,
        quick_python,
    ]
