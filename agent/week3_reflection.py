from __future__ import annotations

import time, json
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from .tools import get_tools, reflect_answer, save_markdown_file

load_dotenv()

@dataclass
class Guardrails:
    max_steps: int = 10
    max_retries: int = 2
    max_total_seconds: int = 60

SYSTEM_PROMPT = """You are a research assistant for arXiv paper discovery.

When the user asks for papers:
- Call arxiv_search_papers first.
- Then write a Markdown list of papers with:
  - Title
  - Authors
  - Published date
  - arXiv link
- Do NOT invent papers. Only use tool output.
- After writing the final Markdown, call save_markdown_file if the user requests saving.
"""


def _clean_text(content) -> str:
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
        return " ".join(parts).strip()
    return str(content).strip()

def _force_write_answer(llm, messages, query: str) -> str:
    want_papers = any(k in query.lower() for k in ["arxiv", "paper", "papers"])

    if want_papers:
        requirement = (
            "Write the final answer NOW in Markdown.\n"
            "No tool calls.\n"
            "Return a list of papers. For EACH paper include:\n"
            "- Title\n- Authors (all)\n- Published date\n- arXiv link\n"
            "Do NOT invent papers; only use tool outputs already in the chat.\n"
        )
    else:
        requirement = (
            "Write the final answer NOW in plain text.\n"
            "No tool calls.\n"
            "Include at least one source URL for factual claims.\n"
        )

    for _ in range(3):
        ai = llm.invoke(messages + [HumanMessage(content=requirement)])
        text = _clean_text(ai.content)
        if text:
            return text
    return ""


def run_week3_reflection_agent(
    query: str,
    output_md: str = "report_week3.md",
    guards: Guardrails = Guardrails(),
):
    
    from .tools import arxiv_search_papers, search_tool, save_markdown_file, reflect_answer

    want_papers = any(k in query.lower() for k in ["arxiv", "paper", "papers"])
    if want_papers:
        tools = [arxiv_search_papers, save_markdown_file, reflect_answer]
    else:
        tools = [search_tool, save_markdown_file, reflect_answer]
        tool_map = {t.name: t for t in tools}

        llm = ChatGoogleGenerativeAI(model="gemini-pro-latest", temperature=0).bind_tools(tools)

        start = time.time()
        retries = 0
        steps = 0
        best_draft = ""

        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)]

        print("---- WEEK 3 IMPROVED AGENT START ----")

        while True:
            if time.time() - start > guards.max_total_seconds:
                print("⛔ TIMEOUT reached. Finalizing best draft.")
                break
            if steps >= guards.max_steps:
                print("⛔ MAX STEPS reached. Finalizing best draft.")
                break

            steps += 1
            print(f"=== STEP {steps}: DRAFT ===")

            ai: AIMessage = llm.invoke(messages)
            messages.append(ai)

            # Execute tool calls (can be multiple)
            if ai.tool_calls:
                for call in ai.tool_calls:
                    tool_name = call["name"]
                    args = call.get("args", {})
                    print(f"🛠️ TOOL CALL: {tool_name} args={args}")

                    tool = tool_map.get(tool_name)
                    obs = tool.invoke(args) if tool else f"Unknown tool: {tool_name}"
                    obs_text = _clean_text(obs)

                    messages.append(ToolMessage(content=obs_text, tool_call_id=call["id"]))
                    print(f"👀 OBSERVE: got {len(obs_text)} chars")

                # IMPORTANT: after tools, force an answer
                draft = _force_write_answer(llm, messages, query)
            else:
                # No tools called; still force answer if content is empty
                draft = _clean_text(ai.content) or _force_write_answer(llm, messages)

            if draft:
                best_draft = draft

            print("🧾 DRAFT (preview):")
            print((best_draft[:400] + "...") if len(best_draft) > 400 else best_draft)

            print("=== REFLECTION GATE ===")
            reflection = json.loads(reflect_answer.invoke({"question": query, "draft_answer": best_draft}))
            print(reflection)

            sufficient = bool(reflection.get("sufficient"))
            next_action = reflection.get("next_action", "rewrite")

            if sufficient or next_action == "finalize":
                break

            if retries >= guards.max_retries:
                print("⛔ MAX RETRIES reached. Finalizing best draft anyway.")
                break

            retries += 1
            print(f"🔁 RETRY {retries}/{guards.max_retries} ({next_action})")

            if next_action == "search_more":
                suggested = reflection.get("suggested_queries") or []
                hint = f"Suggested queries: {suggested}" if suggested else ""
                messages.append(HumanMessage(content=f"Do more research with search. {hint} Then answer with source URLs."))
            else:
                issues = reflection.get("issues", [])
                messages.append(HumanMessage(content=f"Rewrite and fix: {issues}. Include source URLs."))

            save_markdown_file.invoke({"content": best_draft, "filename": output_md})
        print("---- WEEK 3 IMPROVED AGENT END ----")
        print(
            f"Saved: {output_md} | "
            f"Steps={steps}, Retries={retries}, Elapsed={int(time.time()-start)}s"
        )
        return {
            "final": best_draft,
            "steps": steps,
            "retries": retries,
        }


if __name__ == "__main__":
    print("week3_reflection main is running...")
    
    run_week3_reflection_agent(
        "Find 5 papers related to computer vision on arXiv. "
        "For each paper list title, all authors, published date, and arXiv link. "
        "Save to cv_papers.md",
        output_md="cv_papers.md",
    )

