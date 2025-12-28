from __future__ import annotations

from .week3_reflection import run_week3_reflection_agent
import importlib


def run_baseline(query: str, out_md: str):
    """
    Calls your week2 baseline script function if you have one.
    If your baseline is only runnable as a script, simplest is:
    - copy its core logic into a function called run_baseline(query, out_md)
    inside agent_baseline.py
    """
    baseline = importlib.import_module("agent_baseline")

    if hasattr(baseline, "run_agent_with_logs"):
        # If your week2 function exists but doesn’t save, you can modify baseline to save.
        baseline.run_agent_with_logs(query, thread_id="baseline-exp")
        # Minimal fallback: create a placeholder (so experiment file always completes)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("Baseline ran; please update agent_baseline.py to return/save final output.\n")
        return {"final": "Baseline output not captured (update baseline to return final text)."}
    elif hasattr(baseline, "run_baseline"):
        return baseline.run_baseline(query, out_md)
    else:
        raise RuntimeError("Please expose a function in agent_baseline.py: run_baseline(query, out_md) returning {'final': text}.")


def make_table(rows):
    lines = []
    lines.append("# Week 3 Comparison (Before vs After)\n")
    lines.append("| Task | Baseline chars | Improved chars | Improved steps | Improved retries |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in rows:
        lines.append(f"| {r['task']} | {r['b_chars']} | {r['i_chars']} | {r['i_steps']} | {r['i_retries']} |")
    return "\n".join(lines)


def main():
    tasks = [
            ("Find 5 papers related to computer vision on arXiv. List title, authors, published date, and arXiv link. Save to cv_papers.md", "cv"),
            ("Find 5 arXiv papers about vision transformers (ViT). List title, authors, published date, and arXiv link. Save to vit_papers.md", "vit"),
    ]

    rows = []

    for q, tag in tasks:
        baseline_out = f"{tag}_baseline.md"
        improved_out = f"{tag}_improved.md"

        b = run_baseline(q, baseline_out)
        i = run_week3_reflection_agent(q, improved_out)

        b_text = (b.get("final") or "")
        i_text = (i.get("final") or "")

        rows.append({
            "task": tag,
            "b_chars": len(b_text),
            "i_chars": len(i_text),
            "i_steps": i.get("steps", 0),
            "i_retries": i.get("retries", 0),
        })

    table = make_table(rows)
    with open("week3_comparison.md", "w", encoding="utf-8") as f:
        f.write(table)

    print("Saved: week3_comparison.md")


if __name__ == "__main__":
    main()
