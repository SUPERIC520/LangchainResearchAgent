from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class WorkingMemory:
    """Short-term memory for one run."""
    notes: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)

    def add_note(self, text: str):
        self.notes.append(text)

    def add_source(self, text: str):
        self.sources.append(text)

    def snapshot(self) -> Dict[str, Any]:
        return {"notes": self.notes[-20:], "sources": self.sources[-20:]}
