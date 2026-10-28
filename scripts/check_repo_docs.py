"""Validate intra-repository Markdown links without network access.

Usage (from any working directory): python scripts/check_repo_docs.py
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "TUTORIAL.md",
    ROOT / "docs" / "FACTORY_MODE.md",
    ROOT / "docs" / "ABLATION_RESEARCH.md",
    ROOT / "configs" / "ablations" / "README.md",
]
LINK = re.compile(r"!?\[[^\]]+\]\(([^)]+)\)")
REMOTE = ("https://", "http://", "mailto:", "data:")


def main() -> int:
    problems = []
    count = 0
    for document in DOCS:
        if not document.is_file():
            problems.append(f"Missing document: {document.relative_to(ROOT)}")
            continue
        text = document.read_text(encoding="utf-8")
        for destination in LINK.findall(text):
            destination = destination.split("#", 1)[0].strip()
            if not destination or destination.startswith(REMOTE):
                continue
            # Basic Markdown links only: refuse out-of-tree links and missing paths.
            path = (document.parent / unquote(destination)).resolve()
            count += 1
            if not path.is_relative_to(ROOT):
                problems.append(f"{document.relative_to(ROOT)}: link escapes repository: {destination}")
            elif not path.exists():
                problems.append(f"{document.relative_to(ROOT)}: missing target: {destination}")
    for problem in problems:
        print("ERROR:", problem)
    print(f"Checked {count} local links in {len(DOCS)} documentation files.")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
