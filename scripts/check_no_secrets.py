#!/usr/bin/env python3
"""Fail if anything that looks like a credential reached the repo or the built site.

The published site is static and world-readable, so a key that lands in it is exposed
to anyone who opens the page. This is a cheap tripwire, not a substitute for keeping
the key in an encrypted secret - it catches the accident of a key being pasted into a
config file, baked into a data payload, or echoed into a build artifact.

    python scripts/check_no_secrets.py [paths...]
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Directories that are never scanned - dependencies, caches and version-control data.
SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache",
}

#: Only text-like files can leak a key in a readable form.
TEXT_SUFFIXES = {
    ".py", ".js", ".mjs", ".ts", ".json", ".html", ".css", ".md", ".yml", ".yaml",
    ".txt", ".toml", ".cfg", ".ini", ".sh", ".webmanifest", "",
}

#: Patterns that indicate a credential rather than a reference to one. Deliberately
#: narrow: `apikey=` followed by a literal, not the word "apikey" on its own.
PATTERNS = [
    (re.compile(r"apikey=[A-Za-z0-9]{12,}", re.I), "literal apikey= value"),
    (re.compile(r"api_key=[A-Za-z0-9]{12,}", re.I), "literal api_key= value"),
    (re.compile(r"token=[A-Za-z0-9]{20,}", re.I), "literal token= value"),
    (
        re.compile(r"""(FMP_API_KEY|API_KEY)\s*[:=]\s*["'][A-Za-z0-9]{12,}["']"""),
        "hardcoded key assignment",
    ),
    (re.compile(r"\bghp_[A-Za-z0-9]{30,}"), "GitHub personal access token"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key id"),
]

#: Lines carrying these markers are documentation or detection logic, not secrets.
ALLOW_MARKERS = ("noqa: secrets", "check_no_secrets")


def iter_files(paths: list[Path]):
    for base in paths:
        if base.is_file():
            yield base
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                path = Path(dirpath) / name
                if path.suffix.lower() in TEXT_SUFFIXES:
                    yield path


def scan(paths: list[Path]) -> list[str]:
    findings: list[str] = []

    # The live key, if this runs somewhere it is set. Catches the exact value even if
    # it appears in a shape the generic patterns miss.
    live_key = os.environ.get("FMP_API_KEY") or os.environ.get("API_KEY") or ""

    for path in iter_files(paths):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path

        if live_key and len(live_key) >= 12 and live_key in text:
            findings.append(f"{rel}: contains the live API key")

        for lineno, line in enumerate(text.splitlines(), start=1):
            if any(marker in line for marker in ALLOW_MARKERS):
                continue
            for pattern, label in PATTERNS:
                if pattern.search(line):
                    findings.append(f"{rel}:{lineno}: {label}")
                    break
    return findings


def main() -> int:
    args = sys.argv[1:]
    targets = [Path(a) for a in args] if args else [
        ROOT / "backend", ROOT / "frontend", ROOT / "scripts", ROOT / ".github",
        ROOT / "README.md", ROOT / "requirements.txt",
    ]
    targets = [t for t in targets if t.exists()]

    # The built site, when present, is the highest-risk artifact: it is served publicly.
    site = ROOT / "site"
    if site.exists():
        targets.append(site)

    findings = scan(targets)
    if findings:
        print("Potential credentials found:", file=sys.stderr)
        for f in findings:
            print(f"  {f}", file=sys.stderr)
        return 1

    scanned = ", ".join(str(t.relative_to(ROOT)) for t in targets)
    print(f"No credentials found in: {scanned}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
