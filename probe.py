#!/usr/bin/env python3
"""
probe.py — Scan Cython .pxd files for ``cdef extern from`` headers,
extract every ``#define`` macro from those headers, and emit a JSON
inventory that ``build.sh --list-args`` can display.

Usage::

    python probe.py [--output macros.json] [--project-root /path/to/repo]

The output JSON is committed to the repo and treated as the canonical
inventory.  ``build.sh`` regenerates it automatically if it is missing.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PACKAGE_NAME = "algo_engine"

# Directories (relative to project root) that contain Cython modules we
# actually build.  Everything else (deprecated, profile, …) is skipped.
CYTHON_DIRS: List[str] = [
    "algo_engine",
    "algo_engine/base",
    "algo_engine/base/c_market_data",
    "algo_engine/exchange_profile",
    "algo_engine/engine",
]

# Headers we skip because they are system / third-party headers that we
# don't own (and wouldn't contain useful build-time macros).
SKIP_HEADER_PREFIXES: Tuple[str, ...] = (
    "Python.h",
    "pthread.h",
    "std",
    "cbase/",
    "sys/",
    "math.h",
    "time.h",
    "sched.h",
    "stdlib.h",
    "string.h",
    "stdatomic.h",
    "stdbool.h",
    "stddef.h",
    "stdint.h",
)

# Regex patterns
_RE_CDEF_EXTERN = re.compile(
    r'cdef\s+extern\s+from\s*"([^"]+)"',
    re.MULTILINE,
)
_RE_DEFINE = re.compile(
    r'^\s*#\s*define\s+([a-zA-Z_]\w*)'  # macro name
    r'(?:\([^)]*\))?'  # optional function-like params (skip)
    r'\s+(.*?)'  # value
    r'\s*$',  # trailing whitespace / comment
    re.MULTILINE,
)
_RE_INCLUDE_GUARD = re.compile(r'^[A-Z_]+_H(_.*)?$')
# Check if the char immediately after the macro name is '(' — that means it's a
# function-like macro e.g. ``#define FOO(x) ...``, not a value macro.
_RE_IS_FUNC_MACRO = re.compile(r'#\s*define\s+[a-zA-Z_]\w*\(')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_header_path(include_str: str, project_root: Path) -> Optional[Path]:
    """Convert a ``"algo_engine/.../foo.h"`` string to an absolute Path.

    Returns *None* for system / third-party headers.
    """
    if include_str.startswith(SKIP_HEADER_PREFIXES):
        return None

    # Are we pointing at a local algo_engine header?
    if include_str.startswith(PACKAGE_NAME + "/"):
        return project_root / include_str

    # Relative include inside the same directory (e.g. "c_market_data_config.h"
    # from a pxd in c_market_data/) — we can't resolve without context, so
    # try the project root as a base.
    if "/" not in include_str:
        return None  # ambiguous without the pxd's directory; skip for safety

    return None


def header_is_deprecated(header_path: Path) -> bool:
    """Skip the deprecated c_market_data_deprecated/ and profile/ subtrees."""
    parts = str(header_path).split("/")
    return "c_market_data_deprecated" in parts or "profile" in parts


def extract_macros(header_path: Path) -> List[Dict[str, str]]:
    """Parse a C header and return a list of macro dicts."""
    macros: List[Dict[str, str]] = []
    try:
        text = header_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return macros

    relative = str(header_path.relative_to(header_path.anchor))
    # Make it repo-relative if possible
    try:
        # Walk up to find project root marker
        p = header_path.parent
        while p != p.parent:
            if (p / PACKAGE_NAME).is_dir() and (p / "setup.py").is_file():
                relative = str(header_path.relative_to(p))
                break
            p = p.parent
    except ValueError:
        pass

    for match in _RE_DEFINE.finditer(text):
        name = match.group(1)

        # Skip include-guard-looking macros
        if _RE_INCLUDE_GUARD.match(name):
            continue

        # Skip function-like macros (e.g. ``#define FOO(x) ...``)
        if _RE_IS_FUNC_MACRO.search(match.group(0)):
            continue

        value = match.group(2).strip()

        # Skip empty defines (e.g. ``#define _GNU_SOURCE``) and regex
        # artifacts where a line-continuation or preprocessor directive
        # leaked into the value.
        if not value or value.startswith('#'):
            continue

        # Strip trailing comments from value
        if "//" in value:
            value = value.split("//")[0].strip()
        # Strip block comments
        while "/*" in value:
            start = value.index("/*")
            end = value.find("*/", start)
            if end == -1:
                value = value[:start].strip()
                break
            value = (value[:start] + value[end + 2:]).strip()

        macros.append({
            "name": name,
            "default": value,
            "header": relative,
            "raw_line": match.group(0).strip(),
        })

    return macros


def find_pxd_files(project_root: Path) -> List[Path]:
    """Return all .pxd files inside the directories we build."""
    pxd_files: List[Path] = []
    for d in CYTHON_DIRS:
        dir_path = project_root / d
        if not dir_path.is_dir():
            continue
        for f in sorted(dir_path.rglob("*.pxd")):
            # Skip __infra__.pxd — it's a meta-file, not a real module pxd
            if f.name == "__infra__.pxd":
                continue
            # Skip deprecated
            if header_is_deprecated(f):
                continue
            pxd_files.append(f)
    return pxd_files


def probe(project_root: Path) -> Dict[str, List[Dict[str, str]]]:
    """Main probe function.

    Returns a dict ready to be serialised as JSON.
    """
    pxd_files = find_pxd_files(project_root)
    seen_headers: set[str] = set()
    all_macros: List[Dict[str, str]] = []

    for pxd_path in pxd_files:
        try:
            text = pxd_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        for m in _RE_CDEF_EXTERN.finditer(text):
            include_str = m.group(1)
            header_path = resolve_header_path(include_str, project_root)
            if header_path is None:
                continue
            if not header_path.exists():
                continue
            if header_is_deprecated(header_path):
                continue

            header_key = str(header_path)
            if header_key in seen_headers:
                continue
            seen_headers.add(header_key)

            macros = extract_macros(header_path)
            all_macros.extend(macros)

    # Deduplicate by macro name (first seen wins — deterministic because we
    # process in sorted order).
    seen_names: set[str] = set()
    unique_macros: List[Dict[str, str]] = []
    for m in all_macros:
        if m["name"] not in seen_names:
            seen_names.add(m["name"])
            unique_macros.append(m)

    # Sort by name for stable output
    unique_macros.sort(key=lambda m: m["name"])

    return {"macros": unique_macros}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Probe Cython .pxd extern headers for #define macros"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to write the JSON output (default: <project_root>/macros.json)",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Path to the project root (default: auto-detect from this script's location)",
    )
    args = parser.parse_args()

    # Auto-detect project root
    if args.project_root:
        project_root = Path(args.project_root).resolve()
    else:
        project_root = Path(__file__).resolve().parent

    if not (project_root / PACKAGE_NAME).is_dir():
        print(f"Error: '{PACKAGE_NAME}/' not found under {project_root}", file=sys.stderr)
        sys.exit(1)

    result = probe(project_root)

    output_path = Path(args.output) if args.output else project_root / "macros.json"

    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[probe] Wrote {len(result['macros'])} macros -> {output_path}")


if __name__ == "__main__":
    main()
