#!/usr/bin/env python3
from __future__ import annotations
import re, bisect
from pathlib import Path
from typing import List, Dict, Optional

# Third-party
from pygments import lex
from pygments.token import Token
from pygments.lexers import get_lexer_by_name, guess_lexer_for_filename
from pygments.util import ClassNotFound

class BuildCommentExtractor:
    # -------- Build file detection --------
    BUILD_FILES = {
        # Java / Gradle / Maven
        "pom.xml",
        "build.gradle", "build.gradle.kts",
        "settings.gradle", "settings.gradle.kts",
        # Go
        "go.mod",
        # Python
        "pyproject.toml", "setup.cfg", "setup.py",
        # JS/TS
        "tsconfig.json",  # JSONC (comments allowed) -> use JS lexer
        # Ruby
        "gemfile", "rakefile",
        # Generic build
        "makefile",
    }
    IGNORE_DIRS = {
        "vendor", "node_modules", "dist", "build", "__pycache__", ".venv", "venv",
        "target", ".tox", ".cache"
    }

    def __init__(self, source: Optional[str] = None, *, filename: Optional[str] = None, path: Optional[str | Path] = None):
        """
        Default: provide `source` (string) and optional `filename` for lexer choice.
        Alternative: set `path` to a file or directory to scan.
        """
        self.source = source
        self.filename = (filename or "UNKNOWN").strip()
        self.path = Path(path) if path is not None else None

    # ---------- Public API ----------
    def extract(self) -> List[Dict]:
        """
        Returns a list of dicts with keys: {"comment","start_line","end_line"}.
        """
        if self.source is not None:
            return self._extract_from_text(self.source, self.filename)

        if self.path is None:
            raise ValueError("Provide `source` (preferred) or a filesystem `path` (file/dir).")

        if self.path.is_file():
            text = self.path.read_text(encoding="utf-8", errors="replace")
            return self._extract_from_text(text, self.path.name)

        if self.path.is_dir():
            out: List[Dict] = []
            for f in self.path.rglob("*"):
                if f.is_file() and self._is_build_file(f):
                    text = f.read_text(encoding="utf-8", errors="replace")
                    out.extend(self._extract_from_text(text, f.name))
            return out

        raise FileNotFoundError(f"Path not found: {self.path}")

    # ---------- Helpers ----------
    @classmethod
    def _is_ignored(cls, path: Path) -> bool:
        return any(part.lower() in cls.IGNORE_DIRS for part in path.parts)

    @classmethod
    def _is_requirements_file(cls, name: str) -> bool:
        n = name.lower()
        return n.startswith("requirements") and n.endswith(".txt")

    @classmethod
    def _is_build_file(cls, path: Path) -> bool:
        if cls._is_ignored(path):
            return False
        name = path.name.lower()
        return (name in cls.BUILD_FILES) or cls._is_requirements_file(name)

    @staticmethod
    def _line_starts(s: str) -> List[int]:
        starts = [0]
        for m in re.finditer(r"\n", s):
            starts.append(m.end())
        return starts

    @staticmethod
    def _pos_to_line(starts: List[int], pos: int) -> int:
        return bisect.bisect_right(starts, pos)

    # ---------- Dispatch by filename ----------
    def _extract_from_text(self, text: str, filename: str) -> List[Dict]:
        name = Path(filename).name.lower()
        if name == "go.mod":
            return self._extract_comments_gomod(text)
        if self._is_requirements_file(name):
            return self._extract_comments_requirements(text)
        return self._extract_comments_pygments(text, filename)

    # ---------- Extractors ----------
    def _extract_comments_gomod(self, text: str) -> List[Dict]:
        rows: List[Dict] = []
        starts = self._line_starts(text)

        # // line comments
        for m in re.finditer(r'//.*?$', text, flags=re.M):
            rows.append({
                "comment": m.group(),
                "start_line": self._pos_to_line(starts, m.start()),
                "end_line": self._pos_to_line(starts, m.end())
            })

        # /* block comments */
        for m in re.finditer(r'/\*.*?\*/', text, flags=re.S):
            rows.append({
                "comment": m.group(),
                "start_line": self._pos_to_line(starts, m.start()),
                "end_line": self._pos_to_line(starts, m.end() - 1)
            })

        rows.sort(key=lambda r: (r["start_line"], r["end_line"]))
        return rows

    @staticmethod
    def _extract_comments_requirements(text: str) -> List[Dict]:
        rows: List[Dict] = []
        for i, line in enumerate(text.splitlines(keepends=False), start=1):
            pos = line.find("#")
            if pos != -1:
                rows.append({"comment": line[pos:].rstrip(), "start_line": i, "end_line": i})
        return rows

    @staticmethod
    def _pick_lexer(filename: str, text: str):
        name = Path(filename).name.lower()
        try:
            if name == "pom.xml":
                return get_lexer_by_name("XML")
            if name.endswith(".gradle"):
                return get_lexer_by_name("Groovy")
            if name.endswith(".kts"):
                return get_lexer_by_name("Kotlin")
            if name == "pyproject.toml":
                return get_lexer_by_name("TOML")
            if name == "setup.cfg":
                return get_lexer_by_name("Ini")
            if name == "makefile":
                return get_lexer_by_name("Makefile")
            if name in ("gemfile", "rakefile"):
                return get_lexer_by_name("Ruby")
            if name == "tsconfig.json":
                return get_lexer_by_name("JavaScript")  # JSONC
            if name == "package.json":
                return None  # JSON: no comments by spec
            return guess_lexer_for_filename(filename, text)
        except ClassNotFound:
            return None

    def _extract_comments_pygments(self, text: str, filename: str) -> List[Dict]:
        lexer = self._pick_lexer(filename, text)
        if not lexer:
            return []
        rows: List[Dict] = []
        line = 1
        for ttype, value in lex(text, lexer):
            newlines = value.count("\n")
            if ttype in Token.Comment or (hasattr(ttype, "parent") and ttype.parent == Token.Comment):
                rows.append({
                    "comment": value.rstrip("\n"),
                    "start_line": line,
                    "end_line": line + newlines
                })
            line += newlines
        return rows

# -------- Example --------
if __name__ == "__main__":
    src = """\
# Build config
[tool.black]  # formatter
line-length = 100
"""
    extractor = BuildCommentExtractor(source=src, filename="pyproject.toml")
    print(extractor.extract())
