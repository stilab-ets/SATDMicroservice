# pip install comment-parser python-magic
from typing import List, Dict, Iterable, Optional
from comment_parser import comment_parser
import pathlib

class MultiLangCommentExtractor:
    """
    Returns comments as {"comment": str, "start_line": int, "end_line": int}.
    Supports: Java, Python, Go, JS/TS/TSX, C/C++/Headers, PHP, C#.

    Behavior:
      - Preserves multi-line blocks (/* ... */)
      - Coalesces only *full-line* single-line comments
      - NEW: can merge across blank lines (Go header blocks)
      - JS/TS fix: drops '//' inside regex literals
    """

    MIME_MAP = {
        ".java": "text/x-java-source",
        ".py":   "text/x-python",
        ".go":   "text/x-go",
        ".js":   "application/javascript",
        ".ts":   "application/javascript",
        ".tsx":  "application/javascript",
        ".c":    "text/x-c", ".h": "text/x-c",
        ".cc":   "text/x-c++", ".cpp": "text/x-c++", ".cxx": "text/x-c++",
        ".hpp":  "text/x-c++", ".hxx": "text/x-c++",
        ".php":  "text/x-php",
        ".cs":   "text/x-csharp",
    }

    JS_EXTS = {".js", ".ts", ".tsx"}
    LINE_MARKERS = {
        ".py":  ("#",),
        ".php": ("//", "#"),
        ".js":  ("//",),
        ".ts":  ("//",),
        ".tsx": ("//",),
        ".java":("//",),
        ".go":  ("//",),
        ".c":   ("//",),
        ".h":   ("//",),
        ".cc":  ("//",),
        ".cpp": ("//",),
        ".cxx": ("//",),
        ".hpp": ("//",),
        ".hxx": ("//",),
        ".cs":  ("//",),
    }

    def __init__(
        self,
        merge_line_blocks: bool = True,
        skip_directives: bool = False,
        filter_js_regex_fp: bool = True,
        merge_across_blank_lines: bool = True,   # <-- NEW
    ):
        self.merge_line_blocks = merge_line_blocks
        self.skip_directives = skip_directives
        self.filter_js_regex_fp = filter_js_regex_fp
        self.merge_across_blank_lines = merge_across_blank_lines
        self.DIRECTIVE_PREFIXES = ("go:", "go:build", "+build", "@ts-", "eslint-", "jshint")

    # ---------- Public API ----------

    def extract_from_text(self, text: str, path_hint: str) -> List[Dict]:
        p = pathlib.Path(path_hint)
        ext = p.suffix.lower()
        mime = self.MIME_MAP.get(ext)
        if not mime or not text:
            return []
        try:
            raw = comment_parser.extract_comments_from_str(text, mime=mime)
        except Exception:
            return []
        return self._process(raw, ext, text)

    def extract_from_path(self, path: str) -> List[Dict]:
        p = pathlib.Path(path)
        ext = p.suffix.lower()
        mime = self.MIME_MAP.get(ext)
        if not mime:
            return []
        try:
            src = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            src = None
        try:
            raw = comment_parser.extract_comments(str(p), mime=mime)
        except Exception:
            return []
        return self._process(raw, ext, src)

    # ---------- Internals ----------

    def _process(self, raw, ext: str, full_text: Optional[str]) -> List[Dict]:
        items = []
        for c in raw:
            t = c.text()
            start = c.line_number()
            end = start + t.count("\n") if c.is_multiline() else start
            items.append({"kind": "block" if c.is_multiline() else "line",
                          "start_line": start, "end_line": end, "comment": t})
        items.sort(key=lambda x: x["start_line"])

        if self.skip_directives:
            items = [it for it in items if not self._looks_like_directive(it["comment"])]

        # JS/TS: drop false positives when '//' is inside regex literals
        if self.filter_js_regex_fp and full_text and ext in self.JS_EXTS:
            lines = full_text.splitlines()
            filtered = []
            for it in items:
                if it["kind"] == "line":
                    ln = it["start_line"]
                    line = lines[ln-1] if 1 <= ln <= len(lines) else ""
                    if not self._js_has_real_line_comment(line):
                        continue
                filtered.append(it)
            items = filtered

        # Merge only *full-line* single-line comments; allow gaps of blank lines
        if self.merge_line_blocks and full_text:
            lines = full_text.splitlines()
            items = self._coalesce_full_line_only(
                items,
                lines,
                self.LINE_MARKERS.get(ext, ("//","#")),
                allow_blank_gap=self.merge_across_blank_lines
            )

        return [{"comment": it["comment"], "start_line": it["start_line"], "end_line": it["end_line"]}
                for it in items]

    def _coalesce_full_line_only(
        self,
        items: List[Dict],
        lines: List[str],
        markers: Iterable[str],
        allow_blank_gap: bool = True
    ) -> List[Dict]:
        def is_full_line_comment(line_no: int) -> bool:
            if line_no < 1 or line_no > len(lines): return False
            s = lines[line_no - 1]
            idxs = [s.find(m) for m in markers if s.find(m) != -1]
            if not idxs: return False
            i = min(idxs)
            return s[:i].strip() == ""  # nothing before marker

        def only_blank_between(a: int, b: int) -> bool:
            # True if all lines strictly between a and b are whitespace-only
            for ln in range(a + 1, b):
                if 1 <= ln <= len(lines):
                    if lines[ln - 1].strip() != "":
                        return False
            return True

        out, cur = [], []
        def flush():
            nonlocal cur
            if not cur: return
            out.append(cur[0] if len(cur) == 1 else {
                "kind": "line_block",
                "start_line": cur[0]["start_line"],
                "end_line":   cur[-1]["end_line"],
                "comment":    "\n".join(x["comment"] for x in cur)
            })
            cur = []

        for it in items:
            if it["kind"] == "block":
                flush(); out.append(it); continue
            if cur:
                prev = cur[-1]
                consecutive = (it["start_line"] == prev["end_line"] + 1)
                gap_blank = allow_blank_gap and (it["start_line"] > prev["end_line"] + 1) and only_blank_between(prev["end_line"], it["start_line"])
                if (consecutive or gap_blank) and is_full_line_comment(prev["start_line"]) and is_full_line_comment(it["start_line"]):
                    cur.append(it)
                else:
                    flush(); cur = [it]
            else:
                cur = [it]
        flush()
        return out

    def _looks_like_directive(self, text: str) -> bool:
        t = text.strip().lower()
        return any(t.startswith(pref) for pref in self.DIRECTIVE_PREFIXES)

    # ---------- JS-specific helper ----------
    def _js_has_real_line_comment(self, s: str) -> bool:
        i, n = 0, len(s)
        in_str: Optional[str] = None
        in_regex = False
        in_class = False
        def prev_nonspace(ix):
            j = ix - 1
            while j >= 0 and s[j].isspace(): j -= 1
            return s[j] if j >= 0 else None

        while i < n:
            ch = s[i]
            if in_str:
                if ch == "\\": i += 2; continue
                if ch == in_str: in_str = None
                i += 1; continue
            if in_regex:
                if ch == "\\": i += 2; continue
                if ch == "[": in_class = True; i += 1; continue
                if ch == "]" and in_class: in_class = False; i += 1; continue
                if ch == "/" and not in_class:
                    i += 1
                    while i < n and s[i].isalpha(): i += 1
                    in_regex = False
                    continue
                i += 1; continue
            if ch in ("'", '"', "`"): in_str = ch; i += 1; continue
            if ch == "/":
                if i + 1 < n and s[i+1] == "/": return True
                if i + 1 < n and s[i+1] == "*": return True
                prev = prev_nonspace(i)
                if prev is None or prev in "([{:;,=!?&|^~+-*%<>":
                    in_regex = True; i += 1; continue
            i += 1
        return False

""""
with open("./tmp/services.go", "r", encoding="utf-8") as f:
    js_code = f.read()

ex = MultiLangCommentExtractor(merge_line_blocks=True, skip_directives=False, filter_js_regex_fp=True)
rows = ex.extract_from_text(js_code, "services.go")
for r in rows:
    print(r)
    print("-----")
"""