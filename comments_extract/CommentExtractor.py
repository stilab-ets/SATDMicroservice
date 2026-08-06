from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass
from typing import List, Iterable, Optional,Any, Dict, List, Union
from comments_extract.commentDocker import  commentExtractorDocker
import subprocess
import os
from comments_extract.extratctor_comments import MultiLangCommentExtractor
import json
CommentItem = Dict[str, Union[int, str]]
from pathlib import Path
from comments_extract.build_comments import BuildCommentExtractor
THIS_DIR = Path(__file__).resolve().parent      # .../comments_extract
PROJECT_ROOT = THIS_DIR.parent                  # repo root


class CommentExtractor:
    """
    Extracts comments from Python source code.

    Captures:
      - Line comments beginning with '#', including inline comments after code.
      - Block comments:
          * Module, class, and (async) function docstrings
          * Bare triple-quoted strings used as standalone comments (configurable)

    Parameters
    ----------
    include_bare_strings : bool
        If True, also treat standalone string expressions (not only docstrings)
        as block comments. Defaults to True.
    keep_raw_block_quotes : bool
        If True, keep the raw quoted text for block comments (including quotes).
        If False, keep only the string content without surrounding quotes.
        Defaults to True.
    """

    @dataclass
    class Comment:
        text: str
        line: int
        is_block: bool  # False = line '# ...', True = block/docstring

    def __init__(self, include_bare_strings: bool = True, keep_raw_block_quotes: bool = True):
        self.include_bare_strings = include_bare_strings
        self.keep_raw_block_quotes = keep_raw_block_quotes

    # -------- Public API --------

    def extract(self, code: str) -> List[Comment]:
        """Extract comments from a code string."""
        comments: List[CommentExtractor.Comment] = []
        comments.extend(self._extract_line_comments(code))
        comments.extend(self._extract_block_comments(code))
        return self._dedup_and_sort(comments)

    def extract_from_path(self, path: str, encoding: Optional[str] = "utf-8") -> List[Comment]:
        """Extract comments from a file on disk."""
        with open(path, "r", encoding=encoding or "utf-8") as f:
            code = f.read()
        return self.extract(code)

    # -------- Internals --------

    def _extract_line_comments(self, code: str) -> List[Comment]:
        out: List[CommentExtractor.Comment] = []
        tokens = tokenize.tokenize(io.BytesIO(code.encode()).readline)
        for toknum, tokstring, tokloc, _, _ in tokens:
            if toknum == tokenize.COMMENT:
                # strip leading '#' and left space
                text = tokstring[1:].lstrip()
                out.append(self.Comment(text=text, line=tokloc[0], is_block=False))
        return out

    def _extract_block_comments(self, code: str) -> List[Comment]:
        """
        Use AST to find:
          - module/class/def/async def docstrings
          - (optionally) other bare string expressions considered as comments
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Return nothing; line comments were already collected
            return []

        out: List[CommentExtractor.Comment] = []

        def get_raw_or_clean(node_parent: ast.AST, str_node: ast.AST) -> tuple[str, int]:
            """
            Prefer raw source (keeps original quotes) if available
            and requested; otherwise use the literal value.
            """
            # Line number: prefer string node, fallback to parent
            line = getattr(str_node, "lineno", getattr(node_parent, "lineno", 1))

            # Extract raw source (includes quotes) if available
            raw = ast.get_source_segment(code, str_node)
            if self.keep_raw_block_quotes and raw is not None:
                return raw, line

            # Fallback: actual string value (no quotes)
            val = getattr(str_node, "value", None)
            if isinstance(val, str):
                return val, line
            # Py<3.8: ast.Str
            s = getattr(str_node, "s", None)
            if isinstance(s, str):
                return s, line

            # Worst case: empty text
            return "", line

        # 1) Docstrings on Module / Class / (Async)Function
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.body:
                    first = node.body[0]
                    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), (ast.Constant, ast.Str)):
                        text, line = get_raw_or_clean(node, first.value)
                        out.append(self.Comment(text=text, line=line, is_block=True))

        # 2) Bare string expressions treated as comments (optional)
        if self.include_bare_strings:
            for parent in ast.walk(tree):
                body: Optional[Iterable[ast.stmt]] = getattr(parent, "body", None)
                if not body:
                    continue
                body = list(body)
                for idx, child in enumerate(body):
                    if isinstance(child, ast.Expr) and isinstance(getattr(child, "value", None), (ast.Constant, ast.Str)):
                        is_first_stmt = (idx == 0)  # already captured as docstring above
                        if not is_first_stmt:
                            text, line = get_raw_or_clean(parent, child.value)
                            out.append(self.Comment(text=text, line=line, is_block=True))

        return out

    @staticmethod
    def _dedup_and_sort(comments: List[Comment]) -> List[Comment]:
        seen = set()
        unique: List[CommentExtractor.Comment] = []
        for c in comments:
            key = (c.line, c.text, c.is_block)
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return sorted(unique, key=lambda c: c.line)
     
    def extract_nirjas(self, file_path, source_code):

        """
         This fucntion use nirjas librairy
         args: file_path
        """
        filename = os.path.basename(file_path) # Returns "test.py"
        
        with open(filename, 'w', encoding='utf-8', errors='replace') as file:
            file.write(source_code)
        command = ['nirjas', filename]
        

        try:
            # capture_output=True saves the command's output
            # text=True decodes the output as a string
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True # Raise an exception if the command returns a non-zero exit code
            )

            if os.path.exists(filename):
                os.remove(filename)
            
            return (CommentExtractor.extract_comments(result.stdout))
    
        except FileNotFoundError:
            print(f"Error: The command '{command[0]}' was not found. Check your system's PATH.")
        except subprocess.CalledProcessError as e:
            print(f"Error: Command failed with return code {e.returncode}")
            print("--- Standard Error ---")
            print(e.stderr)
        
   
    @staticmethod
    def extract_comments(obj: Union[str, Dict[str, Any]]) -> List[CommentItem]:
        """
        Normalize comments from a structure like:
        {
        "metadata": {...},
        "single_line_comment": [{"line_number": 8, "comment": "..."}, ...],
        "cont_single_line_comment": [...],
        "multi_line_comment": [{"start_line": 3, "end_line": 6, "comment": "..."}]
        }

        Returns a list of dicts:
        {"start_line": int, "end_line": int, "comment": str}
        """
        
        if isinstance(obj, str):
            data = json.loads(obj)
        else:
            data = obj
       
        results: List[CommentItem] = []

        # 1) Single-line comments
        for item in data.get("single_line_comment", []) or []:
            line = int(item.get("line_number"))
            text = str(item.get("comment", "")).strip()
            results.append({"comment": text, "start_line": line, "end_line": line})

        # 2) Multi-line comments
        for item in data.get("multi_line_comment", []) or []:
            start = int(item.get("start_line"))
            end = int(item.get("end_line"))
            text = str(item.get("comment", "")).strip()
            results.append({"comment": text,"start_line": start, "end_line": end})

        # 3) Continued single-line blocks (various possible shapes handled)
        for block in data.get("cont_single_line_comment", []) or []:
            if isinstance(block, dict):
                # Common shapes:
                # a) {"start_line": X, "end_line": Y, "comment": "..."}
                if "start_line" in block and "end_line" in block:
                    start = int(block["start_line"])
                    end = int(block["end_line"])
                    text = str(block.get("comment", "")).strip()
                    results.append({"comment": text,"start_line": start, "end_line": end})
                    continue

                # b) {"lines": [X, X+1, ...], "comment": "..."}
                if "lines" in block and isinstance(block["lines"], list) and block["lines"]:
                    lines = sorted(int(x) for x in block["lines"])
                    start, end = lines[0], lines[-1]
                    text = str(block.get("comment", "")).strip()
                    results.append({"comment": text,"start_line": start, "end_line": end})
                    continue

                # c) {"items": [{"line_number": X, "comment": "..."}, ...]}
                if "items" in block and isinstance(block["items"], list) and block["items"]:
                    items = block["items"]
                    lines = sorted(int(i.get("line_number")) for i in items if "line_number" in i)
                    # Join comments in order of line number
                    line_to_text = {int(i["line_number"]): str(i.get("comment", "")).strip() for i in items if "line_number" in i}
                    merged_text = "\n".join(line_to_text[l] for l in lines if line_to_text.get(l) is not None)
                    if lines:
                        results.append({"comment": text,"start_line": lines[0], "end_line": lines[-1]})
                    continue

                # Fallback: if dict has 'comment' only, we can't infer lines—skip or set to -1
                if "comment" in block:
                    results.append({"comment": str(block["comment"]).strip(),"start_line": -1, "end_line": -1})
                    continue

            elif isinstance(block, list) and block:
                # d) A list of single-line items for one block:
                #    [{"line_number": X, "comment": "..."}, ...]
                try:
                    lines = sorted(int(i.get("line_number")) for i in block if "line_number" in i)
                    line_to_text = {int(i["line_number"]): str(i.get("comment", "")).strip() for i in block if "line_number" in i}
                    merged_text = "\n".join(line_to_text[l] for l in lines if line_to_text.get(l) is not None)
                    if lines:
                        results.append({"comment": merged_text, "start_line": lines[0], "end_line": lines[-1]})
                except Exception:
                    # If unexpected content, ignore gracefully
                    pass

        # Sort by start_line, then end_line
        results.sort(key=lambda x: (x["start_line"], x["end_line"]))
        
        return results
    
    def extract_comments_form_anyFile(self, file_path: str, source_code: Optional[str] = None) -> List[CommentItem]: 

        """
        This function extract all comments from any type of file
        args:
          - file: file containing source code OR
          - source_code: string containing source code
        return: list of comments
        """
        filename = os.path.basename(file_path) # Returns "test.py"
        _, extension =  os.path.splitext(file_path)
        name = os.path.basename(file_path)       
        if name.lower() == "dockerfile" or extension ==".yml" or extension ==".yaml":
            # Call of docker file comment extractor
            extractor = commentExtractorDocker()
            return extractor.extract_comments(source_code)
        if  BuildCommentExtractor. _is_build_file(Path(file_path)) :
                ex = BuildCommentExtractor(source=source_code, filename= file_path)
                return ex.extract() 
        else: # Other files like java, python, js  
            ex = MultiLangCommentExtractor(merge_line_blocks=True, skip_directives=False, filter_js_regex_fp=True)
            return ex.extract_from_text(source_code, file_path)
            #return self.extract_nirjas(file_path, source_code)
        # Cas of build files like Makefile, CMakeLists.txt
        
if __name__ == "__main__": 

    extractor = CommentExtractor(include_bare_strings=True, keep_raw_block_quotes=False)
    print(extractor.extract_comments_form_anyFile("../docker-compose.yml"))


