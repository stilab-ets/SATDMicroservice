from difflib import SequenceMatcher
import re
import subprocess, sys
from urllib.parse import urlparse
from pathlib import Path
from pathlib import PurePosixPath
import fnmatch
import re
from typing import List, Dict, Tuple

DEFAULT_EXCLUDE_COMPONENTS = {
    "godeps", "vendor", "node_modules", "dist", "build", 
    ".git", ".hg", ".svn", "__pycache__", ".tox", ".venv", "venv", ".cache",
    "target"  # Rust
}
DEFAULT_EXCLUDE_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".jar", ".wasm", ".bin"
}

def normf(s: str) -> str:
    return " ".join(s.split()).strip().lower()

def simf(a: str, b: str) -> float:
    return SequenceMatcher(None, normf(a), normf(b)).ratio()

def parse_hunks(unified: str):
    # -> list of (old_start, old_end, new_start, new_end)
    H = []
    for hdr in re.finditer(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', unified, flags=re.M):
        o0, ol, n0, nl = hdr.groups()
        o0, n0 = int(o0), int(n0)
        ol = int(ol) if ol else 1
        nl = int(nl) if nl else 1
        H.append((o0, o0+ol-1, n0, n0+nl-1))
    return H

def overlaps(a0, a1, b0, b1):
    return not (a1 < b0 or b1 < a0)

def hunk_hits_old(c, hunks):
    return [h for h in hunks if overlaps(c["start_line"], c["end_line"], h[0], h[1])]
def hunk_hits_new(c, hunks):
    return [h for h in hunks if overlaps(c["start_line"], c["end_line"], h[2], h[3])]


def gh_owner_repo(repo_path: str, remote: str = "origin"):
    url = subprocess.check_output(["git","-C",repo_path,"remote","get-url",remote], text=True).strip()
    if url.startswith("git@"):
        host = url.split("@",1)[1].split(":",1)[0]
        path = url.split(":",1)[1]
    else:
        u = urlparse(url); host, path = u.hostname, u.path.lstrip("/")
    if not host or "github.com" not in host.lower():
        raise RuntimeError("Remote is not GitHub")
    if path.endswith(".git"): path = path[:-4]
    owner, repo = path.split("/", 1)
    return owner, repo

def gh_blob_url(owner: str, repo: str, sha: str, path: str, start: int, end: int | None = None) -> str:
    anchor = f"#L{start}" + (f"-L{end}" if end and end != start else "")
    return f"https://github.com/{owner}/{repo}/blob/{sha}/{path}{anchor}"

def gh_commit_url(owner: str, repo: str, sha: str) -> str:
    return f"https://github.com/{owner}/{repo}/commit/{sha}"

def attach_github_links(rec: dict, owner: str, repo: str, commit_sha: str, parent_sha: str | None):
    """Mutates/returns rec with 'url' (new side) and 'old_url' (old side, if any)."""
    r = dict(rec)
    
    if r["state"] in {"new","unchanged","modified"} and r.get("path") and r.get("start_line"):
        r["url"] = gh_blob_url(owner, repo, commit_sha, r["path"], int(r["start_line"]), int(r.get("end_line") or r["start_line"]))
    if parent_sha and r["state"] in {"removed","unchanged","modified"} and r.get("old_path") and r.get("old_start_line"):
        r["old_url"] = gh_blob_url(owner, repo, parent_sha, r["old_path"], int(r["old_start_line"]), int(r.get("old_end_line") or r["old_start_line"]))
    return r


def want_path(
    path: str | None,
    *,
    exclude_components: set[str] = DEFAULT_EXCLUDE_COMPONENTS,
    exclude_globs: list[str] | None = None,
    exclude_exts: set[str] = DEFAULT_EXCLUDE_EXTS,
    top_level_only: bool = False,
    casefold: bool = True,
) -> bool:
    """
    Return True if a repo-relative path should be analyzed.

    - Filters by directory components (e.g., vendor dirs).
    - Optional glob patterns (e.g., 'third_party/**', '**/*.min.*').
    - Optional extension filter for obvious binary/assets.
    - Set top_level_only=True to match only the first path component.
    """
    if not path:
        return False
    # Normalize to POSIX semantics for Git paths
    norm = path.replace("\\", "/")
    p = PurePosixPath(norm)
    parts = list(p.parts)

    # Case-fold for robust matching (Windows/Mac FS vs Git case sensitivity)
    parts_cf = [c.casefold() for c in parts] if casefold else parts
    excl_cf = {c.casefold() for c in exclude_components} if casefold else exclude_components

    scan = parts_cf[:1] if top_level_only else parts_cf
    if any(comp in excl_cf for comp in scan):
        return False

    # Glob filters (match on normalized POSIX path)
    if exclude_globs and any(fnmatch.fnmatch(norm, pat) for pat in exclude_globs):
        return False

    # Extension filter
    if p.suffix.lower() in exclude_exts:
        return False

    return True


def _norm_text(s: str) -> str:
    # normalize: trim, collapse spaces, lowercase
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def dedup_comments_list(comments: List[Dict], *, by_path: bool = True) -> List[Dict]:
    """
    Deduplicate comments in a single list.
    - by_path=True  -> same text must be in the same 'path' to be considered duplicate
    - by_path=False -> duplicates anywhere by text are removed
    Keeps the FIRST occurrence encountered.
    """
    seen: set[Tuple[str, str]] = set()
    out: List[Dict] = []
    for c in comments:
        text_key = _norm_text(c.get("comment", ""))
        path_key = (c.get("path") or "").lower() if by_path else ""
        key = (text_key, path_key)
        if text_key and key not in seen:
            seen.add(key)
            out.append(c)
    return out

