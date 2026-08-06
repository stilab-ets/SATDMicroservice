from __future__ import annotations
import pandas as pd
from pathlib import Path
import os
import numpy as np

import json, csv, re
from typing import Iterable, Dict, Any, List, Tuple


root = Path(__file__).resolve().parents[0] if '__file__' in globals() else Path.cwd()
print(root)


# ---------- Keyword handling ----------

def load_keywords(path: Path | None = None, keywords_debt: Iterable[str] | None = None) -> List[str]:
    """Load SATD keywords from a text file (one per line) and/or a provided list."""
    kw: List[str] = []
    if path:
        for line in path.read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if t and not t.startswith("#"):
                kw.append(t)
    if keywords_debt:
        kw.extend([k.strip() for k in keywords_debt if k and k.strip()])
    # de-dup while preserving order
    seen = set()
    out = []
    for k in kw:
        if k.lower() not in seen:
            seen.add(k.lower()); out.append(k)
    if not out:
        raise ValueError("No SATD keywords provided. Pass keywords file or a list.")
    return out



def compile_keyword_regex(keywords: Iterable[str]) -> re.Pattern:
    """
    Build a single regex that matches any keyword (case-insensitive),
    with word boundaries at the ends. Internal spaces become \\s+.
    Also:
      - treat ASCII/curly apostrophes as equivalent
      - add exceptions to reduce false positives (e.g., 'broken down')
    """

    # Exception overrides (lowercased keyword -> custom regex fragment)
    # Example: match 'broken' but NOT 'broken down'
    exceptions = {
        "broken": r"\bbroken\b(?!\s+down\b)",
    }

    parts = []
    for k in keywords:
        if not k:
            continue
        k = k.strip()
        if not k:
            continue

        kl = k.lower()
        if kl in exceptions:
            parts.append(exceptions[kl])
            continue

        # Escape literal chars, then:
        #  - allow flexible whitespace (\s+)
        #  - make apostrophes tolerant to ' or ’
        k_esc = re.escape(k)
        k_esc = re.sub(r"\s+", r"\\s+", k_esc)                  # spaces -> \s+
        k_esc = k_esc.replace("\\'", r"(?:'|’)")                # ASCII apostrophe
        k_esc = k_esc.replace("’", r"(?:'|’)")                  # curly apostrophe if present

        parts.append(rf"\b{k_esc}\b")

    if not parts:
        # Fallback that never matches
        return re.compile(r"$(?=a)")

    pattern = r"(?is)(" + "|".join(parts) + r")"
    return re.compile(pattern)


def is_satd(text: str, rx: re.Pattern) -> bool:
    return bool(rx.search(text or ""))

# ---------- JSON loading & extraction ----------

def load_records(p: Path) -> List[Dict[str, Any]]:
    """Load a JSON array or JSONL file into a list of dicts."""
    txt = p.read_text(encoding="utf-8")
    try:
        data = json.loads(txt)
        if isinstance(data, dict):
            for k in ("items", "data", "commits"):
                if k in data and isinstance(data[k], list):
                    return data[k]
            return [data]
        return data
    except json.JSONDecodeError:
        # JSON Lines fallback
        return [json.loads(line) for line in txt.splitlines() if line.strip()]

def extract_rows(records: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, str]]:
    """Yield rows: url, comment, language from commit->files->comments."""
    for rec in records or []:
        for f in (rec.get("files") or []):
            lang = f.get("language", "") or ""
            for c in (f.get("comments") or []):
                url = (c.get("url") or "").strip()
                txt = (c.get("comment") or "").strip()
                if url and txt:
                    yield {"url": url, "comment": txt, "language": lang}

# ---------- File discovery ----------

def gather_json_files(inputs: Iterable[str | Path]) -> List[Path]:
    """
    Accept files and/or directories. For directories, recurse **/*.json.
    Returns a de-duplicated list of Paths.
    """
    files: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(p.rglob("*.json"))
        elif p.is_file() and p.suffix.lower() == ".json":
            files.append(p)
        else:
            # ignore non-existent or non-json files silently
            pass
    # de-dup / stable
    seen = set()
    out = []
    for f in files:
        rp = f.resolve()
        if rp not in seen:
            seen.add(rp); out.append(rp)
    return out

# ---------- Main ----------

def main(
    inputs: Iterable[str | Path],
    out_dir: str | Path,
    keywords_file: str | Path | None = None,
    keywords_debt: Iterable[str] | None = None,
) -> Tuple[Path, Path, int, int, int]:
    """
    Read JSON files, extract (url, comment, language), label SATD via keywords,
    and write two CSVs: SATD_comments.csv and non_SATD_comments.csv.

    Returns: (satd_csv_path, nonsatd_csv_path, satd_count, nonsatd_count, total)
    """
    json_files = gather_json_files(inputs)
    if not json_files:
        raise FileNotFoundError("No JSON files found in inputs.")

    kws = load_keywords(Path(keywords_file) if keywords_file else None, keywords_debt)
    rx = compile_keyword_regex(kws)

    satd_rows: List[Dict[str, str]] = []
    nonsatd_rows: List[Dict[str, str]] = []

    for jf in json_files:
        for r in extract_rows(load_records(jf)):
            row = {
                "url": r["url"],
                "comment": r["comment"],
                "language": r["language"],
                "label": "SATD" if is_satd(r["comment"], rx) else "no-SATD",
            }
            (satd_rows if row["label"] == "SATD" else nonsatd_rows).append(row)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    satd_csv = out_dir / "SATD_comments.csv"
    nonsatd_csv = out_dir / "non_SATD_comments.csv"

    headers = ["url", "comment", "language", "label"]
    for path, rows in ((satd_csv, satd_rows), (nonsatd_csv, nonsatd_rows)):
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            w.writerows(rows)

    return satd_csv, nonsatd_csv, len(satd_rows), len(nonsatd_rows), len(satd_rows) + len(nonsatd_rows)


if __name__ == "__main__":

    comments_dir = Path(root) / "dataset" / "comments" " 
    keywords_list = root / "dataset" / "keywords_list.txt"
    #html_url_repo = pd.read_csv(root + '/Dataset/3out.csv', header=None)


    files = list (comments_dir.glob("*.json"))

    SATD_comments = pd.DataFrame(columns=['Link Location', 'Comment', 'Keywords'])
    Comments_with_no_keywords = pd.DataFrame(columns=['Link Location', 'Comment'])
    revision_list = pd.DataFrame(columns=['Repository ID', 'Revison'])
    with open(keywords_list, 'r') as file:
        keywords_debt = file.read().split(', ')

    maven_comments = 0
    maven_repo = 0
    pom_file = 0
    error = []
    
    satd_path, nonsatd_path, n_satd, n_non, n_total = main(
        inputs=files,
        out_dir="./outputs",
        keywords_debt=keywords_debt
    )
    print(f"Wrote {n_total} rows  →  {satd_path} (SATD: {n_satd})  |  {nonsatd_path} (non-SATD: {n_non})")

    #out_path, n = main(files[0], "comments.csv")

