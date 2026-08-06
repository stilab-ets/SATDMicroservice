import os
import re
import subprocess
from pydriller import Repository, Git
from pydriller.domain.commit import Commit
from pathlib import Path
from utils.utils import *
from collections import defaultdict, deque
import json
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from comments_extract.build_comments import BuildCommentExtractor
from comments_extract.CommentExtractor import CommentExtractor

class CommitAnalyzer:
    """
    Analyzes a Git repository to extract specific information about commits and files.
    """

    def __init__(self, repo_path, month=2, allowed_extensions=None, mode="patch"):

        self.repo_path = repo_path
        self._months = month # None means extract comments from all commits
        if mode not in {"patch", "file", "both"}:
            raise ValueError("mode must be one of: patch, file, both")
        self.mode = mode
        self._language_map = {
            '.py': 'Python', '.js': 'JavaScript', '.java': 'Java',
            '.c': 'C', '.cpp': 'C++', '.h': 'C/C++', '.cs': 'C#',
            '.go': 'Go', '.ts': 'TypeScript', '.php': 'PHP',
            '.yml': 'YAML', '.yaml': 'YAML', 'dockerfile': 'Dockerfile'
        }
        self._comment_syntax = {
            'Python': ['#'], 'JavaScript': ['//', '/*'], 'Java': ['//', '/*'],
            'C': ['//', '/*'], 'C++': ['//', '/*'], 'C#': ['//', '/*'],
            'Go': ['//', '/*'], 'TypeScript': ['//', '/*'], 'PHP': ['#', '//', '/*'],
            'YAML': ['#'], 'Dockerfile': ['#']
        }
       
        if allowed_extensions is None:
            self.allowed_extensions = {'.py', '.js', '.java', '.c', '.cpp', '.h', '.cs', '.go', '.ts', '.php', '.yml', '.yaml', 'Dockerfile'}
        else:        
          self.allowed_extensions = allowed_extensions

    def _get_language(self, file_path):
        """Infers the programming language from a file's extension."""
        file_name = os.path.basename(file_path).lower()
        if file_name in self._language_map:
            return self._language_map[file_name]
        
        _, ext = os.path.splitext(file_name)
        return self._language_map.get(ext, 'Unknown')
    
    def extract_comments_by_file(self, mod, sim_threshold: float = 0.7):
        """
        Return a flat list of comments for one PyDriller Modification with state in:
        {'new','removed','modified','unchanged'}.

        For 'unchanged' we now include both old/new paths/lines and a 'renamed' flag.
        For 'modified' we keep both old and new versions; for 'removed' only the old side.
        """
        old_path = (mod.old_path or mod.filename or "").replace("\\", "/")
        new_path = (mod.new_path or mod.filename or "").replace("\\", "/")
        renamed_flag = (getattr(mod, "change_type", None) and mod.change_type.name == "RENAME" and old_path != new_path)
        
        old_text, new_text = mod.source_code_before or "", mod.source_code or ""
        extractor = CommentExtractor()
       
        old_raw = extractor.extract_comments_form_anyFile(old_path, old_text)
        new_raw = extractor.extract_comments_form_anyFile(new_path, new_text)

        # annotate
        old_cs = [dict(c, _t=normf(c["comment"]), path=old_path) for c in old_raw]
        new_cs = [dict(c, _t=normf(c["comment"]), path=new_path) for c in new_raw]

        # ---- 1) UNCHANGED by exact normalized text (multiset-aware) ----
        old_idx, new_idx = defaultdict(deque), defaultdict(deque)
        for i, c in enumerate(old_cs): old_idx[c["_t"]].append(i)
        for i, c in enumerate(new_cs): new_idx[c["_t"]].append(i)

        used_old, used_new = set(), set()
        unchanged_pairs = []
        for tok in set(old_idx) & set(new_idx):
            k = min(len(old_idx[tok]), len(new_idx[tok]))
            for _ in range(k):
                oi = old_idx[tok].popleft()
                ni = new_idx[tok].popleft()
                used_old.add(oi); used_new.add(ni)
                unchanged_pairs.append((old_cs[oi], new_cs[ni]))

        # ---- 2) candidates for removed/new ----
        removed_cands    = [c for i, c in enumerate(old_cs) if i not in used_old]
        introduced_cands = [c for i, c in enumerate(new_cs) if i not in used_new]

        # ---- 3) MODIFIED via hunk overlap + similarity ----
        hunks = parse_hunks(mod.diff)
        def old_hits(c): return [h for h in hunks if overlaps(c["start_line"], c["end_line"], h[0], h[1])]
        def new_hits(c): return [h for h in hunks if overlaps(c["start_line"], c["end_line"], h[2], h[3])]

        modified_pairs, removed_left, introduced_left = [], removed_cands[:], introduced_cands[:]
        for rc in removed_cands:
            H = old_hits(rc)
            if not H: continue
            cands = [nc for nc in introduced_left if any(overlaps(nc["start_line"], nc["end_line"], h[2], h[3]) for h in H)]
            if not cands: continue
            best = max(cands, key=lambda nc: simf(rc["comment"], nc["comment"]))
            score = simf(rc["comment"], best["comment"])
            if score >= sim_threshold:
                modified_pairs.append((rc, best, score))
                removed_left.remove(rc)
                introduced_left.remove(best)

        # ---- 4) Flatten output ----
        out = []

        # unchanged → include both sides + renamed flag
        for old_c, new_c in unchanged_pairs:
            out.append({
                "state": "unchanged",
                "comment": new_c["comment"],
                "start_line": new_c["start_line"],
                "end_line": new_c["end_line"],
                "path": new_c["path"],
                "old_comment": old_c["comment"],
                "old_start_line": old_c["start_line"],
                "old_end_line": old_c["end_line"],
                "old_path": old_c["path"],
                "renamed": str(renamed_flag),
            })

        # modified → both sides
        for old_c, new_c, score in modified_pairs:
            out.append({
                "state": "modified",
                "comment": new_c["comment"],
                "start_line": new_c["start_line"],
                "end_line": new_c["end_line"],
                "path": new_c["path"],
                "old_comment": old_c["comment"],
                "old_start_line": old_c["start_line"],
                "old_end_line": old_c["end_line"],
                "old_path": old_c["path"],
                "similarity": score,
                "renamed": str(renamed_flag),
            })

        # new (introduced)
        for nc in introduced_left:
            out.append({
                "state": "new",
                "comment": nc["comment"],
                "start_line": nc["start_line"],
                "end_line": nc["end_line"],
                "path": nc["path"],
                "renamed": str(renamed_flag),
            })

        # removed → old side only
        for rc in removed_left:
            out.append({
                "state": "removed",
                "comment": None,
                "start_line": None,
                "end_line": None,
                "path": None,
                "old_comment": rc["comment"],
                "old_start_line": rc["start_line"],
                "old_end_line": rc["end_line"],
                "old_path": rc["path"],
                "renamed": str(renamed_flag),
            })

        out.sort(key=lambda r: (
            {"removed":0,"modified":1,"unchanged":2,"new":3}.get(r["state"], 9),
            (r.get("path") or r.get("old_path") or ""),
            r.get("start_line") or r.get("old_start_line") or 0
        ))
        return out #test
    
    def extract_comments_by_diff(self, mod, sim_threshold: float = 0.7):
        """
        Return comments changed by a patch.

        Added comments become state='new', removed comments become
        state='removed', and similar removed/new comments in the same hunk
        become state='modified'.
        """
        old_path = (mod.old_path or mod.filename or "").replace("\\", "/")
        new_path = (mod.new_path or mod.old_path or mod.filename or "").replace("\\", "/")
        renamed_flag = (
            getattr(mod, "change_type", None)
            and mod.change_type.name == "RENAME"
            and (mod.old_path or "") != (mod.new_path or "")
        )

        diff_text = mod.diff or ""
        if not diff_text.strip():
            return []

        extractor = CommentExtractor()
        old_comments = self._extract_comments_from_diff_side(
            extractor=extractor,
            file_path=old_path,
            diff_text=diff_text,
            side="old",
        )
        new_comments = self._extract_comments_from_diff_side(
            extractor=extractor,
            file_path=new_path,
            diff_text=diff_text,
            side="new",
        )

        hunks = parse_hunks(diff_text)
        def old_hits(c): return [h for h in hunks if overlaps(c["start_line"], c["end_line"], h[0], h[1])]

        out = []
        seen = set()
        removed_left = old_comments[:]
        introduced_left = new_comments[:]

        for old_c in old_comments:
            H = old_hits(old_c)
            if not H:
                continue
            cands = [
                new_c for new_c in introduced_left
                if any(overlaps(new_c["start_line"], new_c["end_line"], h[2], h[3]) for h in H)
            ]
            if not cands:
                continue
            best = max(cands, key=lambda new_c: simf(old_c["comment"], new_c["comment"]))
            score = simf(old_c["comment"], best["comment"])
            if score >= sim_threshold:
                row = {
                    "state": "modified",
                    "comment": best["comment"],
                    "start_line": best["start_line"],
                    "end_line": best["end_line"],
                    "path": best["path"],
                    "old_comment": old_c["comment"],
                    "old_start_line": old_c["start_line"],
                    "old_end_line": old_c["end_line"],
                    "old_path": old_c["path"],
                    "similarity": score,
                    "renamed": str(renamed_flag),
                }
                key = (
                    row["state"], row["comment"], row["start_line"], row["end_line"], row["path"],
                    row["old_comment"], row["old_start_line"], row["old_end_line"], row["old_path"],
                )
                if key not in seen:
                    seen.add(key)
                    out.append(row)
                if old_c in removed_left:
                    removed_left.remove(old_c)
                if best in introduced_left:
                    introduced_left.remove(best)

        for c in introduced_left:
            row = {
                "state": "new",
                "comment": c["comment"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "path": new_path,
                "renamed": str(renamed_flag),
            }
            key = (row["comment"], row["start_line"], row["end_line"], row["path"])
            if key not in seen:
                seen.add(key)
                out.append(row)

        for c in removed_left:
            row = {
                "state": "removed",
                "comment": None,
                "start_line": None,
                "end_line": None,
                "path": None,
                "old_comment": c["comment"],
                "old_start_line": c["start_line"],
                "old_end_line": c["end_line"],
                "old_path": c["path"],
                "renamed": str(renamed_flag),
            }
            key = (row["state"], row["old_comment"], row["old_start_line"], row["old_end_line"], row["old_path"])
            if key not in seen:
                seen.add(key)
                out.append(row)

        out.sort(key=lambda r: (
            {"removed":0,"modified":1,"new":2}.get(r["state"], 9),
            (r.get("path") or r.get("old_path") or ""),
            r.get("start_line") or r.get("old_start_line") or 0
        ))
        return out

    def _extract_comments_from_diff_side(self, extractor, file_path: str, diff_text: str, side: str):
        assert side in {"old", "new"}

        results = []
        seen = set()

        old_line = None
        new_line = None
        side_lines = []
        side_line_map = []
        side_changed_map = []

        def flush_hunk():
            nonlocal side_lines, side_line_map, side_changed_map
            if not side_lines:
                return

            snippet = "\n".join(side_lines)
            extracted = extractor.extract_comments_form_anyFile(file_path, snippet) or []

            for c in extracted:
                rel_start = c.get("start_line")
                rel_end = c.get("end_line")
                if not rel_start or not rel_end:
                    continue
                if rel_start < 1 or rel_end > len(side_line_map):
                    continue
                if not any(side_changed_map[rel_start - 1:rel_end]):
                    continue

                abs_start = side_line_map[rel_start - 1]
                abs_end = side_line_map[rel_end - 1]

                row = {
                    "comment": c["comment"],
                    "start_line": abs_start,
                    "end_line": abs_end,
                    "path": file_path,
                }

                key = (row["comment"], row["start_line"], row["end_line"], row["path"])
                if key not in seen:
                    seen.add(key)
                    results.append(row)

            side_lines = []
            side_line_map = []
            side_changed_map = []

        for raw in diff_text.splitlines():
            if raw.startswith("@@"):
                flush_hunk()
                m = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
                if not m:
                    old_line = None
                    new_line = None
                    continue
                old_line = int(m.group(1))
                new_line = int(m.group(2))
                continue

            if old_line is None or new_line is None:
                continue

            if raw.startswith("---") or raw.startswith("+++"):
                continue

            if raw.startswith(" "):
                text = raw[1:]
                if side == "new":
                    side_lines.append(text)
                    side_line_map.append(new_line)
                    side_changed_map.append(False)
                elif side == "old":
                    side_lines.append(text)
                    side_line_map.append(old_line)
                    side_changed_map.append(False)
                old_line += 1
                new_line += 1

            elif raw.startswith("-"):
                text = raw[1:]
                if side == "old":
                    side_lines.append(text)
                    side_line_map.append(old_line)
                    side_changed_map.append(True)
                old_line += 1

            elif raw.startswith("+"):
                text = raw[1:]
                if side == "new":
                    side_lines.append(text)
                    side_line_map.append(new_line)
                    side_changed_map.append(True)
                new_line += 1

        flush_hunk()
        return results

    def _dedup_comment_events(self, comments):
        """
        Deduplicate events without dropping removed comments, which store text in
        old_comment instead of comment.
        """
        seen = set()
        out = []
        for c in comments:
            text = c.get("comment") or c.get("old_comment") or ""
            path = c.get("path") or c.get("old_path") or ""
            key = (
                c.get("state"),
                normf(text),
                path.lower(),
                c.get("start_line") or c.get("old_start_line"),
                c.get("end_line") or c.get("old_end_line"),
            )
            if normf(text) and key not in seen:
                seen.add(key)
                out.append(c)
        return out

    def _git(self, repo, *args) -> str:
        return subprocess.check_output(['git', '-C', repo, *args],
                                    stderr=subprocess.DEVNULL).decode().strip()

    def _ref_exists(self, repo, ref: str) -> bool:
        # works for both local (refs/heads/*) and remote (refs/remotes/origin/*)
        return subprocess.call(['git', '-C', repo, 'show-ref', '--verify', ref],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0

    def _sha_exists(self, repo, sha: str) -> bool:
        return subprocess.call(['git', '-C', repo, 'cat-file', '-e', f'{sha}^{{commit}}'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0

    def tip_of_default_branch(self, repo_path: str, *, remote: str = 'origin', fetch: bool = False):
        """
        Returns (sha, ref_used) for tip of main/master. Tries:
        1) local refs/heads/main → master
        2) refs/remotes/origin/HEAD (remote default)
        3) refs/remotes/origin/main → master
        Set fetch=True to fetch before resolving remote refs.
        """
        if fetch:
            try:
                subprocess.check_call(['git', '-C', repo_path, 'fetch', remote, '--prune', '--tags'],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                pass  # continue with whatever we have locally

        # 1) Local branches
        for local in ('refs/heads/main', 'refs/heads/master'):
            if self._ref_exists(repo_path, local):
                sha = self._git(repo_path, 'rev-parse', local)
                return sha, local

        # 2) Remote default HEAD (e.g., refs/remotes/origin/HEAD -> refs/remotes/origin/main)
        try:
            remote_head_sym = self._git(repo_path, 'symbolic-ref', f'refs/remotes/{remote}/HEAD')  # returns a full ref
            if remote_head_sym and self._ref_exists(repo_path, remote_head_sym):
                sha = self._git(repo_path, 'rev-parse', remote_head_sym)
                return sha, remote_head_sym
        except subprocess.CalledProcessError:
            pass

        # 3) Explicit remote names
        for remote_ref in (f'refs/remotes/{remote}/main', f'refs/remotes/{remote}/master'):
            if self._ref_exists(repo_path, remote_ref):
                sha = self._git(repo_path, 'rev-parse', remote_ref)
                return sha, remote_ref

        raise RuntimeError("Could not find main/master locally or on the remote. "
                        "Try: `git fetch origin` or verify the default branch name.")

    def get_tip_commit(self, repo_path: str, *, remote: str = 'origin', fetch: bool = False):
        sha, ref_used = self.tip_of_default_branch(repo_path, remote=remote, fetch=fetch)

        # Ensure the commit object exists locally (in very shallow clones this can fail)
        if not self._sha_exists(repo_path, sha):
            # Try fetching just that tip if missing
            try:
                subprocess.check_call(['git', '-C', repo_path, 'fetch', remote, ref_used.split('/')[-1]],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                pass
            if not self._sha_exists(repo_path, sha):
                raise RuntimeError(f"Commit {sha} not present locally; run `git fetch {remote}`.")

        it = Repository(repo_path, only_commits=[sha]).traverse_commits()
        try:
            return next(it)  # PyDriller Commit
        except StopIteration:
            raise RuntimeError(f"PyDriller could not load commit {sha}. "
                            "Check that the SHA is reachable and not filtered elsewhere.")


    def analyze_repository(self, output_file, commit_hashs=None):
        """
        Analyzes commits and extracts comments.
        If self._months is None, analyze all commits. Otherwise, analyze from
        the HEAD commit date back self._months months.
        """
        owner, repo = gh_owner_repo(self.repo_path)
  
        #for commit in Repository(self.repo_path, only_commits=[hexsha]).traverse_commits():
        repo_data = []  # ensure you reset this before each run
        if commit_hashs:
            commit_iter = Repository(self.repo_path, only_commits=commit_hashs, only_no_merge=True).traverse_commits()
        elif self._months is None:
            commit_iter = Repository(self.repo_path, only_no_merge=True).traverse_commits()
        else:
            c = self.get_tip_commit(self.repo_path)
            since = c.committer_date - relativedelta(months=self._months)
            until = datetime.now(timezone.utc)
            commit_iter = Repository(self.repo_path, since=since, to=until, only_no_merge=True).traverse_commits()

        for commit in commit_iter:
        #for commit in Repository(self.repo_path, only_commits=[c.hash]).traverse_commits():
            commit_info = {
                'hash': commit.hash,
                'msg': commit.msg.splitlines()[0],
                'author': commit.author.name,
                'date': commit.author_date.isoformat(),
                'files': []
            }

            for mod in commit.modified_files:
                try:
                    # Normalize the full file path using mod.new_path                    
                    file_path = (mod.new_path or mod.old_path or "").replace("\\", "/")  # Use new_path, fallback to old_path, normalize slashes
                    name = os.path.basename(file_path)
                    parent = commit.parents[0] if commit.parents else None
                    # Exclude vendored deps so they don’t pollute SATD results:
                    if not want_path(file_path):
                        continue 
                    
                    if not (file_path.endswith(tuple(self.allowed_extensions)) or name.lower() == "dockerfile"  or BuildCommentExtractor. _is_build_file(Path(file_path))):
                        continue  # Skip files that don't match the allowed extensions
                    
                    if self.mode == "patch":
                        rows = self.extract_comments_by_diff(mod)
                    elif self.mode == "file":
                        rows = self.extract_comments_by_file(mod)
                    else:
                        rows = self.extract_comments_by_diff(mod) + self.extract_comments_by_file(mod)
                    if rows is None or len(rows) == 0:
                        continue
                    # remove duplicate comment events in the same file
                    rows = self._dedup_comment_events(rows)
                    #print(self._get_language(mod.new_path))
                    file_info = {
                                'path': file_path,
                                'language': self._get_language(file_path),
                                'change_type': str(mod.change_type.name),
                                'comments': [attach_github_links(r, owner, repo, commit.hash, parent) for r in rows]
                            }                
                    commit_info['files'].append(file_info)
                except Exception as e:
                    print(f"An error occurred test: {e}")
                    continue
            if commit_info['files'] is None or len(commit_info['files']) == 0:
                continue # Skip commits with no relevant files
            repo_data.append(commit_info)
              
        # Convert the result to a JSON string
        repo_data_json = json.dumps(repo_data, indent=4, ensure_ascii=False)

        # Write the result to a JSON file
       # Write the result to a JSON file
        with open(output_file, 'w', encoding='utf-8') as json_file:
            json_file.write(repo_data_json)
        return repo_data_json

# --- Example Usage ---
if __name__ == "__main__":
    # Replace 'path/to/your/repo' with the actual path to your Git repository.
    repo_path = "./repos/clones/taskcluster/taskcluster/"
    languages ={".java", ".py", ".yaml", ".yml", ".js", ".rb", ".go", ".ts","Dockerfile"}
    analyzer = CommitAnalyzer(repo_path, month=None, allowed_extensions=languages, mode="patch")
    output_file =f"./results/comments/comments_.json"
    analysis_results = analyzer.analyze_repository(output_file)
   
    
    
