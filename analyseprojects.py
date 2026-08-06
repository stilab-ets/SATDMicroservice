import pandas as pd
import subprocess
import json
from pathlib import Path
from pydriller import Repository
from tqdm import tqdm

# =====================================
# CONFIGURATION
# =====================================
repos_file = Path("./dataset/projects.xlsx")
PROJECT_COLUMN = "repo_full_name"
CLOC_PATH = Path("./lib/cloc-2.06.exe")

REPOS = Path("./repos/clones")
OUTPUT_FILE = Path("./RQ1/project_metrics.xlsx")

# =====================================
# UTILITIES
# =====================================
def repo_to_folder(repo: str) -> Path:
    """
    owner/repo -> repos/owner/repo
    """
    owner, name = repo.split("/")
    return REPOS / owner / name

# =====================================
# METRICS (OFFLINE)
# =====================================
def count_files(repo_path: Path) -> int:
    """Count tracked files only"""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_path,
        capture_output=True,
        text=True
    )
    return len(result.stdout.splitlines())


def compute_loc(repo_path: Path):
    """
    Compute logical LOC using cloc (snapshot-based).
    
    Returns:
        int  -> LOC if successfully computed
        None -> if LOC cannot be reliably computed
    """
    if not repo_path.exists():
        return None

    try:
        result = subprocess.run(
            [
                str(CLOC_PATH),
                str(repo_path),
                "--json",
                "--quiet",
                "--exclude-dir=.git,node_modules,vendor,dist,build,target,out"
            ],
            capture_output=True,
            text=True,
            timeout=300  # safety for very large repos
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return None

    # Robust JSON extraction (handles noisy output)
    stdout = result.stdout.strip()
    json_start = stdout.find("{")
    json_end = stdout.rfind("}") + 1

    if json_start == -1 or json_end == -1:
        return None

    try:
        data = json.loads(stdout[json_start:json_end])
        return data.get("SUM", {}).get("code")
    except json.JSONDecodeError:
        return None



def extract_commit_metrics(repo_path: Path):
    """
    Uses PyDriller to extract:
    - number of unique developers
    - number of commits
    """
    developers = set()
    commit_count = 0

    for commit in Repository(str(repo_path)).traverse_commits():
        commit_count += 1
        if commit.author and commit.author.email:
            developers.add(commit.author.email.lower())

    return {
        "developers": len(developers),
        "commits": commit_count
    }


# =====================================
# MAIN PIPELINE
# =====================================
def main():
    df = pd.read_excel(repos_file)
    projects = df[PROJECT_COLUMN].dropna().unique()

    results = []

    for repo in tqdm(projects, desc="Mining local repositories"):
        repo_path = repo_to_folder(repo)

        if not repo_path.exists():
            print(f"[SKIP] Repository not found locally: {repo}")
            continue

        files = count_files(repo_path)
        loc = compute_loc(repo_path)
        commit_metrics = extract_commit_metrics(repo_path)

        results.append({
            "project": repo,
            "files": files,
            "loc": loc,
            "developers": commit_metrics["developers"],
            "commits": commit_metrics["commits"]
        })

    out_df = pd.DataFrame(results)
    out_df.to_excel(OUTPUT_FILE, index=False)
    print(f"\n✅ Metrics saved to {OUTPUT_FILE}")

# =====================================
if __name__ == "__main__":
    main()
