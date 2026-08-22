"""Churn co-change clustering over a local git repo.

Pure pipeline: git log -> per-file commit rows -> co-change graph -> Louvain clusters.
No streamlit imports here so this stays testable (`python3 churn.py` runs the self-check).
"""

import itertools
import subprocess
from fnmatch import fnmatch

import networkx as nx
import pandas as pd

COMMIT_MARK = "__C__"

# Files that create false coupling: one dependency bump touches every module's lockfile
# in a single commit, so every module looks tightly coupled to every other one.
EXCLUDE = (
    "*.lock",
    "*.lockfile",
    "*-lock.json",
    "*.sum",
    "*.min.js",
    "*.map",
    "*.snap",
    "*/dist/*",
    "*/build/*",
    "*/vendor/*",
    "*/node_modules/*",
    "dist/*",
    "build/*",
    "vendor/*",
    "node_modules/*",
)


def excluded(path, patterns=EXCLUDE):
    return any(fnmatch(path, pat) for pat in patterns)


def resolve_rename(path):
    """Normalize git's rename notation to the post-rename path.

    ponytail: a renamed file's history splits at the rename, so churn before and
    after land on two nodes. Fixing that needs --follow per path (one git call per
    file); do it only if rename-heavy repos actually distort the clusters.
    """
    if " => " not in path:
        return path
    if "{" in path:
        prefix, rest = path.split("{", 1)
        mid, suffix = rest.split("}", 1)
        return prefix + mid.split(" => ")[1] + suffix
    return path.split(" => ")[1]


def git_log(repo, since):
    return subprocess.run(
        [
            "git", "-C", repo, "log", "--numstat", "--no-merges",
            f"--since={since}", f"--pretty=format:{COMMIT_MARK}%H\t%aI\t%s",
        ],
        capture_output=True, text=True, check=True,
    ).stdout


def git_line_counts(repo):
    """Lines per file at HEAD in one call. `-c ''` counts every line, `-I` skips
    binaries. Exits 1 on no matches, so no check=True."""
    return subprocess.run(
        ["git", "-C", repo, "grep", "-I", "-c", "", "HEAD"],
        capture_output=True, text=True,
    ).stdout


def parse_line_counts(text):
    """`HEAD:path:count` lines -> {path: lines}. Paths may contain colons."""
    counts = {}
    for line in text.splitlines():
        rest, count = line.rsplit(":", 1)
        counts[rest.split(":", 1)[1]] = int(count)
    return counts


def parse_numstat(text, patterns=EXCLUDE):
    """git log --numstat output -> DataFrame(path, sha, date, added, deleted)."""
    rows = []
    sha = date = subject = None
    for line in text.splitlines():
        if line.startswith(COMMIT_MARK):
            # A subject can be empty, so pad rather than unpack strictly.
            parts = (line[len(COMMIT_MARK):].split("\t", 2) + ["", ""])[:3]
            sha, date, subject = parts
        elif "\t" in line:
            added, deleted, path = line.split("\t", 2)
            if added == "-":  # binary diff
                continue
            path = resolve_rename(path)
            if excluded(path, patterns):
                continue
            rows.append((path, sha, date, subject, int(added), int(deleted)))
    df = pd.DataFrame(
        rows, columns=["path", "sha", "date", "subject", "added", "deleted"]
    )
    df["churn"] = df.added + df.deleted
    return df


def cochange_graph(commits, max_files_per_commit):
    """Files co-changed in a commit get an edge; weight = number of such commits.

    ponytail: pairs are O(n^2) per commit, so max_files_per_commit is load-bearing,
    not a nicety. It also drops the mass renames and dependency bumps that would
    otherwise dominate the graph.
    """
    graph = nx.Graph()
    graph.add_nodes_from(commits.path.unique())
    for _, group in commits.groupby("sha"):
        files = sorted(group.path.unique())
        if not 2 <= len(files) <= max_files_per_commit:
            continue
        for a, b in itertools.combinations(files, 2):
            weight = graph.edges[a, b]["weight"] + 1 if graph.has_edge(a, b) else 1
            graph.add_edge(a, b, weight=weight)
    return graph


def cluster(graph):
    """path -> cluster_id. Edgeless files all land in cluster -1."""
    assignment = {}
    linked = graph.subgraph([n for n, d in graph.degree if d > 0])
    for cluster_id, members in enumerate(
        nx.community.louvain_communities(linked, weight="weight", seed=0)
    ):
        for path in members:
            assignment[path] = cluster_id
    return {n: assignment.get(n, -1) for n in graph}


def edge_table(graph):
    """Co-change edges, both directions, so a partner lookup is one filter on `a`."""
    rows = [(a, b, d["weight"]) for a, b, d in graph.edges(data=True)]
    rows += [(b, a, w) for a, b, w in rows]
    return pd.DataFrame(rows, columns=["a", "b", "weight"])


def analyze(repo, since, max_files_per_commit, patterns=EXCLUDE):
    commits = parse_numstat(git_log(repo, since), patterns)
    if commits.empty:
        empty = pd.DataFrame()
        return empty, empty, commits, empty

    graph = cochange_graph(commits, max_files_per_commit)
    assignment = cluster(graph)

    files = commits.groupby("path").agg(
        churn=("churn", "sum"),
        added=("added", "sum"),
        deleted=("deleted", "sum"),
        commits=("sha", "nunique"),
        last_month=("date", "max"),
    ).reset_index()
    files["last_month"] = files.last_month.str[:7]

    # Churn alone ranks big files highest just for being big. Churn per current line
    # says how many times the file has effectively been rewritten. Files missing from
    # HEAD were deleted (or are empty), so they get no density rather than a fake one.
    files["lines"] = files.path.map(parse_line_counts(git_line_counts(repo)))
    files["density"] = (files.churn / files.lines).round(1)
    files["cluster_id"] = files.path.map(assignment)
    files["module"] = files.path.str.split("/").str[0]

    clusters = files.groupby("cluster_id").agg(
        total_churn=("churn", "sum"),
        num_files=("path", "nunique"),
        total_commits=("commits", "sum"),
        dominant_module=("module", lambda s: s.mode().iat[0]),
        num_modules=("module", "nunique"),
    ).reset_index()
    # cluster_id alone renders as "0"/"1"/"2" in a treemap, which tells you nothing.
    clusters["label"] = clusters.apply(
        lambda r: "unclustered (%d files)" % r.num_files
        if r.cluster_id == -1
        else "#%d %s (%d files)" % (r.cluster_id, r.dominant_module, r.num_files),
        axis=1,
    )
    return (
        clusters.sort_values("total_churn", ascending=False),
        files,
        commits,
        edge_table(graph),
    )


def summarise(clusters, files, commits):
    """Reduce one repo's analysis to the portfolio row. Pure — no git access."""
    real = clusters[clusters.cluster_id != -1] if not clusters.empty else clusters
    live = files[files.lines.notna()]
    lines_now = int(live.lines.sum())
    top_module = None
    if not real.empty:
        top_module = real.groupby("dominant_module").total_churn.sum().idxmax()
    return {
        "total_churn": int(files.churn.sum()),
        # From commits, not files.commits: a commit touching three files shows up in
        # three file rows, so summing that column multiplies every commit by its fan-out.
        "total_commits": int(commits.sha.nunique()),
        "num_files": len(files),
        "num_clusters": len(real),
        "cross_module": int((real.num_modules > 1).sum()) if not real.empty else 0,
        "lines_now": lines_now,
        # Live files on both sides. Dividing *total* churn by HEAD lines would count
        # deleted files in the numerator only, inflating every repo.
        "density": round(live.churn.sum() / lines_now, 2) if lines_now else None,
        "top_module": top_module,
    }


def demo():
    log = (
        f"{COMMIT_MARK}aaa\t2026-01-05T10:00:00+00:00\tadd users\n"
        "10\t2\tapi/users.py\n"
        "3\t1\tapi/schema.py\n"
        "-\t-\tassets/logo.png\n"
        "9\t9\tpoetry.lock\n"
        f"{COMMIT_MARK}bbb\t2026-02-05T10:00:00+00:00\t\n"
        "5\t0\tapi/{old => users}.py\n"
        "1\t1\tapi/schema.py\n"
        f"{COMMIT_MARK}ccc\t2026-03-05T10:00:00+00:00\ttouch web\n"
        "7\t0\tweb/app.js\n"
    )
    df = parse_numstat(log)
    assert set(df.path) == {"api/users.py", "api/schema.py", "web/app.js"}, df.path.tolist()
    assert df.churn.sum() == 10 + 2 + 3 + 1 + 5 + 1 + 1 + 7
    assert df.added.sum() == 10 + 3 + 5 + 1 + 7 and df.deleted.sum() == 2 + 1 + 1
    assert set(df.subject) == {"add users", "", "touch web"}, set(df.subject)

    graph = cochange_graph(df, max_files_per_commit=30)
    assert graph.edges["api/users.py", "api/schema.py"]["weight"] == 2
    assert graph.degree["web/app.js"] == 0

    # The commit cap drops the pair entirely rather than partially connecting it.
    assert cochange_graph(df, max_files_per_commit=1).number_of_edges() == 0

    assignment = cluster(graph)
    assert assignment["web/app.js"] == -1
    assert assignment["api/users.py"] == assignment["api/schema.py"] >= 0

    edges = edge_table(graph)
    partners = edges[edges.a == "api/users.py"]
    assert list(partners.b) == ["api/schema.py"] and partners.weight.iat[0] == 2
    assert edges[edges.a == "web/app.js"].empty  # edgeless file has no partners

    counts = parse_line_counts(
        "HEAD:api/users.py:120\nHEAD:weird/na:me.py:7\n"
    )
    assert counts == {"api/users.py": 120, "weird/na:me.py": 7}, counts

    assert resolve_rename("a/b/{x => y}/c.py") == "a/b/y/c.py"
    assert resolve_rename("old.py => new.py") == "new.py"
    assert resolve_rename("plain.py") == "plain.py"
    assert excluded("web/node_modules/x.js") and not excluded("web/app.js")

    # summarise: build the frames analyze() would produce from this fixture.
    files = df.groupby("path").agg(
        churn=("churn", "sum"), commits=("sha", "nunique")
    ).reset_index()
    files["cluster_id"] = files.path.map(assignment)
    files["lines"] = files.path.map({"api/users.py": 100, "api/schema.py": 50})
    clusters = pd.DataFrame([
        {"cluster_id": 0, "total_churn": 22, "dominant_module": "api", "num_modules": 2},
        {"cluster_id": -1, "total_churn": 7, "dominant_module": "web", "num_modules": 1},
    ])
    s = summarise(clusters, files, df)
    # Three commits, not 5. Summing files.commits would give 5, because commits aaa and
    # bbb each touch two files.
    assert files.commits.sum() == 5, files.commits.sum()
    assert s["total_commits"] == 3, s["total_commits"]
    assert s["total_churn"] == 30, s["total_churn"]      # includes web/app.js
    assert s["lines_now"] == 150, s["lines_now"]          # web/app.js has no lines
    assert s["num_clusters"] == 1 and s["cross_module"] == 1, s
    assert s["top_module"] == "api", s["top_module"]
    # Live churn (23) over live lines (150), not total churn (30) over 150.
    assert s["density"] == round(23 / 150, 2), s["density"]

    gone = files.assign(lines=float("nan"))
    assert summarise(clusters, gone, df)["density"] is None
    assert summarise(pd.DataFrame(), files, df)["top_module"] is None
    print("ok")


if __name__ == "__main__":
    demo()
