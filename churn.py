"""Churn co-change clustering over a local git repo.

Pure pipeline: git log -> per-file commit rows -> co-change graph -> Louvain clusters.
No streamlit imports here so this stays testable (`python3 churn.py` runs the self-check).
"""

import itertools
import os
import re
import subprocess
import tempfile
from concurrent import futures
from fnmatch import fnmatch

import networkx as nx
import numpy as np
import pandas as pd

COMMIT_MARK = "__C__"

# Extension -> one of git's built-in diff drivers. The driver's xfuncname regex is what
# turns "@@ -10,2 +10,3 @@" into "@@ ... @@ public void save(...)", which is the whole
# trick behind the per-region view. Git ships these but enables none by default, and a
# hand-rolled regex does worse: git truncates the heading where the match ends, so a
# loose pattern yields headings like "export". Anything unlisted still gets git's
# generic fallback, which finds the nearest unindented line — usable for YAML and JSON.
FUNC_DRIVERS = {
    ".java": "java", ".py": "python", ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".cjs": "javascript", ".go": "golang", ".rb": "ruby", ".rs": "rust",
    ".cs": "csharp", ".kt": "kotlin", ".kts": "kotlin", ".php": "php",
    ".c": "cpp", ".h": "cpp", ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp",
    ".m": "objc", ".scss": "css", ".css": "css", ".less": "css",
    ".html": "html", ".htm": "html", ".vue": "html", ".md": "markdown",
    ".sh": "bash", ".bash": "bash", ".pl": "perl", ".pm": "perl",
    ".ex": "elixir", ".exs": "elixir", ".tex": "bibtex", ".el": "scheme",
}

_ATTRIBUTES_PATH = None


def attributes_file():
    """Path to a throwaway gitattributes file that switches the drivers on.

    Git reads diff drivers only from an attributes file, never from `-c` alone, so
    there has to be a real file. It goes in the temp dir: turning this on must not mean
    writing a .gitattributes into 64 repos the user did not ask us to touch.
    """
    global _ATTRIBUTES_PATH
    if _ATTRIBUTES_PATH is None or not os.path.exists(_ATTRIBUTES_PATH):
        handle, path = tempfile.mkstemp(prefix="churn-attrs-", suffix=".txt")
        with os.fdopen(handle, "w") as out:
            out.writelines(f"*{ext} diff={driver}\n" for ext, driver in FUNC_DRIVERS.items())
        _ATTRIBUTES_PATH = path
    return _ATTRIBUTES_PATH

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


# Automation that commits like a person. Left out of everything, not just the head
# count: a code-sync agent mirroring files between repos creates churn that no one
# wrote, so counting it as rework would misrank the very files it copies.
# Glob patterns, matched case-insensitively against the git author name.
IGNORE_AUTHORS = (
    "root",
    "*bot*",
    "*agent*",
    "*pipelines*",
    "*jenkins*",
    "*noreply*",
    "system administrator",
)


def is_bot(author, patterns=IGNORE_AUTHORS):
    name = (author or "").lower()
    return any(fnmatch(name, pattern.lower()) for pattern in patterns)


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
            f"--since={since}", f"--pretty=format:{COMMIT_MARK}%H\t%aI\t%an\t%s",
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


def git_file_hunks(repo, path, since):
    """Every diff hunk for one file, with git's guess at the enclosing function.

    -U0 keeps the hunks tight, so a heading names the region actually edited rather
    than a neighbour dragged in by context. --follow keeps history across renames --
    affordable here because this is one file, on demand, not all of them up front.
    No check=True: a path git cannot resolve is an empty result, not a crash.
    """
    return subprocess.run(
        [
            "git", "-c", f"core.attributesfile={attributes_file()}", "-C", repo,
            "log", "-p", "-U0", "--no-merges", "--follow", f"--since={since}",
            f"--pretty=format:{COMMIT_MARK}%H\t%aI\t%an\t%s", "--", path,
        ],
        capture_output=True, text=True,
    ).stdout


# "@@ -12,3 +12,5 @@ def save(self):" -- counts are omitted when they are 1.
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)$")


def parse_hunks(text, ignore_authors=IGNORE_AUTHORS):
    """git log -p -U0 output -> DataFrame(region, sha, date, subject, added, removed).

    `region` is git's section heading, which is only as good as the language's diff
    driver. Empty means git could not name it: a whole-file add or delete has nothing
    above the hunk to match, and an unsupported format has no pattern to match with.
    """
    rows = []
    sha = date = author = subject = None
    skip = False
    for line in text.splitlines():
        if line.startswith(COMMIT_MARK):
            parts = (line[len(COMMIT_MARK):].split("\t", 3) + ["", "", ""])[:4]
            sha, date, author, subject = parts
            skip = is_bot(author, ignore_authors)
            continue
        found = HUNK.match(line)
        if skip:
            continue
        if found:
            _, removed, _, added, region = found.groups()
            rows.append((
                region.strip(), sha, date, author, subject,
                1 if removed is None else int(removed),
                1 if added is None else int(added),
            ))
    df = pd.DataFrame(
        rows,
        columns=["region", "sha", "date", "author", "subject", "removed", "added"],
    )
    df["churn"] = df.added + df.removed
    return df


UNNAMED = "(unnamed region)"

# A declaration keyword names what follows it; check that before the call form, or
# `const total = useMemo(...)` would be filed under "useMemo".
DECLARED = re.compile(
    r"\b(?:function|class|interface|type|enum|struct|trait|def|const|let|var)"
    r"\s+([A-Za-z_$][\w$]*)"
)
CALLED = re.compile(r"([A-Za-z_$][\w$]*)\s*\(")


def region_symbol(heading):
    """Short name for a hunk heading, used to group its edits together.

    The heading is a whole signature line, so editing an argument rewrites it and the
    same method then splits into two rows that each look half as hot as it really is.
    Collapsing to the identifier keeps one region one row.
    """
    if not heading:
        return UNNAMED
    for pattern in (DECLARED, CALLED):
        found = pattern.search(heading)
        if found:
            return found.group(1)
    # Git truncates a heading at 80 characters, which can cut off the "(" the call form
    # needs. The trailing identifier of a signature line is still the method name.
    words = re.findall(r"[A-Za-z_$][\w$]*", heading)
    return words[-1] if words else heading.strip()[:60]


def region_stats(hunks, top=30):
    """Churn per named region inside one file — the treemap idea, one level down.

    A file that churns evenly is just busy. A file where one method takes most of the
    churn has a specific design problem, and this is the view that separates the two.
    """
    if hunks.empty:
        return pd.DataFrame()
    named = hunks.assign(
        symbol=hunks.region.map(region_symbol),
        # Per hunk, not per commit: with -U0 a hunk that both adds and removes IS a
        # replacement. A region has no line count of its own, so rework stays absolute.
        rework=rework(hunks.added, hunks.removed),
    )
    out = named.groupby("symbol").agg(
        churn=("churn", "sum"),
        rework=("rework", "sum"),
        added=("added", "sum"),
        deleted=("removed", "sum"),
        # Separate edits from commits: 5 edits over 5 commits is a region under steady
        # pressure; 5 edits in one commit is a single sweep.
        edits=("sha", "size"),
        commits=("sha", "nunique"),
        last=("date", "max"),
        authors=("author", "nunique"),
        # Bus factor: one name doing nearly all of it is knowledge in one head. Many
        # names on one region is the opposite risk -- nobody owns it.
        owner=("author", lambda s: s.mode().iat[0] if len(s) else ""),
        owner_share=("author", lambda s: round(s.value_counts(normalize=True).iat[0], 2)),
        # Longest heading seen for this symbol: the fullest signature, for the tooltip.
        signature=("region", lambda s: max(s, key=len)),
    ).reset_index().rename(columns={"symbol": "region"})
    out["signature"] = out.signature.where(out.signature != "", out.region)
    out["last"] = out["last"].str[:7]
    out["share"] = (out.churn / hunks.churn.sum()).round(3)
    out["growth"] = ((out.added - out.deleted) / out.churn).round(2)
    out = out[["region", "churn", "share", "rework", "edits", "commits", "growth",
               "authors", "owner", "owner_share", "last", "signature"]]
    return out.sort_values(["rework", "churn"], ascending=False).head(top).reset_index(drop=True)


def region_cochange(hunks, max_regions_per_commit=40):
    """Which regions inside one file keep changing together.

    The same co-change idea as the repo view, one level down: if two methods are always
    edited in the same commit they are one unit of behaviour split across two names.
    That is the shape of a class that wants to be two classes, or of a missing
    abstraction sitting between them.
    """
    if hunks.empty:
        return pd.DataFrame()
    # cochange_graph works on (path, sha) rows, and a region here plays the part a file
    # plays there -- so reuse it rather than writing the same pair-counting twice.
    as_paths = hunks.assign(path=hunks.region.map(region_symbol))
    graph = cochange_graph(as_paths, max_regions_per_commit)
    edges = [
        (a, b, data["weight"]) for a, b, data in graph.edges(data=True)
        if UNNAMED not in (a, b)  # an unnamed hunk is not a region to pair with
    ]
    if not edges:
        return pd.DataFrame()
    touched = as_paths.groupby("path").sha.nunique()
    out = pd.DataFrame(edges, columns=["region", "partner", "together"])
    # Together-count alone favours whichever pair is simply busiest. The share of the
    # rarer one's commits is what says "these two never move apart".
    out["of the rarer"] = (
        out.together / out.apply(lambda r: min(touched[r.region], touched[r.partner]), axis=1)
    ).round(2)
    return out.sort_values(["of the rarer", "together"], ascending=False).reset_index(drop=True)


def git_band_commits(repo, path, since, start, end):
    """Commits touching lines start..end of one file. `git log -L` follows the range
    backwards through history as lines move, which plain hunk positions cannot do:
    a line number from ten commits ago does not mean the same place in today's file."""
    return subprocess.run(
        [
            "git", "-C", repo, "log", f"-L{start},{end}:{path}", "-s",
            "--no-merges", f"--since={since}", f"--pretty=format:{COMMIT_MARK}%H\t%aI",
        ],
        capture_output=True, text=True,
    ).stdout


def git_file_source(repo, path):
    """The file as it stands at HEAD. Empty for a path that is no longer there."""
    return subprocess.run(
        ["git", "-C", repo, "show", f"HEAD:{path}"],
        capture_output=True, text=True,
    ).stdout


def line_bands(repo, path, since, lines, bands=48, fetch=git_band_commits, span=8):
    """Commit count per slice of the current file — where in the file the work lands.

    Language-independent, so it still says something for the files where no diff driver
    produces a function name. Costs one git call per band, which is why the bands are
    capped and the calls run concurrently. `span` is the smallest slice worth asking
    about: finer than this and the git calls outnumber the insight.
    """
    if not lines or lines < 2:
        return pd.DataFrame()
    bands = max(1, min(bands, lines // span or 1))
    size = -(-lines // bands)  # ceil, so the last band is the short one
    ranges = [(i + 1, min(i + size, lines)) for i in range(0, lines, size)]

    def scan(span):
        text = fetch(repo, path, since, *span)
        rows = [l[len(COMMIT_MARK):].split("\t") for l in text.splitlines()
                if l.startswith(COMMIT_MARK)]
        return {
            "start": span[0], "end": span[1],
            "commits": len({r[0] for r in rows}),
            "last": max((r[1][:7] for r in rows if len(r) > 1), default=""),
        }

    with futures.ThreadPoolExecutor(max_workers=12) as pool:
        out = pd.DataFrame(list(pool.map(scan, ranges)))
    out["band"] = out.apply(lambda r: "%d-%d" % (r.start, r.end), axis=1)
    return out[["band", "start", "end", "commits", "last"]]


def refine_bands(repo, path, since, bands, top=6, into=16, widest=40,
                 fetch=git_band_commits):
    """Re-scan the hottest bands at finer resolution, and splice the result back in.

    A flat scan has to trade width against call count, and on a 10,000-line file that
    means 215-line slices — enough to say "somewhere in here", not enough to point at a
    function. Spending the second pass only where the heat already is buys ~14-line
    slices in the part anyone will actually read, for the price of scanning 6 bands
    rather than all 48. Files whose bands are already narrow skip the pass entirely.
    """
    if bands.empty or (bands.end - bands.start + 1).max() <= widest:
        return bands
    hot = bands[bands.commits > 0].nlargest(top, "commits")
    if hot.empty:
        return bands

    spans = []
    for row in hot.itertuples():
        size = max(1, -(-(row.end - row.start + 1) // into))
        spans += [(i, min(i + size - 1, row.end))
                  for i in range(row.start, row.end + 1, size)]
    if not spans:
        return bands

    def scan(span):
        text = fetch(repo, path, since, *span)
        rows = [l[len(COMMIT_MARK):].split("\t") for l in text.splitlines()
                if l.startswith(COMMIT_MARK)]
        return {"start": span[0], "end": span[1], "commits": len({r[0] for r in rows}),
                "last": max((r[1][:7] for r in rows if len(r) > 1), default="")}

    with futures.ThreadPoolExecutor(max_workers=12) as pool:
        finer = pd.DataFrame(list(pool.map(scan, spans)))
    kept = bands[~bands.start.isin(hot.start)]
    out = pd.concat([kept, finer]).sort_values("start").reset_index(drop=True)
    out["band"] = out.apply(lambda r: "%d-%d" % (r.start, r.end), axis=1)
    return out[["band", "start", "end", "commits", "last"]]


def merge_bands(bands, into=20):
    """Fold fine bands into fewer, wider ones for the chart.

    The code view wants slices small enough to point at a function; a bar chart with
    that many rows is unreadable. Rather than scan twice, scan fine once and add up.
    """
    if bands.empty or len(bands) <= into:
        return bands
    group = bands.index // -(-len(bands) // into)
    out = bands.groupby(group).agg(
        start=("start", "min"), end=("end", "max"),
        # Sum, not max: a commit spanning two fine bands really did touch both. This
        # slightly over-counts a commit that straddles a boundary, which is the honest
        # price of not running a second, coarser sweep. Fine bands stay exact.
        commits=("commits", "sum"), last=("last", "max"),
    ).reset_index(drop=True)
    out["band"] = out.apply(lambda r: "%d-%d" % (r.start, r.end), axis=1)
    return out[["band", "start", "end", "commits", "last"]]


def line_heat(bands, lines):
    """Per-line commit count, spread from the band that covers each line.

    One value per line of the file at HEAD, so the source can be painted directly.
    Resolution is the band width, not the line — a line is as hot as its neighbourhood.
    """
    heat = [0] * lines
    for row in bands.itertuples():
        for i in range(row.start - 1, min(row.end, lines)):
            heat[i] = row.commits
    return heat


TICKET = re.compile(r"\b([A-Z][A-Z0-9]{1,9})-\d+\b")


def ticket_projects(subjects):
    """Which tracker projects show up in these commit subjects, and how often.

    An intent classifier ("is this a fix or a feature?") guesses from English and gets
    it wrong on real subjects: "Merged in FIX-131405-..." means the Jira project is
    called FIX, not that anything was fixed. Ticket prefixes are facts. What they buy
    is a different and better question anyway — how many separate workstreams keep
    landing in this one file. Several is a shared-ownership smell.
    """
    counts = {}
    for subject in subjects:
        for key in set(TICKET.findall(subject or "")):
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def parse_numstat(text, patterns=EXCLUDE, ignore_authors=IGNORE_AUTHORS):
    """git log --numstat output -> DataFrame(path, sha, date, added, deleted)."""
    rows = []
    sha = date = author = subject = None
    skip = False
    for line in text.splitlines():
        if line.startswith(COMMIT_MARK):
            # Subject goes last and keeps maxsplit, so a tab inside it stays in it.
            # A subject can also be empty, so pad rather than unpack strictly.
            parts = (line[len(COMMIT_MARK):].split("\t", 3) + ["", "", ""])[:4]
            sha, date, author, subject = parts
            skip = is_bot(author, ignore_authors)
        elif "\t" in line and not skip:
            added, deleted, path = line.split("\t", 2)
            if added == "-":  # binary diff
                continue
            path = resolve_rename(path)
            if excluded(path, patterns):
                continue
            rows.append((path, sha, date, author, subject, int(added), int(deleted)))
    df = pd.DataFrame(
        rows, columns=["path", "sha", "date", "author", "subject", "added", "deleted"]
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


def rework(added, deleted):
    """Lines replaced rather than added or removed: 2 * min(added, deleted).

    Churn cannot tell writing from rewriting. A 1,600-line plan committed once has 1,600
    churn and zero rework, because nothing was replaced; a 200-line class edited thirty
    times has modest churn and high rework. Only the second one is a design problem.

    Measure it per commit and sum, never over the whole window: a file added wholesale
    and later deleted wholesale has min(added, deleted) large across the window but zero
    inside any single commit, and it is not being reworked -- it is being created and
    then removed. That distinction is not academic. In whatfix-api it is the difference
    between ranking a migration component first (25,162 churn, 64 rework) and ranking
    the class that was genuinely rewritten eight times over.
    """
    return 2 * pd.concat([added, deleted], axis=1).min(axis=1)


def rework_per_file(commits):
    """path -> rework, summed over each commit separately. See rework()."""
    return commits.assign(
        rw=rework(commits.added, commits.deleted)
    ).groupby("path").rw.sum()


def debt_score(rework_lines, lines, commits):
    """Rework per surviving line, damped by how many commits it took.

    Two signals, because either alone misleads. The rate says how much of the file has
    been written over; the commit count says whether that happened once or kept
    happening. Log, not linear: the thirtieth commit is weaker evidence than the third,
    and without damping any config file touched on every build wins outright.
    """
    rate = rework_lines / lines
    return (rate * np.log2(1 + commits)).round(2)


def analyze(repo, since, max_files_per_commit, patterns=EXCLUDE,
            ignore_authors=IGNORE_AUTHORS):
    commits = parse_numstat(git_log(repo, since), patterns, ignore_authors)
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
    files["rework"] = files.path.map(rework_per_file(commits)).fillna(0).astype(int)
    files["debt"] = debt_score(files.rework, files.lines, files.commits)
    files["cluster_id"] = files.path.map(assignment)
    files["module"] = files.path.str.split("/").str[0]

    clusters = files.groupby("cluster_id").agg(
        total_churn=("churn", "sum"),
        num_files=("path", "nunique"),
        dominant_module=("module", lambda s: s.mode().iat[0]),
        num_modules=("module", "nunique"),
    ).reset_index()
    # Distinct commits, not sum(files.commits): a commit touching four files in this
    # cluster would otherwise count four times. Same trap summarise() documents.
    clusters["total_commits"] = clusters.cluster_id.map(
        commits.merge(files[["path", "cluster_id"]], on="path")
        .groupby("cluster_id").sha.nunique()
    ).fillna(0).astype(int)
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


def dev_days(commits, mapping=None):
    """Distinct (author, calendar day) pairs — a cheap stand-in for person-days.

    Neither churn nor commit count measures effort. Churn counts lines, and one
    regenerated fixture outweighs a fortnight of careful work. Commits count habits:
    the same day's work is ten small commits from one person and one big commit from
    the next. Days on which a person touched the code is much closer to time actually
    spent, and git already carries everything it needs.

    Still a proxy, and it has a known bias: a day someone spent ten minutes here counts
    the same as a day they spent eight hours. It is right about where attention goes,
    not about how many hours went there.
    """
    stamped = commits.assign(
        who_when=commits.author.fillna("") + "|" + commits.date.str[:10]
    )
    if mapping is None:
        return stamped.who_when.nunique()
    grouped = stamped.assign(group=stamped.path.map(mapping)).dropna(subset=["group"])
    return grouped.groupby("group").who_when.nunique()


# What a path is *for*. Order matters: the first match wins, so tests/build.gradle is
# build work, not test work. Deliberately coarse -- the question these answer is "did
# the day go on the product or on the machinery around it", not a taxonomy.
WORK_KINDS = (
    ("build/CI", r"(^|/)(build\.gradle|settings\.gradle|extensions\.gradle|pom\.xml"
                 r"|Makefile|Dockerfile|docker-compose|Jenkinsfile|bitbucket-pipelines"
                 r"|\.gitlab-ci|package\.json|tsconfig|webpack|babel|vite\.config"
                 r"|jest\.config|\.eslintrc|eslint\.config)"
                 r"|(^|/)(ci-cd|\.github|\.circleci|scripts|gradle)/"),
    ("deploy/config", r"(^|/)(charts?|helm|k8s|kubernetes|terraform|deploy|conf|config"
                      r"|configs|local-conf)/|\.(properties|env)$"
                      r"|(^|/)(mounts\.template|entrypoint\.sh|nginx)"
                      r"|application.*\.(ya?ml|properties)$"),
    ("tests", r"(^|/)(tests?|spec|__tests__|__mocks__|e2e)/|\.(test|spec)\.[jt]sx?$"
              r"|Test\.java$|_test\.py$|\.stories\.[jt]sx?$"),
    ("docs", r"\.(md|rst|adoc)$|(^|/)docs?/"),
)
PRODUCT = "product code"
INFRA = ("build/CI", "deploy/config")


def work_kind(path):
    """Which bucket a path belongs to. Everything unmatched is product code."""
    for name, pattern in WORK_KINDS:
        if re.search(pattern, path, re.I):
            return name
    return PRODUCT


def effort_split(commits):
    """Person-days per work kind — the infra-versus-features answer for one repo.

    A person-day is indivisible, so it is shared out across the kinds that person
    touched that day rather than counted once per kind. Otherwise someone who edits a
    Dockerfile and six classes on Tuesday adds two full days to the total and the
    percentages stop meaning anything. Weighted by distinct files, so the totals still
    add up to the real number of days worked.
    """
    kinds = [name for name, _ in WORK_KINDS] + [PRODUCT]
    if commits.empty:
        return dict.fromkeys(kinds, 0.0)
    touched = commits.assign(
        kind=commits.path.map(work_kind), day=commits.date.str[:10]
    ).drop_duplicates(["author", "day", "path"])
    per = touched.groupby(["author", "day", "kind"]).size()
    share = per / per.groupby(level=[0, 1]).sum()
    total = share.groupby("kind").sum()
    return {name: round(float(total.get(name, 0.0)), 1) for name in kinds}


def cluster_stats(clusters, files, edges, commits=None):
    """One row per cluster, for the table under the cluster treemap. Pure — no git.

    The treemap answers "which cluster is big". This answers "and is it worth opening":
    concentration says whether the churn is one file or spread across the cluster, and
    cohesion says whether the cluster is a real unit or a slice of something larger.
    """
    if clusters.empty:
        return pd.DataFrame()

    by_cluster = files.groupby("cluster_id")
    live = files[files.lines.notna()].groupby("cluster_id")

    # Module names, not just a count: "src + tests" and "client + server" both read as
    # 2 modules, but only one of them is a problem.
    def module_list(group):
        ranked = group.groupby("module").churn.sum().sort_values(ascending=False)
        head = ", ".join(ranked.index[:3])
        return head + (" +%d" % (len(ranked) - 3) if len(ranked) > 3 else "")

    # Share of the cluster's churn sitting in its single worst file. High means the
    # cluster is really one hotspot wearing a cluster's clothes.
    # nan, not pd.NA: a cluster whose churn sums to zero would otherwise divide into
    # an NAType, and NAType has no __round__ -- the whole screen raises rather than
    # showing one blank cell.
    top = by_cluster.churn.max() / by_cluster.churn.sum().replace(0, float("nan"))

    # Cohesion: of all co-change weight touching this cluster's files, how much stays
    # inside it. Edges are stored both ways, so anchoring on `a` counts each once.
    member = files.set_index("path").cluster_id
    cohesion = pd.Series(dtype=float)
    if not edges.empty:
        linked = edges.assign(
            ca=edges.a.map(member), cb=edges.b.map(member)
        ).dropna(subset=["ca", "cb"])
        total = linked.groupby("ca").weight.sum()
        inside = linked[linked.ca == linked.cb].groupby("ca").weight.sum()
        cohesion = (inside.reindex(total.index).fillna(0) / total).round(2)

    # Person-days beat both churn and commits as a read on effort; see dev_days().
    # Optional so the frame stays usable without a commits table to hand.
    blank = pd.Series(0, index=by_cluster.size().index)
    days = blank if commits is None else dev_days(
        commits, files.set_index("path").cluster_id).reindex(blank.index).fillna(0).astype(int)
    people = blank if commits is None else (
        commits.assign(c=commits.path.map(files.set_index("path").cluster_id))
        .dropna(subset=["c"]).groupby("c").author.nunique()
        .reindex(blank.index).fillna(0).astype(int))

    out = pd.DataFrame({
        "cluster": clusters.set_index("cluster_id").label,
        "modules": by_cluster.apply(module_list),
        "files": by_cluster.path.nunique(),
        "churn": by_cluster.churn.sum(),
        "share": (by_cluster.churn.sum() / files.churn.sum()).round(3),
        "rework": by_cluster.rework.sum(),
        "churn/line": (live.churn.sum() / live.lines.sum()).round(2),
        "rework/line": (live.rework.sum() / live.lines.sum()).round(2),
        "commits": clusters.set_index("cluster_id").total_commits,
        # Where the team's attention went, as opposed to where the lines went.
        "devs": people,
        "dev-days": days,
        "effort": (days / days.sum()).round(3) if days.sum() else days,
        "cohesion": cohesion,
        "hotspot": by_cluster.apply(lambda g: g.path.iat[g.churn.values.argmax()]),
        "in hotspot": top.round(2),
        "last": by_cluster.last_month.max(),
    })
    out = out.reindex(clusters.cluster_id)
    # Aligning against by_cluster (which still holds every cluster, including any the
    # caller filtered out) turns whole-number columns into floats via NaN. Undo that
    # after the reindex has dropped the extra rows, so counts render as counts.
    for column in ("files", "churn", "commits", "rework", "devs", "dev-days"):
        out[column] = out[column].fillna(0).astype(int)
    return out.reset_index(drop=True)


def file_stats(files, edges, cluster_id, commits=None):
    """One row per file in a cluster, for the table under the file treemap. Pure.

    Deliberately keeps files the treemap has to drop (deleted ones, which have no
    per-line figure). A table can show a blank cell; a treemap cannot draw a box of
    unknown size, so without this the churn simply vanishes from the screen.
    """
    members = files[files.cluster_id == cluster_id]
    if members.empty:
        return pd.DataFrame()
    inside = set(members.path)

    # Leakage: the share of a file's co-change weight that lands outside its own
    # cluster. High on a big file means the cluster boundary runs through it.
    outward_share = pd.Series(dtype=float)
    strongest = pd.Series(dtype=object)
    mine = edges[edges.a.isin(inside)] if not edges.empty else edges
    if len(mine):
        total = mine.groupby("a").weight.sum()
        away = mine[~mine.b.isin(inside)].groupby("a").weight.sum()
        outward_share = (away.reindex(total.index).fillna(0) / total).round(2)
        best = mine.sort_values("weight", ascending=False).groupby("a", sort=False).head(1)
        strongest = best.set_index("a").apply(
            lambda r: "%s  x%d" % (r.b, r.weight), axis=1
        )

    rows = members.set_index("path")
    # Person-days per file: the mapping is the identity, since here a file is the group.
    # Optional, like it is for a cluster, so the frame works without a commits table.
    blank = pd.Series(0, index=rows.index)
    days = blank if commits is None else dev_days(
        commits, pd.Series(rows.index, index=rows.index)
    ).reindex(rows.index).fillna(0).astype(int)
    people = blank if commits is None else (
        commits[commits.path.isin(rows.index)].groupby("path").author.nunique()
        .reindex(rows.index).fillna(0).astype(int))

    out = pd.DataFrame({
        "debt": rows.debt,
        "churn": rows.churn,
        "share": (rows.churn / rows.churn.sum()).round(3),
        # Churn answers "how much was written here"; rework answers "how much was
        # written OVER". A file created in one commit has churn but no rework.
        "rework": rows.rework,
        "rework/line": (rows.rework / rows.lines).round(2),
        "lines": rows.lines,
        "churn/line": rows.density,
        "commits": rows.commits,
        # Separates a fiddly hotspot (many small edits) from a rewrite (few huge ones).
        "per commit": (rows.churn / rows.commits.clip(lower=1)).round().astype(int),
        "devs": people,
        "dev-days": days,
        # Share of the cluster's person-days, so files compare against their neighbours
        # rather than against the whole repo.
        "effort": (days / days.sum()).round(3) if days.sum() else days,
        # Near 0 means the file was rewritten in place rather than grown: the same
        # lines kept being replaced. -1 is pure deletion, +1 pure addition.
        "growth": ((rows.added - rows.deleted) / rows.churn).round(2),
        "outside": outward_share.reindex(rows.index),
        "top partner": strongest.reindex(rows.index),
        "last": rows.last_month,
    }).sort_values(["debt", "churn"], ascending=False)
    return out.reset_index().rename(columns={"path": "file"})


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
        "rework": int(live.rework.sum()),
        "dev_days": int(dev_days(commits)),
        "devs": int(commits.author.nunique()),
        **{f"days_{name}": value for name, value in zip(
            ("build", "config", "tests", "docs", "product"),
            [effort_split(commits)[k] for k in
             ("build/CI", "deploy/config", "tests", "docs", PRODUCT)])},
        # The repo-level answer to "how much of this was rewriting, not writing".
        "rework_density": round(live.rework.sum() / lines_now, 2) if lines_now else None,
        "top_module": top_module,
    }


def demo():
    log = (
        f"{COMMIT_MARK}aaa\t2026-01-05T10:00:00+00:00\tAva\tadd users\n"
        "10\t2\tapi/users.py\n"
        "3\t1\tapi/schema.py\n"
        "-\t-\tassets/logo.png\n"
        "9\t9\tpoetry.lock\n"
        f"{COMMIT_MARK}bbb\t2026-02-05T10:00:00+00:00\tBo\t\n"
        "5\t0\tapi/{old => users}.py\n"
        "1\t1\tapi/schema.py\n"
        f"{COMMIT_MARK}ccc\t2026-03-05T10:00:00+00:00\tAva\ttouch web\n"
        "7\t0\tweb/app.js\n"
    )
    df = parse_numstat(log)
    assert set(df.path) == {"api/users.py", "api/schema.py", "web/app.js"}, df.path.tolist()
    assert df.churn.sum() == 10 + 2 + 3 + 1 + 5 + 1 + 1 + 7
    assert df.added.sum() == 10 + 3 + 5 + 1 + 7 and df.deleted.sum() == 2 + 1 + 1
    assert set(df.subject) == {"add users", "", "touch web"}, set(df.subject)
    assert set(df.author) == {"Ava", "Bo"}, set(df.author)

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

    # Bots are dropped whole, not just from the head count: a sync agent's churn is
    # code nobody wrote, so leaving it in would misrank the files it copies.
    assert is_bot("root") and is_bot("Code Sync Agent (Production)")
    assert is_bot("bitbucket-pipelines") and is_bot("dependabot[bot]")
    assert not is_bot("Deep Nandi") and not is_bot("releng-whatfix")
    assert not is_bot("root", ignore_authors := ("*bot*",)), "list must be honoured"
    botty = (
        f"{COMMIT_MARK}h1\t2026-01-05T10:00:00+00:00\tAva\treal work\n"
        "5\t1\tapp.py\n"
        f"{COMMIT_MARK}h2\t2026-01-06T10:00:00+00:00\tCode Sync Agent\tmirror\n"
        "900\t900\tapp.py\n"
    )
    assert parse_numstat(botty).churn.sum() == 6, parse_numstat(botty)
    assert set(parse_numstat(botty).author) == {"Ava"}
    # Keeping them is one argument away, for anyone who wants the raw picture.
    assert parse_numstat(botty, ignore_authors=()).churn.sum() == 1806

    # summarise: build the frames analyze() would produce from this fixture.
    files = df.groupby("path").agg(
        churn=("churn", "sum"), commits=("sha", "nunique"),
        added=("added", "sum"), deleted=("deleted", "sum"),
    ).reset_index()
    files["cluster_id"] = files.path.map(assignment)
    files["lines"] = files.path.map({"api/users.py": 100, "api/schema.py": 50})
    files["density"] = (files.churn / files.lines).round(1)
    files["rework"] = files.path.map(rework_per_file(df)).fillna(0).astype(int)
    files["debt"] = debt_score(files.rework, files.lines, files.commits)
    files["module"] = files.path.str.split("/").str[0]
    files["last_month"] = files.path.map(
        {"api/users.py": "2026-02", "api/schema.py": "2026-02", "web/app.js": "2026-03"}
    )
    clusters = pd.DataFrame([
        {"cluster_id": 0, "total_churn": 23, "dominant_module": "api", "num_modules": 2,
         "num_files": 2, "total_commits": 2, "label": "#0 api (2 files)"},
        {"cluster_id": -1, "total_churn": 7, "dominant_module": "web", "num_modules": 1,
         "num_files": 1, "total_commits": 1, "label": "unclustered (1 files)"},
    ])

    # cluster_stats: the table under the treemap.
    stats = cluster_stats(clusters, files, edges).set_index("cluster")
    api, un = stats.loc["#0 api (2 files)"], stats.loc["unclustered (1 files)"]
    assert api.files == 2 and api.churn == 23, stats
    assert api.share == round(23 / 30, 3), api.share       # of all churn, unclustered included
    assert api["churn/line"] == round(23 / 150, 2), api["churn/line"]
    # users.py is 17 of the cluster's 23 -- one file, not a spread-out cluster.
    assert api.hotspot == "api/users.py" and api["in hotspot"] == round(17 / 23, 2), api
    assert api.cohesion == 1.0, api.cohesion               # both partners inside
    assert api.modules == "api" and api.last == "2026-02", api
    # An edgeless file has no co-change weight at all, so cohesion is unknown, not 0.
    assert pd.isna(un.cohesion), un.cohesion
    assert un["in hotspot"] == 1.0, un["in hotspot"]        # a lone file IS its hotspot
    # Row order follows the clusters frame, so table and treemap agree.
    assert list(cluster_stats(clusters, files, edges).cluster) == list(clusters.label)
    assert cluster_stats(pd.DataFrame(), files, edges).empty
    # A cluster that churned zero lines must render a blank cell, not raise: NAType
    # has no __round__, so dividing by a nullable zero takes the screen down.
    idle = files.assign(churn=0, rework=0)
    assert pd.isna(cluster_stats(clusters, idle, edges)["in hotspot"].iat[0])

    # file_stats: the table under the file treemap.
    fs = file_stats(files, edges, 0).set_index("file")
    # Ordered by debt, not churn: both had 4 lines replaced, but schema.py is half the
    # size, so the same rework covers twice as much of it.
    assert list(fs.index) == ["api/schema.py", "api/users.py"], list(fs.index)
    assert fs.loc["api/schema.py"].debt > fs.loc["api/users.py"].debt, fs.debt
    assert fs.loc["api/users.py"]["rework/line"] == round(4 / 100, 2), fs

    # Person-days per file. users.py moved on two days (Ava, then Bo); schema.py the
    # same; web/app.js on one. Effort is the share of the cluster's days, not the repo's.
    withdays = file_stats(files, edges, 0, df).set_index("file")
    assert withdays.loc["api/users.py"]["dev-days"] == 2, withdays
    assert withdays.loc["api/users.py"].devs == 2, withdays
    assert withdays.effort.sum() == 1.0, withdays.effort
    lone = file_stats(files, edges, -1, df).set_index("file").loc["web/app.js"]
    assert lone["dev-days"] == 1 and lone.devs == 1 and lone.effort == 1.0, lone
    # Optional, same as for a cluster: no commits table means zeros, not a crash.
    assert file_stats(files, edges, 0)["dev-days"].sum() == 0
    u = fs.loc["api/users.py"]
    assert u.churn == 17 and u.commits == 2 and u["per commit"] == 8, u  # 17/2, banker's
    assert u.share == round(17 / 23, 3), u.share
    assert u["churn/line"] == round(17 / 100, 1) == 0.2, u["churn/line"]
    # +15 added, -2 deleted over 17 churn: grown, not rewritten in place.
    assert u.growth == round(13 / 17, 2), u.growth
    # Its only partner is inside the cluster, so nothing leaks out.
    assert u.outside == 0.0 and u["top partner"] == "api/schema.py  x2", u
    # A deleted file keeps its row -- blank per-line figure, churn still visible.
    web = file_stats(files, edges, -1).set_index("file").loc["web/app.js"]
    assert web.churn == 7 and pd.isna(web.lines), web
    assert pd.isna(web.outside) and pd.isna(web["top partner"]), web  # edgeless
    assert file_stats(files, edges, 99).empty

    # parse_hunks / region_stats: the region view on the detail screen.
    patch = (
        f"{COMMIT_MARK}aaa\t2026-01-05T10:00:00+00:00\tAva\tfix crash on save\n"
        "@@ -10,4 +10,6 @@ def save(self):\n"
        "@@ -80 +80 @@ def load(self):\n"          # omitted count means 1
        f"{COMMIT_MARK}bbb\t2026-03-05T10:00:00+00:00\tBo\trefactor save path\n"
        "@@ -10,2 +10,0 @@ def save(self):\n"      # pure deletion
        "@@ -0,0 +1,7 @@\n"                         # whole-file add: git names nothing
    )
    hunks = parse_hunks(patch)
    assert len(hunks) == 4 and hunks.churn.sum() == 10 + 2 + 2 + 7, hunks
    assert hunks.sha.nunique() == 2 and set(hunks.subject) == {
        "fix crash on save", "refactor save path"}, hunks

    regions = region_stats(hunks).set_index("region")
    assert list(regions.index[:1]) == ["save"], list(regions.index)
    save = regions.loc["save"]
    assert save.signature == "def save(self):", save.signature
    # Two edits across two commits: sustained pressure, not one sweep.
    assert save.churn == 12 and save.edits == 2 and save.commits == 2, save
    assert save.share == round(12 / 21, 3), save.share
    # +6 added, -6 deleted: rewritten in place, not grown.
    assert save.growth == 0.0, save.growth
    assert save["last"] == "2026-03", save["last"]
    # One method, two signatures: the same region, not two half-as-hot ones.
    split = parse_hunks(
        f"{COMMIT_MARK}ccc\t2026-04-05T10:00:00+00:00\tAva\twiden save\n"
        "@@ -10,3 +10,3 @@ def save(self):\n"
        "@@ -40,1 +40,1 @@ def save(self, force=False):\n"
    )
    merged = region_stats(split).set_index("region")
    assert list(merged.index) == ["save"], list(merged.index)
    assert merged.loc["save"].churn == 8 and merged.loc["save"].edits == 2, merged
    assert merged.loc["save"].signature == "def save(self, force=False):", merged

    assert region_symbol("public Map<String, Object> entQuery(String locale,") == "entQuery"
    assert region_symbol("const Widget: React.FC = () => {") == "Widget"
    assert region_symbol("public class AdminGestor extends DirectGestor {") == "AdminGestor"
    # A declaration wins over the call on the same line, or this reads as "useMemo".
    assert region_symbol("const total = useMemo(() => sum(rows), [rows])") == "total"
    assert region_symbol("") == UNNAMED
    # Git cut this heading at 80 chars, so there is no "(" left to anchor on.
    assert region_symbol(
        "public static Map<String, List<Map<String, String>>> getIntegrationPanelCondi"
    ) == "getIntegrationPanelCondi"
    assert regions.loc[UNNAMED].churn == 7, regions.loc[UNNAMED]
    assert region_stats(parse_hunks("")).empty

    keys = ticket_projects([
        "Merged in FIX-131405-flow-name (pull request #20491)",
        "Merged in feature/SUCC-38578-fab-label-fix (pull request #20156)",
        "Merged in FIX-130318 (pull request #20367)",
        "no ticket here",
    ])
    assert keys == {"FIX": 2, "SUCC": 1}, keys          # ordered by count
    # One subject naming the same project twice still counts once for that subject.
    assert ticket_projects(["FIX-1 and FIX-2"]) == {"FIX": 1}
    assert ticket_projects([None, ""]) == {}

    # Author spread: two people on save, one of them holding half of it.
    assert save.authors == 2 and save.owner_share == 0.5, save
    assert regions.loc[UNNAMED].authors == 1, regions.loc[UNNAMED]

    # region_cochange: save and load moved together in commit aaa.
    pairs = region_cochange(hunks)
    assert len(pairs) == 1, pairs                       # one row per pair, not two
    top_pair = pairs.iloc[0]
    assert {top_pair.region, top_pair.partner} == {"save", "load"}, top_pair
    assert top_pair.together == 1
    # load only ever moved once, and that once was with save, so it never moves apart.
    assert top_pair["of the rarer"] == 1.0, top_pair["of the rarer"]
    # An unnamed hunk is not a region, so it never becomes a pair.
    assert UNNAMED not in set(pairs.region) | set(pairs.partner), pairs
    assert region_cochange(parse_hunks("")).empty

    # line_bands: fake the git calls, so the banding maths is what gets tested.
    def fake(repo, path, since, start, end):
        hot = start <= 60 <= end
        return "".join(
            f"{COMMIT_MARK}sha{i}\t2026-0{i + 1}-01T00:00:00\n"
            for i in range(3 if hot else 1)
        )

    bands = line_bands("r", "f.py", "12 months ago", lines=100, bands=4, fetch=fake)
    assert list(bands.band) == ["1-25", "26-50", "51-75", "76-100"], list(bands.band)
    assert list(bands.commits) == [1, 1, 3, 1], list(bands.commits)
    assert bands.iloc[2]["last"] == "2026-03", bands.iloc[2]["last"]
    # A short file gets fewer bands, not 20 one-line ones: span is the floor width.
    assert len(line_bands("r", "f.py", "s", lines=40, bands=20, span=40, fetch=fake)) == 1
    # The last band is short rather than running past the end of the file.
    ragged = line_bands("r", "f.py", "s", lines=110, bands=4, fetch=fake)
    assert ragged.iloc[-1].end == 110, ragged
    assert line_bands("r", "f.py", "s", lines=0, fetch=fake).empty

    # refine_bands: narrow slices where the heat is, leave the cold parts coarse.
    wide = line_bands("r", "f.py", "s", lines=400, bands=4, span=100, fetch=fake)
    assert list(wide.commits) == [3, 1, 1, 1], list(wide.commits)   # 100-line bands
    finer = refine_bands("r", "f.py", "s", wide, top=1, into=5, widest=40, fetch=fake)
    # The one hot band (1-100) is replaced by 5 sub-bands; the other three stay whole.
    assert len(finer) == 3 + 5, list(finer.band)
    assert list(finer.band)[:5] == ["1-20", "21-40", "41-60", "61-80", "81-100"], list(finer.band)
    assert list(finer.band)[5:] == ["101-200", "201-300", "301-400"], list(finer.band)
    # Only the sub-band actually holding line 60 stays hot; the rest of the old band
    # was never that hot, which is the whole point of looking closer.
    assert list(finer.commits)[:5] == [1, 1, 3, 1, 1], list(finer.commits)
    assert list(finer.start) == sorted(finer.start), list(finer.start)
    # Already-narrow bands are left alone rather than scanned a second time.
    assert refine_bands("r", "f.py", "s", wide, widest=1000, fetch=fake) is wide

    # merge_bands: fold the fine scan into chart-sized rows without scanning twice.
    fine = line_bands("r", "f.py", "s", lines=120, bands=6, span=20, fetch=fake)
    assert len(fine) == 6, list(fine.band)
    rolled = merge_bands(fine, into=3)
    assert list(rolled.band) == ["1-40", "41-80", "81-120"], list(rolled.band)
    assert rolled.commits.sum() == fine.commits.sum(), (rolled.commits, fine.commits)
    assert merge_bands(fine, into=99) is fine        # already coarse enough
    assert merge_bands(pd.DataFrame(), into=3).empty

    # line_heat: every line gets its band's count, and the array is exactly file-long.
    heat = line_heat(fine, 120)
    assert len(heat) == 120 and heat[0] == fine.commits.iat[0], heat[:3]
    assert heat[59] == fine.commits.iat[2], heat[59]   # line 60 sits in band 41-60
    # A band running past the file end must not stretch the array.
    assert len(line_heat(fine, 50)) == 50

    assert os.path.exists(attributes_file())
    assert "*.java diff=java" in open(attributes_file()).read()
    # rework() only counts what a single commit both added and removed. users.py grew
    # by 10/-2 then 5/-0, so 2 of its lines were replaced; schema.py 3/-1 then 1/-1
    # gives 2 more. Nothing else in the fixture replaces anything.
    assert list(files.rework) == [4, 4, 0], list(files.rework)
    # web/app.js was pure addition -- churn 7, rework 0. That is the whole point.
    assert files[files.path == "web/app.js"].churn.iat[0] == 7

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

    assert s["rework"] == 8 and s["rework_density"] == round(8 / 150, 2), s
    # Three commits, but Ava made two of them on different days and Bo one: 3 person-
    # days, 2 people. Same-day commits by one person would have collapsed to one.
    assert s["dev_days"] == 3 and s["devs"] == 2, s
    # Three product-code days in the fixture, nothing else -- poetry.lock is excluded
    # before this ever runs and the .png is a binary diff.
    assert s["days_product"] == 3.0 and s["days_build"] == 0.0, s

    # effort_split: a day shared between kinds must not become two days.
    mixed = parse_numstat(
        f"{COMMIT_MARK}zzz\t2026-05-05T09:00:00+00:00\tAva\tship it\n"
        "4\t0\tbuild.gradle\n"
        "9\t1\tsrc/main/java/App.java\n"
        "2\t0\tsrc/main/java/Util.java\n"
    )
    split = effort_split(mixed)
    assert round(sum(split.values()), 3) == 1.0, split      # one person, one day
    assert split["build/CI"] == round(1 / 3, 1), split      # one of three files
    assert split[PRODUCT] == round(2 / 3, 1), split
    assert effort_split(parse_numstat("")) == {
        "build/CI": 0.0, "deploy/config": 0.0, "tests": 0.0, "docs": 0.0, PRODUCT: 0.0}
    assert work_kind("tests/build.gradle") == "build/CI"    # first match wins
    same_day = df.assign(author="Ava", date="2026-01-05T10:00:00+00:00")
    assert dev_days(same_day) == 1, dev_days(same_day)

    # dev_days per cluster, and effort as a share of the repo's days.
    stats = cluster_stats(clusters, files, edges, df).set_index("cluster")
    api, un = stats.loc["#0 api (2 files)"], stats.loc["unclustered (1 files)"]
    # api/*.py moved on two days (Ava then Bo); web/app.js on one (Ava).
    assert api["dev-days"] == 2 and api.devs == 2, api
    assert un["dev-days"] == 1 and un.devs == 1, un
    assert api.effort == round(2 / 3, 3), api.effort
    # Effort is optional: without a commits table the columns are zero, not missing.
    assert cluster_stats(clusters, files, edges)["dev-days"].sum() == 0

    gone = files.assign(lines=float("nan"))
    assert summarise(clusters, gone, df)["density"] is None
    assert summarise(pd.DataFrame(), files, df)["top_module"] is None
    print("ok")


if __name__ == "__main__":
    demo()
