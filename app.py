"""Churn explorer: portfolio -> cluster treemap -> file treemap -> per-file detail."""

import html
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import churn
import store

st.set_page_config(page_title="Churn clusters", layout="wide")

# Chart palette, kept in step with .streamlit/config.toml (editorial theme).
HOT = "#c41e3a"      # theme primaryColor
COOL = "#c9c9c9"     # light enough that plotly picks dark label text
# The darkest two OrRd steps are near-black, and a global theme font colour stops
# plotly from flipping label text to white on them. Drop them.
SCALE = px.colors.sequential.OrRd[:-2]

st.session_state.setdefault("screen", "portfolio")
st.session_state.setdefault("repo_path", None)
st.session_state.setdefault("cluster_id", None)
st.session_state.setdefault("file_path", None)
st.session_state.setdefault("person", None)


@st.cache_resource
def db():
    """One connection for the app. check_same_thread=False because Streamlit reruns
    the script on a different thread than the one that opened it."""
    return store.connect(same_thread=False)


@st.cache_data(show_spinner="Analyzing repo...")
def analyze(repo, since, max_files_per_commit, patterns, ignore_authors):
    return churn.analyze(repo, since, max_files_per_commit, patterns, ignore_authors)


@st.cache_data(show_spinner="Reading hunks...")
def file_hunks(repo, path, since, ignore_authors):
    """Per-hunk history for one file. Deliberately not part of analyze(): it is one
    extra `git log -p` per file, so it runs when someone opens that file's page rather
    than 64 times up front. Around 0.1-0.5s on the biggest files measured."""
    return churn.parse_hunks(churn.git_file_hunks(repo, path, since), ignore_authors)


@st.cache_data(show_spinner="Scanning line bands...")
def line_bands(repo, path, since, lines):
    """Up to 48 `git log -L` calls, run concurrently. Fine enough to paint the source
    line by line; merge_bands folds them down for the chart, so this scan happens once.
    Cached, and only ever for the one file being looked at."""
    bands = churn.line_bands(repo, path, since, lines)
    # Second pass only where the heat already is: a flat scan of a 10k-line file gives
    # 215-line slices, too coarse to point at a function.
    return churn.refine_bands(repo, path, since, bands)


@st.cache_data(show_spinner=False)
def file_source(repo, path):
    return churn.git_file_source(repo, path)


MAX_PAINTED = 4000


def heat_source(repo, path, bands, lines):
    """The file itself, each line tinted by how often its neighbourhood changed.

    The bar chart says a slice is hot; this says which code is in it. Colour is the
    band's commit count, so resolution is the band width -- a line is as hot as its
    neighbourhood, not measured on its own. Plain monospace rather than syntax
    highlighting: two colour systems on the same characters and neither reads.
    """
    source = file_source(repo, path).splitlines()
    if not source:
        return
    heat = churn.line_heat(bands, len(source))
    hottest = max(heat) or 1
    clipped = len(source) > MAX_PAINTED

    rows = []
    for number, (text, hits) in enumerate(zip(source[:MAX_PAINTED], heat), start=1):
        # Alpha, not a palette lookup: it tints whatever background the viewer's theme
        # paints, and leaves the text colour alone so the code stays readable.
        shade = f"rgba(196,30,58,{0.07 + 0.5 * hits / hottest:.3f})" if hits else "transparent"
        rows.append(
            f'<div style="background:{shade}" title="line {number} &middot; '
            f'{hits} commits"><span style="display:inline-block;width:4.5em;'
            f'text-align:right;padding-right:1em;opacity:.45">{number}</span>'
            f"{html.escape(text) or '&nbsp;'}</div>"
        )

    with st.expander(f"Show the code, painted ({len(source):,} lines)"):
        st.html(
            '<div style="max-height:640px;overflow:auto;font-family:ui-monospace,'
            'SFMono-Regular,Menlo,monospace;font-size:12px;line-height:1.5;'
            'white-space:pre;border:1px solid rgba(128,128,128,.25);border-radius:6px">'
            + "".join(rows) + "</div>"
        )
        st.caption(
            f"Darker = more commits touching that slice, up to {hottest} at the peak. "
            f"Slices are {bands.end.sub(bands.start).add(1).max()} lines wide, so the "
            "shade is the neighbourhood's, not the single line's. Untinted lines were "
            "never touched in the window."
            + (f" Showing the first {MAX_PAINTED:,} lines." if clipped else "")
        )


KINDS = {"days_product": "product code", "days_build": "build/CI",
         "days_config": "deploy/config", "days_tests": "tests", "days_docs": "docs"}


DEBT_HELP = """
**debt = (rework ÷ lines now) × log₂(1 + commits)**

- **rework** — lines a commit both added *and* removed, counted as `2 × min(added, deleted)`
  per commit and summed. A file written once has churn but no rework. It is measured
  inside each commit and never across the window, so a file added wholesale and later
  deleted wholesale scores zero: that is creation and removal, not rework.
- **rework ÷ lines now** — how many times each line *still in the file* has been written
  over. 1.0 means the file has effectively been rewritten once.
- **log₂(1 + commits)** — a damper, not a size bonus. It separates one big rewrite from
  edits that kept coming back. Log rather than linear, because the thirtieth commit is
  weaker evidence than the third; without it, a config file touched on every build wins
  outright.

**Reading it:** under 0.5 is settled · around 1 has been written over once · 3 and up has
been rewritten again and again and is still moving. It ranks files against each other. It
is not a unit, so no absolute value is a threshold on its own.

**Who is eligible:** production files only — tests, build/CI, deploy, config and docs are
all excluded. The file must still exist at HEAD and be at least 50 lines. Each repo
contributes its single highest-scoring file, and the six worst of those are shown.

**What it misses:** one sweeping rewrite and twenty small ones can land on a similar score
for different reasons, and a file that shrank scores high on a small absolute amount of
rework. Open the file and look at the commit count before acting on the rank.

Code: `churn.debt_score`, and the eligibility filter in `churn.summarise`.
"""

# One place for every term the screens use, so a number never has to be explained twice.
# (heading, standfirst, [(term, meaning)]) -- rendered on the definitions screen.
DEFINITIONS = [
    (
        "Counting change",
        "All of it comes from `git log --numstat` over the window in the sidebar. "
        "No tickets, no timesheets.",
        [
            ("churn", "Lines added plus lines deleted, summed over commits. Honest "
                      "about how much writing happened, but it cannot tell writing "
                      "from rewriting."),
            ("rework", "`2 × min(added, deleted)` per commit, summed. Lines replaced "
                       "rather than added. A 1,600-line document committed once has "
                       "1,600 churn and zero rework."),
            ("lines now", "Lines the file has at HEAD today. Blank, or *gone*, means "
                          "the file was deleted or emptied during the window, so its "
                          "churn is history rather than a live hotspot."),
            ("churn / line", "churn ÷ lines now. How many times the file has "
                             "effectively been rewritten, with size divided out — big "
                             "files churn more just for being big."),
            ("rework / line", "rework ÷ lines now. The same idea, counting only "
                              "replaced lines."),
            ("debt", "rework ÷ lines now, damped by log₂(1 + commits). The redesign "
                     "ranking. Full derivation at the bottom of this page."),
            ("growth", "(added − deleted) ÷ churn. +1 is pure addition, −1 pure "
                       "deletion, near 0 means the same lines were replaced in place."),
            ("churn / commit", "Separates a fiddly hotspot — many small edits — from a "
                               "rewrite, which is a few huge ones."),
        ],
    ),
    (
        "Effort",
        "Lines are not effort. These columns are the closest git gets to time spent.",
        [
            ("person-day", "One person, one calendar day with a commit. Two repos on "
                           "one day is still one day. It cannot tell ten minutes from "
                           "eight hours, so it is right about where attention went, "
                           "not how many hours went there."),
            ("work kind", "Every path is product code, build/CI, deploy/config, tests "
                          "or docs. First match wins, so `tests/build.gradle` counts as "
                          "build work. Deliberately coarse."),
            ("infra", "build/CI plus deploy/config — the machinery around the product, "
                      "as opposed to the product."),
            ("effort", "A file's or cluster's share of its group's person-days, so rows "
                       "compare against their neighbours rather than the whole repo."),
            ("split percentages", "A person-day is indivisible, so it is shared out "
                                  "across the kinds that person touched that day, "
                                  "weighted by distinct files. Each row sums to 100% "
                                  "and the totals still add up to real days worked."),
            ("repo memberships", "Per-repo identity counts, summed. Someone working in "
                                 "five repos counts five times. It is not a head count "
                                 "— the people screen has that."),
            ("person", "Git identities merged into one when the names match ignoring "
                       "punctuation and case, or when the local part of the email "
                       "does. So `Ronak Parmaar` and `ronak.parmaar` are one row."),
            ("top dev / bus factor", "The largest share of a repo's person-days held "
                                     "by one person. Above 40% is worth a second "
                                     "owner."),
        ],
    ),
    (
        "Clusters",
        "Clusters are read off how the code actually changes, not off the folder tree.",
        [
            ("co-change", "Two files edited in the same commit get an edge; its weight "
                          "is the number of such commits. Commits touching more files "
                          "than the sidebar cap are skipped, which also drops mass "
                          "renames and dependency bumps."),
            ("cluster", "A Louvain community over that graph — a set of files that "
                        "keep moving together, whatever directory they sit in."),
            ("unclustered", "Files that never co-changed inside the cap. Usually most "
                            "of the repo, so they are hidden by default."),
            ("modules", "The top-level directory of a file. A cluster spanning several "
                        "modules is a boundary that does not match how the work "
                        "happens."),
            ("cohesion", "Of all co-change weight touching a cluster's files, the share "
                         "that stays inside it. Low means the cluster is a slice of "
                         "something larger."),
            ("in hotspot", "Share of the cluster's churn sitting in its single worst "
                           "file. High means one hotspot wearing a cluster's clothes."),
            ("outside", "Share of one file's co-change weight landing outside its own "
                        "cluster. High on a big file means the cluster boundary runs "
                        "through it."),
            ("top partner", "The file it co-changed with most, and how many times."),
        ],
    ),
    (
        "Inside a file",
        "The same questions one level down: which parts of the file, and whose.",
        [
            ("region", "The function, method or declaration git names in the hunk "
                       "header. Edits to one symbol are grouped, so changing a "
                       "signature does not split a method into two rows. "
                       "*(unnamed region)* is a hunk git could not label — often a "
                       "whole-file rewrite, or a format with no diff driver."),
            ("edits", "Separate hunks touching a region. Many edits over few commits is "
                      "one sweep; the same count spread over many commits is pressure."),
            ("owned", "Share of a region's edits made by one person. Near 100% on a hot "
                      "region is knowledge held in one head."),
            ("where in the file", "Commits per slice of the file **as it stands "
                                  "today**, from `git log -L`. It follows each range "
                                  "back through history, so a line that moved is "
                                  "counted where it now lives. Works on any language."),
            ("of the rarer", "For two regions that change together: their shared "
                             "commits as a share of the less-touched region's own "
                             "commits. 100% means it never moves alone."),
            ("tracker projects", "Ticket keys parsed out of commit subjects. Several "
                                 "projects landing in one file is shared ownership."),
        ],
    ),
    (
        "What shapes every number above",
        "Four sidebar settings decide what git is even asked. Change one and every "
        "figure moves.",
        [
            ("months of history", "The window. Nothing before it exists here, so a file "
                                  "rewritten two years ago looks calm."),
            ("max files per commit", "Commits above the cap are ignored **when building "
                                     "the co-change graph only**. Their churn, rework "
                                     "and days still count."),
            ("exclude globs", "Generated files and lockfiles, dropped everywhere. They "
                              "dominate churn and fake cross-module coupling."),
            ("ignore authors", "Name patterns whose commits are dropped from "
                               "everything, head count included. A sync agent's churn "
                               "is code nobody wrote."),
        ],
    ),
]


def portfolio_review(done):
    """The whole portfolio on one screen: scale, then the four things worth acting on.

    Everything here comes off the stored rows, so it costs nothing to draw. The four
    panels are deliberately different questions — an effort sink is not a debt hotspot
    and neither is a bus factor — because ranking a portfolio on one number is how you
    end up funding the biggest repo instead of the worst one.
    """
    if done.empty or not done[list(KINDS)].sum().sum():
        return
    kind_days = done[list(KINDS)].sum().rename(KINDS)
    total = kind_days.sum()
    infra = kind_days[["build/CI", "deploy/config"]].sum()
    share = done[list(KINDS)].div(done[list(KINDS)].sum(axis=1).replace(0, float("nan")), axis=0)

    if st.button("Who spends time on what", type="tertiary", icon=":material/group:"):
        goto("people")
    st.markdown("##### Portfolio review")
    tiles = st.columns(5)
    tiles[0].metric("Person-days", f"{total:,.0f}", delta=f"{len(done)} repos",
                    delta_color="off")
    # Identities, not people: one person with two git configs is counted twice, and
    # saying "developers" here would be a number someone repeats in a board deck.
    # Summed across repos, so someone working in five of them still counts five times.
    tiles[1].metric("Repo memberships", f"{int(done.devs.sum()):,}",
                    delta="see the people screen for a head count", delta_color="off")
    tiles[2].metric("Infra", f"{infra / total:.0%}",
                    delta="build, deploy, config", delta_color="off")
    tiles[3].metric("Tests", f"{kind_days['tests'] / total:.1%}",
                    delta="of all effort", delta_color="off")
    tiles[4].metric("Rework", f"{done.rework.sum() / done.total_churn.sum():.0%}",
                    delta="of churn was rewriting", delta_color="off")

    left, right = st.columns(2)
    named = done.assign(
        infra_pct=share[["days_build", "days_config"]].sum(axis=1),
        test_pct=share.days_tests,
    )

    with left:
        st.caption("**Effort sinks** — the file that ate the most days in each repo")
        sinks = named[named.top_effort_file.notna()].nlargest(6, "top_effort_days")
        st.dataframe(
            sinks[["name", "top_effort_file", "top_effort_days", "top_effort_devs"]]
            .rename(columns={"name": "project", "top_effort_file": "file",
                             "top_effort_days": "days", "top_effort_devs": "devs"}),
            hide_index=True, width="stretch",
            column_config={"file": st.column_config.TextColumn(width="medium")},
        )
        st.caption("**Infra-heavy repos** — over a quarter of their days on plumbing")
        heavy = named[(named.infra_pct > 0.25) & (named.dev_days >= 20)].nlargest(6, "infra_pct")
        st.dataframe(
            heavy[["name", "dev_days", "infra_pct"]].rename(
                columns={"name": "project", "dev_days": "days", "infra_pct": "infra"}),
            hide_index=True, width="stretch",
            column_config={"infra": st.column_config.ProgressColumn(
                "infra", format="percent", min_value=0.0, max_value=1.0)},
        )

    with right:
        st.caption("**Redesign candidates** — worst production file in each repo")
        # The one column on this page nobody can read off its name. Say what it is
        # before the table, not in a tooltip nobody hovers.
        st.markdown(
            ":gray[**debt** = rework ÷ lines now × log₂(1 + commits) — how many times "
            "each surviving line has been written over, damped by how many sittings it "
            "took. **Under 0.5 settled · about 1 rewritten once over · 3+ rewritten "
            "again and again.** Production files only, 50 lines or more, still alive "
            "at HEAD.]"
        )
        with st.expander("How debt is scored"):
            st.markdown(DEBT_HELP)
        debt = named[named.top_debt_file.notna()].nlargest(6, "top_debt")
        st.dataframe(
            debt[["name", "top_debt_file", "top_debt"]].rename(
                columns={"name": "project", "top_debt_file": "file", "top_debt": "debt"}),
            hide_index=True, width="stretch",
            column_config={"file": st.column_config.TextColumn(width="medium"),
                           "debt": st.column_config.NumberColumn(
                               format="%.1f",
                               help="rework ÷ lines now × log2(1 + commits). About 1 "
                                    "means the file has been written over once; 3 and "
                                    "up means it kept being rewritten.")},
        )
        st.caption("**Thin on tests** — under 1% of effort, and busy enough to matter")
        thin = named[(named.test_pct < 0.01) & (named.dev_days >= 50)].nlargest(6, "dev_days")
        st.dataframe(
            thin[["name", "dev_days", "test_pct", "solo_share"]].rename(
                columns={"name": "project", "dev_days": "days", "test_pct": "tests",
                         "solo_share": "top dev"}),
            hide_index=True, width="stretch",
            column_config={
                "tests": st.column_config.NumberColumn(format="percent"),
                "top dev": st.column_config.NumberColumn(
                    format="percent",
                    help="Largest share of the repo's days held by one person — "
                         "bus factor. Above 40% is worth a second owner."),
            },
        )


@st.cache_data(show_spinner="Reading every repo's authors...")
def all_commits(repo_paths, since, max_files, patterns, ignore_authors):
    """Every repo's commit rows in one frame, tagged with the repo.

    Per-repo rows cannot answer a per-person question: days have to be deduplicated
    across the whole portfolio before they are counted. Each analyze() call is itself
    cached, so this is cheap right after Analyze all and slow only on a cold start.
    """
    frames = []
    for path in repo_paths:
        try:
            _, _, commits, _ = analyze(path, since, max_files, patterns, ignore_authors)
        except (subprocess.CalledProcessError, OSError):
            continue
        if not commits.empty:
            frames.append(commits.assign(repo=repo_name(path)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def repo_name(path):
    """Last two segments, so sibling projects stay distinguishable."""
    parts = Path(path).parts[-2:]
    return "/".join(parts) if parts else path


def patterns_from(text, fallback):
    """Lines of a settings box, falling back to the built-in list when it is empty.

    An empty box means "I have not set this", not "filter nothing". Read the other way
    round, clearing it by accident silently drops every exclusion and the numbers that
    come back still look plausible — bot churn quietly returns and nothing says so.
    A single line reading "none" is how to actually turn a list off.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return tuple(fallback)
    if len(lines) == 1 and lines[0].lower() == "none":
        return ()
    return tuple(lines)


def tail_of(path, keep=2):
    """Last few segments of a path. A deep Java path wraps the breadcrumb onto two
    lines and breaks its alignment; the full path still goes in the tooltip."""
    parts = path.split("/")
    return path if len(parts) <= keep else ".../" + "/".join(parts[-keep:])


def picked_row(selection, name, guard, column):
    """Row position of a freshly clicked cell in `column`, or None.

    Cell selection rather than row selection, because row selection makes Streamlit
    draw a checkbox down the left edge -- `rowMarkers: checkbox` is hardcoded in the
    grid and painted on a canvas, so neither a column config nor CSS can touch it.
    Selecting cells needs no marker, and it says which column was hit, so only the
    name column navigates and the numbers stay ordinary selectable cells.

    Positions are the frame's own, not the browser's sort order, so .iat is safe here.
    """
    cells = selection.get("selection", {}).get("cells") or []
    token = (guard, tuple(map(tuple, cells)))
    seen = st.session_state.get(f"picked_{name}")
    # Record every change, the empty one included. Clicking a selected cell deselects it
    # and clicking again re-selects it, so without the empty step in between the second
    # click matches the token from the first and the cell can never be opened twice.
    st.session_state[f"picked_{name}"] = token
    if seen == token:
        return None
    return next((row for row, col in cells if col == column), None)



def goto(screen, **kwargs):
    st.session_state.screen = screen
    st.session_state.update(kwargs)
    st.rerun()


def set_root_hover(fig, text):
    """A treemap needs one explicit root row, or plotly draws its own and paints the
    header dark — `root.color`, `depthfade` and `tiling.pad` all leave it alone (only
    a real row picks up the colour scale). The root has no source row though, so its
    customdata is empty and its hover would read "(?)". Fill it in."""
    trace = fig.data[0]
    data = trace.customdata
    for i, parent in enumerate(trace.parents):
        if not parent:  # the root is the only node without a parent
            data[i] = [text]
    trace.customdata = data
    return fig


def clicked_label(key, known):
    """Label of the clicked treemap box, or None if there is no usable selection.

    A selection survives in session_state across re-analysis, so a label from the
    previous repo or date window can still be sitting there. Ignore anything the
    current data doesn't contain.
    """
    points = st.session_state.get(key, {}).get("selection", {}).get("points") or []
    for point in points:
        label = point.get("label")
        if label and label in known:
            return label
    return None


conn = db()
saved = lambda key, default: store.get_setting(conn, key, default)

with st.sidebar:
    # Reachable from every screen, because the term you cannot read is never on the
    # page you happen to be standing on.
    if st.button("Definitions", type="tertiary", icon=":material/help:"):
        goto("definitions")
    paths_text = st.text_area(
        "Repo paths", value=saved("paths", ""), height=160,
        help="One absolute path per line. Saved to churn.db, so paste them once.",
    )
    repo_paths = [p.strip() for p in paths_text.splitlines() if p.strip()]
    months = st.number_input(
        "Months of history", 1, 24, int(saved("months", 12)),
        help="How far back to read git log.",
    )
    since = f"{int(months)} months ago"
    max_files = st.number_input(
        "Max files per commit", 2, 500, int(saved("max_files", 30)),
        help="Commits touching more files are ignored when building the co-change graph.",
    )
    excludes = st.text_area(
        "Exclude globs", key="excludes_box",
        value=saved("excludes", "\n".join(churn.EXCLUDE)), height=140,
        help="Generated files and lockfiles. They dominate churn and fake cross-module "
             "coupling. Empty falls back to the built-in list; type `none` to exclude "
             "nothing at all.",
    )
    ignore_text = st.text_area(
        "Ignore authors", key="ignore_box",
        value=saved("ignore_authors", "\n".join(churn.IGNORE_AUTHORS)),
        height=110,
        help="Glob patterns, matched case-insensitively against the git author name. "
             "Their commits are dropped from everything — churn and rework as well as "
             "the head count — because a sync agent's churn is code nobody wrote. "
             "Empty falls back to the built-in list; type `none` to keep every author.",
    )
    # Render-time only: not analyze() arguments, so moving these does not bust the cache.
    hot_pct = st.slider("Highlight top % churn", 1, 100, 20)
    size_by = st.radio(
        "Size by", ["rework", "churn", "churn / line"], horizontal=True,
        help=(
            "- **rework** — *what is being rewritten?* Lines a commit both added and "
            "removed. A file written once has churn but no rework, so one-shot "
            "additions stop crowding out the real hotspots. Start here.\n"
            "- **churn** — *where did our effort actually go?* Honest about time "
            "spent, but it cannot tell writing from rewriting.\n"
            "- **churn / line** — *which files are unstable for their size?* Divides "
            "out the fact that big files churn more just for being big.\n\n"
            "Rework is the one that points at redesign candidates."
        ),
    )
    show_unclustered = st.checkbox(
        "Show unclustered", value=False,
        help="Files that never co-changed inside the commit-size cap. Usually most of "
             "the repo, so they crowd out the real clusters.",
    )
    for label, chosen, box in [("exclude glob", patterns_from(excludes, churn.EXCLUDE), excludes),
                               ("ignored author", patterns_from(ignore_text, churn.IGNORE_AUTHORS), ignore_text)]:
        if not box.strip():
            st.caption(f"Using the {len(chosen)} built-in {label} patterns.")
        elif not chosen:
            st.caption(f"No {label} filtering — every match is kept.")

    analyze_all = st.button("Analyze all", type="primary")

patterns = patterns_from(excludes, churn.EXCLUDE)
bots = patterns_from(ignore_text, churn.IGNORE_AUTHORS)
for key, value in [
    ("paths", paths_text), ("months", int(months)),
    ("max_files", int(max_files)), ("excludes", excludes),
    ("ignore_authors", ignore_text),
]:
    if saved(key, None) != str(value):
        store.set_setting(conn, key, value)


def analyze_one(path):
    """Analyse one repo and persist its summary row. Never raises: a bad path among 60
    must cost that row, not the whole run."""
    row = {"path": path, "name": repo_name(path),
           "analyzed_at": datetime.now().isoformat(timespec="seconds"),
           "since_months": int(months), "max_files": int(max_files)}
    try:
        clusters, files, commits, _ = analyze(path, since, int(max_files), patterns, bots)
        if clusters.empty:
            row["error"] = "no commits in window"
        else:
            row.update(churn.summarise(clusters, files, commits))
    except subprocess.CalledProcessError as exc:
        row["error"] = (exc.stderr or "git failed").strip().splitlines()[-1]
    except OSError as exc:
        row["error"] = str(exc)
    store.save_repo(conn, row)


if analyze_all and repo_paths:
    bar = st.progress(0.0)
    for i, path in enumerate(repo_paths, 1):
        bar.progress((i - 1) / len(repo_paths), text=f"{i}/{len(repo_paths)} {repo_name(path)}")
        analyze_one(path)
    bar.empty()
    goto("portfolio")

screen = st.session_state.screen
repo = st.session_state.repo_path

if screen == "definitions":
    if st.button("projects", type="tertiary"):
        goto("portfolio")
    st.subheader("Definitions")
    st.caption(
        "Every number this app shows, what it is made of, and what it cannot see. "
        "Everything is derived from `git log` alone — no tickets, no timesheets, and "
        "no way to tell a good line from a bad one."
    )
    for heading, standfirst, terms in DEFINITIONS:
        st.markdown(f"##### {heading}")
        if standfirst:
            st.caption(standfirst)
        st.markdown("\n".join(f"- **{term}** — {meaning}" for term, meaning in terms))
    st.markdown("##### Debt, in full")
    st.caption("The redesign ranking, and the only column here that is a formula "
               "rather than a count.")
    st.markdown(DEBT_HELP)
    st.stop()

if screen == "portfolio":
    st.subheader("Projects")
    if not repo_paths:
        st.info("Paste repo paths in the sidebar — one absolute path per line — then Analyze all.")
        st.stop()
    portfolio = store.load_repos(conn, repo_paths)
    done = portfolio[portfolio.total_churn.notna()]
    # The sidebar drives the live per-file screens, but the portfolio rows are whatever
    # window they were last analysed in. Silently mixing the two makes the drill-down
    # disagree with the table it came from, and nothing on screen says why.
    stale = done[done.since_months.notna() & (done.since_months != int(months))]
    if not stale.empty:
        windows = ", ".join(f"{int(m)}" for m in sorted(stale.since_months.unique()))
        st.warning(
            f"Sidebar is set to **{int(months)} months**, but {len(stale)} row(s) were "
            f"analysed over {windows}. The cluster and file screens read the sidebar "
            "value live, so they will not match this table until you re-run.",
            icon=":material/history:",
        )
    if done.empty:
        st.info(f"{len(repo_paths)} repo(s) listed, none analysed yet. Press Analyze all.")
    else:
        metric = {"rework": "rework", "churn": "total_churn"}.get(size_by, "density")
        ranked = done[done[metric].notna()].assign(
            hover=lambda d: d.apply(
                lambda r: f"{r['name']}<br>churn {int(r.total_churn):,}"
                f"<br>rework {int(r.rework):,} ({r.rework_density} /line)"
                f"<br>density {r.density}<br>{int(r.num_files)} files"
                f" · {int(r.total_commits)} commits", axis=1)
        )
        fig = px.treemap(
            ranked, path=[px.Constant("projects"), "name"], values=metric,
            color="rework_density", color_continuous_scale=SCALE, custom_data=["hover"],
        )
        fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>",
                          pathbar_visible=False)
        set_root_hover(fig, f"{len(ranked)} projects<br>churn "
                            f"{int(ranked.total_churn.sum()):,}")
        fig.update_coloraxes(colorbar_title_text="rework/line")
        fig.update_layout(height=520, margin=dict(t=30, l=0, r=0, b=0))
        st.plotly_chart(fig, on_select="rerun", key="portfolio_tm", width="stretch")
        picked = clicked_label("portfolio_tm", set(ranked.name))
        if picked:
            goto("clusters", repo_path=ranked.loc[ranked.name == picked, "path"].iat[0],
                 cluster_id=None, file_path=None)

    portfolio_review(done)

    KIND_COLS = {"days_product": "product code", "days_build": "build/CI",
                 "days_config": "deploy/config", "days_tests": "tests",
                 "days_docs": "docs"}
    split = done[list(KIND_COLS)].sum().rename(KIND_COLS)
    if split.sum():
        st.markdown("##### Where the time goes")
        total = split.sum()
        infra = split[["build/CI", "deploy/config"]].sum()
        band = pd.DataFrame({"kind": split.index, "days": split.values, "all": ""})
        fig = px.bar(band[band.days > 0], x="days", y="all", color="kind",
                     orientation="h", text="kind",
                     color_discrete_sequence=[HOT, "#e06377", "#efa9a9", COOL, "#eeeeee"],
                     custom_data=["kind", "days"])
        fig.update_traces(
            hovertemplate="%{customdata[0]}<br>%{customdata[1]:.0f} person-days"
                          "<extra></extra>", textposition="inside", insidetextanchor="middle")
        fig.update_yaxes(visible=False)
        fig.update_xaxes(title=None)
        fig.update_layout(height=110, showlegend=False, bargap=0.05,
                          margin=dict(t=0, l=0, r=0, b=0))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"**{total:,.0f} person-days** across {len(done)} repos, "
            f"{int(done.devs.sum())} author identities. "
            f"**{infra / total:.0%} went to build, deploy and config**; "
            f"{split['product code'] / total:.0%} to product code. A person-day is one "
            "author on one calendar day, shared across the kinds of file they touched "
            "that day — so a Dockerfile edit alongside six classes does not count twice."
        )

    shown = portfolio.rename(columns={
        "name": "project", "total_churn": "churn", "num_files": "files",
        "total_commits": "commits", "num_clusters": "clusters",
        "cross_module": "cross-module", "analyzed_at": "analyzed",
        "rework_density": "rework/line", "lines_now": "lines"}).sort_values(
            "dev_days", ascending=False)
    kind_total = shown[list(KIND_COLS)].sum(axis=1)
    shown["infra %"] = (
        shown[["days_build", "days_config"]].sum(axis=1) / kind_total.replace(0, float("nan"))
    ).round(3)
    picked = st.dataframe(
        shown[["project", "dev_days", "devs", "infra %", "rework", "rework/line", "lines",
               "churn", "density", "files", "commits", "clusters", "cross-module",
               "top_module", "analyzed"]
              + (["error"] if shown.error.notna().any() else [])],
        hide_index=True, width="stretch",
        on_select="rerun", selection_mode="single-cell", key="portfolio_tbl",
        column_config={
            "rework": st.column_config.NumberColumn(
                format="%d",
                help="Lines a single commit both added and removed — rewriting, not "
                     "writing. Blank means this repo predates the metric; re-run it.",
            ),
            "rework/line": st.column_config.NumberColumn(
                format="%.2f", help="Rework per surviving line."
            ),
            "dev_days": st.column_config.NumberColumn(
                "dev-days", format="%d",
                help="Distinct (person, day) pairs — where the team's time went. Rank "
                     "by this to fund a team; rank by rework/line to fund a refactor.",
            ),
            "devs": st.column_config.NumberColumn(
                format="%d", help="Author identities that touched this repo. Counts "
                                  "identities, not people — one person with two git "
                                  "configs counts twice."
            ),
            "infra %": st.column_config.ProgressColumn(
                "infra %", format="percent", min_value=0.0, max_value=1.0,
                help="Share of this repo's person-days spent on build, CI, deploy and "
                     "config rather than product code.",
            ),
            "lines": st.column_config.NumberColumn(
                format="%d",
                help="Lines at HEAD across the files touched in the window. A high "
                     "rework/line on a tiny codebase is one volatile file, not a "
                     "portfolio signal — read the two together.",
            ),
        },
    )
    st.caption("Click a project name to open it.")
    hit = picked_row(picked, "portfolio_tbl", tuple(repo_paths), "project")
    if hit is not None:
        goto("clusters", repo_path=shown.path.iat[hit],
             cluster_id=None, file_path=None)
    st.stop()

if screen == "person":
    everyone = all_commits(tuple(repo_paths), since, int(max_files), patterns, bots)
    person = st.session_state.person
    mine = everyone[churn._who(everyone) == person] if not everyone.empty else everyone
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        if st.button("projects", type="tertiary"):
            goto("portfolio")
        st.markdown(":gray[/]")
        if st.button("people", type="tertiary"):
            goto("people")
        st.markdown(":gray[/]")
        st.markdown(f"**{person}**")
    if mine.empty:
        st.info(f"No commits for {person} in this window.")
        st.stop()

    day = mine.date.str[:10]
    tiles = st.columns(6)
    tiles[0].metric("Active days", f"{day.nunique():,}")
    tiles[1].metric("Repos", f"{mine.repo.nunique():,}")
    tiles[2].metric("Commits", f"{mine.sha.nunique():,}")
    tiles[3].metric("Churn", f"{int(mine.churn.sum()):,}")
    tiles[4].metric("Files touched", f"{mine.path.nunique():,}")
    tiles[5].metric("Active", f"{day.min()[:7]} to {day.max()[:7]}")
    aliases = sorted(set(mine.author))
    if len(aliases) > 1:
        st.caption("Git identities merged into this person: " + ", ".join(f"`{a}`" for a in aliases))

    st.caption("Active days per month")
    st.bar_chart(mine.assign(month=day.str[:7]).drop_duplicates(["month", "date"])
                 .groupby("month").size().rename("days"), color=HOT, height=180)

    st.markdown("##### Where their time goes")
    st.dataframe(
        churn.person_repos(everyone, person).rename(columns={"product code": "product"}),
        hide_index=True, width="stretch",
        column_config={
            "days": st.column_config.NumberColumn(format="%d"),
            "commits": st.column_config.NumberColumn(format="%d"),
            "churn": st.column_config.NumberColumn(format="%d"),
            **{k: st.column_config.NumberColumn(format="percent")
               for k in ("product", "build/CI", "deploy/config", "tests", "docs")},
        },
    )

    left, right = st.columns(2)
    with left:
        st.markdown("##### What only they touch")
        own = churn.person_ownership(everyone, person)
        if own.empty:
            st.caption("No file where they did most of the work over several days.")
        else:
            st.dataframe(
                own.head(12), hide_index=True, width="stretch",
                column_config={
                    "path": st.column_config.TextColumn(width="medium"),
                    "days": st.column_config.NumberColumn(format="%d"),
                    "share": st.column_config.ProgressColumn(
                        "share", format="percent", min_value=0.0, max_value=1.0,
                        help="Their share of that file's person-days."),
                    "others": st.column_config.NumberColumn(
                        format="%d", help="Other people who touched it. Zero is a "
                                          "single point of knowledge."),
                },
            )
            alone = own[own.others == 0]
            if not alone.empty:
                st.caption(f"**{len(alone)} file(s) nobody else touched all window.**")
    with right:
        st.markdown("##### Who they work alongside")
        peers = churn.person_peers(everyone, person)
        if peers.empty:
            st.caption("No files shared with anyone else.")
        else:
            st.dataframe(
                peers, hide_index=True, width="stretch",
                column_config={
                    "shared files": st.column_config.NumberColumn(
                        format="%d", help="Files both people touched in the window. "
                                          "Collaboration read off the code rather than "
                                          "the org chart."),
                    "shared repos": st.column_config.NumberColumn(format="%d"),
                },
            )
    st.stop()

if screen == "people":
    if st.button("projects", type="tertiary"):
        goto("portfolio")
    st.subheader("Who spends time on what")
    everyone = all_commits(tuple(repo_paths), since, int(max_files), patterns, bots)
    people = churn.people_effort(everyone)
    if people.empty:
        st.info("Nothing analysed yet. Press Analyze all on the projects screen.")
        st.stop()

    tiles = st.columns(4)
    tiles[0].metric("People", f"{len(people):,}",
                    delta="git identities merged", delta_color="off")
    tiles[1].metric("Person-days", f"{int(people.days.sum()):,}",
                    delta="deduplicated across repos", delta_color="off")
    tiles[2].metric("Repos per person", f"{people.repos.median():.0f}",
                    delta=f"median · max {int(people.repos.max())}", delta_color="off")
    tiles[3].metric("Median days", f"{people.days.median():.0f}",
                    delta=f"top author {int(people.days.max())}", delta_color="off")

    who = st.text_input("Filter by name", placeholder="type part of a name")
    shown_people = people[people.author.str.contains(who, case=False, na=False)] if who else people
    ratio = st.columns(2)
    ratio[0].caption(
        f"Showing {len(shown_people)} of {len(people)}. Percentages are that person's "
        "own days, so each row sums to 100%."
    )
    picked = st.dataframe(
        shown_people.rename(columns={"product code": "product"}),
        hide_index=True, width="stretch",
        on_select="rerun", selection_mode="single-cell", key="people_tbl",
        column_config={
            "author": st.column_config.TextColumn(width="medium"),
            "days": st.column_config.NumberColumn(
                format="%d", help="Distinct calendar days with a commit, across all "
                                  "repos. Two repos on one day is still one day."),
            "repos": st.column_config.NumberColumn(format="%d"),
            "top repo": st.column_config.TextColumn(width="medium"),
            **{name: st.column_config.NumberColumn(format="percent")
               for name in ("product", "build/CI", "deploy/config", "tests", "docs")},
        },
    )
    hit = picked_row(picked, "people_tbl", who, "author")
    if hit is not None:
        goto("person", person=shown_people.author.iat[hit])
    st.caption(
        "Click a name for their detail. "
        "A person-day is one author on one calendar day, split across the kinds of "
        "file they touched. It cannot tell ten minutes from eight hours, and it counts "
        "Git identities are merged into one person when their names match ignoring "
        "punctuation, or the local part of their email does — so Ronak Parmaar and "
        "ronak.parmaar are one row."
    )
    st.stop()

if not repo:
    goto("portfolio")

try:
    clusters, files, commits, edges = analyze(repo, since, int(max_files), patterns, bots)
except subprocess.CalledProcessError as exc:
    st.error(f"git log failed for `{repo}`:\n\n```\n{exc.stderr.strip()}\n```")
    st.stop()

if clusters.empty:
    st.info("No commits found. Check the repo path and the 'since' window.")
    st.stop()

cluster_id = st.session_state.cluster_id
cluster_row = clusters[clusters.cluster_id == cluster_id]
cluster_label = cluster_row.label.iat[0] if len(cluster_row) else None
if screen != "clusters" and cluster_label is None:
    goto("clusters")  # cluster ids are not stable across re-analysis

# (label, screen to jump to or None for the current page, tooltip)
trail = [("projects", "portfolio", None), (repo_name(repo), "clusters", repo)]
if cluster_label:
    # "(94 files)" is already in the metrics below; the crumb only needs identity.
    trail.append((cluster_label.split(" (")[0], "files", cluster_label))
if screen == "detail":
    file_path = st.session_state.file_path
    trail.append((tail_of(file_path), None, file_path))

# A flex row, not st.columns: four equal columns space the crumbs at 25% intervals,
# which reads as four separate controls rather than one path. Tertiary buttons render
# as links, because a breadcrumb is navigation and not a row of actions.
with st.container(horizontal=True, gap="small", vertical_alignment="center"):
    for step, (text, target, tip) in enumerate(trail):
        if step:
            st.markdown(":gray[/]")
        if target is None or target == screen:
            st.markdown(f"**{text}**", help=tip)  # you are here
        elif st.button(text, key=f"crumb{step}", type="tertiary", help=tip):
            goto(target)

if screen == "clusters":
    shown = clusters if show_unclustered else clusters[clusters.cluster_id != -1]
    # One prebuilt hover string per row, rather than several customdata fields, so the
    # root can be given a sensible line too (see set_root_hover).
    shown = shown.assign(
        hover=shown.apply(
            lambda r: f"{r.label}<br>churn {r.total_churn:,}<br>{r.total_commits} commits"
            f"<br>{r.num_files} files across {r.num_modules} module(s)",
            axis=1,
        )
    )
    fig = px.treemap(
        shown,
        path=[px.Constant(repo_name(repo)), "label"],
        values="total_churn",
        # Modules spanned, not file count: area already shows size, so spend the color
        # channel on the thing you'd act on. 1 = contained, higher = coupling to chase.
        color="num_modules",
        color_continuous_scale=SCALE,
        range_color=(1, max(2, int(shown.num_modules.max()))),
        custom_data=["hover"],
    )
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>",
        pathbar_visible=False,  # the app's own breadcrumb row does this job
    )
    set_root_hover(
        fig,
        f"{len(shown)} clusters<br>churn {int(shown.total_churn.sum()):,}"
        f"<br>{int(shown.num_files.sum())} files",
    )
    fig.update_coloraxes(colorbar_title_text="modules")
    fig.update_layout(height=560, margin=dict(t=30, l=0, r=0, b=0))
    st.plotly_chart(fig, on_select="rerun", key="clusters_tm", width="stretch")
    label = clicked_label("clusters_tm", set(shown.label))
    if label:
        goto("files", cluster_id=int(clusters.loc[clusters.label == label, "cluster_id"].iat[0]))

    real = clusters[clusters.cluster_id != -1]
    loose = clusters[clusters.cluster_id == -1]
    top = files.loc[files.churn.idxmax()]
    cards = st.columns(5)
    cards[0].metric("Churn", f"{int(files.churn.sum()):,}")
    cards[1].metric("Commits", f"{commits.sha.nunique():,}")
    cards[2].metric("Person-days", f"{churn.dev_days(commits):,}",
                    delta=f"{commits.author.nunique()} people", delta_color="off")
    cards[3].metric(
        "Clusters", len(real),
        delta=f"{int((real.num_modules > 1).sum())} cross-module",
        delta_color="off",
    )
    # Churn that never co-changed with anything is churn the cluster view cannot explain.
    cards[4].metric(
        "Unclustered", f"{int(loose.total_churn.sum() / files.churn.sum() * 100)}%",
        delta=f"{int(loose.num_files.iat[0]) if len(loose) else 0} files",
        delta_color="off",
    )
    st.caption(
        f"Biggest single file: `{top.path}` — {int(top.churn):,} churn "
        f"({top.churn / files.churn.sum():.0%} of the repo) over {int(top.commits)} commits."
    )

    st.markdown("##### Clusters")
    table = churn.cluster_stats(shown, files, edges, commits)
    picked = st.dataframe(
        table, hide_index=True, width="stretch",
        on_select="rerun", selection_mode="single-cell", key="clusters_tbl",
        column_config={
            "cluster": st.column_config.TextColumn(width="medium"),
            "modules": st.column_config.TextColumn(
                help="Top-level directories the cluster spans, biggest churn first."
            ),
            "churn": st.column_config.NumberColumn(format="%d"),
            "share": st.column_config.ProgressColumn(
                "share", format="percent", min_value=0.0,
                max_value=float(table.share.max()) if len(table) else 1.0,
                help="This cluster's churn as a fraction of the whole repo's.",
            ),
            "rework": st.column_config.NumberColumn(
                format="%d", help="Lines a single commit both added and removed."
            ),
            "rework/line": st.column_config.NumberColumn(
                format="%.2f",
                help="Rework per surviving line. The redesign signal — unlike "
                     "churn/line it ignores code that was only ever written once.",
            ),
            "churn/line": st.column_config.NumberColumn(
                format="%.2f",
                help="Rewrites per surviving line. High = rework, not growth. "
                     "Deleted files are left out of both sides.",
            ),
            "commits": st.column_config.NumberColumn(
                format="%d", help="Distinct commits touching the cluster."
            ),
            "devs": st.column_config.NumberColumn(
                format="%d", help="People who touched this cluster."
            ),
            "dev-days": st.column_config.NumberColumn(
                format="%d",
                help="Distinct (person, day) pairs — the closest cheap read on effort. "
                     "Churn measures lines, and one regenerated fixture outweighs a "
                     "fortnight of careful work.",
            ),
            "effort": st.column_config.ProgressColumn(
                "effort", format="percent", min_value=0.0,
                max_value=float(table.effort.max()) if len(table) else 1.0,
                help="This cluster's share of the repo's person-days. Sort by this to "
                     "find where the team's time is going, which is often not where "
                     "the churn is.",
            ),
            "cohesion": st.column_config.NumberColumn(
                format="%.2f",
                help="Of all co-change pairs touching these files, the fraction that "
                     "stay inside the cluster. 1.00 = self-contained. Low = the real "
                     "unit of change is bigger than this cluster.",
            ),
            "hotspot": st.column_config.TextColumn(
                width="large", help="Highest-churn file in the cluster."
            ),
            "in hotspot": st.column_config.NumberColumn(
                format="percent",
                help="Share of the cluster's churn in that one file. Near 100% means "
                     "this is one hot file, not a cluster worth opening.",
            ),
            "last": st.column_config.TextColumn(help="Most recent month touched."),
        },
    )
    hit = picked_row(picked, "clusters_tbl", repo, "cluster")
    if hit is not None:
        goto("files", cluster_id=int(
            clusters.loc[clusters.label == table.cluster.iat[hit], "cluster_id"].iat[0]))
    st.caption(
        "Click a cluster name to open it. Read it as: **share** = how much of the "
        "repo, **churn/line** = "
        "how much rework, **cohesion** = whether the cluster is a real boundary, "
        "**in hotspot** = whether it is really just one file."
    )

elif screen == "files":
    in_cluster = files[files.cluster_id == cluster_id]
    metric = {"rework": "rework", "churn": "churn"}.get(size_by, "density")
    # Deleted files have no density, so they drop out of the per-line view entirely
    # rather than being sized as if they were still live code.
    sized = in_cluster.dropna(subset=[metric])
    cutoff = sized[metric].quantile(1 - hot_pct / 100)
    sized = sized.assign(
        hot=(sized[metric] >= cutoff).map({True: "hot", False: "normal"}),
        # NaN would reach the hover template as "null".
        lines_txt=sized.lines.map(lambda n: "gone" if pd.isna(n) else f"{int(n)}"),
    )
    sized = sized.assign(
        hover=sized.apply(
            lambda r: f"{r.path}<br>{metric} {r[metric]:,}<br>churn {r.churn:,}"
            f" · lines {r.lines_txt}<br>{r.commits} commits",
            axis=1,
        )
    )
    dropped = len(in_cluster) - len(sized)
    if dropped:
        st.caption(f"{dropped} file(s) hidden: not in HEAD, so no per-line figure.")
    fig = px.treemap(
        sized,
        path=[px.Constant(cluster_label), "path"],
        values=metric,
        color="hot",
        # "(?)" is the root, which px leaves uncoloured.
        color_discrete_map={"hot": HOT, "normal": COOL, "(?)": COOL},
        custom_data=["hover"],
    )
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>",
        pathbar_visible=False,
    )
    set_root_hover(
        fig, f"{cluster_label}<br>{len(sized)} files<br>churn {int(sized.churn.sum()):,}"
    )
    fig.update_layout(height=560, margin=dict(t=30, l=0, r=0, b=0))
    st.plotly_chart(fig, on_select="rerun", key="files_tm", width="stretch")
    label = clicked_label("files_tm", set(sized.path))
    if label:
        goto("detail", file_path=label)

    summary = churn.cluster_stats(cluster_row, files, edges, commits).iloc[0]
    cluster_commits = commits[commits.path.isin(in_cluster.path)].sha.nunique()
    cards = st.columns(6)
    cards[0].metric("Churn", f"{int(summary.churn):,}",
                    delta=f"{summary.share:.0%} of repo", delta_color="off")
    cards[1].metric("Files", int(summary.files))
    cards[2].metric("Commits", f"{cluster_commits:,}")
    cards[3].metric("Churn / line", "—" if pd.isna(summary["churn/line"])
                    else f"{summary['churn/line']:.2f}")
    cards[4].metric("Cohesion", "—" if pd.isna(summary.cohesion)
                    else f"{summary.cohesion:.0%}",
                    delta="stays in cluster", delta_color="off")
    cards[5].metric("Modules", summary.modules.split(",")[0].strip(),
                    delta=summary.modules, delta_color="off")

    st.markdown("##### Files")
    table = churn.file_stats(files, edges, cluster_id, commits)
    hidden = table.lines.isna().sum()
    if hidden:
        st.caption(
            f"{hidden} of {len(table)} file(s) are not in HEAD. The treemap has to drop "
            "them — no size to draw — but their churn is real, so they are listed here "
            "with a blank per-line figure."
        )
    picked = st.dataframe(
        table, hide_index=True, width="stretch",
        on_select="rerun", selection_mode="single-cell", key=f"files_tbl_{cluster_id}",
        column_config={
            "file": st.column_config.TextColumn(width="large"),
            "debt": st.column_config.NumberColumn(
                format="%.2f",
                help="rework/line x log2(1 + commits). Rewriting, weighted by how many "
                     "separate commits it took. The sort order — highest is the best "
                     "redesign candidate.",
            ),
            "churn": st.column_config.NumberColumn(format="%d"),
            "rework": st.column_config.NumberColumn(
                format="%d",
                help="Lines a single commit both added and removed. A file written "
                     "once has churn but zero rework.",
            ),
            "rework/line": st.column_config.NumberColumn(
                format="%.2f", help="Rework per surviving line."
            ),
            "share": st.column_config.ProgressColumn(
                "share", format="percent", min_value=0.0,
                max_value=float(table.share.max()) if len(table) else 1.0,
                help="This file's churn as a fraction of the cluster's.",
            ),
            "lines": st.column_config.NumberColumn(
                format="%d", help="Lines at HEAD. Blank means the file is gone."
            ),
            "churn/line": st.column_config.NumberColumn(
                format="%.1f", help="Effective rewrites. 20 = replaced ~20 times over."
            ),
            "commits": st.column_config.NumberColumn(format="%d"),
            "per commit": st.column_config.NumberColumn(
                format="%d",
                help="Churn per commit. Small = fiddly hotspot, many little edits. "
                     "Large = rewritten in a few big drops.",
            ),
            "devs": st.column_config.NumberColumn(
                format="%d", help="People who touched this file."
            ),
            "dev-days": st.column_config.NumberColumn(
                format="%d",
                help="Distinct (person, day) pairs on this file — time spent, not "
                     "lines moved. A small config file can outrank a big class.",
            ),
            "effort": st.column_config.ProgressColumn(
                "effort", format="percent", min_value=0.0,
                max_value=float(table.effort.max()) if len(table) else 1.0,
                help="Share of this cluster's person-days.",
            ),
            "growth": st.column_config.NumberColumn(
                format="%.2f",
                help="(added - deleted) / churn. Near 0 means the file was rewritten "
                     "in place, not grown — the same lines kept being replaced. "
                     "+1 is pure addition, -1 pure deletion.",
            ),
            "outside": st.column_config.NumberColumn(
                format="percent",
                help="Share of this file's co-change weight going to files in OTHER "
                     "clusters. High on a big file means the cluster boundary runs "
                     "through it.",
            ),
            "top partner": st.column_config.TextColumn(
                width="large", help="Most frequent co-change partner, and how many "
                                    "commits they shared."
            ),
            "last": st.column_config.TextColumn(help="Most recent month touched."),
        },
    )
    hit = picked_row(picked, "files_tbl", cluster_id, "file")
    if hit is not None:
        goto("detail", file_path=table.file.iat[hit])
    st.caption(
        "Click a file name to open it. Read it as: **growth** near 0 = rework not "
        "new code, "
        "**per commit** small = a file people keep poking, **outside** high = this "
        "file belongs to a different cluster than the algorithm put it in."
    )

elif screen == "detail":
    path = st.session_state.file_path
    row = files[files.path == path]
    if row.empty:
        goto("clusters")
    row = row.iloc[0]
    history = commits[commits.path == path].sort_values("date", ascending=False)
    monthly = (
        history.assign(month=history.date.str[:7])
        .groupby("month")[["added", "deleted"]].sum()
    )

    gone = pd.isna(row.lines)
    cols = st.columns(6)
    cols[0].metric("Total churn", int(row.churn))
    cols[1].metric("Commits", int(row.commits))
    # Distinguishes a fiddly hotspot (small, often) from a rewrite (huge, rarely).
    cols[2].metric("Churn / commit", round(row.churn / max(1, row.commits)))
    cols[3].metric("Lines now", "gone" if gone else int(row.lines))
    # Effective rewrites: 20 means the file has been replaced ~20 times over.
    cols[4].metric("Rework / line", "—" if gone else round(row.rework / row.lines, 2),
                   delta=f"{int(row.rework):,} lines rewritten", delta_color="off")
    cols[5].metric("Last touched", row.last_month)
    if gone:
        st.caption(
            "Not in HEAD — deleted (or empty) since the window started, so its churn "
            "is history rather than a live hotspot."
        )

    st.caption(f"+{int(row.added):,} / -{int(row.deleted):,} lines by month")
    st.bar_chart(monthly, color=[COOL, HOT])

    st.subheader("Hot regions inside the file")
    hunks = file_hunks(repo, path, since, bots)
    regions = churn.region_stats(hunks)
    if regions.empty:
        st.caption("No diff hunks for this path in the window.")
    else:
        unnamed = regions.loc[regions.region == churn.UNNAMED, "share"].sum()
        top_region = regions.iloc[0]
        st.caption(
            f"`{top_region.region}` takes **{top_region.share:.0%}** of this file's "
            f"churn across {int(top_region.commits)} commits."
            + (f" {unnamed:.0%} of the churn is in hunks git could not name — usually "
               "whole-file rewrites, or a format with no diff driver."
               if unnamed >= 0.1 else "")
        )
        st.dataframe(
            regions, hide_index=True, width="stretch",
            column_config={
                "region": st.column_config.TextColumn(
                    width="medium",
                    help="Function, method or declaration git names in the hunk header. "
                         "Edits to the same symbol are grouped, so changing a signature "
                         "does not split one method into two rows.",
                ),
                "churn": st.column_config.NumberColumn(format="%d"),
                "share": st.column_config.ProgressColumn(
                    "share", format="percent", min_value=0.0,
                    max_value=float(regions.share.max()),
                    help="This region's churn as a fraction of the file's.",
                ),
                "rework": st.column_config.NumberColumn(
                    format="%d",
                    help="Lines a hunk both added and removed — this region being "
                         "rewritten rather than grown. The sort order.",
                ),
                "edits": st.column_config.NumberColumn(
                    format="%d",
                    help="Separate hunks touching this region. Many edits over few "
                         "commits is one sweep; spread over many commits is pressure.",
                ),
                "commits": st.column_config.NumberColumn(format="%d"),
                "growth": st.column_config.NumberColumn(
                    format="%.2f",
                    help="(added - deleted) / churn. Near 0 means the region was "
                         "rewritten in place — the same lines replaced again and again.",
                ),
                "authors": st.column_config.NumberColumn(
                    format="%d", help="Distinct people who edited this region."
                ),
                "owner": st.column_config.TextColumn(
                    help="Whoever made the most edits here."
                ),
                "owner_share": st.column_config.NumberColumn(
                    "owned", format="percent",
                    help="Share of edits by that one person. Near 100% on a hot region "
                         "is knowledge held in one head.",
                ),
                "last": st.column_config.TextColumn(help="Most recent month touched."),
                "signature": st.column_config.TextColumn(
                    width="large", help="Fullest signature line git reported."
                ),
            },
        )
        lonely = regions[(regions.owner_share >= 0.9) & (regions.share >= 0.05)]
        if not lonely.empty:
            who = ", ".join(f"`{r.region}` ({r.owner})" for r in lonely.head(3).itertuples())
            st.caption(f"One person has made 90%+ of the edits to: {who}.")

        pairs = churn.region_cochange(hunks)
        if not pairs.empty:
            st.markdown("##### Regions that change together")
            st.caption(
                "The repo-level co-change idea, one level down. Two methods always "
                "edited in the same commit are one unit of behaviour under two names."
            )
            st.dataframe(
                pairs.head(15), hide_index=True, width="stretch",
                column_config={
                    "region": st.column_config.TextColumn(width="medium"),
                    "partner": st.column_config.TextColumn(width="medium"),
                    "together": st.column_config.NumberColumn(
                        format="%d", help="Commits touching both."
                    ),
                    "of the rarer": st.column_config.NumberColumn(
                        format="percent",
                        help="Those shared commits as a share of the less-touched "
                             "region's own commits. 100% means it never moves alone.",
                    ),
                },
            )

        if not pd.isna(row.lines) and int(row.lines) >= 50:
            st.markdown("##### Where in the file")
            fine = line_bands(repo, path, since, int(row.lines))
            bands = churn.merge_bands(fine)
            if not bands.empty:
                quiet = fine[fine.commits == 0]
                fig = px.bar(
                    bands, x="commits", y="band", orientation="h", color="commits",
                    color_continuous_scale=SCALE,
                    custom_data=["band", "commits", "last"],
                )
                fig.update_traces(hovertemplate=(
                    "lines %{customdata[0]}<br>%{customdata[1]} commits"
                    "<br>last %{customdata[2]}<extra></extra>"))
                # Line 1 belongs at the top: the bar chart should read like the file.
                fig.update_yaxes(autorange="reversed", title=None)
                fig.update_layout(height=max(260, 22 * len(bands)), coloraxis_showscale=False,
                                  margin=dict(t=10, l=0, r=0, b=0))
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    "Commits per slice of the file **as it stands today** — `git log -L` "
                    "follows each range back through history, so a line that moved is "
                    "still counted in the place it now lives. Works on any language, "
                    "including the files no diff driver can name."
                    + (f" {len(quiet)} of {len(fine)} slices "
                       f"({int(quiet.end.sub(quiet.start).add(1).sum()):,} lines) went "
                       "untouched all window." if not quiet.empty else "")
                )
                heat_source(repo, path, fine, int(row.lines))

        projects = churn.ticket_projects(hunks.drop_duplicates("sha").subject)
        if projects:
            named = ", ".join(f"**{key}** ({n})" for key, n in list(projects.items())[:6])
            st.caption(
                f"Tracker projects landing here: {named}. Several projects hitting one "
                "file is shared ownership — a design boundary that does not match how "
                "the work is split."
            )

    st.subheader("Changed together with")
    partners = (
        edges[edges.a == path]
        .merge(files[["path", "cluster_id"]], left_on="b", right_on="path")
        .sort_values("weight", ascending=False)
        .head(10)
    )
    if partners.empty:
        st.caption(
            "No co-change partners — this file only ever changed alone, or only in "
            "commits above the max-files cap."
        )
    else:
        shown_partners = partners.assign(
            **{"% of its commits": (partners.weight / row.commits * 100).round().astype(int)}
        ).rename(columns={"b": "file", "weight": "commits together"})[
            ["file", "commits together", "% of its commits"]
        ]
        picked = st.dataframe(
            shown_partners, hide_index=True, width="stretch",
            on_select="rerun", selection_mode="single-cell", key=f"partners_{path}",
        )
        hit = picked_row(picked, "partners", path, "file")
        if hit is not None:
            jump = partners.iloc[hit]
            goto("detail", file_path=jump.b, cluster_id=int(jump.cluster_id))

    st.subheader(f"Commits ({len(history)})")
    st.dataframe(
        history.assign(commit=history.sha.str[:8], date=history.date.str[:10])[
            ["date", "commit", "added", "deleted", "subject"]
        ],
        hide_index=True, width="stretch",
    )
