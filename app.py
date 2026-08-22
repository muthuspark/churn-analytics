"""Churn explorer: portfolio -> cluster treemap -> file treemap -> per-file detail."""

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


@st.cache_resource
def db():
    """One connection for the app. check_same_thread=False because Streamlit reruns
    the script on a different thread than the one that opened it."""
    return store.connect(same_thread=False)


@st.cache_data(show_spinner="Analyzing repo...")
def analyze(repo, since, max_files_per_commit, patterns):
    return churn.analyze(repo, since, max_files_per_commit, patterns)


def repo_name(path):
    """Last two segments, so sibling projects stay distinguishable."""
    parts = Path(path).parts[-2:]
    return "/".join(parts) if parts else path


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
        "Exclude globs", value=saved("excludes", "\n".join(churn.EXCLUDE)), height=140,
        help="Generated files and lockfiles. They dominate churn and fake cross-module coupling.",
    )
    # Render-time only: not analyze() arguments, so moving these does not bust the cache.
    hot_pct = st.slider("Highlight top % churn", 1, 100, 20)
    size_by = st.radio(
        "Size by", ["churn", "churn / line"], horizontal=True,
        help=(
            "- **churn** — *where did our effort actually go?* Honest about time "
            "spent. Use it to orient on a new repo.\n"
            "- **churn / line** — *which files are unstable for their size?* This is "
            "the one that finds rework: code being rewritten rather than extended. "
            "Use it when hunting for something to redesign.\n\n"
            "Raw churn correlates with file size, so big files rank high just for "
            "being big. Per-line divides that out."
        ),
    )
    show_unclustered = st.checkbox(
        "Show unclustered", value=False,
        help="Files that never co-changed inside the commit-size cap. Usually most of "
             "the repo, so they crowd out the real clusters.",
    )
    analyze_all = st.button("Analyze all", type="primary")

patterns = tuple(line.strip() for line in excludes.splitlines() if line.strip())
for key, value in [
    ("paths", paths_text), ("months", int(months)),
    ("max_files", int(max_files)), ("excludes", excludes),
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
        clusters, files, commits, _ = analyze(path, since, int(max_files), patterns)
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

if screen == "portfolio":
    st.subheader("Projects")
    if not repo_paths:
        st.info("Paste repo paths in the sidebar — one absolute path per line — then Analyze all.")
        st.stop()
    portfolio = store.load_repos(conn, repo_paths)
    done = portfolio[portfolio.total_churn.notna()]
    if done.empty:
        st.info(f"{len(repo_paths)} repo(s) listed, none analysed yet. Press Analyze all.")
    else:
        metric = "total_churn" if size_by == "churn" else "density"
        ranked = done[done[metric].notna()].assign(
            hover=lambda d: d.apply(
                lambda r: f"{r['name']}<br>churn {int(r.total_churn):,}"
                f"<br>density {r.density}<br>{int(r.num_files)} files"
                f" · {int(r.total_commits)} commits", axis=1)
        )
        fig = px.treemap(
            ranked, path=[px.Constant("projects"), "name"], values=metric,
            color="density", color_continuous_scale=SCALE, custom_data=["hover"],
        )
        fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>",
                          pathbar_visible=False)
        set_root_hover(fig, f"{len(ranked)} projects<br>churn "
                            f"{int(ranked.total_churn.sum()):,}")
        fig.update_coloraxes(colorbar_title_text="churn/line")
        fig.update_layout(height=520, margin=dict(t=30, l=0, r=0, b=0))
        st.plotly_chart(fig, on_select="rerun", key="portfolio_tm", width="stretch")
        picked = clicked_label("portfolio_tm", set(ranked.name))
        if picked:
            goto("clusters", repo_path=ranked.loc[ranked.name == picked, "path"].iat[0],
                 cluster_id=None, file_path=None)

    shown = portfolio.rename(columns={
        "name": "project", "total_churn": "churn", "num_files": "files",
        "total_commits": "commits", "num_clusters": "clusters",
        "cross_module": "cross-module", "analyzed_at": "analyzed"})
    st.dataframe(
        shown[["project", "churn", "density", "files", "commits", "clusters",
               "cross-module", "top_module", "analyzed", "error"]],
        hide_index=True, width="stretch",
    )
    st.stop()

if not repo:
    goto("portfolio")

try:
    clusters, files, commits, edges = analyze(repo, since, int(max_files), patterns)
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

crumbs = st.columns(4)
if crumbs[0].button("projects"):
    goto("portfolio")
if crumbs[1].button(repo_name(repo), disabled=screen == "clusters"):
    goto("clusters")
if cluster_label and crumbs[2].button(cluster_label, disabled=screen == "files"):
    goto("files")
if screen == "detail":
    crumbs[3].markdown(f"**{st.session_state.file_path}**")

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

elif screen == "files":
    in_cluster = files[files.cluster_id == cluster_id]
    metric = "churn" if size_by == "churn" else "density"
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
    cols[4].metric("Churn / line", "—" if gone else row.density)
    cols[5].metric("Last touched", row.last_month)
    if gone:
        st.caption(
            "Not in HEAD — deleted (or empty) since the window started, so its churn "
            "is history rather than a live hotspot."
        )

    st.caption(f"+{int(row.added):,} / -{int(row.deleted):,} lines by month")
    st.bar_chart(monthly, color=[COOL, HOT])

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
            on_select="rerun", selection_mode="single-row", key=f"partners_{path}",
        )
        # A dataframe selection persists, so on arrival the destination's own stale
        # row would fire again and bounce straight back. Act on each selection once:
        # per-path keys stop the first bounce, the token stops a revisit re-firing.
        rows = picked.get("selection", {}).get("rows") or []
        token = (path, tuple(rows))
        if rows and st.session_state.get("partner_token") != token:
            st.session_state.partner_token = token
            jump = partners.iloc[rows[0]]
            goto("detail", file_path=jump.b, cluster_id=int(jump.cluster_id))

    st.subheader(f"Commits ({len(history)})")
    st.dataframe(
        history.assign(commit=history.sha.str[:8], date=history.date.str[:10])[
            ["date", "commit", "added", "deleted", "subject"]
        ],
        hide_index=True, width="stretch",
    )
