"""Churn cluster treemap: cluster treemap -> file treemap -> per-file detail."""

import subprocess

import pandas as pd
import plotly.express as px
import streamlit as st

import churn

st.set_page_config(page_title="Churn clusters", layout="wide")

st.session_state.setdefault("screen", "clusters")
st.session_state.setdefault("cluster_id", None)
st.session_state.setdefault("file_path", None)


@st.cache_data(show_spinner="Analyzing repo...")
def analyze(repo, since, max_files_per_commit, patterns):
    return churn.analyze(repo, since, max_files_per_commit, patterns)


def goto(screen, **kwargs):
    st.session_state.screen = screen
    st.session_state.update(kwargs)
    st.rerun()


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


with st.sidebar:
    repo = st.text_input("Repo path", value=".")
    since = st.text_input("Since", value="12 months ago")
    max_files = st.number_input(
        "Max files per commit", 2, 500, 30,
        help="Commits touching more files are ignored when building the co-change graph.",
    )
    excludes = st.text_area(
        "Exclude globs", value="\n".join(churn.EXCLUDE), height=140,
        help="Generated files and lockfiles. They dominate churn and fake cross-module coupling.",
    )
    analyzed = st.button("Analyze", type="primary")
    # Render-time only: not analyze() arguments, so moving these does not bust the cache.
    hot_pct = st.slider("Highlight top % churn", 1, 100, 20)
    size_by = st.radio(
        "Size files by", ["churn", "churn / line"], horizontal=True,
        help="Raw churn ranks big files highest just for being big. Per-line ranks by "
             "how often a file has effectively been rewritten.",
    )
    show_unclustered = st.checkbox(
        "Show unclustered", value=False,
        help="Files that never co-changed inside the commit-size cap. Usually most of "
             "the repo, so they crowd out the real clusters.",
    )

if analyzed:
    # Louvain renumbers clusters on every run, so a surviving cluster_id would point
    # at an unrelated cluster. Drop the drill-down state rather than mislabel it.
    st.session_state.update(screen="clusters", cluster_id=None, file_path=None)

try:
    patterns = tuple(line.strip() for line in excludes.splitlines() if line.strip())
    clusters, files, commits, edges = analyze(repo, since, int(max_files), patterns)
except subprocess.CalledProcessError as exc:
    st.error(f"git log failed for `{repo}`:\n\n```\n{exc.stderr.strip()}\n```")
    st.stop()

if clusters.empty:
    st.info("No commits found. Check the repo path and the 'since' window.")
    st.stop()

screen = st.session_state.screen
cluster_id = st.session_state.cluster_id
cluster_row = clusters[clusters.cluster_id == cluster_id]
cluster_label = cluster_row.label.iat[0] if len(cluster_row) else None
if screen != "clusters" and cluster_label is None:
    goto("clusters")  # cluster ids are not stable across re-analysis

crumbs = st.columns(4)
if crumbs[0].button("repo", disabled=screen == "clusters"):
    goto("clusters")
if cluster_label and crumbs[1].button(cluster_label, disabled=screen == "files"):
    goto("files")
if screen == "detail":
    crumbs[2].markdown(f"**{st.session_state.file_path}**")

if screen == "clusters":
    shown = clusters if show_unclustered else clusters[clusters.cluster_id != -1]
    # No px.Constant root: the breadcrumb already names the level, and a synthetic
    # root has no row behind it, so its hover renders "(?)" for every custom field.
    fig = px.treemap(
        shown,
        path=["label"],
        values="total_churn",
        # Modules spanned, not file count: area already shows size, so spend the color
        # channel on the thing you'd act on. 1 = contained, higher = coupling to chase.
        color="num_modules",
        color_continuous_scale="OrRd",
        range_color=(1, max(2, int(shown.num_modules.max()))),
        custom_data=["total_commits", "num_modules", "num_files"],
    )
    fig.update_traces(
        hovertemplate="%{label}<br>churn %{value}<br>%{customdata[0]} commits"
        "<br>%{customdata[2]} files across %{customdata[1]} module(s)<extra></extra>"
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
    dropped = len(in_cluster) - len(sized)
    if dropped:
        st.caption(f"{dropped} file(s) hidden: not in HEAD, so no per-line figure.")
    fig = px.treemap(
        sized,
        path=["path"],
        values=metric,
        color="hot",
        color_discrete_map={"hot": "#d1495b", "normal": "#8d99ae", "(?)": "#e9ecef"},
        custom_data=["commits", "churn", "lines_txt"],
    )
    fig.update_traces(
        hovertemplate="%{label}<br>" + metric + " %{value}"
        "<br>churn %{customdata[1]} · lines %{customdata[2]}"
        "<br>%{customdata[0]} commits<extra></extra>"
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
    st.bar_chart(monthly, color=["#8d99ae", "#d1495b"])

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
