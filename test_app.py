"""UI wiring check: run with `python test_app.py <repo> [<second-repo>]`.

Covers the navigation logic and the persistence, which is where the non-obvious bugs
live: treemap and dataframe selections survive in session_state across a re-analysis,
so a stale label can retarget or ping-pong the drill-down.

Uses a temp database via CHURN_DB, so it never touches the real churn.db.
"""

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

APP = str(Path(__file__).with_name("app.py"))
SINCE = "12 months ago"


def main(repo, other):
    import churn
    import store
    from streamlit.testing.v1 import AppTest

    clusters, files, _, _ = churn.analyze(repo, SINCE, 30)
    assert not clusters.empty, f"no commits in {repo} since {SINCE}"
    top = clusters[clusters.cluster_id != -1].iloc[0]
    top_file = (
        files[files.cluster_id == top.cluster_id]
        .sort_values("churn", ascending=False)
        .path.iat[0]
    )

    def start(paths):
        """Fresh app instance with the given repo path list already in the sidebar."""
        at = AppTest.from_file(APP, default_timeout=600).run()
        assert not at.exception, at.exception
        at.text_area[0].set_value("\n".join(paths)).run()
        assert not at.exception, at.exception
        return at

    def button(at, label):
        """Address buttons by label. Positional indexing silently retargets when the
        sidebar is reordered, and "Analyze all" and the "projects" crumb both land on
        the portfolio, so a mix-up between them passes for the wrong reason."""
        return [b for b in at.button if b.label == label][0]

    bad = tempfile.mkdtemp()  # exists, but is not a git repo

    # --- portfolio: analyse a good repo and a bad path in one run -----------------
    at = start([repo, bad, other])
    button(at, "Analyze all").click().run()
    assert not at.exception, at.exception
    assert at.session_state["screen"] == "portfolio"

    conn = store.connect()
    rows = store.load_repos(conn, [repo, bad, other])
    assert len(rows) == 3, len(rows)
    good = rows[rows.path == repo].iloc[0]
    assert good.total_churn > 0, good.total_churn
    assert pd.isna(good.error), good.error
    # A bad path records its error and must NOT stop the others being saved.
    assert rows[rows.path == bad].iloc[0].error, "bad path should record an error"
    assert rows[rows.path == other].iloc[0].total_churn > 0, "third repo lost to the bad one"
    # Effort band plus treemap.
    assert len(at.get("plotly_chart")) == 2, len(at.get("plotly_chart"))
    # Four review panels plus the portfolio table itself, and a fifth panel whenever a
    # repo has a file young enough to qualify. Counted as a range, because whether that
    # panel draws depends on the age of whatever repo the test was pointed at.
    assert 5 <= len(at.dataframe) <= 6, len(at.dataframe)
    assert len(at.metric) == 5, "review tiles missing"
    split = next(d.value for d in at.dataframe if "infra %" in d.value.columns)
    # NaN for the bad path: no days recorded, so no split to show. Never out of range.
    infra = split["infra %"]
    assert infra.notna().sum() == 2, infra
    assert infra.dropna().between(0, 1).all(), infra

    # --- persistence: a brand new app instance sees the stored rows ---------------
    at2 = start([repo, bad, other])
    assert not at2.exception, at2.exception
    assert len(at2.get("plotly_chart")) == 2, "stored rows should render without re-analysing"
    table = next(d.value for d in at2.dataframe if "churn" in d.value.columns)
    assert (table["churn"].notna()).sum() >= 2, table[["project", "churn"]]

    # --- the people screen: days deduplicated across repos ------------------------
    at = start([repo, other])
    at.session_state["screen"] = "people"
    at.run()
    assert not at.exception, at.exception
    ppl = next(d.value for d in at.dataframe if "top repo" in d.value.columns)
    assert not ppl.empty and ppl.days.is_monotonic_decreasing, ppl.head()
    kinds = ["product", "build/CI", "deploy/config", "tests", "docs"]
    # Each row is that person's own split, so it sums to 1 — not to the org's.
    assert ppl[kinds].sum(axis=1).round(2).between(0.99, 1.01).all(), ppl[kinds].sum(axis=1)
    # Someone in both repos on one day is one day, so the total cannot exceed the
    # number of distinct author-days and must be below the sum of the per-repo counts.
    assert ppl.repos.max() <= 2, ppl.repos.max()
    assert ppl.days.sum() > 0

    # Clicking a name opens that person, and their detail adds up to their row.
    row = ppl.iloc[0]
    at.session_state["people_tbl"] = {"selection": {"cells": [[0, "author"]]}}
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["screen"] == "person", at.session_state["screen"]
    assert at.session_state["person"] == row.author, at.session_state["person"]
    where = next(d.value for d in at.dataframe if "commits" in d.value.columns
                 and "repo" in d.value.columns)
    # One row per repo they worked in, matching the count on the people table.
    assert len(where) == row.repos, (len(where), row.repos)
    # Per-repo days can exceed the person's own total: two repos on one day is one day
    # for them but a day in each repo. It must never be lower.
    assert where.days.sum() >= row.days, (where.days.sum(), row.days)
    assert int(at.metric[1].value.replace(",", "")) == row.repos, at.metric[1].value

    # --- the org-wide file table: filter, sort, page, then open a row -------------
    at = start([repo, other])
    at.session_state["screen"] = "rework"
    at.run()
    assert not at.exception, at.exception

    def widget(at, kind, label):
        return [w for w in getattr(at, kind) if w.label == label][0]

    org = next(d.value for d in at.dataframe
               if "project" in d.value.columns and "rework/line" in d.value.columns)
    # Default order is the whole list by rework, so the first page is the real top.
    assert not org.empty and org.rework.is_monotonic_decreasing, org.head()
    everything = int(at.metric[0].value.replace(",", ""))
    assert everything >= len(org), (everything, len(org))

    # Filtering narrows the list, and every surviving row matches.
    widget(at, "text_input", "Filter by path").set_value(".py").run()
    assert not at.exception, at.exception
    narrowed = next(d.value for d in at.dataframe if "project" in d.value.columns)
    assert not narrowed.empty and narrowed.file.str.endswith(".py").all(), narrowed.file

    # Sorting runs over the filtered list before the page is cut, so reversing it must
    # change the first row rather than reordering the 50 rows already on screen.
    widget(at, "toggle", "Ascending").set_value(True).run()
    flipped = next(d.value for d in at.dataframe if "project" in d.value.columns)
    assert flipped.rework.is_monotonic_increasing, flipped.rework.tolist()

    # Paging, and a filter change putting you back on page 1.
    widget(at, "toggle", "Ascending").set_value(False).run()
    widget(at, "text_input", "Filter by path").set_value("").run()
    widget(at, "selectbox", "Rows per page").set_value(25).run()
    first_page = next(d.value for d in at.dataframe if "project" in d.value.columns)
    assert len(first_page) <= 25, len(first_page)
    page = widget(at, "number_input", "Page")
    if page.max > 1:
        page.set_value(2).run()
        assert not at.exception, at.exception
        second = next(d.value for d in at.dataframe if "project" in d.value.columns)
        # Compared as frames, not by file name: the same path can appear in two repos,
        # so a name that shows up on both pages proves nothing either way.
        assert not second.empty and not second.reset_index(drop=True).equals(
            first_page.reset_index(drop=True)), second
        widget(at, "text_input", "Filter by path").set_value("a").run()
        assert widget(at, "number_input", "Page").value == 1, "filter must reset the page"
        widget(at, "text_input", "Filter by path").set_value("").run()

    # Clicking a file name opens it, in its own repo, with its cluster in hand.
    shown = next(d.value for d in at.dataframe if "project" in d.value.columns)
    at.session_state["rework_tbl"] = {"selection": {"cells": [[0, "file"]]}}
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["screen"] == "detail", at.session_state["screen"]
    assert at.session_state["file_path"] == shown.file.iat[0], at.session_state["file_path"]
    assert at.session_state["repo_path"] in (repo, other), at.session_state["repo_path"]

    # A number is just a number, here as everywhere else.
    at.session_state["screen"] = "rework"
    at.session_state["rework_tbl"] = {"selection": {"cells": []}}
    at.run()
    at.session_state["rework_tbl"] = {"selection": {"cells": [[0, "churn"]]}}
    at.run()
    assert at.session_state["screen"] == "rework", at.session_state["screen"]

    # --- an emptied settings box falls back, it does not disable ------------------
    # Clearing a box used to mean "filter nothing", which silently let bot churn and
    # lockfiles back into every number while the screen still looked right.
    at = start([repo])
    at.session_state["excludes_box"] = ""
    at.session_state["ignore_box"] = ""
    at.run()
    assert not at.exception, at.exception
    captions = [c.value for c in at.sidebar.caption]
    assert any("built-in exclude glob" in c for c in captions), captions
    assert any("built-in ignored author" in c for c in captions), captions

    # "none" is the way to actually turn one off, and it says so on screen.
    at.session_state["ignore_box"] = "none"
    at.run()
    assert any("No ignored author filtering" in c.value for c in at.sidebar.caption), \
        [c.value for c in at.sidebar.caption]

    # These boxes persist to the database, so put them back: a later assertion that
    # silently ran with bots re-enabled would be testing something else entirely.
    at.session_state["excludes_box"] = "\n".join(churn.EXCLUDE)
    at.session_state["ignore_box"] = "\n".join(churn.IGNORE_AUTHORS)
    at.run()
    assert not at.sidebar.caption or all(
        "built-in" not in c.value and "No " not in c.value for c in at.sidebar.caption)

    # --- the same three hops, driven by the TABLES rather than the treemaps -------
    # Every table that names a project, cluster or file navigates the same way the
    # treemap above it does; the row order is the table's, not the frame it came from.
    at = start([repo, bad, other])
    at.session_state["portfolio_tbl"] = {"selection": {"cells": [[0, "project"]]}}
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["screen"] == "clusters", at.session_state["screen"]
    picked_repo = at.session_state["repo_path"]
    assert picked_repo in (repo, other), picked_repo

    at.session_state["clusters_tbl"] = {"selection": {"cells": [[0, "cluster"]]}}
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["screen"] == "files", at.session_state["screen"]
    landed = at.session_state["cluster_id"]

    files_table = next(d.value for d in at.dataframe if "rework/line" in d.value.columns)
    at.session_state["files_tbl_%d" % landed] = {"selection": {"cells": [[1, "file"]]}}
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["screen"] == "detail", at.session_state["screen"]
    assert at.session_state["file_path"] == files_table.file.iat[1], (
        at.session_state["file_path"], files_table.file.iat[1])

    # A selection persists, so the destination's own stale row must not fire and bounce
    # us straight back out of the page we just landed on.
    at.run()
    assert at.session_state["screen"] == "detail", at.session_state["screen"]

    # Going back must not re-fire the row that is still selected there...
    at.session_state["screen"] = "files"
    at.run()
    assert at.session_state["screen"] == "files", at.session_state["screen"]
    # ...but the same row must still be openable a second time. Clicking a selected row
    # deselects it first, and that empty step is what makes the next click count.
    at.session_state["files_tbl_%d" % landed] = {"selection": {"cells": []}}
    at.run()
    assert at.session_state["screen"] == "files", at.session_state["screen"]
    at.session_state["files_tbl_%d" % landed] = {"selection": {"cells": [[1, "file"]]}}
    at.run()
    assert at.session_state["screen"] == "detail", at.session_state["screen"]

    # Clicking a number is just selecting a number. Only the name column navigates.
    at.session_state["screen"] = "files"
    at.run()
    at.session_state["files_tbl_%d" % landed] = {"selection": {"cells": [[3, "churn"]]}}
    at.run()
    assert at.session_state["screen"] == "files", at.session_state["screen"]

    # --- drill down from the portfolio -------------------------------------------
    at = start([repo, bad, other])
    at.session_state["portfolio_tm"] = {
        "selection": {"points": [{"label": store.load_repos(conn, [repo]).name.iat[0]}]}
    }
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["screen"] == "clusters", at.session_state["screen"]
    assert at.session_state["repo_path"] == repo, at.session_state["repo_path"]

    at.session_state["clusters_tm"] = {"selection": {"points": [{"label": top.label}]}}
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["screen"] == "files", at.session_state["screen"]

    at.session_state["files_tm"] = {"selection": {"points": [{"label": top_file}]}}
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["screen"] == "detail"
    assert len(at.metric) == 6
    # Regions, region pairs, partners, commits. Find each by its columns rather than by
    # position, so adding another table above them does not silently retarget this test.
    assert len(at.dataframe) >= 3, len(at.dataframe)
    partners = next(d for d in at.dataframe if "commits together" in d.value.columns)
    regions = next(d for d in at.dataframe if "signature" in d.value.columns)
    assert not regions.value.empty and regions.value.share.iat[0] > 0, regions.value
    assert regions.value.owner_share.between(0, 1).all(), regions.value.owner_share

    # Clicking a co-change partner navigates there and STAYS: a dataframe selection
    # persists, so a naive handler bounces back and forth forever.
    partner = partners.value.iloc[0]["file"]
    at.session_state[f"partners_{top_file}"] = {"selection": {"cells": [[0, "file"]]}}
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["file_path"] == partner, at.session_state["file_path"]
    at.run()
    assert at.session_state["file_path"] == partner, "bounced back to the previous file"

    # Stale selection must fall back, not raise.
    at.session_state["file_path"] = "does/not/exist.py"
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["screen"] == "clusters", at.session_state["screen"]

    # Breadcrumbs.
    at.session_state["screen"] = "detail"
    at.session_state["file_path"] = top_file
    at.run()
    button(at, "projects").click().run()
    assert not at.exception, at.exception
    assert at.session_state["screen"] == "portfolio"

    # --- per-line sizing drops files missing from HEAD ----------------------------
    at.session_state["repo_path"] = repo
    at.session_state["cluster_id"] = int(top.cluster_id)
    at.session_state["screen"] = "files"
    at.radio[0].set_value("churn / line").run()
    assert not at.exception, at.exception
    assert at.session_state["screen"] == "files"
    assert len(at.get("plotly_chart")) == 1
    conn.close()
    print("ok")


if __name__ == "__main__":
    args = sys.argv[1:]
    default = str(Path(__file__).resolve().parents[1])
    repo = args[0] if args else default
    other = args[1] if len(args) > 1 else str(Path(__file__).resolve().parent)
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["CHURN_DB"] = str(Path(tmp) / "test.db")
        main(repo, other)
