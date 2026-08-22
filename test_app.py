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
    assert len(at.get("plotly_chart")) == 1, "portfolio treemap missing"
    assert len(at.dataframe) == 1, "portfolio table missing"

    # --- persistence: a brand new app instance sees the stored rows ---------------
    at2 = start([repo, bad, other])
    assert not at2.exception, at2.exception
    assert len(at2.get("plotly_chart")) == 1, "stored rows should render without re-analysing"
    table = at2.dataframe[0].value
    assert (table["churn"].notna()).sum() >= 2, table[["project", "churn"]]

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
    # Partner table drives navigation, so it must render with the commit list.
    assert len(at.dataframe) == 2, len(at.dataframe)

    # Clicking a co-change partner navigates there and STAYS: a dataframe selection
    # persists, so a naive handler bounces back and forth forever.
    partner = at.dataframe[0].value.iloc[0]["file"]
    at.session_state[f"partners_{top_file}"] = {"selection": {"rows": [0]}}
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
