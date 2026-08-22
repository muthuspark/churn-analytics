"""UI wiring check: run with `python test_app.py <repo-path>` (defaults to this repo's parent).

Covers the navigation logic, which is where the non-obvious bugs live: treemap
selections persist in session_state across a re-analysis, so a label from a
previous repo or date window can still be sitting there on the next run.
"""

import sys
import tempfile
from pathlib import Path

from streamlit.testing.v1 import AppTest

import churn

APP = str(Path(__file__).with_name("app.py"))
SINCE = "12 months ago"


def main(repo):
    clusters, files, _, _ = churn.analyze(repo, SINCE, 30)
    assert not clusters.empty, f"no commits in {repo} since {SINCE}"
    top = clusters[clusters.cluster_id != -1].iloc[0]
    top_file = (
        files[files.cluster_id == top.cluster_id]
        .sort_values("churn", ascending=False)
        .path.iat[0]
    )

    def button(label):
        """Address buttons by label. Positional indexing silently retargets when the
        sidebar is reordered, and "Analyze" and the "repo" crumb both land on the
        cluster screen, so a mix-up between them passes for the wrong reason."""
        return [b for b in at.button if b.label == label][0]

    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception, at.exception

    # A path that is definitely not a repo must surface git's message, not raise.
    # (Don't lean on the default "." — this project is itself a git repo.)
    with tempfile.TemporaryDirectory() as not_a_repo:
        at.text_input[0].set_value(not_a_repo).run()
        assert not at.exception, at.exception
        assert at.error, "expected a git error for a non-repo path"

    at.text_input[0].set_value(repo)
    button("Analyze").click().run()
    assert not at.exception, at.exception
    assert len(at.get("plotly_chart")) == 1

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

    at.session_state["file_path"] = top_file
    at.run()

    # Stale selection must fall back to the cluster view, not raise.
    at.session_state["file_path"] = "does/not/exist.py"
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["screen"] == "clusters", at.session_state["screen"]

    at.session_state["screen"] = "detail"
    at.session_state["file_path"] = top_file
    at.run()
    button("repo").click().run()
    assert not at.exception, at.exception
    assert at.session_state["screen"] == "clusters"

    # Re-analyzing must drop the drill-down: Louvain renumbers clusters, so a
    # surviving cluster_id would silently label a different cluster.
    at.session_state["clusters_tm"] = {"selection": {"points": [{"label": top.label}]}}
    at.run()
    assert at.session_state["screen"] == "files"
    at.text_input[0].set_value(repo)
    button("Analyze").click().run()
    assert not at.exception, at.exception
    assert at.session_state["cluster_id"] is None, at.session_state["cluster_id"]
    assert at.session_state["screen"] == "clusters"

    # Per-line sizing drops files missing from HEAD; the treemap must still render
    # and its click target set must follow the filtered frame, not the full cluster.
    at.session_state["cluster_id"] = int(top.cluster_id)
    at.session_state["screen"] = "files"
    at.radio[0].set_value("churn / line").run()
    assert not at.exception, at.exception
    assert at.session_state["screen"] == "files"
    assert len(at.get("plotly_chart")) == 1
    print("ok")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1]))
