# Portfolio layer — design

**Date:** 2026-08-22
**Status:** approved

## Problem

The app analyses one repo at a time. With ~60 local projects there is no way to ask
"which project is churning most?" without running each by hand and remembering the
numbers. Results also die with the Streamlit process, because `st.cache_data` is
in-memory only.

## Goal

A portfolio screen above the existing three, ranking projects by churn, with results
persisted to disk so they survive restarts.

## Non-goals

- **History.** Each analysis replaces a repo's row. No run history, no trends, no run
  picker. Explicitly deferred — revisit later if wanted.
- **Persisted drill-down.** Clusters, files, commits and edges are not stored. Drilling
  into a repo recomputes live and is cached for the session.
- **Remote repos.** Local paths only, as today.

## Storage

SQLite via stdlib `sqlite3`, in a new `store.py`. Database at `churn.db` in the project
directory, gitignored.

```sql
CREATE TABLE IF NOT EXISTS repos (
  path            TEXT PRIMARY KEY,
  name            TEXT,      -- last two path segments, e.g. "jovis.ai/agent"
  analyzed_at     TEXT,      -- ISO 8601
  since_months    INTEGER,
  max_files       INTEGER,
  total_churn     INTEGER,
  total_commits   INTEGER,
  num_files       INTEGER,
  num_clusters    INTEGER,   -- excludes the unclustered bucket
  cross_module    INTEGER,   -- count of clusters spanning >1 top-level module
  lines_now       INTEGER,   -- sum of per-file HEAD line counts
  density         REAL,      -- live-file churn / lines_now (see Summarising)
  top_module      TEXT,      -- most common dominant_module across clusters
  error           TEXT       -- git's stderr; NULL on success
);

CREATE TABLE IF NOT EXISTS settings (
  key   TEXT PRIMARY KEY,
  value TEXT
);
```

`repos` is upserted by `path`: re-analysing a repo replaces its row. Rows for paths
removed from the list stay on disk but are not displayed — removing a path from the
list must not destroy data.

`settings` holds the repo path list, `since_months`, `max_files` and the exclude globs,
so a 60-line path list is pasted once and survives restarts.

### `store.py` interface

- `connect(db_path)` — open, apply schema, return connection. Idempotent.
- `save_repo(conn, row: dict)` — upsert one repo summary.
- `load_repos(conn, paths: list[str]) -> DataFrame` — rows for these paths, in list order.
- `get_setting(conn, key, default)` / `set_setting(conn, key, value)`.

Depends only on `sqlite3`, `json` and `pandas`. No Streamlit import, so it is testable
standalone — same rule as `churn.py`.

## Summarising a repo

New in `churn.py`:

`summarise(clusters, files, commits) -> dict` reduces one repo's analysis to the `repos`
columns. Pure, takes the existing dataframes, no git access:

- `total_churn` = `files.churn.sum()` — every file, including ones deleted since
- `total_commits` = `commits.sha.nunique()`. It must come from `commits`, not from
  summing `files.commits`: a commit touching three files appears in three file rows, so
  summing would triple-count it.
- `num_files` = `len(files)`
- `num_clusters` = clusters excluding `cluster_id == -1`
- `cross_module` = those with `num_modules > 1`
- `lines_now` = `files.lines.sum()`, NaN-safe — deleted files contribute nothing
- `density` = churn **of live files only** divided by `lines_now`, so numerator and
  denominator cover the same files. Using `total_churn` here would inflate it, because
  that includes churn from files with no lines left at HEAD. `None` when `lines_now` is 0.
- `top_module` = the `dominant_module` with the highest summed `total_churn` across
  clusters — churn-weighted, not a plain count, so one huge cluster outranks several tiny
  ones.

`churn.analyze()` is unchanged. The portfolio calls it per repo and reduces the result.

## Screens

```
portfolio  ->  clusters  ->  files  ->  detail
  (new)          (existing, unchanged)
```

Breadcrumb gains a level: `projects / jovis.ai/agent / #12 app (158 files) / app/services/chat.py`

`st.session_state` gains `repo_path`; `screen` gains `"portfolio"` and becomes the
default. The existing screens read the selected repo from session state instead of the
sidebar text input.

### Portfolio screen

- Treemap sized by `total_churn`, coloured by `density`, one box per repo. Clicking a
  box selects that repo and moves to its cluster screen.
- Table below: name, churn, density, files, commits, clusters, cross-module,
  analyzed_at, error. Sortable.
- Same churn / churn-per-line toggle as the file screen, for the same reason: raw churn
  ranks the biggest repo first, density finds the one being rewritten.
- Repos never analysed show as empty rows, so the list doubles as a to-do.

### Sidebar

- `Repo paths` — textarea, one absolute path per line, loaded from and saved to settings.
- `Months of history` (1-24) and `Max files per commit` — as today, persisted.
- `Exclude globs` — as today, persisted.
- `Analyze all` — iterates the list.

## Analyze flow

`Analyze all` walks the path list showing `st.progress` with `12/60 jovis.ai/agent`.
Per repo: `churn.analyze()` → `summarise()` → `store.save_repo()`.

**A failing repo must not abort the run.** `subprocess.CalledProcessError` (bad path,
not a repo) and an empty result (no commits in the window) are caught per repo, recorded
in `error`, and the loop continues. One bad path in 60 cannot cost the other 59. Failed
repos appear in the table with the git message rather than silently vanishing.

Each table row also gets an `Analyze` button to refresh one repo without redoing all 60.

## Performance

Sequential, roughly 2-5 minutes for 60 repos over a 12-month window. Ship it simple.
If that proves too slow, the fix is a thread pool over repos — `git` is subprocess-bound
so it parallelises well — but that is a follow-up, not day one. Do not pre-build it.

## Error handling

| Case | Behaviour |
|---|---|
| Path is not a git repo | `error` = git stderr, row saved, run continues |
| No commits in window | `error` = "no commits in window", row saved, run continues |
| Path missing from disk | Same as not-a-git-repo |
| `git grep` finds no files | `lines_now` = 0, `density` = NULL |
| Empty path list | Portfolio shows a hint to paste paths |
| DB file unwritable | Surface the OSError; do not silently run without persisting |

## Testing

`store.py` gets a `demo()` self-check run by `python store.py`, against a temp DB:

- schema applies twice without error (idempotent `connect`)
- `save_repo` inserts, then replaces on the same path rather than duplicating
- `load_repos` returns rows in the requested order and skips unknown paths
- settings round-trip, including a value containing newlines (the path list)

`churn.py`'s `demo()` gains a `summarise()` case on its existing fixture, asserting:

- `total_commits` counts commits, not file-rows (the fixture has a commit touching two
  files, so a triple-counting bug would show up as 4 instead of 3)
- `lines_now` is NaN-safe
- `density is None` when `lines_now` is 0

`test_app.py` gains a portfolio case:

- two real repo paths analysed → two rows persisted, portfolio table shows both
- clicking a repo box reaches its cluster screen
- a deliberately bad path records an `error` and does not prevent the good repo's row
- state survives a fresh `AppTest` against the same DB (proves persistence, which is
  the whole point of the feature)

## Files

| File | Change |
|---|---|
| `store.py` | new, ~60 lines of SQL plus `demo()` |
| `churn.py` | add `summarise()`, extend `demo()` |
| `app.py` | add portfolio screen and analyze-all loop, ~80 lines; read repo from session state |
| `test_app.py` | add portfolio case |
| `.gitignore` | add `churn.db` |
| `README.md` | document the portfolio layer and persistence |

`app.py` goes to roughly 360 lines. Acceptable as one file, but the portfolio screen is
the natural seam if it grows again — that split is the next refactor, not this one.
