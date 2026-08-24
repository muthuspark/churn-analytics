"""Disk persistence for the portfolio view.

One summary row per repo, replaced when that repo is re-analysed. No history — see
docs/superpowers/specs/2026-08-22-portfolio-layer-design.md. No streamlit import, so
`python store.py` runs the self-check.
"""

import os
import sqlite3

import pandas as pd

DB_NAME = "churn.db"


def db_path():
    """CHURN_DB lets the tests point at a temp file instead of the real database."""
    return os.environ.get("CHURN_DB") or DB_NAME

COLUMNS = (
    "path", "name", "analyzed_at", "since_months", "max_files",
    "total_churn", "total_commits", "num_files", "num_clusters",
    "cross_module", "lines_now", "density", "rework", "rework_density",
    "dev_days", "devs", "days_build", "days_config", "days_tests", "days_docs",
    "days_product", "top_module", "error",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS repos (
  path            TEXT PRIMARY KEY,
  name            TEXT,
  analyzed_at     TEXT,
  since_months    INTEGER,
  max_files       INTEGER,
  total_churn     INTEGER,
  total_commits   INTEGER,
  num_files       INTEGER,
  num_clusters    INTEGER,
  cross_module    INTEGER,
  lines_now       INTEGER,
  density         REAL,
  rework          INTEGER,
  rework_density  REAL,
  dev_days        INTEGER,
  devs            INTEGER,
  days_build      REAL,
  days_config     REAL,
  days_tests      REAL,
  days_docs       REAL,
  days_product    REAL,
  top_module      TEXT,
  error           TEXT
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
"""

# column -> declared type, read straight off SCHEMA so adding a column there is the
# only edit needed. connect() backfills whatever an older database is missing.
COLUMN_TYPES = {
    parts[0]: parts[1]
    for parts in (
        line.strip().rstrip(",").split()
        for line in SCHEMA.splitlines()
        if line.startswith("  ")
    )
    if parts[0] in COLUMNS
}


def connect(path=None, same_thread=True):
    """Open the database and apply the schema. Safe to call repeatedly.

    Streamlit runs each rerun on a script thread that need not be the one that opened
    the connection, so the app passes same_thread=False.
    """
    conn = sqlite3.connect(path or db_path(), check_same_thread=same_thread)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    # CREATE TABLE IF NOT EXISTS leaves an older table alone, so a database written
    # before a column existed would still be missing it -- and save_repo would fail on
    # every write. Add what is missing instead of asking anyone to delete their data.
    have = {r["name"] for r in conn.execute("PRAGMA table_info(repos)")}
    for column, kind in COLUMN_TYPES.items():
        if column not in have:
            conn.execute(f"ALTER TABLE repos ADD COLUMN {column} {kind}")
    conn.commit()
    return conn


def save_repo(conn, row):
    """Upsert one repo summary, keyed on path."""
    values = [row.get(c) for c in COLUMNS]
    placeholders = ",".join("?" * len(COLUMNS))
    conn.execute(
        f"INSERT OR REPLACE INTO repos ({','.join(COLUMNS)}) VALUES ({placeholders})",
        values,
    )
    conn.commit()


def load_repos(conn, paths):
    """Rows for these paths, in the order given.

    Paths never analysed come back as empty rows rather than being dropped, so the
    portfolio table doubles as a to-do list. Rows for paths no longer in the list stay
    on disk untouched — editing the list must not destroy data.
    """
    if not paths:
        return pd.DataFrame(columns=COLUMNS)
    marks = ",".join("?" * len(paths))
    stored = {
        r["path"]: dict(r)
        for r in conn.execute(f"SELECT * FROM repos WHERE path IN ({marks})", paths)
    }
    rows = [stored.get(p, {"path": p}) for p in paths]
    df = pd.DataFrame(rows, columns=COLUMNS)
    # Never-analysed rows have no stored name, and a table of "None" is unreadable.
    df["name"] = df.name.fillna(df.path.map(lambda p: "/".join(p.split("/")[-2:])))
    return df


def get_setting(conn, key, default=None):
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return default if row is None else row["value"]


def set_setting(conn, key, value):
    conn.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value))
    )
    conn.commit()


def demo():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "t.db")
        conn = connect(db)
        connect(db).close()  # schema must be idempotent

        row = dict(
            path="/repos/a", name="org/a", analyzed_at="2026-08-22T10:00:00",
            since_months=12, max_files=30, total_churn=100, total_commits=9,
            num_files=5, num_clusters=2, cross_module=1, lines_now=50,
            density=2.0, top_module="app", error=None,
        )
        save_repo(conn, row)
        save_repo(conn, {**row, "total_churn": 250})  # same path -> replace, not append
        got = load_repos(conn, ["/repos/a"])
        assert len(got) == 1, got
        assert got.total_churn.iat[0] == 250, got.total_churn.iat[0]

        # Unknown paths come back as empty rows, in the order asked for.
        got = load_repos(conn, ["/repos/zz", "/repos/a"])
        assert list(got.path) == ["/repos/zz", "/repos/a"], list(got.path)
        assert pd.isna(got.total_churn.iat[0])
        assert got.total_churn.iat[1] == 250

        assert load_repos(conn, []).empty

        # The path list is a setting, so newlines have to survive the round trip.
        paths = "/repos/a\n/repos/b\n/repos/c"
        set_setting(conn, "paths", paths)
        assert get_setting(conn, "paths") == paths
        set_setting(conn, "paths", "/repos/a")
        assert get_setting(conn, "paths") == "/repos/a"
        assert get_setting(conn, "missing", "fallback") == "fallback"

        # An error row is still a row: a failed repo must stay visible.
        save_repo(conn, {"path": "/repos/bad", "name": "bad", "error": "not a git repo"})
        got = load_repos(conn, ["/repos/bad"])
        assert got.error.iat[0] == "not a git repo"
        conn.close()

        # Reopening the same file must see the saved rows — the whole point.
        conn = connect(db)
        assert load_repos(conn, ["/repos/a"]).total_churn.iat[0] == 250
        conn.close()

        # A database written before a column existed must gain it, not blow up on the
        # next save. Drop one and reopen.
        old = str(Path(tmp) / "old.db")
        legacy = sqlite3.connect(old)
        legacy.executescript(
            "CREATE TABLE repos (path TEXT PRIMARY KEY, name TEXT, total_churn INTEGER);"
            "CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT);"
            "INSERT INTO repos VALUES ('/repos/old', 'org/old', 7);"
        )
        legacy.commit()
        legacy.close()
        conn = connect(old)
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(repos)")}
        assert set(COLUMNS) <= columns, set(COLUMNS) - columns
        assert load_repos(conn, ["/repos/old"]).total_churn.iat[0] == 7
        save_repo(conn, {**row, "path": "/repos/old", "rework": 12})
        assert load_repos(conn, ["/repos/old"]).rework.iat[0] == 12
        conn.close()
    print("ok")


if __name__ == "__main__":
    demo()
