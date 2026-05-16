import os, sqlite3, logging

DB_PATH = "data.db"

log = logging.getLogger("db")

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id   TEXT PRIMARY KEY,
            filename     TEXT NOT NULL,
            content_type TEXT NOT NULL,
            raw          BLOB NOT NULL,
            uploaded_at  TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id         TEXT NOT NULL,
            orientation        TEXT NOT NULL,
            score_orientation  TEXT NOT NULL,
            tp INTEGER NOT NULL, tn INTEGER NOT NULL,
            fp INTEGER NOT NULL, fn INTEGER NOT NULL,
            rows_total   INTEGER NOT NULL,
            rows_valid   INTEGER NOT NULL,
            rows_invalid INTEGER NOT NULL,
            accuracy    REAL NOT NULL,
            sensitivity REAL NOT NULL,
            specificity REAL NOT NULL,
            precision   REAL NOT NULL,
            roc_auc     REAL NOT NULL,
            computed_at TEXT NOT NULL,
            FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_metrics_dataset_time ON metrics(dataset_id, computed_at DESC)")
    conn.commit()
    conn.close()
    log.info("database initialized")
