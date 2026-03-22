#!/usr/bin/env python3

import argparse
import hashlib
import json
import logging
import os
import shlex
import signal
import sqlite3
import threading
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Sequence
    from types import FrameType


COMMIT_BATCH_SIZE = 20000
PARTIAL_SIZE = 64 * 1024  # 64KB
CHUNK_SIZE = 1024 * 1024  # 1MB
DB = "dedup.db"
REPORT_DB = "report.db"


parser = argparse.ArgumentParser(description="Deduplicate files by inode and hash.")
parser.add_argument("root", nargs="*", help="Root directories to scan")
parser.add_argument(
    "--database",
    default=DB,
    help=f"Path to SQLite database file (default: {DB})",
)
parser.add_argument(
    "--report-database",
    default=REPORT_DB,
    help=f"Path to SQLite database file for report (default: {REPORT_DB})",
)
parser.add_argument(
    "--report-only",
    action="store_true",
    help="Skip scanning and hashing, only generate report",
)

log_format = "[%(asctime)s : %(levelname)s : %(pathname)s : %(lineno)s : %(funcName)s] %(message)s"
logging.basicConfig(format=log_format, datefmt="%Y-%m-%dT%H:%M:%S%z", level=logging.INFO)
logger = logging.getLogger(__name__)

stop_requested = threading.Event()


def handle_shutdown(signum: int, frame: FrameType | None) -> None:
    s = signal.Signals(signum)
    logger.warning("Shutdown signal received: %s", s.name)
    stack = "".join(traceback.format_stack(frame))
    logger.warning("Stack frame at time of signal:\n%s", stack)
    stop_requested.set()


_sigint_handler = signal.signal(signal.SIGINT, handle_shutdown)
_sigterm_handler = signal.signal(signal.SIGTERM, handle_shutdown)


def main() -> None:
    args = parser.parse_args()
    db = connect(args.database)
    init_db(db)
    if args.root:
        items = [item for root_dir in args.root if (item := Path(root_dir).absolute()).is_dir()]
        print("Scanning filesystem...")  # noqa: T201
        if not scan_files(db, items):
            return
    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigterm_handler)
    if not args.report_only:
        print("Computing partial hashes...")  # noqa: T201
        compute_partial_hashes(db)
        print("Computing full hashes...")  # noqa: T201
        compute_full_hashes(db)
    print("Duplicate report:")  # noqa: T201
    report_duplicates(db, args.report_database)


def connect(database: str) -> sqlite3.Connection:
    db = sqlite3.connect(database=database)
    db.execute("PRAGMA cache_size=-200000")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA mmap_size=30000000000")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA temp_store=MEMORY")
    return db


def init_db(db: sqlite3.Connection) -> None:
    db.execute(CREATE_TABLE_INODE_QUERY)
    db.execute(CREATE_TABLE_PATH_QUERY)
    db.execute(CREATE_INDEX_INODE_SIZE_QUERY)
    db.execute(CREATE_INDEX_INODE_PARTIAL_HASH_QUERY)
    db.execute(CREATE_INDEX_INODE_FULL_HASH_QUERY)
    db.execute(CREATE_INDEX_PATH_DEVICE_INODE_QUERY)
    db.commit()


def scan_files(db: sqlite3.Connection, root_dirs: Sequence[Path]) -> bool:
    cur = db.cursor()
    for i, p in enumerate(tqdm(scan_tree(root_dirs), ncols=80)):
        if stop_requested.is_set():
            return False
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except Exception:
            logger.exception("Failed to stat file: %s", p)
            continue
        cur.execute(
            INSERT_INODE_QUERY,
            (st.st_dev, st.st_ino, st.st_size, int(st.st_mtime)),
        )
        cur.execute(INSERT_PATH_QUERY, (str(p), st.st_dev, st.st_ino))
        if i % COMMIT_BATCH_SIZE == 0:
            db.commit()
    db.commit()
    return True


def compute_partial_hashes(db: sqlite3.Connection) -> None:
    cur = db.cursor()
    cur.execute(COMPUTE_PARTIAL_HASH_QUERY)
    for i, (p, device, inode) in enumerate(tqdm(cur.fetchall(), ncols=80)):
        ph = partial_hash(Path(p))
        cur.execute(UPDATE_INODE_PARTIAL_HASH_QUERY, (ph, device, inode))
        if i % COMMIT_BATCH_SIZE == 0:
            db.commit()
    db.commit()


def compute_full_hashes(db: sqlite3.Connection) -> None:
    cur = db.cursor()
    cur.execute(COMPUTE_FULL_HASH_QUERY)
    for i, (p, device, inode) in enumerate(tqdm(cur.fetchall(), ncols=80)):
        ph = full_hash(Path(p))
        cur.execute(UPDATE_INODE_FULL_HASH_QUERY, (ph, device, inode))
        if i % COMMIT_BATCH_SIZE == 0:
            db.commit()
    db.commit()


def report_duplicates(db: sqlite3.Connection, report_db: str) -> None:
    cur = db.cursor()
    cur.execute(ATTACH_REPORT_DATABASE_QUERY, (report_db,))
    cur.execute(DROP_REPORT_TABLE_INODE_QUERY)
    cur.execute(DROP_REPORT_TABLE_PATH_QUERY)
    cur.execute(DROP_REPORT_TABLE_DUPLICATE_QUERY)
    cur.execute(CREATE_REPORT_TABLE_INODE_QUERY)
    cur.execute(CREATE_REPORT_TABLE_PATH_QUERY)
    cur.execute(CREATE_REPORT_TABLE_DUPLICATE_QUERY)
    cur.execute(REPORT_QUERY)
    items = cur.fetchall()
    total = len(items)
    with Path("report.sh").open("w", encoding="utf-8") as f:
        for i, (full_hash, size, path_json) in enumerate(items, start=1):
            items: list[str] = json.loads(path_json)
            step_msg = f"Files with hash={full_hash} size={size} found={len(items)} progress={i}/{total}"
            step_cmd = ["echo", step_msg]
            jdupes_cmd = ["jdupes", "--one-file-system", "--linkhard", "--", *items]
            print(shlex.join(step_cmd), file=f)
            print(shlex.join(jdupes_cmd), file=f)


def scan_tree(root_dirs: Sequence[Path]) -> Generator[Path]:
    stack = [*root_dirs]
    while stack:
        path = stack.pop()
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    yield Path(path) / entry


def partial_hash(path: Path) -> str | None:
    h = hashlib.sha256()
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            h.update(f.read(PARTIAL_SIZE))
            if size > PARTIAL_SIZE * 2:
                f.seek(size // 2)
                h.update(f.read(PARTIAL_SIZE))
                f.seek(-PARTIAL_SIZE, 2)
                h.update(f.read(PARTIAL_SIZE))
    except Exception:
        logger.exception("Failed to read file: %s", path)
        return None
    return h.hexdigest()


def full_hash(path: Path) -> str | None:
    h = hashlib.sha256()
    try:
        with path.open("rb", buffering=CHUNK_SIZE) as f:
            for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                h.update(chunk)
    except Exception:
        logger.exception("Failed to read file: %s", path)
        return None
    return h.hexdigest()


CREATE_TABLE_INODE_QUERY = """
CREATE TABLE IF NOT EXISTS inodes (
    device INTEGER,
    inode INTEGER,
    size INTEGER,
    mtime INTEGER,
    partial_hash TEXT,
    full_hash TEXT,
    PRIMARY KEY(device, inode)
)
"""

CREATE_TABLE_PATH_QUERY = """
CREATE TABLE IF NOT EXISTS paths (
    path TEXT PRIMARY KEY,
    device INTEGER,
    inode INTEGER,
    FOREIGN KEY(device, inode) REFERENCES inodes(device, inode)
)"""

CREATE_INDEX_INODE_SIZE_QUERY = "CREATE INDEX IF NOT EXISTS idx_inode_size ON inodes(size)"

CREATE_INDEX_INODE_PARTIAL_HASH_QUERY = "CREATE INDEX IF NOT EXISTS idx_inode_partial ON inodes(partial_hash)"

CREATE_INDEX_INODE_FULL_HASH_QUERY = "CREATE INDEX IF NOT EXISTS idx_inode_full ON inodes(full_hash)"

CREATE_INDEX_PATH_DEVICE_INODE_QUERY = "CREATE INDEX IF NOT EXISTS idx_paths_inode ON paths(device, inode)"

INSERT_INODE_QUERY = """
INSERT INTO inodes(device, inode, size, mtime)
VALUES (?, ?, ?, ?)
ON CONFLICT(device, inode) DO UPDATE
SET size = excluded.size, mtime = excluded.mtime
WHERE size != excluded.size OR mtime != excluded.mtime
"""

INSERT_PATH_QUERY = """
INSERT INTO paths(path, device, inode)
VALUES (?, ?, ?)
ON CONFLICT(path) DO UPDATE
SET device = excluded.device, inode = excluded.inode
WHERE device != excluded.device OR inode != excluded.inode
"""

COMPUTE_PARTIAL_HASH_QUERY = """
SELECT path, device, inode
FROM inodes
JOIN (
    SELECT size
    FROM inodes
    WHERE size > 16384
    GROUP BY size
    HAVING COUNT(*) > 1
) AS t USING (size)
JOIN (
    SELECT path, device, inode
    FROM (
        SELECT path, device, inode, ROW_NUMBER() OVER (PARTITION by device, inode ORDER BY PATH) AS p
        FROM paths
    ) AS t1
    WHERE p = 1
) AS t2 USING (device, inode)
WHERE partial_hash IS NULL
"""

UPDATE_INODE_PARTIAL_HASH_QUERY = "UPDATE inodes SET partial_hash=? WHERE device=? AND inode=?"

COMPUTE_FULL_HASH_QUERY = """
SELECT path, device, inode
FROM inodes
JOIN (
    SELECT partial_hash
    FROM inodes
    WHERE partial_hash IS NOT NULL
    GROUP BY partial_hash
    HAVING COUNT(*) > 1
) AS t USING (partial_hash)
JOIN (
    SELECT path, device, inode
    FROM (
        SELECT path, device, inode, ROW_NUMBER() OVER (PARTITION by device, inode ORDER BY PATH) AS p
        FROM paths
    ) AS t1
    WHERE p = 1
) AS t2 USING (device, inode)
WHERE full_hash IS NULL AND size > 16384
"""

UPDATE_INODE_FULL_HASH_QUERY = "UPDATE inodes SET full_hash=? WHERE device=? AND inode=?"

ATTACH_REPORT_DATABASE_QUERY = "ATTACH DATABASE ? AS report"

DROP_REPORT_TABLE_INODE_QUERY = "DROP TABLE IF EXISTS report.inodes"

DROP_REPORT_TABLE_PATH_QUERY = "DROP TABLE IF EXISTS report.paths"

DROP_REPORT_TABLE_DUPLICATE_QUERY = "DROP TABLE IF EXISTS report.duplicates"

CREATE_REPORT_TABLE_INODE_QUERY = """
CREATE TABLE report.inodes AS
SELECT device, inode, size, full_hash
FROM inodes
JOIN (
    SELECT full_hash
    FROM inodes
    WHERE full_hash IS NOT NULL
    GROUP BY full_hash
    HAVING COUNT(*) > 1
) AS t USING (full_hash)
"""

CREATE_REPORT_TABLE_PATH_QUERY = """
CREATE TABLE report.paths AS
SELECT path, device, inode
FROM (
    SELECT path, device, inode, ROW_NUMBER() OVER (PARTITION by device, inode ORDER BY PATH) AS p
    FROM paths
) AS t
WHERE p = 1
"""

CREATE_REPORT_TABLE_DUPLICATE_QUERY = """
CREATE TABLE report.duplicates AS
SELECT device, inode, size, full_hash, path
FROM report.inodes
JOIN report.paths USING (device, inode)
ORDER BY full_hash
"""

REPORT_QUERY = """
SELECT full_hash, size, json_group_array(path ORDER BY path) AS path_json
FROM report.duplicates
GROUP BY full_hash, size
HAVING COUNT(*) > 1
ORDER BY size DESC
"""


if __name__ == "__main__":
    main()
