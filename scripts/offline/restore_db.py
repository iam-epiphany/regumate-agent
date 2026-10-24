"""Restore an offline seed into a new SQLite volume without replacing user data."""
from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path
import sqlite3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/seed/app.db'))
    parser.add_argument('--destination', type=Path, default=Path('/app/db_runtime/app.db'))
    args = parser.parse_args()
    target = args.destination
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size:
        with closing(sqlite3.connect(target)) as db:
            if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise RuntimeError('Existing database failed quick_check; refusing to overwrite it.')
        print('Existing database preserved.', flush=True)
        return
    if not args.source.is_file() or not args.source.stat().st_size:
        raise RuntimeError('Offline SQLite seed is missing or empty.')
    pending = target.with_suffix('.restore-tmp')
    # The delivered seed is a completed immutable backup. SQLite otherwise
    # tries to create WAL/SHM sidecars even for a read-only source mount.
    with closing(sqlite3.connect(args.source.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)) as source:
        with closing(sqlite3.connect(pending)) as destination:
            source.backup(destination)
            if destination.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise RuntimeError('Restored database failed quick_check.')
    pending.replace(target)
    print('Offline SQLite seed restored.', flush=True)


if __name__ == '__main__':
    main()
