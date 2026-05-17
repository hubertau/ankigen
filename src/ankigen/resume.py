"""Resumable, durable CSV writing for interrupted LLM generate runs.

Both the vocab (``generate_csv``) and grammar (``generate_grammar_csv``)
pipelines make one expensive LLM call per row. If the connection drops
mid-run we want to (a) keep every row already written safely on disk and
(b) skip the finished rows on the next run instead of re-spending API
budget. This module centralises both concerns.
"""

import csv
import os
from pathlib import Path
from typing import TextIO

from ankigen.anki_db import normalize_anki_term


def completed_csv_keys(path: Path, key_column: str) -> set[str]:
    """Return the NFC-normalised values already written under ``key_column``.

    Returns an empty set when the file is absent, empty, or has no usable
    header — i.e. there is nothing to resume from. Used so a re-run skips
    rows that finished before an interruption.
    """
    if not path.exists() or path.stat().st_size == 0:
        return set()
    done: set[str] = set()
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or key_column not in reader.fieldnames:
            return set()
        for row in reader:
            value = (row.get(key_column) or "").strip()
            if value:
                done.add(normalize_anki_term(value))
    return done


def durable_write(f: TextIO) -> None:
    """Flush Python and OS buffers so rows written so far survive a hard kill.

    A clean exception already flushes on file close, but a SIGKILL / power
    loss / laptop-sleep-death does not. fsync per row costs ~ms, negligible
    against the multi-second LLM call that produced the row.
    """
    f.flush()
    os.fsync(f.fileno())
