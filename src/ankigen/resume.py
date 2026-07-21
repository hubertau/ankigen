"""Resumable, durable CSV writing for interrupted LLM generate runs.

Both the vocab (``generate_csv``) and grammar (``generate_grammar_csv``)
pipelines make one expensive LLM call per row. If the connection drops
mid-run we want to (a) keep every row already written safely on disk and
(b) skip the finished rows on the next run instead of re-spending API
budget. This module centralises both concerns.
"""

import csv
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from ankigen.anki_db import normalize_anki_term


def write_anki_header(
    f: TextIO,
    fieldnames: Sequence[str],
    *,
    separator: str = "comma",
    html: bool = True,
) -> None:
    """Write the Anki import header block for a generated CSV.

    Emits the directive lines Anki reads on import, e.g.::

        #separator:comma
        #html:true
        #columns:Hanzi,Pinyin,Jyutping,English,Sentence
        #Hanzi,#Pinyin,#Jyutping,#English,#Sentence

    Every line starts with ``#`` so Anki treats them as directives/comments
    rather than data. ``#columns:`` names the fields; the trailing
    ``#<field>,...`` line is a human-readable echo of the same names.
    """
    f.write(f"#separator:{separator}\n")
    if html:
        f.write("#html:true\n")
    f.write("#columns:" + ",".join(fieldnames) + "\n")
    f.write(",".join(f"#{name}" for name in fieldnames) + "\n")


def completed_csv_keys(path: Path, key_column: str) -> set[str]:
    """Return the NFC-normalised values already written under ``key_column``.

    Returns an empty set when the file is absent, empty, or has no usable
    header — i.e. there is nothing to resume from. Used so a re-run skips
    rows that finished before an interruption.

    Handles the Anki header block written by :func:`write_anki_header`: field
    names are recovered from the ``#columns:`` directive and all other
    ``#``-prefixed lines are dropped before the data rows are parsed. Files
    without a directive (legacy plain-header CSVs) fall back to treating the
    first line as the header.
    """
    if not path.exists() or path.stat().st_size == 0:
        return set()

    columns: list[str] | None = None
    data_lines: list[str] = []
    with open(path, encoding="utf-8", newline="") as f:
        for line in f:
            if line.startswith("#columns:"):
                spec = line[len("#columns:") :].strip()
                columns = next(csv.reader([spec]))
            elif line.startswith("#"):
                continue
            else:
                data_lines.append(line)

    if columns is not None:
        reader = csv.DictReader(data_lines, fieldnames=columns)
    else:
        # Legacy plain-header file: first data line is the header row.
        reader = csv.DictReader(data_lines)

    if reader.fieldnames is None or key_column not in reader.fieldnames:
        return set()

    done: set[str] = set()
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
