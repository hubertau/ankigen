"""Staging checkpoints for folder/watch extract runs.

Persists extracted text, per-chunk LLM results, and a manifest so interrupted
runs can resume without re-parsing documents or re-calling finished chunks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ankigen.models import GrammarItem
from ankigen.resume import durable_write

ExtractMode = Literal["vocab", "grammar", "all"]

_DEFAULT_OUTPUT_DIR = "./inputs"

logger = logging.getLogger("ankigen.extract_checkpoint")

FileStatus = Literal["pending", "text_done", "vocab_done", "grammar_done", "failed"]

_DEFAULT_STAGING_SUBDIR = ".staging"


class FileCheckpoint(BaseModel):
    """Per-source-file state within an extract run."""

    path: str
    name: str
    status: FileStatus = "pending"
    source_mtime: float | None = None
    file_key: str = ""
    text_path: str | None = None
    vocab_chunks: int = 0
    grammar_chunks: int = 0
    last_error: str | None = None


class ExtractManifest(BaseModel):
    """Top-level manifest for one extract run."""

    lang: str
    mode: str
    source_dir: str
    date: str
    run_id: str
    started_at: str
    complete: bool = False
    files: list[FileCheckpoint] = Field(default_factory=list)


def _get_output_dir() -> Path:
    return Path(os.getenv("ANKIGEN_OUTPUT_DIR", _DEFAULT_OUTPUT_DIR))


def get_staging_dir() -> Path:
    """Root directory for extract staging data."""
    raw = os.getenv("ANKIGEN_STAGING_DIR")
    if raw:
        return Path(raw)
    return _get_output_dir() / _DEFAULT_STAGING_SUBDIR


def file_key_for(path: Path) -> str:
    """Stable short id for staging filenames (unique per resolved path)."""
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]


def compute_run_id(source_dir: Path, lang: str, mode: str, date: str) -> str:
    """Deterministic run folder name for a source dir + lang + mode + date."""
    key = f"{source_dir.resolve()}|{lang}|{mode}|{date}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return f"{date}_{digest}"


def _run_root(lang: str, run_id: str) -> Path:
    return get_staging_dir() / lang / run_id


def load_manifest(lang: str, run_id: str) -> ExtractManifest | None:
    """Load manifest if the run directory exists."""
    path = _run_root(lang, run_id) / "manifest.json"
    if not path.exists():
        return None
    try:
        return ExtractManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read manifest %s: %s", path, exc)
        return None


def save_manifest(manifest: ExtractManifest) -> None:
    """Write manifest.json with fsync."""
    root = _run_root(manifest.lang, manifest.run_id)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))
        f.write("\n")
        durable_write(f)


def init_manifest(
    *,
    lang: str,
    mode: ExtractMode,
    source_dir: Path,
    date: str,
    file_paths: list[Path],
    fresh: bool = False,
) -> ExtractManifest:
    """Create or resume a manifest for ``file_paths``."""
    run_id = compute_run_id(source_dir, lang, mode, date)
    root = _run_root(lang, run_id)

    if fresh and root.exists():
        import shutil

        shutil.rmtree(root)
        logger.info("Fresh extract run: cleared staging %s", root)

    existing = None if fresh else load_manifest(lang, run_id)
    if existing is not None:
        by_path = {f.path: f for f in existing.files}
        ordered: list[FileCheckpoint] = []
        for fp in file_paths:
            resolved = str(fp.resolve())
            if resolved in by_path:
                ordered.append(by_path[resolved])
            else:
                ordered.append(_file_entry(fp))
        existing.files = ordered
        save_manifest(existing)
        return existing

    manifest = ExtractManifest(
        lang=lang,
        mode=mode,
        source_dir=str(source_dir.resolve()),
        date=date,
        run_id=run_id,
        started_at=datetime.now().isoformat(timespec="seconds"),
        files=[_file_entry(fp) for fp in file_paths],
    )
    save_manifest(manifest)
    return manifest


def _file_entry(path: Path) -> FileCheckpoint:
    resolved = path.resolve()
    mtime = resolved.stat().st_mtime if resolved.exists() else None
    return FileCheckpoint(
        path=str(resolved),
        name=path.name,
        file_key=file_key_for(resolved),
        source_mtime=mtime,
    )


def find_file_entry(manifest: ExtractManifest, path: Path) -> FileCheckpoint | None:
    key = str(path.resolve())
    for entry in manifest.files:
        if entry.path == key:
            return entry
    return None


def source_changed(entry: FileCheckpoint, path: Path) -> bool:
    """True when the source file mtime differs from what we cached."""
    if not path.exists():
        return True
    current = path.stat().st_mtime
    if entry.source_mtime is None:
        return True
    return current != entry.source_mtime


class ExtractRunCheckpoint:
    """Read/write staging artifacts for one extract run."""

    def __init__(self, manifest: ExtractManifest) -> None:
        self.manifest = manifest
        self.root = _run_root(manifest.lang, manifest.run_id)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "text").mkdir(exist_ok=True)
        (self.root / "vocab").mkdir(exist_ok=True)
        (self.root / "grammar").mkdir(exist_ok=True)

    def _text_path(self, entry: FileCheckpoint) -> Path:
        return self.root / "text" / f"{entry.file_key}.txt"

    def _vocab_chunks_path(self, entry: FileCheckpoint) -> Path:
        return self.root / "vocab" / f"{entry.file_key}.jsonl"

    def _grammar_chunks_path(self, entry: FileCheckpoint) -> Path:
        return self.root / "grammar" / f"{entry.file_key}.jsonl"

    def load_cached_text(self, entry: FileCheckpoint, path: Path) -> str | None:
        """Return cached text when valid; None if missing or stale."""
        if entry.status == "pending" or source_changed(entry, path):
            return None
        text_file = self._text_path(entry)
        if not text_file.exists():
            return None
        return text_file.read_text(encoding="utf-8")

    def save_text(self, entry: FileCheckpoint, path: Path, text: str) -> None:
        """Cache extracted text and mark ``text_done``."""
        text_file = self._text_path(entry)
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(text)
            durable_write(f)
        entry.text_path = str(text_file.relative_to(self.root))
        entry.source_mtime = path.stat().st_mtime
        if entry.status == "pending" or entry.status == "failed":
            entry.status = "text_done"
            entry.last_error = None
        self._persist_entry(entry)

    def _read_chunk_jsonl(self, path: Path) -> dict[int, dict[str, Any]]:
        """Parse chunk index → payload from a JSONL checkpoint file."""
        out: dict[int, dict[str, Any]] = {}
        if not path.exists():
            return out
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    out[int(row["chunk"])] = row
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
        return out

    def load_vocab_chunk(self, entry: FileCheckpoint, chunk_index: int) -> list[str] | None:
        rows = self._read_chunk_jsonl(self._vocab_chunks_path(entry))
        row = rows.get(chunk_index)
        if row is None:
            return None
        words = row.get("words")
        if not isinstance(words, list):
            return None
        return [str(w) for w in words]

    def save_vocab_chunk(self, entry: FileCheckpoint, chunk_index: int, words: list[str]) -> None:
        path = self._vocab_chunks_path(entry)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"chunk": chunk_index, "words": words}, ensure_ascii=False))
            f.write("\n")
            durable_write(f)
        entry.vocab_chunks = max(entry.vocab_chunks, chunk_index + 1)
        self._persist_entry(entry)

    def load_all_vocab_chunks(self, entry: FileCheckpoint) -> dict[int, list[str]]:
        rows = self._read_chunk_jsonl(self._vocab_chunks_path(entry))
        result: dict[int, list[str]] = {}
        for idx, row in sorted(rows.items()):
            words = row.get("words")
            if isinstance(words, list):
                result[idx] = [str(w) for w in words]
        return result

    def load_all_grammar_chunks(self, entry: FileCheckpoint) -> dict[int, list[GrammarItem]]:
        rows = self._read_chunk_jsonl(self._grammar_chunks_path(entry))
        result: dict[int, list[GrammarItem]] = {}
        for idx, row in sorted(rows.items()):
            items_raw = row.get("items")
            if not isinstance(items_raw, list):
                continue
            items: list[GrammarItem] = []
            for raw in items_raw:
                if isinstance(raw, dict):
                    items.append(GrammarItem.model_validate(raw))
                else:
                    items.append(GrammarItem.model_validate_json(raw))
            result[idx] = items
        return result

    def load_grammar_chunk(
        self, entry: FileCheckpoint, chunk_index: int
    ) -> list[GrammarItem] | None:
        rows = self._read_chunk_jsonl(self._grammar_chunks_path(entry))
        row = rows.get(chunk_index)
        if row is None:
            return None
        items_raw = row.get("items")
        if not isinstance(items_raw, list):
            return None
        items: list[GrammarItem] = []
        for raw in items_raw:
            if isinstance(raw, dict):
                items.append(GrammarItem.model_validate(raw))
            else:
                items.append(GrammarItem.model_validate_json(raw))
        return items

    def save_grammar_chunk(
        self, entry: FileCheckpoint, chunk_index: int, items: list[GrammarItem]
    ) -> None:
        path = self._grammar_chunks_path(entry)
        payload = {
            "chunk": chunk_index,
            "items": [it.model_dump() for it in items],
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")
            durable_write(f)
        entry.grammar_chunks = max(entry.grammar_chunks, chunk_index + 1)
        self._persist_entry(entry)

    def mark_vocab_done(self, entry: FileCheckpoint) -> None:
        entry.status = "vocab_done"
        entry.last_error = None
        self._persist_entry(entry)

    def mark_grammar_done(self, entry: FileCheckpoint) -> None:
        if entry.status == "vocab_done" or entry.status == "grammar_done":
            entry.status = "grammar_done"
        elif entry.status == "text_done":
            entry.status = "grammar_done"
        entry.last_error = None
        self._persist_entry(entry)

    def mark_failed(self, entry: FileCheckpoint, error: str) -> None:
        entry.status = "failed"
        entry.last_error = error
        self._persist_entry(entry)

    def mark_run_complete(self) -> None:
        self.manifest.complete = True
        save_manifest(self.manifest)

    def _persist_entry(self, entry: FileCheckpoint) -> None:
        for i, f in enumerate(self.manifest.files):
            if f.path == entry.path:
                self.manifest.files[i] = entry
                break
        save_manifest(self.manifest)

    def all_files_complete(self, mode: ExtractMode) -> bool:
        """True when every file reached the terminal status for ``mode``."""
        if not self.manifest.files:
            return False
        for f in self.manifest.files:
            if mode == "vocab" and f.status != "vocab_done":
                return False
            if mode == "grammar" and f.status != "grammar_done":
                return False
            if mode == "all" and f.status != "grammar_done":
                return False
        return True

    def count_resumable(self, mode: ExtractMode) -> tuple[int, int]:
        """Return (completed, total) file counts for logging."""
        total = len(self.manifest.files)
        done = 0
        for f in self.manifest.files:
            if mode == "vocab" and f.status == "vocab_done":
                done += 1
            elif mode == "grammar" and f.status == "grammar_done":
                done += 1
            elif mode == "all" and f.status == "grammar_done":
                done += 1
        return done, total

    def should_skip_file(self, entry: FileCheckpoint, mode: ExtractMode) -> bool:
        if mode == "vocab":
            return entry.status == "vocab_done"
        if mode == "grammar":
            return entry.status == "grammar_done"
        return entry.status == "grammar_done"


def clear_vocab_chunks(entry: FileCheckpoint, checkpoint: ExtractRunCheckpoint) -> None:
    """Remove vocab chunk cache (e.g. when source file changed)."""
    path = checkpoint._vocab_chunks_path(entry)
    if path.exists():
        path.unlink()
    entry.vocab_chunks = 0


def clear_grammar_chunks(entry: FileCheckpoint, checkpoint: ExtractRunCheckpoint) -> None:
    path = checkpoint._grammar_chunks_path(entry)
    if path.exists():
        path.unlink()
    entry.grammar_chunks = 0
