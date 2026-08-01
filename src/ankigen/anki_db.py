"""Read vocabulary words from an Anki database for deduplication filtering."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
import unicodedata
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

from ankigen.llm import Language

logger = logging.getLogger("ankigen.anki_db")

# ASCII unit separator used by Anki to separate fields within a note
_FIELD_SEP = chr(31)


class AnkiNote(NamedTuple):
    """A full Anki note read from a deck — used by the audit/backfill pipeline.

    Unlike :func:`load_anki_words`, which returns only one field of each note
    as a flat ``set[str]`` for dedup filtering, :class:`AnkiNote` keeps the
    fields that matter for round-tripping a note back into Anki via a
    GUID-keyed update CSV (see ``ankigen audit`` / ``ankigen backfill``).
    """

    nid: int  # notes.id
    guid: str  # notes.guid (stable across syncs)
    mid: int  # notes.mid (note type id)
    model_name: str  # e.g. "Korean Vocab"
    deck_id: int  # cards.did (deck of the first card)
    fields: dict[str, str]  # field name -> value (NFC-normalised values)
    field_order: list[str]  # field names in note-type order


def normalize_anki_term(s: str) -> str:
    """Normalize a string for comparison with words loaded from Anki (NFC + strip)."""
    return unicodedata.normalize("NFC", s.strip())


def get_anki_db_path() -> Path | None:
    """Read ANKIGEN_ANKI_DB from environment, returning None if not set."""
    value = os.environ.get("ANKIGEN_ANKI_DB", "").strip()
    if not value:
        return None
    return Path(value).expanduser()


def get_anki_deck_name(lang: Language) -> str | None:
    """Read ANKIGEN_ANKI_DECK_{LANG} from environment, returning None if not set."""
    key = f"ANKIGEN_ANKI_DECK_{lang.upper()}"
    value = os.environ.get(key, "").strip()
    return value if value else None


def get_anki_field(lang: Language) -> int | str:
    """
    Read ANKIGEN_ANKI_FIELD_{LANG} from environment.

    Returns:
        int: if the value is a non-negative integer string, or if unset (defaults to 0)
        str: if the value is non-numeric — treated as a field name to look up in
             the Anki note type schema (e.g. "Hanzi", "Korean", "Front")

    Examples in .env:
        ANKIGEN_ANKI_FIELD_ZH=0        # first field by position
        ANKIGEN_ANKI_FIELD_ZH=Hanzi    # field named "Hanzi"
    """
    key = f"ANKIGEN_ANKI_FIELD_{lang.upper()}"
    value = os.environ.get(key, "").strip()
    if not value:
        return 0
    try:
        index = int(value)
        if index < 0:
            logger.warning("Field index for %s is negative (%d); using 0.", key, index)
            return 0
        return index
    except ValueError:
        return value  # treat as a field name


def load_anki_words(
    db_path: Path,
    deck_name: str,
    field: int | str = 0,
) -> set[str]:
    """
    Load words from a specific Anki deck (including all sub-decks).

    Supports both .anki2 (direct SQLite) and .apkg (zip containing SQLite).

    Args:
        db_path: Path to .anki2 or .apkg file
        deck_name: Name of the Anki deck to read (e.g. "Chinese::Vocab").
                   Cards in sub-decks (e.g. "Chinese::Vocab::HSK1") are included.
        field: Which note field to extract — either a 0-based integer index or a
               field name string (e.g. "Hanzi"). Field names are resolved from the
               Anki note type schema stored in the database; notes whose note type
               does not contain the named field are skipped gracefully.

    Returns:
        Set of NFC-normalized word strings from the specified deck and field.
        Returns an empty set (with a warning) if the file is missing, the extension
        is unsupported, or the named deck is not found.
    """
    if isinstance(field, int) and field < 0:
        logger.warning("Field index is negative (%d); using 0.", field)
        field = 0

    try:
        with _open_collection(db_path) as conn:
            return _extract_words(conn, deck_name, field)
    except _CollectionOpenError:
        return set()


def load_anki_notes(db_path: Path, deck_name: str) -> list[AnkiNote]:
    """Load full :class:`AnkiNote` records for every note in the deck.

    Sub-decks (``deck_name + "::..."``) are included, mirroring
    :func:`load_anki_words`. Supports ``.anki2``/``.anki21`` SQLite files and
    ``.apkg`` zips (the embedded collection is extracted to a tempdir).

    Returns an empty list (with a warning) if the file is missing, the
    extension is unsupported, or the named deck is not found. The audit
    pipeline treats "no notes" and "deck missing" alike: nothing to score.
    """
    try:
        with _open_collection(db_path) as conn:
            deck_ids = _get_deck_ids(conn, deck_name)
            if not deck_ids:
                logger.warning("Deck '%s' not found in Anki database", deck_name)
                return []
            notes = _get_notes_from_deck(conn, deck_ids)
            sub_info = f" (+{len(deck_ids) - 1} sub-deck(s))" if len(deck_ids) > 1 else ""
            logger.info(
                "Loaded %d note(s) from Anki deck '%s'%s",
                len(notes),
                deck_name,
                sub_info,
            )
            return notes
    except _CollectionOpenError:
        return []


def load_deck_names(db_path: Path) -> dict[int, str]:
    """Return a ``{deck_id: deck_name}`` map from the Anki collection.

    Used by ``ankigen backfill`` to put the real deck name in the TSV
    ``#deck column`` instead of the literal ``"deck"`` placeholder, so Anki
    files any newly-created notes — and logs matched updates — under the
    correct deck. Returns an empty dict if the file is missing/unsupported.
    """
    try:
        with _open_collection(db_path) as conn:
            return _get_all_deck_names(conn)
    except _CollectionOpenError:
        return {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _CollectionOpenError(Exception):
    """Raised when ``_open_collection`` cannot return a usable connection.

    Callers translate this into an empty result + a warning log line — the
    warning has already been emitted by ``_open_collection`` itself, so the
    caller only has to decide on the shape of "nothing" to return.
    """


@contextmanager
def _open_collection(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Yield a read-only SQLite connection for a ``.anki2`` or ``.apkg`` file.

    Handles the three common failure modes (missing file, unsupported
    extension, unreadable zip / locked SQLite) by logging a warning and
    raising :class:`_CollectionOpenError`. Callers convert that into an
    empty result for the user-facing API.
    """
    db_path = Path(db_path).expanduser()

    if not db_path.exists():
        logger.warning("Anki database not found: %s — skipping Anki filtering", db_path)
        raise _CollectionOpenError(str(db_path))

    suffix = db_path.suffix.lower()
    if suffix == ".apkg":
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                with zipfile.ZipFile(db_path, "r") as zf:
                    names = zf.namelist()
                    if "collection.anki21" in names:
                        inner_name = "collection.anki21"
                    elif "collection.anki2" in names:
                        inner_name = "collection.anki2"
                    else:
                        logger.warning(
                            "No collection.anki2 or collection.anki21 found inside %s",
                            db_path,
                        )
                        raise _CollectionOpenError(str(db_path))
                    tmp_path = Path(tmp_dir) / inner_name
                    with zf.open(inner_name) as src, open(tmp_path, "wb") as dst:
                        dst.write(src.read())
            except (zipfile.BadZipFile, KeyError, OSError) as exc:
                logger.warning("Could not read .apkg file %s: %s", db_path, exc)
                raise _CollectionOpenError(str(db_path)) from exc
            try:
                conn = sqlite3.connect(f"file:{tmp_path}?mode=ro", uri=True)
            except sqlite3.OperationalError as exc:
                logger.warning("Could not open Anki database inside %s: %s", db_path, exc)
                raise _CollectionOpenError(str(db_path)) from exc
            try:
                yield conn
            finally:
                conn.close()
        return

    if suffix not in (".anki2", ".anki21"):
        logger.warning(
            "Unrecognised Anki file extension '%s' for %s — expected .anki2 or .apkg",
            suffix,
            db_path,
        )
        raise _CollectionOpenError(str(db_path))

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError as exc:
        logger.warning("Could not open Anki database %s: %s", db_path, exc)
        raise _CollectionOpenError(str(db_path)) from exc
    try:
        yield conn
    finally:
        conn.close()


def _extract_words(conn: sqlite3.Connection, deck_name: str, field: int | str) -> set[str]:
    """Core logic: find deck IDs (including sub-decks), then pull words from notes."""
    deck_ids = _get_deck_ids(conn, deck_name)
    if not deck_ids:
        logger.warning("Deck '%s' not found in Anki database", deck_name)
        return set()

    num_decks = len(deck_ids)
    words = _get_words_from_deck(conn, deck_ids, field)
    sub_info = f" (+{num_decks - 1} sub-deck(s))" if num_decks > 1 else ""
    logger.info(
        "Loaded %d words from Anki deck '%s'%s (field: %r)",
        len(words),
        deck_name,
        sub_info,
        field,
    )
    return words


def _get_deck_ids(conn: sqlite3.Connection, deck_name: str) -> set[int]:
    """
    Return the deck ID for `deck_name` and all its sub-decks.

    A sub-deck is any deck whose name starts with ``deck_name + "::"``
    (Anki's hierarchy separator). For example, querying "Chinese" will also
    include cards in "Chinese::Vocabulary" and "Chinese::Vocabulary::HSK1".

    Handles both old schema (col.decks JSON) and new schema (decks table).
    Returns an empty set if no matching deck is found.
    """
    ids: set[int] = set()
    prefix = deck_name + "::"

    # New schema first (Anki 2.1.50+): standalone decks table
    try:
        cursor = conn.execute("SELECT id, name FROM decks")
        for did, name in cursor:
            if name == deck_name or name.startswith(prefix):
                ids.add(int(did))
        if ids:
            return ids
        # Table exists but no match — fall through to JSON check in case data
        # lives in both places (some exported files have both).
    except sqlite3.OperationalError:
        pass  # Old schema — no decks table

    # Old schema: col.decks is a JSON dict of {id_str: {name: str, ...}}
    try:
        cursor = conn.execute("SELECT decks FROM col LIMIT 1")
        row = cursor.fetchone()
        if row:
            decks_json: dict[str, Any] = json.loads(row[0])
            for did_str, info in decks_json.items():
                if isinstance(info, dict):
                    name = info.get("name", "")
                    if name == deck_name or name.startswith(prefix):
                        ids.add(int(did_str))
    except (sqlite3.OperationalError, json.JSONDecodeError, KeyError):
        pass

    return ids


def _get_all_deck_names(conn: sqlite3.Connection) -> dict[int, str]:
    """Return every ``{deck_id: name}`` pair, handling both schema variants."""
    names: dict[int, str] = {}

    # New schema first (Anki 2.1.50+): standalone decks table.
    try:
        cursor = conn.execute("SELECT id, name FROM decks")
        for did, name in cursor:
            names[int(did)] = name
        if names:
            return names
    except sqlite3.OperationalError:
        pass  # Old schema — no decks table

    # Old schema: col.decks is a JSON dict of {id_str: {name: str, ...}}.
    try:
        cursor = conn.execute("SELECT decks FROM col LIMIT 1")
        row = cursor.fetchone()
        if row:
            decks_json: dict[str, Any] = json.loads(row[0])
            for did_str, info in decks_json.items():
                if isinstance(info, dict):
                    names[int(did_str)] = info.get("name", "")
    except (sqlite3.OperationalError, json.JSONDecodeError, KeyError):
        pass

    return names


def _build_model_field_map(conn: sqlite3.Connection, field_name: str) -> dict[int, int]:
    """
    Build a mapping of note-type-id -> field-index for the given field name.

    Anki stores note type (model) definitions in two places depending on schema:
    - Old schema (≤18): col.models JSON — dict of {mid: {"flds": [{"name":..., "ord":...}]}}
    - New schema (21+): standalone `fields` table — columns (ntid, ord, name, ...)

    Tries the old schema first, then falls back to the new schema's `fields` table.
    Returns an empty dict if the field name is not found in any note type.
    """
    result: dict[int, int] = {}

    # Old schema: col.models JSON
    try:
        cursor = conn.execute("SELECT models FROM col LIMIT 1")
        row = cursor.fetchone()
        if row:
            models_json: dict[str, Any] = json.loads(row[0])
            for mid_str, model in models_json.items():
                if not isinstance(model, dict):
                    continue
                for fld in model.get("flds", []):
                    if fld.get("name") == field_name:
                        result[int(mid_str)] = int(fld["ord"])
                        break
            if result:
                return result
    except (sqlite3.OperationalError, json.JSONDecodeError, KeyError, TypeError):
        pass

    # New schema: standalone fields table (ntid, ord, name, ...).
    # Anki uses a custom "unicase" collation on `name`; Python's sqlite3 does not
    # register it, so `WHERE name = ?` raises OperationalError. Scan and match
    # in Python instead.
    try:
        cursor = conn.execute("SELECT ntid, ord, name FROM fields")
        for ntid, ord_, fname in cursor:
            if fname == field_name:
                result[int(ntid)] = int(ord_)
    except sqlite3.OperationalError:
        pass

    return result


def _get_words_from_deck(
    conn: sqlite3.Connection, deck_ids: set[int], field: int | str
) -> set[str]:
    """
    Return the set of words at the specified field for all notes in `deck_ids`.

    `deck_ids` is a set of deck IDs (the named deck plus any sub-decks).
    When `field` is an int, it is used directly as a 0-based field index.
    When `field` is a str, it is treated as a field name and resolved per note type
    via `_build_model_field_map`. Notes whose note type does not contain the named
    field are skipped gracefully.
    """
    if not deck_ids:
        return set()

    placeholders = ",".join("?" * len(deck_ids))
    params = tuple(deck_ids)

    if isinstance(field, int):
        try:
            cursor = conn.execute(
                f"""
                SELECT DISTINCT n.mid, n.flds
                FROM notes n
                JOIN cards c ON c.nid = n.id
                WHERE c.did IN ({placeholders})
                """,
                params,
            )
        except sqlite3.OperationalError as exc:
            logger.warning("Error querying Anki notes: %s", exc)
            return set()

        words: set[str] = set()
        seen_models: set[int] = set()
        for mid, flds in cursor:
            seen_models.add(int(mid))
            fields_list = flds.split(_FIELD_SEP)
            if field < len(fields_list):
                raw = fields_list[field].strip()
                if raw:
                    norm = normalize_anki_term(raw)
                    if norm:
                        words.add(norm)
        if len(seen_models) > 1:
            # A positional index means "field N of whatever note type this is",
            # so a deck holding both vocab and grammar cards yields a mix of the
            # two. Say so — comparing headwords against grammar patterns
            # produces nonsense, and a field *name* would scope it properly.
            names = _model_names_for(conn, seen_models)
            logger.info(
                "Field index %d spans %d note type(s) in this deck (%s); "
                "pass a field NAME instead to read just one of them",
                field,
                len(seen_models),
                ", ".join(names),
            )
        return words

    # Field name path
    model_field_map = _build_model_field_map(conn, field)
    if not model_field_map:
        logger.warning("Field name %r not found in any note type", field)
        return set()

    try:
        cursor = conn.execute(
            f"""
            SELECT n.mid, n.flds
            FROM notes n
            JOIN cards c ON c.nid = n.id
            WHERE c.did IN ({placeholders})
            """,
            params,
        )
    except sqlite3.OperationalError as exc:
        logger.warning("Error querying Anki notes: %s", exc)
        return set()

    words = set()
    skipped_models: set[int] = set()
    skipped_notes = 0
    for mid, flds in cursor:
        field_idx = model_field_map.get(mid)
        if field_idx is None:
            skipped_models.add(int(mid))
            skipped_notes += 1
            continue
        fields_list = flds.split(_FIELD_SEP)
        if field_idx < len(fields_list):
            raw = fields_list[field_idx].strip()
            if raw:
                norm = normalize_anki_term(raw)
                if norm:
                    words.add(norm)

    if skipped_models:
        # Reported at INFO, not DEBUG: naming a field is also how you scope a
        # scan to one note type (only note types carrying that field are read),
        # so this line is the only way to tell "40 grammar cards" apart from
        # "nothing matched and the deck was the wrong one".
        names = _model_names_for(conn, skipped_models)
        logger.info(
            "Skipped %d note(s) from %d note type(s) with no %r field (%s)",
            skipped_notes,
            len(skipped_models),
            field,
            ", ".join(names),
        )
    return words


def _model_names_for(conn: sqlite3.Connection, mids: set[int]) -> list[str]:
    """Human-readable note-type names for ``mids``, sorted; falls back to the id."""
    schemas = _get_model_schemas(conn)
    return sorted(
        (schemas[mid][0] if mid in schemas and schemas[mid][0] else f"model {mid}") for mid in mids
    )


def _get_model_schemas(conn: sqlite3.Connection) -> dict[int, tuple[str, list[str]]]:
    """Return ``{mid: (model_name, [field_name, ...])}`` for every note type.

    Mirrors :func:`_build_model_field_map`'s two-schema handling but returns
    the ordered field list (needed when reconstructing a full note row).

    Both schemas may coexist in some exports; if a model id appears in both
    we keep the entry from whichever schema we read first (old schema wins
    here to match the lookup order used by :func:`_build_model_field_map`).
    """
    result: dict[int, tuple[str, list[str]]] = {}

    # Old schema: col.models JSON
    try:
        cursor = conn.execute("SELECT models FROM col LIMIT 1")
        row = cursor.fetchone()
        if row:
            models_json: dict[str, Any] = json.loads(row[0])
            for mid_str, model in models_json.items():
                if not isinstance(model, dict):
                    continue
                name = str(model.get("name", ""))
                flds = model.get("flds", [])
                # Order by the explicit `ord` field; fall back to source order.
                ordered = sorted(
                    (f for f in flds if isinstance(f, dict) and "name" in f),
                    key=lambda f: (int(f.get("ord", 0)),),
                )
                field_names = [str(f["name"]) for f in ordered]
                try:
                    result[int(mid_str)] = (name, field_names)
                except (TypeError, ValueError):
                    continue
    except (sqlite3.OperationalError, json.JSONDecodeError, KeyError, TypeError):
        pass

    # New schema: standalone `notetypes` + `fields` tables. Even if the old
    # JSON path populated some models, we re-scan here so models that ONLY
    # exist in the new schema also get a name + field list.
    notetype_names: dict[int, str] = {}
    try:
        cursor = conn.execute("SELECT id, name FROM notetypes")
        for ntid, nname in cursor:
            notetype_names[int(ntid)] = str(nname)
    except sqlite3.OperationalError:
        pass

    fields_by_mid: dict[int, list[tuple[int, str]]] = {}
    try:
        cursor = conn.execute("SELECT ntid, ord, name FROM fields")
        for ntid, ord_, fname in cursor:
            fields_by_mid.setdefault(int(ntid), []).append((int(ord_), str(fname)))
    except sqlite3.OperationalError:
        pass

    for ntid, pairs in fields_by_mid.items():
        if ntid in result:
            continue
        pairs.sort(key=lambda p: p[0])
        result[ntid] = (notetype_names.get(ntid, ""), [name for _, name in pairs])

    return result


def _get_notes_from_deck(conn: sqlite3.Connection, deck_ids: set[int]) -> list[AnkiNote]:
    """Return all notes whose first card belongs to one of ``deck_ids``.

    Each note appears at most once even if it has multiple cards in the
    deck (we ``GROUP BY n.id`` and take the ``MIN(c.did)`` to pick a stable
    representative deck). Notes whose ``mid`` is not in the model schema
    cache are still returned with ``model_name=""`` and a positional
    field list (``"field0"``, ``"field1"``, ...) so callers can at least
    look at the raw values.
    """
    if not deck_ids:
        return []

    schemas = _get_model_schemas(conn)
    placeholders = ",".join("?" * len(deck_ids))
    params = tuple(deck_ids)

    try:
        cursor = conn.execute(
            f"""
            SELECT n.id, n.guid, n.mid, n.flds, MIN(c.did) AS did
            FROM notes n
            JOIN cards c ON c.nid = n.id
            WHERE c.did IN ({placeholders})
            GROUP BY n.id, n.guid, n.mid, n.flds
            """,
            params,
        )
    except sqlite3.OperationalError as exc:
        logger.warning("Error querying Anki notes: %s", exc)
        return []

    notes: list[AnkiNote] = []
    for nid, guid, mid, flds, did in cursor:
        raw_values = flds.split(_FIELD_SEP) if flds else []
        normalized = [unicodedata.normalize("NFC", v) for v in raw_values]
        schema = schemas.get(int(mid))
        if schema is None:
            field_order = [f"field{i}" for i in range(len(normalized))]
            model_name = ""
        else:
            model_name, schema_fields = schema
            # If the note has more fields than the schema declares (shouldn't
            # happen with healthy collections), keep the extras with positional
            # placeholder names so we don't lose data.
            if len(schema_fields) >= len(normalized):
                field_order = list(schema_fields[: len(normalized)])
            else:
                field_order = list(schema_fields) + [
                    f"field{i}" for i in range(len(schema_fields), len(normalized))
                ]

        fields = dict(zip(field_order, normalized, strict=False))
        notes.append(
            AnkiNote(
                nid=int(nid),
                guid=str(guid),
                mid=int(mid),
                model_name=model_name,
                deck_id=int(did),
                fields=fields,
                field_order=field_order,
            )
        )
    return notes
