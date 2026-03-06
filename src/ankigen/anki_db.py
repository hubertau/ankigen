"""Read vocabulary words from an Anki database for deduplication filtering."""

import json
import logging
import os
import sqlite3
import tempfile
import zipfile
from pathlib import Path

from ankigen.llm import Language

logger = logging.getLogger("ankigen.anki_db")

# ASCII unit separator used by Anki to separate fields within a note
_FIELD_SEP = chr(31)


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
        Set of word strings from the specified deck and field.
        Returns an empty set (with a warning) if the file is missing, the extension
        is unsupported, or the named deck is not found.
    """
    db_path = Path(db_path).expanduser()

    if not db_path.exists():
        logger.warning("Anki database not found: %s — skipping Anki filtering", db_path)
        return set()

    suffix = db_path.suffix.lower()
    if suffix == ".apkg":
        return _load_from_apkg(db_path, deck_name, field)
    elif suffix in (".anki2", ".anki21"):
        return _load_from_anki2(db_path, deck_name, field)
    else:
        logger.warning(
            "Unrecognised Anki file extension '%s' for %s — expected .anki2 or .apkg",
            suffix,
            db_path,
        )
        return set()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_from_anki2(db_path: Path, deck_name: str, field: int | str) -> set[str]:
    """Open a .anki2/.anki21 SQLite file and extract words."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError as exc:
        logger.warning("Could not open Anki database %s: %s", db_path, exc)
        return set()

    try:
        return _extract_words(conn, deck_name, field)
    finally:
        conn.close()


def _load_from_apkg(apkg_path: Path, deck_name: str, field: int | str) -> set[str]:
    """Extract the embedded SQLite from a .apkg zip and read words from it."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            with zipfile.ZipFile(apkg_path, "r") as zf:
                names = zf.namelist()
                if "collection.anki21" in names:
                    inner_name = "collection.anki21"
                elif "collection.anki2" in names:
                    inner_name = "collection.anki2"
                else:
                    logger.warning(
                        "No collection.anki2 or collection.anki21 found inside %s",
                        apkg_path,
                    )
                    return set()
                tmp_path = Path(tmp_dir) / inner_name
                with zf.open(inner_name) as src, open(tmp_path, "wb") as dst:
                    dst.write(src.read())
        except (zipfile.BadZipFile, KeyError, OSError) as exc:
            logger.warning("Could not read .apkg file %s: %s", apkg_path, exc)
            return set()

        return _load_from_anki2(tmp_path, deck_name, field)


def _extract_words(
    conn: sqlite3.Connection, deck_name: str, field: int | str
) -> set[str]:
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
            decks_json: dict = json.loads(row[0])
            for did_str, info in decks_json.items():
                if isinstance(info, dict):
                    name = info.get("name", "")
                    if name == deck_name or name.startswith(prefix):
                        ids.add(int(did_str))
    except (sqlite3.OperationalError, json.JSONDecodeError, KeyError):
        pass

    return ids


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
            models_json: dict = json.loads(row[0])
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

    # New schema: standalone fields table (ntid, ord, name, ...)
    try:
        cursor = conn.execute(
            "SELECT ntid, ord FROM fields WHERE name = ?",
            (field_name,),
        )
        for ntid, ord_ in cursor:
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
                SELECT DISTINCT n.flds
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
        for (flds,) in cursor:
            fields_list = flds.split(_FIELD_SEP)
            if field < len(fields_list):
                word = fields_list[field].strip()
                if word:
                    words.add(word)
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
    for mid, flds in cursor:
        field_idx = model_field_map.get(mid)
        if field_idx is None:
            skipped_models.add(mid)
            continue
        fields_list = flds.split(_FIELD_SEP)
        if field_idx < len(fields_list):
            word = fields_list[field_idx].strip()
            if word:
                words.add(word)

    if skipped_models:
        logger.debug(
            "%d note type(s) in this deck don't have field %r — those notes skipped",
            len(skipped_models),
            field,
        )
    return words
