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


def get_anki_field_index(lang: Language) -> int:
    """
    Read ANKIGEN_ANKI_FIELD_{LANG} from environment.

    Returns the 0-based index of the note field containing the vocabulary word.
    Defaults to 0 (first field) if not set.
    """
    key = f"ANKIGEN_ANKI_FIELD_{lang.upper()}"
    value = os.environ.get(key, "").strip()
    if not value:
        return 0
    try:
        index = int(value)
    except ValueError:
        logger.warning(
            "Invalid value for %s: %r — must be an integer. Using 0.",
            key,
            value,
        )
        return 0
    if index < 0:
        logger.warning("Field index for %s is negative (%d); using 0.", key, index)
        return 0
    return index


def load_anki_words(
    db_path: Path,
    deck_name: str,
    field_index: int = 0,
) -> set[str]:
    """
    Load words from a specific Anki deck.

    Supports both .anki2 (direct SQLite) and .apkg (zip containing SQLite).
    Returns the set of field values at `field_index` for all notes in `deck_name`.
    If the file is missing or the deck is not found, logs a warning and returns an
    empty set so the calling pipeline can continue unaffected.

    Args:
        db_path: Path to .anki2 or .apkg file
        deck_name: Name of the Anki deck to read (e.g. "Chinese::Vocab")
        field_index: Index of the note field to extract (0 = first/front field)

    Returns:
        Set of word strings from the specified deck
    """
    db_path = Path(db_path).expanduser()

    if not db_path.exists():
        logger.warning("Anki database not found: %s — skipping Anki filtering", db_path)
        return set()

    suffix = db_path.suffix.lower()
    if suffix == ".apkg":
        return _load_from_apkg(db_path, deck_name, field_index)
    elif suffix in (".anki2", ".anki21"):
        return _load_from_anki2(db_path, deck_name, field_index)
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


def _load_from_anki2(db_path: Path, deck_name: str, field_index: int) -> set[str]:
    """Open a .anki2/.anki21 SQLite file and extract words."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError as exc:
        logger.warning("Could not open Anki database %s: %s", db_path, exc)
        return set()

    try:
        return _extract_words(conn, deck_name, field_index)
    finally:
        conn.close()


def _load_from_apkg(apkg_path: Path, deck_name: str, field_index: int) -> set[str]:
    """Extract the embedded SQLite from a .apkg zip and read words from it."""
    try:
        with zipfile.ZipFile(apkg_path, "r") as zf:
            names = zf.namelist()
    except (zipfile.BadZipFile, OSError) as exc:
        logger.warning("Could not open .apkg file %s: %s", apkg_path, exc)
        return set()

    # Prefer newer .anki21 format, fall back to .anki2
    if "collection.anki21" in names:
        inner_name = "collection.anki21"
    elif "collection.anki2" in names:
        inner_name = "collection.anki2"
    else:
        logger.warning(
            "No collection.anki2 or collection.anki21 found inside %s", apkg_path
        )
        return set()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / inner_name
        try:
            with zipfile.ZipFile(apkg_path, "r") as zf:
                with zf.open(inner_name) as src, open(tmp_path, "wb") as dst:
                    dst.write(src.read())
        except (zipfile.BadZipFile, KeyError, OSError) as exc:
            logger.warning("Failed to extract %s from %s: %s", inner_name, apkg_path, exc)
            return set()

        return _load_from_anki2(tmp_path, deck_name, field_index)


def _extract_words(
    conn: sqlite3.Connection, deck_name: str, field_index: int
) -> set[str]:
    """Core logic: find deck ID, then pull words from notes in that deck."""
    deck_id = _get_deck_id(conn, deck_name)
    if deck_id is None:
        logger.warning("Deck '%s' not found in Anki database", deck_name)
        return set()

    words = _get_words_from_deck(conn, deck_id, field_index)
    logger.info(
        "Loaded %d words from Anki deck '%s' (field index %d)",
        len(words),
        deck_name,
        field_index,
    )
    return words


def _get_deck_id(conn: sqlite3.Connection, deck_name: str) -> int | None:
    """
    Find the integer deck ID for a named deck.

    Handles two Anki schema variants:
    - Old (schema ≤ 18): deck metadata is stored as JSON in col.decks
    - New (schema 21+): decks are stored in a standalone `decks` table
    """
    # Try new schema first (Anki 2.1.50+): standalone decks table
    try:
        cursor = conn.execute("SELECT id FROM decks WHERE name = ?", (deck_name,))
        row = cursor.fetchone()
        if row is not None:
            return int(row[0])
        # Deck table exists but name not found — still might be old schema exported
        # as new format. Fall through to JSON check.
    except sqlite3.OperationalError:
        pass  # Table doesn't exist — old schema

    # Old schema: col.decks is a JSON dict of {deck_id_str: {name: str, ...}}
    try:
        cursor = conn.execute("SELECT decks FROM col LIMIT 1")
        row = cursor.fetchone()
        if row is None:
            return None
        decks_json: dict = json.loads(row[0])
        for deck_id_str, deck_info in decks_json.items():
            if isinstance(deck_info, dict) and deck_info.get("name") == deck_name:
                return int(deck_id_str)
    except (sqlite3.OperationalError, json.JSONDecodeError, KeyError):
        pass

    return None


def _get_words_from_deck(
    conn: sqlite3.Connection, deck_id: int, field_index: int
) -> set[str]:
    """
    Return the set of words at `field_index` for all notes whose cards are in `deck_id`.
    """
    try:
        cursor = conn.execute(
            """
            SELECT DISTINCT n.flds
            FROM notes n
            JOIN cards c ON c.nid = n.id
            WHERE c.did = ?
            """,
            (deck_id,),
        )
    except sqlite3.OperationalError as exc:
        logger.warning("Error querying Anki notes: %s", exc)
        return set()

    words: set[str] = set()
    for (flds,) in cursor:
        fields = flds.split(_FIELD_SEP)
        if field_index < len(fields):
            word = fields[field_index].strip()
            if word:
                words.add(word)

    return words
