"""Tests for the anki_db module."""

import json
import sqlite3
import tempfile
import zipfile
from pathlib import Path

import pytest

from ankigen.anki_db import (
    _build_model_field_map,
    _get_deck_id,
    _get_words_from_deck,
    get_anki_db_path,
    get_anki_deck_name,
    get_anki_field,
    load_anki_words,
)

# ASCII unit separator used by Anki between fields
FIELD_SEP = chr(31)

# Model id used by all test helpers — must match the `mid` value in notes rows
_MODEL_ID = 1


# ---------------------------------------------------------------------------
# Helpers to build minimal Anki SQLite databases
# ---------------------------------------------------------------------------


def _make_old_schema_db(
    path: Path,
    deck_name: str,
    words: list[str],
    field_names: list[str] | None = None,
) -> None:
    """
    Create an old-schema (.anki2 schema ≤18) SQLite database.

    Args:
        field_names: Names for the note type fields in order (default: ["word", "translation"]).
                     The first field value is populated from `words`; remaining fields
                     receive a placeholder equal to the field name.
    """
    if field_names is None:
        field_names = ["word", "translation"]

    models_json = json.dumps({
        str(_MODEL_ID): {
            "id": _MODEL_ID,
            "name": "Basic",
            "flds": [{"name": n, "ord": i} for i, n in enumerate(field_names)],
        }
    })

    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE col (
            id INTEGER NOT NULL,
            crt INTEGER NOT NULL,
            mod INTEGER NOT NULL,
            scm INTEGER NOT NULL,
            ver INTEGER NOT NULL,
            dty INTEGER NOT NULL,
            usn INTEGER NOT NULL,
            ls INTEGER NOT NULL,
            conf TEXT NOT NULL,
            models TEXT NOT NULL,
            decks TEXT NOT NULL,
            dconf TEXT NOT NULL,
            tags TEXT NOT NULL
        );
        CREATE TABLE notes (
            id INTEGER NOT NULL PRIMARY KEY,
            guid TEXT NOT NULL,
            mid INTEGER NOT NULL,
            mod INTEGER NOT NULL,
            usn INTEGER NOT NULL,
            tags TEXT NOT NULL,
            flds TEXT NOT NULL,
            sfld INTEGER NOT NULL,
            csum INTEGER NOT NULL,
            flags INTEGER NOT NULL,
            data TEXT NOT NULL
        );
        CREATE TABLE cards (
            id INTEGER NOT NULL PRIMARY KEY,
            nid INTEGER NOT NULL,
            did INTEGER NOT NULL,
            ord INTEGER NOT NULL,
            mod INTEGER NOT NULL,
            usn INTEGER NOT NULL,
            type INTEGER NOT NULL,
            queue INTEGER NOT NULL,
            due INTEGER NOT NULL,
            ivl INTEGER NOT NULL,
            factor INTEGER NOT NULL,
            reps INTEGER NOT NULL,
            lapses INTEGER NOT NULL,
            left INTEGER NOT NULL,
            odue INTEGER NOT NULL,
            odid INTEGER NOT NULL,
            flags INTEGER NOT NULL,
            data TEXT NOT NULL
        );
    """)

    deck_id = 1001
    decks_json = json.dumps({
        str(deck_id): {"id": deck_id, "name": deck_name, "conf": 1, "extendNew": 0}
    })
    conn.execute(
        "INSERT INTO col VALUES (1,0,0,0,11,0,0,0,'{}',?,?,'{}',' ')",
        (models_json, decks_json),
    )

    for i, word in enumerate(words, start=1):
        # First field = the word; remaining fields = field name as placeholder
        field_values = [word] + list(field_names[1:])
        flds = FIELD_SEP.join(field_values)
        conn.execute(
            "INSERT INTO notes VALUES (?,?,?,0,0,'',?,0,0,0,'')",
            (i, f"guid{i}", _MODEL_ID, flds),
        )
        conn.execute(
            "INSERT INTO cards VALUES (?,?,?,0,0,0,0,0,0,0,0,0,0,0,0,0,0,'')",
            (i * 10, i, deck_id),
        )

    conn.commit()
    conn.close()


def _make_new_schema_db(
    path: Path,
    deck_name: str,
    words: list[str],
    field_names: list[str] | None = None,
) -> None:
    """
    Create a new-schema (schema 21+) SQLite database with standalone decks/fields tables.

    Args:
        field_names: Names for the note type fields in order
                     (default: ["word", "translation", "extra"]).
    """
    if field_names is None:
        field_names = ["word", "translation", "extra"]

    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE decks (
            id INTEGER NOT NULL PRIMARY KEY,
            name TEXT NOT NULL,
            mtime_secs INTEGER NOT NULL,
            usn INTEGER NOT NULL,
            common BLOB NOT NULL,
            kind BLOB NOT NULL
        );
        CREATE TABLE fields (
            ntid INTEGER NOT NULL,
            ord INTEGER NOT NULL,
            name TEXT NOT NULL,
            config BLOB NOT NULL
        );
        CREATE TABLE notes (
            id INTEGER NOT NULL PRIMARY KEY,
            guid TEXT NOT NULL,
            mid INTEGER NOT NULL,
            mod INTEGER NOT NULL,
            usn INTEGER NOT NULL,
            tags TEXT NOT NULL,
            flds TEXT NOT NULL,
            sfld INTEGER NOT NULL,
            csum INTEGER NOT NULL,
            flags INTEGER NOT NULL,
            data TEXT NOT NULL
        );
        CREATE TABLE cards (
            id INTEGER NOT NULL PRIMARY KEY,
            nid INTEGER NOT NULL,
            did INTEGER NOT NULL,
            ord INTEGER NOT NULL,
            mod INTEGER NOT NULL,
            usn INTEGER NOT NULL,
            type INTEGER NOT NULL,
            queue INTEGER NOT NULL,
            due INTEGER NOT NULL,
            ivl INTEGER NOT NULL,
            factor INTEGER NOT NULL,
            reps INTEGER NOT NULL,
            lapses INTEGER NOT NULL,
            left INTEGER NOT NULL,
            odue INTEGER NOT NULL,
            odid INTEGER NOT NULL,
            flags INTEGER NOT NULL,
            data TEXT NOT NULL
        );
    """)

    deck_id = 2001
    conn.execute(
        "INSERT INTO decks VALUES (?,?,0,0,'','')",
        (deck_id, deck_name),
    )

    for ord_, name in enumerate(field_names):
        conn.execute(
            "INSERT INTO fields VALUES (?,?,?,'')",
            (_MODEL_ID, ord_, name),
        )

    for i, word in enumerate(words, start=1):
        field_values = [word] + list(field_names[1:])
        flds = FIELD_SEP.join(field_values)
        conn.execute(
            "INSERT INTO notes VALUES (?,?,?,0,0,'',?,0,0,0,'')",
            (i, f"guid{i}", _MODEL_ID, flds),
        )
        conn.execute(
            "INSERT INTO cards VALUES (?,?,?,0,0,0,0,0,0,0,0,0,0,0,0,0,0,'')",
            (i * 10, i, deck_id),
        )

    conn.commit()
    conn.close()


def _make_apkg(apkg_path: Path, deck_name: str, words: list[str]) -> None:
    """Create a .apkg zip containing a minimal collection.anki2 database."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "collection.anki2"
        _make_old_schema_db(db_path, deck_name, words)
        with zipfile.ZipFile(apkg_path, "w") as zf:
            zf.write(db_path, "collection.anki2")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetDeckId:
    def test_old_schema_finds_deck(self, tmp_path):
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(db_path, "MyDeck", ["word1"])
        conn = sqlite3.connect(str(db_path))
        deck_id = _get_deck_id(conn, "MyDeck")
        conn.close()
        assert deck_id == 1001

    def test_new_schema_finds_deck(self, tmp_path):
        db_path = tmp_path / "col.anki2"
        _make_new_schema_db(db_path, "MyDeck", ["word1"])
        conn = sqlite3.connect(str(db_path))
        deck_id = _get_deck_id(conn, "MyDeck")
        conn.close()
        assert deck_id == 2001

    def test_returns_none_for_missing_deck(self, tmp_path):
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(db_path, "MyDeck", ["word1"])
        conn = sqlite3.connect(str(db_path))
        deck_id = _get_deck_id(conn, "NonExistentDeck")
        conn.close()
        assert deck_id is None


class TestBuildModelFieldMap:
    def test_old_schema_finds_field(self, tmp_path):
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(db_path, "Chinese", ["促使"], field_names=["Hanzi", "Jyutping", "English"])
        conn = sqlite3.connect(str(db_path))
        result = _build_model_field_map(conn, "Hanzi")
        conn.close()
        assert result == {_MODEL_ID: 0}

    def test_old_schema_finds_non_first_field(self, tmp_path):
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(db_path, "Chinese", ["促使"], field_names=["Hanzi", "Jyutping", "English"])
        conn = sqlite3.connect(str(db_path))
        result = _build_model_field_map(conn, "English")
        conn.close()
        assert result == {_MODEL_ID: 2}

    def test_new_schema_finds_field(self, tmp_path):
        db_path = tmp_path / "col.anki2"
        _make_new_schema_db(db_path, "Korean", ["편한"], field_names=["Korean", "English", "Notes"])
        conn = sqlite3.connect(str(db_path))
        result = _build_model_field_map(conn, "Korean")
        conn.close()
        assert result == {_MODEL_ID: 0}

    def test_new_schema_finds_non_first_field(self, tmp_path):
        db_path = tmp_path / "col.anki2"
        _make_new_schema_db(db_path, "Korean", ["편한"], field_names=["Korean", "English", "Notes"])
        conn = sqlite3.connect(str(db_path))
        result = _build_model_field_map(conn, "English")
        conn.close()
        assert result == {_MODEL_ID: 1}

    def test_unknown_field_returns_empty(self, tmp_path):
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(db_path, "Chinese", ["促使"], field_names=["Hanzi", "English"])
        conn = sqlite3.connect(str(db_path))
        result = _build_model_field_map(conn, "NonExistent")
        conn.close()
        assert result == {}


class TestGetWordsFromDeck:
    def test_extracts_first_field(self, tmp_path):
        words = ["促使", "归纳", "披露"]
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(db_path, "Chinese", words)
        conn = sqlite3.connect(str(db_path))
        result = _get_words_from_deck(conn, 1001, field=0)
        conn.close()
        assert result == set(words)

    def test_extracts_second_field(self, tmp_path):
        words = ["促使", "归纳"]
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(db_path, "Chinese", words)
        conn = sqlite3.connect(str(db_path))
        # Field 1 is the field name ("translation") used as placeholder in helper
        result = _get_words_from_deck(conn, 1001, field=1)
        conn.close()
        assert result == {"translation"}

    def test_extracts_by_field_name_old_schema(self, tmp_path):
        words = ["促使", "归纳"]
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(db_path, "Chinese", words, field_names=["Hanzi", "Jyutping"])
        conn = sqlite3.connect(str(db_path))
        result = _get_words_from_deck(conn, 1001, field="Hanzi")
        conn.close()
        assert result == set(words)

    def test_extracts_non_first_field_by_name(self, tmp_path):
        words = ["促使", "归纳"]
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(db_path, "Chinese", words, field_names=["Hanzi", "Jyutping"])
        conn = sqlite3.connect(str(db_path))
        # Field "Jyutping" is at index 1, populated with "Jyutping" as placeholder
        result = _get_words_from_deck(conn, 1001, field="Jyutping")
        conn.close()
        assert result == {"Jyutping"}

    def test_extracts_by_field_name_new_schema(self, tmp_path):
        words = ["편한", "추천"]
        db_path = tmp_path / "col.anki2"
        _make_new_schema_db(db_path, "Korean", words, field_names=["Korean", "English"])
        conn = sqlite3.connect(str(db_path))
        result = _get_words_from_deck(conn, 2001, field="Korean")
        conn.close()
        assert result == set(words)

    def test_unknown_field_name_returns_empty(self, tmp_path):
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(db_path, "Chinese", ["促使"], field_names=["Hanzi", "English"])
        conn = sqlite3.connect(str(db_path))
        result = _get_words_from_deck(conn, 1001, field="NonExistent")
        conn.close()
        assert result == set()

    def test_empty_deck(self, tmp_path):
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(db_path, "EmptyDeck", [])
        conn = sqlite3.connect(str(db_path))
        result = _get_words_from_deck(conn, 1001, field=0)
        conn.close()
        assert result == set()


class TestLoadAnkiWords:
    def test_old_schema_anki2(self, tmp_path):
        words = ["促使", "归纳", "披露"]
        db_path = tmp_path / "collection.anki2"
        _make_old_schema_db(db_path, "Chinese", words)
        result = load_anki_words(db_path, "Chinese")
        assert result == set(words)

    def test_new_schema_anki2(self, tmp_path):
        words = ["편한", "추천", "방향"]
        db_path = tmp_path / "collection.anki2"
        _make_new_schema_db(db_path, "Korean", words)
        result = load_anki_words(db_path, "Korean")
        assert result == set(words)

    def test_apkg_format(self, tmp_path):
        words = ["促使", "归纳"]
        apkg_path = tmp_path / "export.apkg"
        _make_apkg(apkg_path, "Chinese", words)
        result = load_anki_words(apkg_path, "Chinese")
        assert result == set(words)

    def test_missing_file_returns_empty_set(self, tmp_path):
        result = load_anki_words(tmp_path / "nonexistent.anki2", "MyDeck")
        assert result == set()

    def test_deck_not_found_returns_empty_set(self, tmp_path):
        db_path = tmp_path / "collection.anki2"
        _make_old_schema_db(db_path, "Chinese", ["促使"])
        result = load_anki_words(db_path, "WrongDeckName")
        assert result == set()

    def test_unsupported_extension_returns_empty_set(self, tmp_path):
        bad_path = tmp_path / "file.db"
        bad_path.write_text("not a real db")
        result = load_anki_words(bad_path, "Deck")
        assert result == set()

    def test_field_int_index(self, tmp_path):
        words = ["促使", "归纳"]
        db_path = tmp_path / "collection.anki2"
        _make_old_schema_db(db_path, "Chinese", words)
        # Field index 1 is the "translation" placeholder in our helper
        result = load_anki_words(db_path, "Chinese", field=1)
        assert result == {"translation"}

    def test_field_name_old_schema(self, tmp_path):
        words = ["促使", "归纳"]
        db_path = tmp_path / "collection.anki2"
        _make_old_schema_db(db_path, "Chinese", words, field_names=["Hanzi", "Jyutping"])
        result = load_anki_words(db_path, "Chinese", field="Hanzi")
        assert result == set(words)

    def test_field_name_new_schema(self, tmp_path):
        words = ["편한", "추천"]
        db_path = tmp_path / "collection.anki2"
        _make_new_schema_db(db_path, "Korean", words, field_names=["Korean", "English"])
        result = load_anki_words(db_path, "Korean", field="Korean")
        assert result == set(words)

    def test_field_name_non_first_field(self, tmp_path):
        words = ["促使", "归纳"]
        db_path = tmp_path / "collection.anki2"
        _make_old_schema_db(db_path, "Chinese", words, field_names=["Hanzi", "English"])
        result = load_anki_words(db_path, "Chinese", field="English")
        assert result == {"English"}

    def test_anki21_extension(self, tmp_path):
        words = ["促使"]
        db_path = tmp_path / "collection.anki21"
        _make_old_schema_db(db_path, "Chinese", words)
        result = load_anki_words(db_path, "Chinese")
        assert result == set(words)


class TestEnvHelpers:
    def test_get_anki_db_path_set(self, monkeypatch, tmp_path):
        db = tmp_path / "collection.anki2"
        monkeypatch.setenv("ANKIGEN_ANKI_DB", str(db))
        assert get_anki_db_path() == db

    def test_get_anki_db_path_unset(self, monkeypatch):
        monkeypatch.delenv("ANKIGEN_ANKI_DB", raising=False)
        assert get_anki_db_path() is None

    def test_get_anki_deck_name_zh(self, monkeypatch):
        monkeypatch.setenv("ANKIGEN_ANKI_DECK_ZH", "Chinese::Vocab")
        assert get_anki_deck_name("zh") == "Chinese::Vocab"

    def test_get_anki_deck_name_ko(self, monkeypatch):
        monkeypatch.setenv("ANKIGEN_ANKI_DECK_KO", "Korean")
        assert get_anki_deck_name("ko") == "Korean"

    def test_get_anki_deck_name_unset(self, monkeypatch):
        monkeypatch.delenv("ANKIGEN_ANKI_DECK_ZH", raising=False)
        assert get_anki_deck_name("zh") is None

    def test_get_anki_field_default(self, monkeypatch):
        monkeypatch.delenv("ANKIGEN_ANKI_FIELD_ZH", raising=False)
        assert get_anki_field("zh") == 0

    def test_get_anki_field_integer_string(self, monkeypatch):
        monkeypatch.setenv("ANKIGEN_ANKI_FIELD_ZH", "2")
        assert get_anki_field("zh") == 2

    def test_get_anki_field_returns_str_for_field_name(self, monkeypatch):
        monkeypatch.setenv("ANKIGEN_ANKI_FIELD_ZH", "Hanzi")
        assert get_anki_field("zh") == "Hanzi"

    def test_get_anki_field_returns_str_for_any_non_numeric(self, monkeypatch):
        monkeypatch.setenv("ANKIGEN_ANKI_FIELD_KO", "Korean")
        assert get_anki_field("ko") == "Korean"

    def test_get_anki_field_negative_falls_back_to_zero(self, monkeypatch):
        monkeypatch.setenv("ANKIGEN_ANKI_FIELD_KO", "-1")
        assert get_anki_field("ko") == 0
