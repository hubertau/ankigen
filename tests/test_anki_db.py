"""Tests for the anki_db module."""

import json
import logging
import sqlite3
import tempfile
import unicodedata
import zipfile
from pathlib import Path

from ankigen.anki_db import (
    AnkiNote,
    _build_model_field_map,
    _get_deck_ids,
    _get_words_from_deck,
    get_anki_db_path,
    get_anki_deck_name,
    get_anki_field,
    load_anki_notes,
    load_anki_words,
    load_deck_names,
    normalize_anki_term,
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
    extra_decks: list[tuple[int, str, list[str]]] | None = None,
) -> None:
    """
    Create an old-schema (.anki2 schema ≤18) SQLite database.

    Args:
        field_names: Names for the note type fields in order (default: ["word", "translation"]).
                     The first field value is populated from `words`; remaining fields
                     receive a placeholder equal to the field name.
        extra_decks: Optional list of (deck_id, deck_name, words) for additional decks
                     (used to simulate sub-decks). Words start at note id 1000+.
    """
    if field_names is None:
        field_names = ["word", "translation"]

    models_json = json.dumps(
        {
            str(_MODEL_ID): {
                "id": _MODEL_ID,
                "name": "Basic",
                "flds": [{"name": n, "ord": i} for i, n in enumerate(field_names)],
            }
        }
    )

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
    all_decks = {str(deck_id): {"id": deck_id, "name": deck_name, "conf": 1, "extendNew": 0}}
    if extra_decks:
        for edid, edname, _ in extra_decks:
            all_decks[str(edid)] = {"id": edid, "name": edname, "conf": 1, "extendNew": 0}

    decks_json = json.dumps(all_decks)
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

    if extra_decks:
        note_id = 1000
        for edid, _edname, ewords in extra_decks:
            for word in ewords:
                note_id += 1
                field_values = [word] + list(field_names[1:])
                flds = FIELD_SEP.join(field_values)
                conn.execute(
                    "INSERT INTO notes VALUES (?,?,?,0,0,'',?,0,0,0,'')",
                    (note_id, f"guid{note_id}", _MODEL_ID, flds),
                )
                conn.execute(
                    "INSERT INTO cards VALUES (?,?,?,0,0,0,0,0,0,0,0,0,0,0,0,0,0,'')",
                    (note_id * 10, note_id, edid),
                )

    conn.commit()
    conn.close()


def _make_new_schema_db(
    path: Path,
    deck_name: str,
    words: list[str],
    field_names: list[str] | None = None,
    extra_decks: list[tuple[int, str, list[str]]] | None = None,
) -> None:
    """
    Create a new-schema (schema 21+) SQLite database with standalone decks/fields tables.

    Args:
        field_names: Names for the note type fields in order
                     (default: ["word", "translation", "extra"]).
        extra_decks: Optional list of (deck_id, deck_name, words) for additional decks.
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
    if extra_decks:
        for edid, edname, _ in extra_decks:
            conn.execute(
                "INSERT INTO decks VALUES (?,?,0,0,'','')",
                (edid, edname),
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

    if extra_decks:
        note_id = 1000
        for edid, _edname, ewords in extra_decks:
            for word in ewords:
                note_id += 1
                field_values = [word] + list(field_names[1:])
                flds = FIELD_SEP.join(field_values)
                conn.execute(
                    "INSERT INTO notes VALUES (?,?,?,0,0,'',?,0,0,0,'')",
                    (note_id, f"guid{note_id}", _MODEL_ID, flds),
                )
                conn.execute(
                    "INSERT INTO cards VALUES (?,?,?,0,0,0,0,0,0,0,0,0,0,0,0,0,0,'')",
                    (note_id * 10, note_id, edid),
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


class TestGetDeckIds:
    def test_old_schema_finds_deck(self, tmp_path):
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(db_path, "MyDeck", ["word1"])
        conn = sqlite3.connect(str(db_path))
        deck_ids = _get_deck_ids(conn, "MyDeck")
        conn.close()
        assert deck_ids == {1001}

    def test_new_schema_finds_deck(self, tmp_path):
        db_path = tmp_path / "col.anki2"
        _make_new_schema_db(db_path, "MyDeck", ["word1"])
        conn = sqlite3.connect(str(db_path))
        deck_ids = _get_deck_ids(conn, "MyDeck")
        conn.close()
        assert deck_ids == {2001}

    def test_returns_empty_set_for_missing_deck(self, tmp_path):
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(db_path, "MyDeck", ["word1"])
        conn = sqlite3.connect(str(db_path))
        deck_ids = _get_deck_ids(conn, "NonExistentDeck")
        conn.close()
        assert deck_ids == set()

    def test_old_schema_includes_sub_decks(self, tmp_path):
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(
            db_path,
            "Chinese",
            ["促使"],
            extra_decks=[
                (1002, "Chinese::Vocabulary", ["归纳"]),
                (1003, "Chinese::Vocabulary::HSK1", ["披露"]),
            ],
        )
        conn = sqlite3.connect(str(db_path))
        deck_ids = _get_deck_ids(conn, "Chinese")
        conn.close()
        assert deck_ids == {1001, 1002, 1003}

    def test_new_schema_includes_sub_decks(self, tmp_path):
        db_path = tmp_path / "col.anki2"
        _make_new_schema_db(
            db_path,
            "Korean",
            ["편한"],
            extra_decks=[(2002, "Korean::Grammar", ["추천"])],
        )
        conn = sqlite3.connect(str(db_path))
        deck_ids = _get_deck_ids(conn, "Korean")
        conn.close()
        assert deck_ids == {2001, 2002}

    def test_does_not_match_partial_prefix(self, tmp_path):
        """Querying 'Chinese' should NOT match 'ChinesePlus'."""
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(
            db_path,
            "ChinesePlus",
            ["word1"],
        )
        conn = sqlite3.connect(str(db_path))
        deck_ids = _get_deck_ids(conn, "Chinese")
        conn.close()
        assert deck_ids == set()


class TestLoadDeckNames:
    """`load_deck_names` returns a {deck_id: name} map for both schemas."""

    def test_old_schema_maps_ids_to_names(self, tmp_path):
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(
            db_path,
            "Chinese",
            ["促使"],
            extra_decks=[(1002, "Chinese::Vocabulary", ["归纳"])],
        )
        names = load_deck_names(db_path)
        assert names[1001] == "Chinese"
        assert names[1002] == "Chinese::Vocabulary"

    def test_new_schema_maps_ids_to_names(self, tmp_path):
        db_path = tmp_path / "col.anki2"
        _make_new_schema_db(
            db_path,
            "Korean",
            ["편한"],
            extra_decks=[(2002, "Korean::Grammar", ["추천"])],
        )
        names = load_deck_names(db_path)
        assert names[2001] == "Korean"
        assert names[2002] == "Korean::Grammar"

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_deck_names(tmp_path / "nope.anki2") == {}


class TestBuildModelFieldMap:
    def test_old_schema_finds_field(self, tmp_path):
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(
            db_path, "Chinese", ["促使"], field_names=["Hanzi", "Jyutping", "English"]
        )
        conn = sqlite3.connect(str(db_path))
        result = _build_model_field_map(conn, "Hanzi")
        conn.close()
        assert result == {_MODEL_ID: 0}

    def test_old_schema_finds_non_first_field(self, tmp_path):
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(
            db_path, "Chinese", ["促使"], field_names=["Hanzi", "Jyutping", "English"]
        )
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
        result = _get_words_from_deck(conn, {1001}, field=0)
        conn.close()
        assert result == set(words)

    def test_extracts_second_field(self, tmp_path):
        words = ["促使", "归纳"]
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(db_path, "Chinese", words)
        conn = sqlite3.connect(str(db_path))
        # Field 1 is the field name ("translation") used as placeholder in helper
        result = _get_words_from_deck(conn, {1001}, field=1)
        conn.close()
        assert result == {"translation"}

    def test_extracts_by_field_name_old_schema(self, tmp_path):
        words = ["促使", "归纳"]
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(db_path, "Chinese", words, field_names=["Hanzi", "Jyutping"])
        conn = sqlite3.connect(str(db_path))
        result = _get_words_from_deck(conn, {1001}, field="Hanzi")
        conn.close()
        assert result == set(words)

    def test_extracts_non_first_field_by_name(self, tmp_path):
        words = ["促使", "归纳"]
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(db_path, "Chinese", words, field_names=["Hanzi", "Jyutping"])
        conn = sqlite3.connect(str(db_path))
        # Field "Jyutping" is at index 1, populated with "Jyutping" as placeholder
        result = _get_words_from_deck(conn, {1001}, field="Jyutping")
        conn.close()
        assert result == {"Jyutping"}

    def test_extracts_by_field_name_new_schema(self, tmp_path):
        words = ["편한", "추천"]
        db_path = tmp_path / "col.anki2"
        _make_new_schema_db(db_path, "Korean", words, field_names=["Korean", "English"])
        conn = sqlite3.connect(str(db_path))
        result = _get_words_from_deck(conn, {2001}, field="Korean")
        conn.close()
        assert result == set(words)

    def test_unknown_field_name_returns_empty(self, tmp_path):
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(db_path, "Chinese", ["促使"], field_names=["Hanzi", "English"])
        conn = sqlite3.connect(str(db_path))
        result = _get_words_from_deck(conn, {1001}, field="NonExistent")
        conn.close()
        assert result == set()

    def test_empty_deck(self, tmp_path):
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(db_path, "EmptyDeck", [])
        conn = sqlite3.connect(str(db_path))
        result = _get_words_from_deck(conn, {1001}, field=0)
        conn.close()
        assert result == set()

    def test_multiple_deck_ids_combines_words(self, tmp_path):
        """Words from multiple deck IDs should all be returned."""
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(
            db_path,
            "Chinese",
            ["促使"],
            extra_decks=[
                (1002, "Chinese::Vocabulary", ["归纳"]),
                (1003, "Chinese::Vocabulary::HSK1", ["披露"]),
            ],
        )
        conn = sqlite3.connect(str(db_path))
        result = _get_words_from_deck(conn, {1001, 1002, 1003}, field=0)
        conn.close()
        assert result == {"促使", "归纳", "披露"}

    def test_empty_deck_ids_returns_empty(self, tmp_path):
        db_path = tmp_path / "col.anki2"
        _make_old_schema_db(db_path, "Chinese", ["促使"])
        conn = sqlite3.connect(str(db_path))
        result = _get_words_from_deck(conn, set(), field=0)
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

    def test_sub_decks_old_schema(self, tmp_path):
        """Words in sub-decks should be included when querying the parent deck."""
        db_path = tmp_path / "collection.anki2"
        _make_old_schema_db(
            db_path,
            "Chinese",
            ["促使"],
            extra_decks=[
                (1002, "Chinese::Vocabulary", ["归纳"]),
                (1003, "Chinese::Vocabulary::HSK1", ["披露"]),
            ],
        )
        result = load_anki_words(db_path, "Chinese")
        assert result == {"促使", "归纳", "披露"}

    def test_sub_decks_new_schema(self, tmp_path):
        """Words in sub-decks should be included when querying the parent deck."""
        db_path = tmp_path / "collection.anki2"
        _make_new_schema_db(
            db_path,
            "Korean",
            ["편한"],
            extra_decks=[(2002, "Korean::Grammar", ["추천"])],
        )
        result = load_anki_words(db_path, "Korean")
        assert result == {"편한", "추천"}

    def test_sub_deck_query_only_returns_sub_deck_words(self, tmp_path):
        """Querying a sub-deck should NOT include words from sibling or parent decks."""
        db_path = tmp_path / "collection.anki2"
        _make_old_schema_db(
            db_path,
            "Chinese",
            ["促使"],
            extra_decks=[(1002, "Chinese::Vocabulary", ["归纳"])],
        )
        result = load_anki_words(db_path, "Chinese::Vocabulary")
        assert result == {"归纳"}
        assert "促使" not in result


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


class TestNormalizeAnkiTerm:
    def test_combining_latin_unifies_with_precomposed(self):
        assert normalize_anki_term("e\u0301") == normalize_anki_term("\u00e9")

    def test_strips_surrounding_whitespace(self):
        assert normalize_anki_term("  abc  ") == "abc"


class TestLoadAnkiWordsFieldClampAndNfc:
    def test_negative_field_index_clamped_like_zero(self, tmp_path, caplog):
        words = ["促使"]
        db_path = tmp_path / "collection.anki2"
        _make_old_schema_db(db_path, "Chinese", words)
        caplog.set_level(logging.WARNING)
        result_neg = load_anki_words(db_path, "Chinese", field=-1)
        result_zero = load_anki_words(db_path, "Chinese", field=0)
        assert result_neg == result_zero == set(words)
        assert "negative" in caplog.text.lower()

    def test_note_field_nfd_normalized_to_nfc(self, tmp_path):
        nfd_ga = unicodedata.normalize("NFD", "\uac00")
        db_path = tmp_path / "collection.anki2"
        _make_new_schema_db(db_path, "Korean", [nfd_ga], field_names=["Korean", "English"])
        result = load_anki_words(db_path, "Korean", field="Korean")
        assert result == {"\uac00"}


# ---------------------------------------------------------------------------
# Helpers to build full-note databases (one row per note, arbitrary field values)
# ---------------------------------------------------------------------------


def _make_full_note_db_old_schema(
    path: Path,
    *,
    deck_id: int,
    deck_name: str,
    model_name: str,
    field_names: list[str],
    notes: list[tuple[int, str, list[str]]],
) -> None:
    """Build an old-schema DB with full control over each note's field values.

    Args:
        notes: list of ``(nid, guid, [field_value, ...])`` — one tuple per
            note. Field values must match the ``field_names`` length.
    """
    models_json = json.dumps(
        {
            str(_MODEL_ID): {
                "id": _MODEL_ID,
                "name": model_name,
                "flds": [{"name": n, "ord": i} for i, n in enumerate(field_names)],
            }
        }
    )

    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE col (
            id INTEGER NOT NULL, crt INTEGER NOT NULL, mod INTEGER NOT NULL,
            scm INTEGER NOT NULL, ver INTEGER NOT NULL, dty INTEGER NOT NULL,
            usn INTEGER NOT NULL, ls INTEGER NOT NULL, conf TEXT NOT NULL,
            models TEXT NOT NULL, decks TEXT NOT NULL, dconf TEXT NOT NULL,
            tags TEXT NOT NULL
        );
        CREATE TABLE notes (
            id INTEGER NOT NULL PRIMARY KEY, guid TEXT NOT NULL,
            mid INTEGER NOT NULL, mod INTEGER NOT NULL, usn INTEGER NOT NULL,
            tags TEXT NOT NULL, flds TEXT NOT NULL, sfld INTEGER NOT NULL,
            csum INTEGER NOT NULL, flags INTEGER NOT NULL, data TEXT NOT NULL
        );
        CREATE TABLE cards (
            id INTEGER NOT NULL PRIMARY KEY, nid INTEGER NOT NULL,
            did INTEGER NOT NULL, ord INTEGER NOT NULL, mod INTEGER NOT NULL,
            usn INTEGER NOT NULL, type INTEGER NOT NULL, queue INTEGER NOT NULL,
            due INTEGER NOT NULL, ivl INTEGER NOT NULL, factor INTEGER NOT NULL,
            reps INTEGER NOT NULL, lapses INTEGER NOT NULL, left INTEGER NOT NULL,
            odue INTEGER NOT NULL, odid INTEGER NOT NULL, flags INTEGER NOT NULL,
            data TEXT NOT NULL
        );
    """)

    decks_json = json.dumps({str(deck_id): {"id": deck_id, "name": deck_name}})
    conn.execute(
        "INSERT INTO col VALUES (1,0,0,0,11,0,0,0,'{}',?,?,'{}',' ')",
        (models_json, decks_json),
    )

    for nid, guid, values in notes:
        flds = FIELD_SEP.join(values)
        conn.execute(
            "INSERT INTO notes VALUES (?,?,?,0,0,'',?,0,0,0,'')",
            (nid, guid, _MODEL_ID, flds),
        )
        conn.execute(
            "INSERT INTO cards VALUES (?,?,?,0,0,0,0,0,0,0,0,0,0,0,0,0,0,'')",
            (nid * 10, nid, deck_id),
        )

    conn.commit()
    conn.close()


class TestLoadAnkiNotes:
    """End-to-end tests for the richer :func:`load_anki_notes` API."""

    def test_returns_anki_note_records(self, tmp_path):
        db_path = tmp_path / "collection.anki2"
        _make_full_note_db_old_schema(
            db_path,
            deck_id=1001,
            deck_name="Korean::Vocab",
            model_name="Korean Vocab",
            field_names=["Korean", "Hanja", "English", "Comments"],
            notes=[
                (1, "guid-aaa", ["음식", "", "food", ""]),
                (2, "guid-bbb", ["飮食", "飮食", "food (sino)", "<span>...</span>"]),
            ],
        )
        notes = load_anki_notes(db_path, "Korean::Vocab")
        assert len(notes) == 2
        by_guid = {n.guid: n for n in notes}
        assert isinstance(by_guid["guid-aaa"], AnkiNote)
        assert by_guid["guid-aaa"].fields["Korean"] == "음식"
        assert by_guid["guid-aaa"].fields["Hanja"] == ""
        assert by_guid["guid-aaa"].fields["English"] == "food"
        assert by_guid["guid-aaa"].model_name == "Korean Vocab"
        assert by_guid["guid-aaa"].deck_id == 1001
        assert by_guid["guid-bbb"].fields["Hanja"] == "飮食"

    def test_returns_field_order_matching_note_type(self, tmp_path):
        db_path = tmp_path / "collection.anki2"
        _make_full_note_db_old_schema(
            db_path,
            deck_id=1001,
            deck_name="Chinese",
            model_name="Chinese Vocab",
            field_names=["Hanzi", "Jyutping", "English", "Sentence"],
            notes=[(1, "g1", ["促使", "", "to urge", ""])],
        )
        notes = load_anki_notes(db_path, "Chinese")
        assert notes[0].field_order == ["Hanzi", "Jyutping", "English", "Sentence"]

    def test_missing_file_returns_empty_list(self, tmp_path):
        assert load_anki_notes(tmp_path / "nope.anki2", "Deck") == []

    def test_deck_not_found_returns_empty_list(self, tmp_path):
        db_path = tmp_path / "collection.anki2"
        _make_full_note_db_old_schema(
            db_path,
            deck_id=1001,
            deck_name="Chinese",
            model_name="Basic",
            field_names=["F"],
            notes=[(1, "g", ["x"])],
        )
        assert load_anki_notes(db_path, "Nonexistent") == []

    def test_apkg_format(self, tmp_path):
        apkg_path = tmp_path / "deck.apkg"
        with tempfile.TemporaryDirectory() as tmp:
            inner = Path(tmp) / "collection.anki2"
            _make_full_note_db_old_schema(
                inner,
                deck_id=1001,
                deck_name="Korean",
                model_name="Korean Vocab",
                field_names=["Korean", "English"],
                notes=[(1, "g1", ["편한", "comfortable"])],
            )
            with zipfile.ZipFile(apkg_path, "w") as zf:
                zf.write(inner, "collection.anki2")
        notes = load_anki_notes(apkg_path, "Korean")
        assert len(notes) == 1
        assert notes[0].fields["Korean"] == "편한"

    def test_nfd_korean_field_is_nfc_normalized(self, tmp_path):
        nfd_ga = unicodedata.normalize("NFD", "\uac00")
        db_path = tmp_path / "collection.anki2"
        _make_full_note_db_old_schema(
            db_path,
            deck_id=1001,
            deck_name="Korean",
            model_name="Korean Vocab",
            field_names=["Korean", "English"],
            notes=[(1, "g1", [nfd_ga, "x"])],
        )
        notes = load_anki_notes(db_path, "Korean")
        assert notes[0].fields["Korean"] == "\uac00"

    def test_unsupported_extension_returns_empty_list(self, tmp_path):
        bad = tmp_path / "x.db"
        bad.write_text("not a db", encoding="utf-8")
        assert load_anki_notes(bad, "Deck") == []

    def test_multiple_cards_per_note_deduped(self, tmp_path):
        """A note with multiple cards in the deck should only return once."""
        db_path = tmp_path / "collection.anki2"
        _make_full_note_db_old_schema(
            db_path,
            deck_id=1001,
            deck_name="Korean",
            model_name="Korean Vocab",
            field_names=["Korean", "English"],
            notes=[(1, "g1", ["편한", "x"])],
        )
        # Add a second card row for the same nid (Anki cards table is per-card)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO cards VALUES (?,?,?,0,0,0,0,0,0,0,0,0,0,0,0,0,0,'')",
            (11, 1, 1001),
        )
        conn.commit()
        conn.close()
        notes = load_anki_notes(db_path, "Korean")
        assert len(notes) == 1
