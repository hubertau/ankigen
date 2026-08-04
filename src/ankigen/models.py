"""Pydantic models for structured LLM responses."""

from functools import cache

from pydantic import BaseModel, Field

# Shared by the sentence-plus-notes response and the notes-only response, so a
# card gets the same notes whether they arrived with generated sentences or
# were backfilled on their own.
_NOTES_FIELD_DESCRIPTION = (
    "Short English usage notes for the target word: closest "
    "similar or easily-confused words and how they differ, the "
    "register the word belongs to (written vs spoken, formal vs "
    "casual), and any collocation quirk worth remembering. "
    "Return an empty string when there is nothing useful to add."
)


@cache
def create_sentence_response(
    num_sentences: int = 3,
    with_notes: bool = False,
) -> type[BaseModel]:
    """Factory to create a SentenceResponse model with dynamic sentence count.

    ``with_notes`` adds a free-form ``notes`` field for learner context
    (confusable words, register, collocation quirks). Left off for
    :func:`ankigen.llm.remark_sentences`, which only re-marks existing text.
    """
    if not with_notes:

        class SentenceResponse(BaseModel):
            """Response model for sentence generation."""

            sentences: list[str] = Field(
                ...,
                min_length=num_sentences,
                max_length=num_sentences,
                description=f"Exactly {num_sentences} example sentences using the target word",
            )

        return SentenceResponse

    class SentenceResponseWithNotes(BaseModel):
        """Response model for sentence generation plus learner context notes."""

        sentences: list[str] = Field(
            ...,
            min_length=num_sentences,
            max_length=num_sentences,
            description=f"Exactly {num_sentences} example sentences using the target word",
        )
        notes: str = Field(
            default="",
            description=_NOTES_FIELD_DESCRIPTION,
        )

    SentenceResponseWithNotes.__name__ = "SentenceResponse"
    SentenceResponseWithNotes.__qualname__ = "SentenceResponse"
    return SentenceResponseWithNotes


class NotesResponse(BaseModel):
    """Response model for generating learner context notes on their own.

    Used when a card's example sentences are already fine and only the
    context-notes block is missing, so there is no sentence generation to
    carry the notes back for free.
    """

    notes: str = Field(
        default="",
        description=_NOTES_FIELD_DESCRIPTION,
    )


# ---------------------------------------------------------------------------
# Content review
# ---------------------------------------------------------------------------


class SentenceVerdict(BaseModel):
    """One sentence's content-review result."""

    index: int = Field(
        ...,
        description="1-based position of the sentence in the list that was reviewed",
    )
    ok: bool = Field(
        ...,
        description=(
            "True when the sentence is grammatical, natural, and uses the target "
            "word with the stated meaning. False only for a clear, describable defect."
        ),
    )
    issue: str = Field(
        default="",
        description=(
            "Short description of the defect when `ok` is false (a few words, e.g. "
            "'wrong particle' or 'uses the noun sense, gloss is the verb'). "
            "Empty string when `ok` is true."
        ),
    )


class ContentReviewResponse(BaseModel):
    """Response model for reviewing a card's example sentences."""

    verdicts: list[SentenceVerdict] = Field(
        default_factory=list,
        description="Exactly one verdict per reviewed sentence, in the same order.",
    )


class TranslationResponse(BaseModel):
    """Response model for word translation."""

    translation: str = Field(
        ...,
        description="English translation of the word, including part of speech and multiple meanings if applicable",
    )


class KoreanTranslationResponse(BaseModel):
    """Response model for Korean word translation that also returns Hanja."""

    translation: str = Field(
        ...,
        description="English translation of the Korean word, including part of speech and multiple meanings if applicable",
    )
    hanja: str = Field(
        default="",
        description=(
            "The canonical Hanja (Chinese-character) form of the word if it is "
            "Sino-Korean. Use the most common single Hanja spelling without spaces. "
            "Return an empty string for native-Korean words that have no Hanja."
        ),
    )


# ---------------------------------------------------------------------------
# Grammar models
# ---------------------------------------------------------------------------


class GrammarExample(BaseModel):
    """A single example sentence for a grammatical construction."""

    target: str = Field(
        ...,
        description=(
            "The example sentence in the target language (e.g., Korean or Chinese), "
            "with the part that realises the grammar pattern wrapped in double "
            "asterisks (e.g. '한국어를 잘**하게 되었어요**.')"
        ),
    )
    english: str = Field(
        default="",
        description="English translation of the example sentence (may be empty if not provided)",
    )


class GrammarItem(BaseModel):
    """A single grammar/construction item extracted from teacher notes."""

    pattern: str = Field(
        ...,
        description=(
            "The grammatical pattern/construction itself, in the target language "
            "(e.g. '~게 되다', '에 + 씩'). Use the canonical form a learner would "
            "look up; do NOT include English translations in this field."
        ),
    )
    meaning: str = Field(
        ...,
        description=(
            "Short English gloss of what the pattern means / how it is used "
            "(1 sentence, like a part-of-speech-style note)."
        ),
    )
    explanation: str = Field(
        default="",
        description=(
            "1-3 sentence usage notes. Prefer copying the teacher's notes "
            "verbatim if the source document explains the pattern."
        ),
    )
    hanja: str = Field(
        default="",
        description=(
            "Optional Hanja annotation for any Sino-Korean roots inside the "
            "pattern (e.g. '博士 課程 中' for '박사 과정 중'). Leave empty when "
            "the pattern contains no Sino-Korean noun roots or has no canonical "
            "Hanja form. Only populated for Korean."
        ),
    )
    examples: list[GrammarExample] = Field(
        default_factory=list,
        description=(
            "Example sentences from the teacher's notes (verbatim). "
            "Each example pairs a target-language sentence with an English translation "
            "if the doc provides one."
        ),
    )


class GrammarExtractionResponse(BaseModel):
    """Response model for extracting grammar items from a document."""

    items: list[GrammarItem] = Field(
        default_factory=list,
        description="Distinct grammatical constructions found in the document.",
    )


@cache
def create_grammar_example_response(num_examples: int = 3) -> type[BaseModel]:
    """Factory to create a GrammarExampleResponse model with a fixed example count."""

    class GrammarExampleResponse(BaseModel):
        """Response model for top-up generation of grammar example sentences."""

        examples: list[GrammarExample] = Field(
            ...,
            min_length=num_examples,
            max_length=num_examples,
            description=f"Exactly {num_examples} example sentences for the grammar pattern",
        )

    return GrammarExampleResponse
