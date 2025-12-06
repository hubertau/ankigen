"""Pydantic models for structured LLM responses."""

from pydantic import BaseModel, Field


class SentenceResponse(BaseModel):
    """Response model for sentence generation."""

    sentences: list[str] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Exactly 3 example sentences using the target word",
    )


class TranslationResponse(BaseModel):
    """Response model for word translation."""

    translation: str = Field(
        ...,
        description="English translation of the word, including part of speech and multiple meanings if applicable",
    )

