"""Pydantic models for structured LLM responses."""

from pydantic import BaseModel, Field


def create_sentence_response(num_sentences: int = 3) -> type[BaseModel]:
    """Factory to create a SentenceResponse model with dynamic sentence count."""

    class SentenceResponse(BaseModel):
        """Response model for sentence generation."""

        sentences: list[str] = Field(
            ...,
            min_length=num_sentences,
            max_length=num_sentences,
            description=f"Exactly {num_sentences} example sentences using the target word",
        )

    return SentenceResponse


class TranslationResponse(BaseModel):
    """Response model for word translation."""

    translation: str = Field(
        ...,
        description="English translation of the word, including part of speech and multiple meanings if applicable",
    )

