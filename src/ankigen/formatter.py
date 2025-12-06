"""HTML formatting for vocabulary sentences."""

import re


def format_sentences(text: str, keyword: str) -> str:
    """
    Takes numbered sentences and formats them as inline HTML.

    Args:
        text: Input text with numbered sentences (e.g., "1. 句子一 2. 句子二")
        keyword: The word to highlight in red

    Returns:
        HTML string with blue text and red keyword, separated by <br>
    """
    # Remove sentence numbers (e.g., "1. ", "2. ", etc.)
    # Split by number-period pattern
    sentences = re.split(r"\d+\.\s*", text)

    # Filter out empty strings
    sentences = [s.strip() for s in sentences if s.strip()]

    # Format each sentence
    formatted = []
    for sentence in sentences:
        # Replace keyword with red-styled span, rest stays blue
        highlighted = sentence.replace(
            keyword,
            f'</span><span style="color: red;">{keyword}</span><span style="color: blue;">',
        )
        # Wrap in blue span
        formatted.append(f'<span style="color: blue;">{highlighted}</span>')

    # Join with <br>
    result = "<br>".join(formatted)

    # Clean up empty spans that might occur
    result = result.replace('<span style="color: blue;"></span>', "")

    return result

