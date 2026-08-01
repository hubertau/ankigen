"""Shared test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _no_llm_retry_backoff(monkeypatch):
    """Make any LLM call that escapes mocking fail fast instead of sleeping.

    The suite is fully mocked, but it is easy to add a code path that reaches a
    *different* LLM helper than the one a test patched. When that happens the
    call hits the network, fails, and the production retry policy backs off for
    5 + 15 + 45 + 90 seconds — turning a leak into a three-minute test rather
    than an obvious one. (This is exactly how four backfill tests quietly grew
    to ~190s each.)

    Disabling retries and shortening the timeout does not hide the leak — the
    call still fails and the assertion still tells you why — it just keeps the
    cost proportionate.
    """
    monkeypatch.setenv("ANKIGEN_LLM_MAX_RETRIES", "0")
    monkeypatch.setenv("ANKIGEN_LLM_TIMEOUT_SEC", "5")


@pytest.fixture
def sample_chinese_words():
    """Sample Chinese vocabulary words for testing."""
    return ["促使", "归纳", "披露"]


@pytest.fixture
def sample_korean_words():
    """Sample Korean vocabulary words for testing."""
    return ["편한", "추천", "방향"]


@pytest.fixture
def mock_sentences_zh():
    """Mock Chinese sentences for a word."""
    return [
        "他的成功促使我更加努力地工作。",
        "这些政策的调整，促使了经济的快速发展。",
        "父母的鼓励促使他下定决心出国留学。",
    ]


@pytest.fixture
def mock_sentences_ko():
    """Mock Korean sentences for a word."""
    return [
        "이 의자는 정말 편한 것 같아요.",
        "편한 옷을 입고 오세요.",
        "집에서 편한 시간을 보냈습니다.",
    ]


@pytest.fixture
def mock_translation_zh():
    """Mock Chinese translation."""
    return "Verb: to urge, to spur, to prompt"


@pytest.fixture
def mock_translation_ko():
    """Mock Korean translation."""
    return "Adjective: comfortable, relaxed, easy"
