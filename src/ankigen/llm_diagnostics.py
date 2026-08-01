"""Connectivity diagnostics for LLM provider / network failures."""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

logger = logging.getLogger("ankigen.llm_diagnostics")

Provider = str

_PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "anthropic": "https://api.anthropic.com",
    "local": "http://localhost:11434/v1",
    "deepseek": "https://api.deepseek.com/v1",
}

_PROBE_TIMEOUT_SEC = 10.0


@dataclass(frozen=True, slots=True)
class DiagnosticProbe:
    """Result of a single connectivity check."""

    name: str
    ok: bool
    detail: str
    latency_ms: float | None = None


def _configured_provider() -> tuple[str, str | None]:
    """Return ``(provider, invalid_raw_value)`` for the configured provider.

    Unlike :func:`ankigen.llm.get_provider` this never raises — ``llm-check`` is
    the tool you reach for *because* something is misconfigured, so a bad value
    is reported as a failed probe and the remaining probes still run against the
    default so the output stays useful.
    """
    raw = os.getenv("LLM_PROVIDER", "openai")
    provider = raw.strip().lower()
    if provider not in _PROVIDER_BASE_URLS:
        return "openai", raw
    return provider, None


def _get_provider() -> str:
    return _configured_provider()[0]


def _provider_base_url(provider: str) -> str:
    return os.getenv("LLM_BASE_URL") or _PROVIDER_BASE_URLS[provider]


def _api_key_for_probe(provider: str) -> str:
    key = os.getenv("LLM_API_KEY", "").strip()
    if provider == "local" and not key:
        return ""
    return key


def _classify_exception(exc: BaseException) -> str:
    """Short label for the failure mode."""
    name = exc.__class__.__name__
    if name == "ValidationError":
        return "invalid_json"
    message = str(exc).lower()
    if "timeout" in name.lower() or "timed out" in message:
        return "timeout"
    if "connection" in name.lower() or "connection error" in message:
        return "connection"
    if "ssl" in message or "certificate" in message:
        return "tls"
    if "401" in message or "unauthorized" in message:
        return "auth"
    if "429" in message or "rate limit" in message:
        return "rate_limit"
    if "name or service not known" in message or "nodename" in message:
        return "dns"
    return "unknown"


def _probe_dns(host: str) -> DiagnosticProbe:
    if not host:
        return DiagnosticProbe("dns", False, "no host in API base URL")
    start = time.perf_counter()
    try:
        socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        ms = (time.perf_counter() - start) * 1000
        return DiagnosticProbe("dns", True, f"resolved {host}", latency_ms=ms)
    except OSError as exc:
        ms = (time.perf_counter() - start) * 1000
        return DiagnosticProbe("dns", False, f"{host}: {exc}", latency_ms=ms)


def _probe_models_endpoint(base_url: str, api_key: str) -> DiagnosticProbe:
    url = base_url.rstrip("/") + "/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = Request(url, headers=headers, method="GET")
    start = time.perf_counter()
    try:
        with urlopen(req, timeout=_PROBE_TIMEOUT_SEC) as resp:
            ms = (time.perf_counter() - start) * 1000
            body = resp.read(512).decode("utf-8", errors="replace")
            status = getattr(resp, "status", 200)
            detail = f"HTTP {status}"
            if status == 200 and body.strip().startswith("{"):
                try:
                    data = json.loads(body)
                    if isinstance(data, dict) and "data" in data:
                        n = len(data["data"]) if isinstance(data["data"], list) else "?"
                        detail = f"HTTP {status}, models list ok ({n} entries in sample)"
                except json.JSONDecodeError:
                    detail = f"HTTP {status}, JSON body"
            return DiagnosticProbe("api_reachable", True, detail, latency_ms=ms)
    except HTTPError as exc:
        ms = (time.perf_counter() - start) * 1000
        hint = "check LLM_API_KEY" if exc.code in (401, 403) else ""
        detail = f"HTTP {exc.code} {exc.reason}"
        if hint:
            detail += f" ({hint})"
        return DiagnosticProbe("api_reachable", False, detail, latency_ms=ms)
    except URLError as exc:
        ms = (time.perf_counter() - start) * 1000
        reason = getattr(exc, "reason", exc)
        return DiagnosticProbe("api_reachable", False, str(reason), latency_ms=ms)
    except OSError as exc:
        ms = (time.perf_counter() - start) * 1000
        return DiagnosticProbe("api_reachable", False, str(exc), latency_ms=ms)


def run_llm_diagnostics(*, provider: str | None = None) -> list[DiagnosticProbe]:
    """Run probes against the configured (or given) LLM provider."""
    invalid_raw: str | None = None
    if provider is None:
        provider, invalid_raw = _configured_provider()
    base_url = _provider_base_url(provider)
    api_key = _api_key_for_probe(provider)
    host = urlparse(base_url).hostname or ""

    probes: list[DiagnosticProbe] = []

    if invalid_raw is not None:
        probes.append(
            DiagnosticProbe(
                "provider",
                False,
                f"Unknown LLM_PROVIDER={invalid_raw!r} — valid: "
                f"{', '.join(sorted(_PROVIDER_BASE_URLS))}. "
                "Probing openai below; generate/extract will refuse to run.",
            )
        )
    else:
        probes.append(DiagnosticProbe("provider", True, provider))

    if provider != "local" and not api_key:
        probes.append(DiagnosticProbe("api_key", False, "LLM_API_KEY is not set"))
    else:
        probes.append(DiagnosticProbe("api_key", True, "LLM_API_KEY is set"))

    probes.append(
        DiagnosticProbe(
            "config",
            True,
            f"provider={provider} base_url={base_url} model env={os.getenv('LLM_MODEL') or '(default)'}",
        )
    )

    if provider == "anthropic":
        probes.append(
            DiagnosticProbe(
                "api_reachable",
                True,
                "Anthropic uses a separate SDK; run a small generate/extract call to verify",
            )
        )
        return probes

    probes.append(_probe_dns(host))
    probes.append(_probe_models_endpoint(base_url, api_key))
    return probes


def format_diagnostics_report(
    probes: list[DiagnosticProbe],
    *,
    exc: BaseException | None = None,
) -> list[str]:
    """Human-readable lines for console or logs."""
    lines: list[str] = []
    if exc is not None:
        from ankigen.llm import format_llm_error

        lines.append(f"Failure: {format_llm_error(exc)} (class={exc.__class__.__name__})")
        lines.append(f"Likely cause: {_classify_exception(exc)}")
    for probe in probes:
        status = "ok" if probe.ok else "FAIL"
        latency = f" ({probe.latency_ms:.0f} ms)" if probe.latency_ms is not None else ""
        lines.append(f"  [{status}] {probe.name}: {probe.detail}{latency}")
    lines.extend(_hints_for_failure(exc, probes))
    return lines


def _hints_for_failure(
    exc: BaseException | None,
    probes: list[DiagnosticProbe],
) -> list[str]:
    hints: list[str] = []
    by_name = {p.name: p for p in probes}

    if exc is not None:
        kind = _classify_exception(exc)
        if kind == "timeout":
            hints.append(
                "Hint: try raising ANKIGEN_LLM_TIMEOUT_SEC or use a faster model "
                "(e.g. deepseek-v4-flash)."
            )
        elif kind == "connection":
            hints.append(
                "Hint: connection errors are often VPN/firewall/DNS or provider outage — "
                "run `ankigen llm-check` and retry when api_reachable is ok."
            )
        elif kind == "auth":
            hints.append("Hint: verify LLM_API_KEY and LLM_BASE_URL match your provider.")
        elif kind == "invalid_json":
            hints.append(
                "Hint: the model returned JSON that does not match the expected schema — "
                "see the 'Invalid JSON snippet' / 'LLM raw response' lines above. "
                "Try deepseek-v4-flash or tighten the prompt."
            )

    if not by_name.get("dns", DiagnosticProbe("dns", True, "")).ok:
        hints.append("Hint: DNS failed — check network or try another resolver.")
    api_probe = by_name.get("api_reachable")
    if api_probe is not None and not api_probe.ok:
        hints.append(
            "Hint: API endpoint unreachable — confirm DeepSeek/status, proxy settings, "
            "and that LLM_BASE_URL is correct."
        )
    if not hints:
        hints.append(
            "Hint: run `ankigen llm-check` for a full connectivity report; "
            "re-run extract to resume from staging checkpoints."
        )
    return hints


def log_llm_failure_diagnostics(exc: BaseException) -> None:
    """Emit connectivity diagnostics after a failed LLM call (INFO/WARNING)."""
    try:
        probes = run_llm_diagnostics()
    except Exception as diag_exc:  # noqa: BLE001
        logger.warning("Could not run LLM diagnostics: %s", diag_exc)
        return
    for line in format_diagnostics_report(probes, exc=exc):
        if line.startswith("  [FAIL]") or line.startswith("Failure:"):
            logger.warning("LLM diagnostics: %s", line)
        else:
            logger.info("LLM diagnostics: %s", line)
