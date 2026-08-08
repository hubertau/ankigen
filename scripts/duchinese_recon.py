#!/usr/bin/env python3
"""
Reconnaissance for the planned `ankigen pull --source duchinese` feature.

Opens a real browser window, waits while you sign in to duchinese.net by hand,
then records what is needed to decide *how* the saved-word list should be read:

  * every network response the wordlist page makes (URL, status, JSON shape)
  * the rendered HTML of that page
  * repeated DOM structures containing Chinese text — i.e. candidate row selectors

The script never types into the login form and never asks for your password.
Network recording is cleared *after* you finish signing in and the page is then
reloaded, so the captured traffic covers the wordlist load only — none of the
login exchange is written to disk.

Session cookies are saved separately (mode 0600) to --state so later runs skip
the manual login. That file is NOT part of the shareable dump.

Usage:
    uv sync --extra web
    uv run playwright install chromium
    uv run python scripts/duchinese_recon.py

Then skim duchinese_recon/summary.md before sharing the folder.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

DEFAULT_START_URL = "https://duchinese.net/"

# Share the session file with `ankigen pull` — including its
# ANKIGEN_DUCHINESE_STATE override — so signing in here signs you in there,
# rather than the two silently writing different files.
try:
    from ankigen.duchinese import get_state_path

    DEFAULT_STATE = get_state_path()
except ModuleNotFoundError:  # running the script outside the installed package
    DEFAULT_STATE = Path.home() / ".config" / "ankigen" / "duchinese_state.json"

# Endpoints whose URL says "this might be the wordlist".
INTERESTING_URL_WORDS = (
    "word",
    "vocab",
    "flashcard",
    "card",
    "review",
    "srs",
    "saved",
    "study",
    "list",
    "deck",
)

# Query-string keys and JSON keys whose values are never worth keeping.
SECRET_WORDS = (
    "token",
    "password",
    "secret",
    "auth",
    "session",
    "jwt",
    "api_key",
    "apikey",
    "api-key",
    "credential",
    "signature",
    "email",
)

MAX_BODY_BYTES = 2_000_000

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_SECRET_KEY_RE = re.compile(
    r'("[^"]*(?:' + "|".join(SECRET_WORDS) + r')[^"]*"\s*:\s*)"[^"]*"',
    re.IGNORECASE,
)
# The JSON form above misses secrets embedded in inline <script> blocks and
# query strings — `auth_token=abc123`, `csrfToken: 'abc'` — which is exactly
# where SPAs bootstrap session state. Over-redacting here is the safe failure.
_SECRET_ASSIGN_RE = re.compile(
    r"((?:" + "|".join(SECRET_WORDS) + r")[\"']?\s*[=:]\s*[\"']?)([^\s\"'&<>;,)]{4,})",
    re.IGNORECASE,
)
_CJK_RE = re.compile(r"[一-鿿]")


@dataclass
class Capture:
    """One recorded HTTP response."""

    index: int
    url: str
    method: str
    status: int
    resource_type: str
    content_type: str
    body_bytes: int
    cjk_chars: int
    body_file: str | None = None
    shape: Any = None
    truncated: bool = False

    def score(self) -> int:
        """Rank how likely this response is to be the saved-word payload."""
        points = 0
        if "json" in self.content_type:
            points += 3
        if self.cjk_chars > 0:
            points += 3
        if self.cjk_chars > 100:
            points += 2
        lowered = self.url.lower()
        if any(word in lowered for word in INTERESTING_URL_WORDS):
            points += 2
        if self.body_bytes > 5000:
            points += 1
        return points


@dataclass
class Recorder:
    """Collects responses; cleared once before the clean wordlist reload."""

    extra_redactions: list[str] = field(default_factory=list)
    captures: list[Capture] = field(default_factory=list)
    bodies: dict[int, str] = field(default_factory=dict)
    enabled: bool = True
    # Strong references to in-flight handler tasks. asyncio keeps only a weak
    # reference to a running task, so without this a handler suspended on
    # `await response.body()` can be collected mid-flight and its capture lost
    # silently — including, potentially, the one response we are looking for.
    pending: set[asyncio.Task[None]] = field(default_factory=set)
    oversized: int = 0

    def track(self, task: asyncio.Task[None]) -> None:
        self.pending.add(task)
        task.add_done_callback(self.pending.discard)

    async def drain(self) -> None:
        """Wait for handlers already in flight to finish."""
        while self.pending:
            await asyncio.gather(*tuple(self.pending), return_exceptions=True)

    async def stop(self) -> None:
        """Stop recording and let in-flight handlers settle."""
        self.enabled = False
        await self.drain()

    def clear(self) -> None:
        self.captures.clear()
        self.bodies.clear()
        self.oversized = 0

    def redact(self, text: str) -> str:
        text = _EMAIL_RE.sub("<email redacted>", text)
        text = _SECRET_KEY_RE.sub(r'\1"<redacted>"', text)
        text = _SECRET_ASSIGN_RE.sub(r"\1<redacted>", text)
        for needle in self.extra_redactions:
            if needle:
                text = text.replace(needle, "<redacted>")
        return text

    def redact_url(self, url: str) -> str:
        parts = urlparse(url)
        if parts.query:
            pairs = [
                (k, "<redacted>" if any(w in k.lower() for w in SECRET_WORDS) else v)
                for k, v in parse_qsl(parts.query, keep_blank_values=True)
            ]
            parts = parts._replace(query=urlencode(pairs))
        return self.redact(urlunparse(parts))


def summarize_json(value: Any, depth: int = 0) -> Any:
    """Describe a JSON payload's *shape* — keys and types, not the data itself."""
    if depth >= 4:
        return "..."
    if isinstance(value, dict):
        return {k: summarize_json(v, depth + 1) for k, v in list(value.items())[:30]}
    if isinstance(value, list):
        if not value:
            return []
        return [f"<list of {len(value)}>", summarize_json(value[0], depth + 1)]
    if isinstance(value, str):
        return "str(cjk)" if _CJK_RE.search(value) else "str"
    if value is None:
        return "null"
    return type(value).__name__


# Groups every element holding Chinese text by tag+class, so a repeating row
# structure shows up as a high count without anyone guessing a selector first.
_DOM_CLUSTER_JS = """
() => {
  const CJK = /[\\u4e00-\\u9fff]/;
  const groups = new Map();
  for (const el of document.querySelectorAll('*')) {
    let own = '';
    for (const node of el.childNodes) {
      if (node.nodeType === 3) own += node.nodeValue;
    }
    own = own.trim();
    if (!own || !CJK.test(own)) continue;
    const cls = (el.getAttribute('class') || '').trim();
    const key = el.tagName.toLowerCase() + (cls ? '.' + cls.split(/\\s+/).join('.') : '');
    if (!groups.has(key)) {
      const chain = [];
      let p = el.parentElement;
      for (let i = 0; i < 3 && p; i++) {
        const pcls = (p.getAttribute('class') || '').trim();
        chain.push(p.tagName.toLowerCase() + (pcls ? '.' + pcls.split(/\\s+/).join('.') : ''));
        p = p.parentElement;
      }
      groups.set(key, { selector: key, count: 0, samples: [], ancestors: chain });
    }
    const rec = groups.get(key);
    rec.count += 1;
    if (rec.samples.length < 5) rec.samples.push(own.slice(0, 40));
  }
  return [...groups.values()].sort((a, b) => b.count - a.count).slice(0, 25);
}
"""

_CJK_ELEMENT_COUNT_JS = """
() => {
  const CJK = /[\\u4e00-\\u9fff]/;
  let n = 0;
  for (const el of document.querySelectorAll('*')) {
    for (const node of el.childNodes) {
      if (node.nodeType === 3 && CJK.test(node.nodeValue)) { n++; break; }
    }
  }
  return n;
}
"""

# Scrolls the window *and* the largest inner scrollable box. A wordlist panel
# that scrolls independently of the page is common, and scrolling only the
# window would silently capture just the first batch of words.
_SCROLL_STEP_JS = """
() => {
  window.scrollTo(0, document.body.scrollHeight);
  let best = null, bestArea = 0;
  for (const el of document.querySelectorAll('*')) {
    if (el.scrollHeight > el.clientHeight + 50 && el.clientHeight > 100) {
      const area = el.clientHeight * el.clientWidth;
      if (area > bestArea) { bestArea = area; best = el; }
    }
  }
  let container = null;
  if (best) {
    best.scrollTop = best.scrollHeight;
    const cls = (best.getAttribute('class') || '').trim();
    container = {
      selector: best.tagName.toLowerCase() + (cls ? '.' + cls.split(/\\s+/).join('.') : ''),
      scroll_height: best.scrollHeight,
    };
  }
  return { page_scroll_height: document.body.scrollHeight, container };
}
"""

# Buttons/links that would page in more words. Reported, never clicked.
_PAGINATION_JS = """
() => {
  const out = [];
  for (const el of document.querySelectorAll('button, a, [role=button]')) {
    const text = (el.innerText || '').trim();
    if (!text || text.length > 40) continue;
    if (!/more|next|load|show|page|\\u66f4\\u591a/i.test(text)) continue;
    const cls = (el.getAttribute('class') || '').trim();
    out.push({
      text,
      tag: el.tagName.toLowerCase(),
      selector: el.tagName.toLowerCase() + (cls ? '.' + cls.split(/\\s+/).join('.') : ''),
    });
  }
  return out.slice(0, 20);
}
"""


async def record_response(recorder: Recorder, response: Any) -> None:
    """Response handler — stores a redacted summary plus JSON/text bodies."""
    if not recorder.enabled:
        return
    try:
        headers = await response.all_headers()
    except Exception:
        headers = {}
    content_type = headers.get("content-type", "")
    body_text = ""
    oversized = False
    if any(kind in content_type for kind in ("json", "javascript", "text/plain")):
        try:
            raw = await response.body()
            if len(raw) <= MAX_BODY_BYTES:
                body_text = raw.decode("utf-8", errors="replace")
            else:
                # Do not let the cap hide the payload we are hunting for: a big
                # word list is *more* likely to be the answer, not less. Keep a
                # prefix so the shape and its Chinese content still score.
                oversized = True
                body_text = raw[:MAX_BODY_BYTES].decode("utf-8", errors="replace")
                recorder.oversized += 1
        except Exception:
            body_text = ""

    # A late handler must not land in the log after recording is stopped —
    # that is what would leak login traffic into a dump meant to be shared.
    if not recorder.enabled:
        return

    index = len(recorder.captures)
    capture = Capture(
        index=index,
        url=recorder.redact_url(response.url),
        method=response.request.method,
        status=response.status,
        resource_type=response.request.resource_type,
        content_type=content_type.split(";")[0],
        body_bytes=len(body_text.encode("utf-8")),
        cjk_chars=len(_CJK_RE.findall(body_text)),
    )

    if body_text:
        safe = recorder.redact(body_text)
        recorder.bodies[index] = safe
        if "json" in content_type:
            try:
                capture.shape = summarize_json(json.loads(safe))
            except (ValueError, TypeError):
                # Expected for a truncated oversized body — the prefix is not
                # valid JSON on its own, but still shows the fields.
                capture.shape = "<truncated json>" if oversized else "<unparseable json>"

    capture.truncated = oversized
    recorder.captures.append(capture)


async def autoscroll(page: Any, rounds: int, pause_ms: int) -> list[dict[str, Any]]:
    """Scroll to the bottom repeatedly so lazy-loaded words are fetched."""
    log: list[dict[str, Any]] = []
    stable = 0
    for step in range(rounds):
        cjk = await page.evaluate(_CJK_ELEMENT_COUNT_JS)
        scrolled = await page.evaluate(_SCROLL_STEP_JS)
        entry: dict[str, Any] = {
            "step": step,
            "cjk_elements": cjk,
            "page_scroll_height": scrolled["page_scroll_height"],
            "container": scrolled["container"],
        }
        log.append(entry)
        where = scrolled["container"]["selector"] if scrolled["container"] else "window"
        print(f"   scroll {step + 1}/{rounds}: {cjk} elements with Chinese (scrolled {where})")

        if step and _scroll_state(entry) == _scroll_state(log[-2]):
            stable += 1
            if stable >= 2:
                print("   content stopped growing")
                break
        else:
            stable = 0

        await page.keyboard.press("End")
        await page.wait_for_timeout(pause_ms)
    return log


def _scroll_state(entry: dict[str, Any]) -> tuple[int, int, int]:
    """The triple that has to stop changing before scrolling is done."""
    container = entry.get("container") or {}
    return (
        entry["cjk_elements"],
        entry["page_scroll_height"],
        int(container.get("scroll_height", 0)),
    )


def write_summary(out_dir: Path, page_url: str, recorder: Recorder, probes: dict[str, Any]) -> None:
    """Write the human-readable digest that gets read first."""
    ranked = sorted(recorder.captures, key=lambda c: (-c.score(), -c.cjk_chars))
    lines = [
        "# DuChinese recon",
        "",
        f"Page: `{page_url}`",
        f"Responses recorded: {len(recorder.captures)} (login traffic excluded)",
        "",
        "## Candidate wordlist endpoints",
        "",
        "Ranked by JSON-ness, Chinese content, and URL keywords.",
        "",
        "| score | status | type | Chinese chars | bytes | URL |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for capture in ranked[:12]:
        note = " (truncated)" if capture.truncated else ""
        lines.append(
            f"| {capture.score()} | {capture.status} | {capture.content_type} | "
            f"{capture.cjk_chars} | {capture.body_bytes}{note} | `{capture.url[:110]}` |"
        )

    if recorder.oversized:
        lines += [
            "",
            f"> {recorder.oversized} response(s) exceeded the {MAX_BODY_BYTES:,}-byte capture "
            "cap and were stored as a prefix. A large word list is a *likelier* candidate, "
            "not a weaker one — check those first.",
        ]

    top = ranked[0] if ranked else None
    lines += [
        "",
        "**Read this as:** a top entry with JSON content type and a high Chinese-character",
        "count means the wordlist arrives as JSON and can be read straight off the network,",
        "no DOM selectors involved. If nothing scores well, the page is server-rendered or",
        "hydrated from inline state and the DOM clusters below are the way in.",
        "",
    ]
    if top is not None and top.score() >= 6:
        lines.append(f"Best candidate: `{top.url}` — body saved as `bodies/{top.index:03d}.json`.")
    else:
        lines.append("No strong JSON candidate; expect to scrape the DOM.")

    clusters = probes.get("dom_clusters", [])
    lines += [
        "",
        "## Repeated DOM structures containing Chinese",
        "",
        "| count | selector | samples |",
        "| --- | --- | --- |",
    ]
    for cluster in clusters[:12]:
        samples = ", ".join(cluster.get("samples", [])[:3])
        lines.append(f"| {cluster['count']} | `{cluster['selector']}` | {samples} |")

    pagination = probes.get("pagination_controls", [])
    if pagination:
        lines += ["", "## Possible pagination controls (not clicked)", ""]
        for control in pagination:
            lines.append(f"- `{control['selector']}` — {control['text']!r}")

    lines += [
        "",
        "## Files",
        "",
        "- `network.json` — every recorded response, redacted",
        "- `bodies/` — JSON/text payloads, redacted",
        "- `page.html` — rendered DOM at snapshot time",
        "- `dom_probe.json` — clusters, pagination controls, scroll log",
        "",
        "## Before sharing",
        "",
        "Emails and secret-looking JSON keys are redacted automatically, but skim",
        "`page.html` for anything personal (display name, subscription details).",
        "Re-run with `--redact 'Your Name'` to scrub extra strings.",
        "The session/cookie file lives outside this folder and should not be shared.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run(args: argparse.Namespace) -> int:
    try:
        from playwright.async_api import async_playwright
    except ModuleNotFoundError:
        print(
            "Playwright is not installed. From the repo root:\n"
            "    uv sync --extra web\n"
            "    uv run playwright install chromium",
            file=sys.stderr,
        )
        return 1

    out_dir: Path = args.out
    (out_dir / "bodies").mkdir(parents=True, exist_ok=True)
    state_path: Path = args.state
    recorder = Recorder(extra_redactions=list(args.redact or []))

    async with async_playwright() as pw:
        browser_path = args.browser_path or os.getenv("ANKIGEN_CHROMIUM_PATH")
        browser = await pw.chromium.launch(
            headless=args.headless,
            executable_path=browser_path or None,
        )
        storage = str(state_path) if state_path.exists() else None
        if storage:
            print(f"Reusing saved session from {state_path}")
        context = await browser.new_context(storage_state=storage)
        page = await context.new_page()
        page.on(
            "response",
            lambda response: recorder.track(
                asyncio.create_task(record_response(recorder, response))
            ),
        )

        await page.goto(args.start_url, wait_until="domcontentloaded")

        print(
            "\nA browser window is open.\n"
            "  1. Sign in to Du Chinese (this script never touches the login form).\n"
            "  2. Navigate to your saved words / flashcards page.\n"
            "  3. Come back here and press Enter.\n"
        )
        await asyncio.to_thread(input, "Press Enter once the wordlist is on screen... ")

        state_path.parent.mkdir(parents=True, exist_ok=True)
        # Create the file at 0600 before writing, so the cookie jar is never
        # briefly world-readable under the usual umask.
        os.close(os.open(state_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600))
        await context.storage_state(path=str(state_path))
        print(f"Session saved to {state_path} (mode 0600, not part of the dump)")

        # Everything captured so far includes the login exchange. Stop
        # recording, let handlers already in flight settle, and only then drop
        # what they wrote — otherwise a late handler appends login traffic into
        # a log that is about to be written out and shared.
        await recorder.stop()
        recorder.clear()
        recorder.enabled = True
        page_url = page.url
        print(f"\nReloading {page_url} with a clean network log...")
        await page.reload(wait_until="networkidle")

        print("Scrolling to pull in lazily-loaded words...")
        scroll_log = await autoscroll(page, args.scroll_rounds, args.scroll_pause)

        print("Snapshotting the DOM...")
        probes: dict[str, Any] = {
            "page_url": page_url,
            "dom_clusters": await page.evaluate(_DOM_CLUSTER_JS),
            "pagination_controls": await page.evaluate(_PAGINATION_JS),
            "scroll_log": scroll_log,
            "cjk_elements_final": await page.evaluate(_CJK_ELEMENT_COUNT_JS),
        }
        html = recorder.redact(await page.content())
        if args.screenshot:
            await page.screenshot(path=str(out_dir / "page.png"), full_page=True)

        # Let any handler still reading a body finish before the page closes,
        # so a slow response is not dropped from the dump.
        await recorder.stop()

        await context.close()
        await browser.close()

    (out_dir / "page.html").write_text(html, encoding="utf-8")
    (out_dir / "dom_probe.json").write_text(
        json.dumps(probes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Record where each body landed *before* serialising the index, otherwise
    # every body_file in network.json is null and the URL→payload mapping is
    # lost.
    for index, body in recorder.bodies.items():
        recorder.captures[index].body_file = f"bodies/{index:03d}.json"
        (out_dir / "bodies" / f"{index:03d}.json").write_text(body, encoding="utf-8")

    (out_dir / "network.json").write_text(
        json.dumps(
            [capture.__dict__ for capture in recorder.captures], ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    write_summary(out_dir, probes["page_url"], recorder, probes)

    print(f"\nDone. {len(recorder.captures)} responses recorded.")
    print(f"Read {out_dir / 'summary.md'} first, then share the {out_dir} folder.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record how duchinese.net serves your saved word list.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("duchinese_recon"),
        help="Output folder for the dump (default: duchinese_recon/)",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=DEFAULT_STATE,
        help=f"Where to save the browser session (default: {DEFAULT_STATE})",
    )
    parser.add_argument(
        "--start-url",
        type=str,
        default=DEFAULT_START_URL,
        help=f"Page to open first (default: {DEFAULT_START_URL})",
    )
    parser.add_argument(
        "--redact",
        action="append",
        metavar="TEXT",
        help="Extra literal string to scrub from the dump (repeatable)",
    )
    parser.add_argument(
        "--scroll-rounds",
        type=int,
        default=30,
        help="Maximum scroll-to-bottom passes (default: 30)",
    )
    parser.add_argument(
        "--scroll-pause",
        type=int,
        default=1200,
        help="Milliseconds to wait after each scroll (default: 1200)",
    )
    parser.add_argument(
        "--screenshot",
        action="store_true",
        help="Also save a full-page screenshot (may show your account details)",
    )
    parser.add_argument(
        "--browser-path",
        type=str,
        default=None,
        metavar="PATH",
        help="Use an existing Chromium binary instead of Playwright's own "
        "(also settable via ANKIGEN_CHROMIUM_PATH), skipping the ~150MB download.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without a visible window. Only useful once --state holds a "
        "valid session, since you cannot sign in to an invisible browser.",
    )
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
