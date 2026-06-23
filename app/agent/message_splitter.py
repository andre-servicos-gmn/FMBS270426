"""Sprint 1.6 — split LLM outputs into WhatsApp-sized message blocks.

The recommend and pitch-consultoria nodes ask the LLM to return JSON of the
shape ``{"messages": ["block 1", "block 2", ...]}``. When the LLM follows the
contract, ``parse_messages`` returns the list verbatim. When it doesn't (plain
string output, malformed JSON), we fall back to splitting by paragraph and
then by sentence so the customer still gets a humanized cadence.

``compute_typing_delay`` picks a 1.0–3.0 second pause between blocks scaled
by message length and with mild randomness, mimicking a person typing.
"""
from __future__ import annotations

import json
import logging
import random
import re

logger = logging.getLogger(__name__)

# Soft caps on block size — keep blocks WhatsApp-readable, never push past 500.
_MIN_BLOCK_CHARS = 1
_MAX_BLOCK_CHARS = 500
# When greedy-packing sentences into blocks we aim for this target size so
# long single paragraphs actually break into multiple WhatsApp-sized blocks
# instead of being collected into one 500-char wall.
_TARGET_BLOCK_CHARS = 250
# A "short" response we don't try to split — keep as a single block.
_SHORT_TOTAL_THRESHOLD = 200
# When falling back from a long single string, prefer paragraphs but force
# sentence split for any paragraph beyond this length.
_PARAGRAPH_FORCE_SENTENCE_SPLIT = 300

# Sentence boundary: split after .!?… + whitespace before an uppercase/number
# start — BUT never right after a list-item marker ("3." / "10)") at the start
# of a line. The negative lookbehind on a short digit-run+dot stops the splitter
# from treating an item number as a sentence end (the production bug: a numbered
# product list broke into balloons mid-item, e.g. "...459,00\n3." | "Raquete…").
_SENTENCE_BOUNDARY = re.compile(
    r"(?<![\d])(?<=[.!?…])\s+(?=[A-Z0-9ÀÁÂÃÉÊÍÓÔÕÚÇ])"
)
_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n+")
# A fragment that is ONLY a list-item marker ("4.", "10)", "-", "•") with no
# content — the sentence splitter must never leave this orphaned at a block
# end; it gets glued back to the following fragment (the actual item).
_ORPHAN_LIST_MARKER = re.compile(r"^\s*(?:\d{1,2}[.)]|[-•*])\s*$")
# A line that begins with a list-item marker — used to detect "this response is
# a list" so we pack whole items into blocks instead of sentence-splitting them.
_LIST_LINE = re.compile(r"^\s*(?:\d{1,2}[.)]|[-–—•*])\s+\S")


def parse_messages(llm_output) -> list[str]:
    """Return a list of message blocks for WhatsApp delivery.

    Args:
        llm_output: One of —
            - ``dict`` already containing a ``messages`` key (list of strings)
            - ``list[str]`` (a pre-split list)
            - ``str`` that is JSON ``{"messages": [...]}`` (LLM JSON mode)
            - any other ``str`` (plain text — falls back to paragraph/sentence split)

    Returns:
        Non-empty list of strings, each between 1 and 500 chars. Short total
        inputs return a single-block list.
    """
    if llm_output is None:
        return []

    # Case 1 — dict already
    if isinstance(llm_output, dict):
        return _clean_blocks(llm_output.get("messages") or [])

    # Case 2 — list already
    if isinstance(llm_output, list):
        return _clean_blocks(llm_output)

    # Case 3 — string. Try JSON first.
    if isinstance(llm_output, str):
        stripped = llm_output.strip()
        if not stripped:
            return []

        # Quick heuristic: only attempt JSON parse if it looks like JSON.
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and "messages" in parsed:
                return _clean_blocks(parsed["messages"])
            if isinstance(parsed, list):
                return _clean_blocks(parsed)

        # Fallback — plain string. Split if long.
        return _fallback_split(stripped)

    # Unknown type — coerce to string then fallback.
    return _fallback_split(str(llm_output))


def _clean_blocks(items) -> list[str]:
    """Coerce to strings, trim, enforce 1..500 chars per block, drop empties."""
    out: list[str] = []
    for item in items:
        if item is None:
            continue
        s = str(item).strip()
        if not s:
            continue
        if len(s) > _MAX_BLOCK_CHARS:
            # Bigger than allowed — sentence-split this single oversize block.
            out.extend(_split_by_sentence(s))
        else:
            out.append(s)
    return out


def _looks_like_list(text: str) -> bool:
    """True when the response is mostly list lines (2+ items) — a product list.
    Such responses must be packed by whole items, never sentence-split."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    list_lines = [ln for ln in lines if _LIST_LINE.match(ln)]
    return len(list_lines) >= 2


def _split_list(text: str) -> list[str]:
    """Pack a list response into blocks WITHOUT ever cutting a line in half.

    A leading intro line (before the first item) stays glued to the first item.
    Items are packed greedily up to the target size; a block boundary only ever
    falls BETWEEN whole lines.
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    blocks: list[str] = []
    current = ""
    for ln in lines:
        candidate = f"{current}\n{ln}" if current else ln
        if current and len(candidate) > _TARGET_BLOCK_CHARS:
            blocks.append(current)
            current = ln
        else:
            current = candidate
    if current:
        blocks.append(current)
    return _cap_block_count(blocks)


def _fallback_split(text: str) -> list[str]:
    """Split a single plain-text response into 1–4 blocks heuristically."""
    if len(text) <= _SHORT_TOTAL_THRESHOLD:
        return [text]

    # A numbered/bulleted product list: pack whole items, never cut mid-item.
    if _looks_like_list(text):
        return _split_list(text)

    # First try paragraph boundaries (\n\n or more)
    paragraphs = [p.strip() for p in _PARAGRAPH_BOUNDARY.split(text) if p.strip()]

    if len(paragraphs) >= 2:
        # Some paragraphs may still be too long — sentence-split those.
        blocks: list[str] = []
        for p in paragraphs:
            if len(p) > _PARAGRAPH_FORCE_SENTENCE_SPLIT:
                blocks.extend(_split_by_sentence(p))
            else:
                blocks.append(p)
        return _cap_block_count(blocks)

    # No paragraphs — fall back to sentence split.
    return _cap_block_count(_split_by_sentence(text))


def _split_by_sentence(text: str) -> list[str]:
    """Split a single string at sentence boundaries.

    Packs sentences greedily, but flushes the current block as soon as it
    crosses ``_TARGET_BLOCK_CHARS`` (≈250 chars) so a 400-500 char paragraph
    breaks into 2 blocks instead of collapsing into one. Hard cap is
    ``_MAX_BLOCK_CHARS``.
    """
    raw_sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
    if not raw_sentences:
        return [text[:_MAX_BLOCK_CHARS]]

    # Glue orphaned list markers ("4.", "10)", "-") back onto the next fragment
    # so a sentence split never separates a number from its item.
    sentences: list[str] = []
    pending_marker = ""
    for s in raw_sentences:
        if _ORPHAN_LIST_MARKER.match(s):
            pending_marker = (pending_marker + " " + s).strip() if pending_marker else s
            continue
        if pending_marker:
            sentences.append(f"{pending_marker} {s}".strip())
            pending_marker = ""
        else:
            sentences.append(s)
    if pending_marker:  # trailing orphan with nothing after — keep it.
        sentences.append(pending_marker)

    blocks: list[str] = []
    current = ""
    for sent in sentences:
        if len(sent) > _MAX_BLOCK_CHARS:
            # Single sentence too long — hard-cut.
            if current:
                blocks.append(current.strip())
                current = ""
            blocks.append(sent[:_MAX_BLOCK_CHARS])
            continue
        if not current:
            current = sent
            # Single sentence already crossed the target — flush it.
            if len(current) >= _TARGET_BLOCK_CHARS:
                blocks.append(current.strip())
                current = ""
        elif len(current) + 1 + len(sent) <= _TARGET_BLOCK_CHARS:
            current = f"{current} {sent}"
        else:
            blocks.append(current.strip())
            current = sent
    if current:
        blocks.append(current.strip())
    return blocks


def _cap_block_count(blocks: list[str], max_blocks: int = 4) -> list[str]:
    """Keep at most ``max_blocks`` — merge any tail blocks into the last one."""
    if len(blocks) <= max_blocks:
        return blocks
    head = blocks[: max_blocks - 1]
    tail = " ".join(blocks[max_blocks - 1 :])
    head.append(tail[:_MAX_BLOCK_CHARS])
    return head


def compute_typing_delay(text: str) -> float:
    """Return a 1.0–3.0s delay calibrated by message length, with mild jitter.

    Buckets (before jitter):
        - short  (< 50 chars):   1.0–1.5s
        - medium (50–150 chars): 1.5–2.5s
        - long   (> 150 chars):  2.0–3.0s

    A ±0.3s jitter is added then the result is clamped to [1.0, 3.0].
    """
    length = len(text)
    if length < 50:
        base = random.uniform(1.0, 1.5)
    elif length <= 150:
        base = random.uniform(1.5, 2.5)
    else:
        base = random.uniform(2.0, 3.0)
    jitter = random.uniform(-0.3, 0.3)
    return max(1.0, min(3.0, base + jitter))
