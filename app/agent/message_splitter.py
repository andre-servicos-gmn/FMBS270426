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

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+(?=[A-Z0-9ÀÁÂÃÉÊÍÓÔÕÚÇ])")
_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n+")


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


def _fallback_split(text: str) -> list[str]:
    """Split a single plain-text response into 1–4 blocks heuristically."""
    if len(text) <= _SHORT_TOTAL_THRESHOLD:
        return [text]

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
    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
    if not sentences:
        return [text[:_MAX_BLOCK_CHARS]]

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
