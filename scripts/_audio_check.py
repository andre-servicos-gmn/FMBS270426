"""Sprint 1.12 — manual end-to-end check of the audio path.

Runs ``transcribe_audio`` against a real audio file passed via CLI, then
prints both the raw Whisper transcription and what the diagnose agent
would respond if that text arrived as the customer's first message.

Usage:
    .venv/Scripts/python scripts/_audio_check.py path/to/audio.ogg

You can record a quick test audio with Audacity or your phone's recorder
and drop the file at any path. WhatsApp natively uses .ogg (opus) — that's
the format you'll see in production.
"""
import asyncio
import logging
import mimetypes
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import AIMessage, HumanMessage

from app.adapters.media_processor import transcribe_audio
from app.agent.checkpointer import close_checkpointer, init_checkpointer
from app.agent.graph import build_graph

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")


async def main() -> None:
    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"audio file not found: {audio_path}")
        sys.exit(1)

    audio_bytes = audio_path.read_bytes()
    mime, _ = mimetypes.guess_type(audio_path.name)
    mime = mime or "audio/ogg"

    print(f"== Transcribing {audio_path.name} ({len(audio_bytes):,} bytes, {mime}) ==")
    text = await transcribe_audio(audio_bytes, mime)
    print(f"\n>>> Whisper transcription:\n{text!r}\n")

    if not text.strip():
        print("(empty transcription — webhook would send 'Não consegui entender o áudio')")
        return

    # Run the agent as if the customer had typed the transcribed text.
    print("== Running diagnose with the transcribed text ==")
    thread_id = uuid.uuid4().hex[:16]
    await init_checkpointer()
    graph = build_graph()
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=text)],
            "phone_hash": f"audiotest{thread_id}",
            "needs_handoff": False,
            "handoff_reason": None,
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    print(f"\nINTENT  : {result.get('intent')}")
    print(f"PROFILE : {result.get('player_profile')}")
    for m in reversed(result.get("messages") or []):
        if isinstance(m, AIMessage):
            print(f"\nAI_REPLY:\n{m.content if isinstance(m.content, str) else str(m.content)}")
            break

    await close_checkpointer()


if __name__ == "__main__":
    asyncio.run(main())
