"""Generate a short test audio (TTS) so we can validate the Whisper path end-to-end.

Writes ``tests/fixtures/test_audio.ogg`` (or similar) using OpenAI TTS with
a Brazilian-Portuguese voice. Run once; the file is reused by ``_audio_check.py``.

Usage:
    .venv/Scripts/python scripts/_gen_test_audio.py "oi quero uma raquete pra beach tennis" tests/fixtures/test_audio.mp3
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import AsyncOpenAI

from app.config import get_settings


async def main() -> None:
    text = sys.argv[1]
    out_path = Path(sys.argv[2])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(api_key=get_settings().openai_api_key)
    response = await client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )
    # response.read() returns the audio bytes (mp3 by default).
    audio_bytes = response.read()
    out_path.write_bytes(audio_bytes)
    print(f"wrote {out_path} ({len(audio_bytes):,} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
