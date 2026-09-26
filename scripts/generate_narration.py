import asyncio
import json
from pathlib import Path

import edge_tts


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_FILE = ROOT / "output" / "DDK001_script.json"
OUTPUT_FILE = ROOT / "output" / "DDK001_test_narration.mp3"

VOICE = "en-US-AvaNeural"
RATE = "-8%"
VOLUME = "+0%"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_text(script):
    scene = script["scenes"][0]
    parts = [
        script.get("narrator_intro", ""),
        scene.get("narration", ""),
        scene.get("dialogue", ""),
        scene.get("song_or_rhyme", ""),
        scene.get("child_interaction", ""),
    ]
    return " ".join(part.strip() for part in parts if part and part.strip())


async def generate(text):
    communicate = edge_tts.Communicate(
        text,
        VOICE,
        rate=RATE,
        volume=VOLUME,
    )
    await communicate.save(str(OUTPUT_FILE))


def main():
    script = load_json(SCRIPT_FILE)
    text = build_text(script)

    if not text:
        raise RuntimeError("No narration text found in DDK001_script.json")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(generate(text))
    print("Narration generated successfully.")
    print("Voice:", VOICE)
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
