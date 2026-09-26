import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_FILE = ROOT / "output" / "DDK001_script.json"
OUTPUT_FILE = ROOT / "output" / "DDK001_subtitles.srt"


def srt_time(seconds):
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3600000
    milliseconds %= 3600000
    minutes = milliseconds // 60000
    milliseconds %= 60000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def main():
    with open(SCRIPT_FILE, "r", encoding="utf-8") as file:
        script = json.load(file)

    scenes = script.get("scenes", [])
    if not scenes:
        raise RuntimeError("No scenes found in DDK001_script.json")

    blocks = []
    current = 0.0

    # Match the current 12-second-per-scene test timing.
    for index, scene in enumerate(scenes, start=1):
        duration = float(scene.get("duration_seconds", 12))
        text_parts = [
            scene.get("narration", ""),
            scene.get("dialogue", ""),
            scene.get("song_or_rhyme", ""),
            scene.get("child_interaction", ""),
        ]
        text = " ".join(p.strip() for p in text_parts if p and p.strip())
        if not text:
            continue

        start = current
        end = current + duration
        blocks.append(
            f"{len(blocks) + 1}\n"
            f"{srt_time(start)} --> {srt_time(end)}\n"
            f"{text}\n"
        )
        current = end

    if not blocks:
        raise RuntimeError("No subtitle text found.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(blocks), encoding="utf-8")
    print(f"Created subtitles: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
