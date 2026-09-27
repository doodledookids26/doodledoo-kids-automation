import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "config" / "bobo_animation_presets.json"
OUTPUT = ROOT / "output" / "DDK001_bobo_animation_plan.json"

with PRESETS.open("r", encoding="utf-8") as f:
    data = json.load(f)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

plan = {
    "episode_id": "DDK001",
    "character": data["character"],
    "fps": data["fps"],
    "animations": [
        {
            "name": name,
            "duration_seconds": preset["duration_seconds"],
            "keyframes": preset["keyframes"]
        }
        for name, preset in data["presets"].items()
    ]
}

with OUTPUT.open("w", encoding="utf-8") as f:
    json.dump(plan, f, indent=2, ensure_ascii=False)

print(f"Created {OUTPUT}")
