import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SCRIPT_FILE = ROOT / "output" / "DDK001_script.json"
OUTPUT_DIR = ROOT / "output"
OUTPUT_FILE = OUTPUT_DIR / "DDK001_scenes.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    script = load_json(SCRIPT_FILE)

    scenes = []

    for scene in script["scenes"]:
        scenes.append({
            "scene_number": scene["scene_number"],
            "duration_seconds": 12,
            "characters": [
                "Bobo",
                "Mimi"
            ],
            "location": "Happy Meadow",

            "visual_prompt": (
                "Friendly stylized 3D preschool animation in the "
                "DoodleDoo Kids Happy Meadow world. "
                "Bobo is a cute brown bear wearing a yellow shirt "
                "and blue shorts. "
                "Mimi is a cheerful pink bunny. "
                + scene["visual_description"]
                + " Bright pleasant colors, rounded shapes, "
                "expressive friendly faces, clean simple environment, "
                "safe for children ages 2-7, no scary elements."
            ),

            "narration": scene["narration"],
            "dialogue": scene["dialogue"],
            "song_or_rhyme": scene["song_or_rhyme"],
            "child_interaction": scene["child_interaction"],

            "camera": {
                "shot": "medium_wide",
                "movement": "slow_push_in",
                "transition": "soft_dissolve"
            }
        })

    result = {
        "episode_id": script["episode_id"],
        "title": script["title"],
        "learning_objective": script["learning_objective"],
        "scene_count": len(scenes),
        "scenes": scenes
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("Scene plan generated successfully.")
    print("Episode:", script["episode_id"])
    print("Scenes:", len(scenes))
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
