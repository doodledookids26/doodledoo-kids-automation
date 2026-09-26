import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CHANNEL_FILE = ROOT / "config" / "channel.json"
CHARACTERS_FILE = ROOT / "config" / "characters.json"
EPISODE_FILE = ROOT / "content" / "episodes" / "DDK001.json"

OUTPUT_DIR = ROOT / "output"
OUTPUT_FILE = OUTPUT_DIR / "DDK001_script_request.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    channel = load_json(CHANNEL_FILE)
    characters_data = load_json(CHARACTERS_FILE)
    episode = load_json(EPISODE_FILE)

    character_lookup = {
        character["name"].split(" the ")[0]: character
        for character in characters_data["characters"]
    }

    selected_characters = []

    for character_name in episode["characters"]:
        if character_name in character_lookup:
            selected_characters.append(
                character_lookup[character_name]
            )

    request = {
        "episode_id": episode["id"],
        "channel": channel["channel_name"],
        "world": channel["world"],
        "language": episode["language"],
        "title": episode["title"],
        "hindi_title": episode["hindi_title"],
        "category": episode["category"],
        "target_age": episode["target_age"],
        "duration_seconds": episode["duration_seconds"],
        "learning_objective": episode["learning_objective"],
        "location": episode["location"],
        "story": episode["story"],
        "educational_points": episode["educational_points"],
        "tone": episode["tone"],
        "characters": selected_characters,

        "script_requirements": {
            "opening_hook": True,
            "simple_language": True,
            "repetition_for_learning": True,
            "child_interaction": True,
            "clear_beginning_middle_end": True,
            "positive_ending": True,
            "short_sentences": True,
            "age_appropriate": True,
            "original_content": True
        },

        "output_requirements": {
            "narrator": True,
            "dialogue": True,
            "song_lyrics": True,
            "scene_breakdown": True,
            "estimated_duration": True
        },

        "safety_requirements": episode["safety"]
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(request, file, ensure_ascii=False, indent=2)

    print("Script request created successfully.")
    print("Episode:", episode["id"])
    print("Title:", episode["title"])
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
