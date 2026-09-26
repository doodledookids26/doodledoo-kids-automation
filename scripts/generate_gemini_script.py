import json
import os
from pathlib import Path

from google import genai


ROOT = Path(__file__).resolve().parents[1]

REQUEST_FILE = ROOT / "output" / "DDK001_script_request.json"
OUTPUT_DIR = ROOT / "output"
OUTPUT_FILE = OUTPUT_DIR / "DDK001_script.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not available.")

    request = load_json(REQUEST_FILE)

    client = genai.Client(api_key=api_key)

    prompt = f"""
Create an original preschool children's rhyme/story script.

You are writing for the YouTube channel:
{request["channel"]}

World:
{request["world"]}

Episode:
{request["title"]}

Target age:
{request["target_age"]}

Learning objective:
{request["learning_objective"]}

Location:
{request["location"]}

Story:
{json.dumps(request["story"], ensure_ascii=False)}

Educational points:
{json.dumps(request["educational_points"], ensure_ascii=False)}

Characters:
{json.dumps(request["characters"], ensure_ascii=False)}

Tone:
{json.dumps(request["tone"], ensure_ascii=False)}

Requirements:
- Create completely original wording.
- Use simple English suitable for ages 2-7.
- Make the story playful and easy to follow.
- Include repetition of the learning concept.
- Include moments where children can answer or participate.
- Include a short catchy original rhyme/song.
- Keep Bobo and Mimi's personalities consistent.
- No scary scenes.
- No violence.
- No dangerous behavior.
- No adult themes.
- No copyrighted song lyrics.
- Do not imitate a specific existing children's show or creator.
- End positively.
- Target approximately 3-4 minutes of spoken/sung content.

Return ONLY valid JSON.
"""

    schema = {
        "type": "object",
        "properties": {
            "episode_id": {"type": "string"},
            "title": {"type": "string"},
            "estimated_duration_seconds": {"type": "integer"},
            "learning_objective": {"type": "string"},
            "narrator_intro": {"type": "string"},
            "scenes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "scene_number": {"type": "integer"},
                        "visual_description": {"type": "string"},
                        "narration": {"type": "string"},
                        "dialogue": {"type": "string"},
                        "song_or_rhyme": {"type": "string"},
                        "child_interaction": {"type": "string"}
                    },
                    "required": [
                        "scene_number",
                        "visual_description",
                        "narration",
                        "dialogue",
                        "song_or_rhyme",
                        "child_interaction"
                    ]
                }
            },
            "ending": {"type": "string"},
            "safety_notes": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": [
            "episode_id",
            "title",
            "estimated_duration_seconds",
            "learning_objective",
            "narrator_intro",
            "scenes",
            "ending",
            "safety_notes"
        ]
    }

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": schema,
            "temperature": 0.8
        }
    )

    result = json.loads(response.text)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print("Gemini script generated successfully.")
    print("Episode:", result["episode_id"])
    print("Title:", result["title"])
    print("Scenes:", len(result["scenes"]))
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
