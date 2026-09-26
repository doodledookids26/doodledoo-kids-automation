import base64
import io
import os
from pathlib import Path

from PIL import Image
from google import genai

MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")
INPUT = Path("assets/characters/bobo.png")
OUTPUT_DIR = Path("output/bobo_gemini_rig")
OUTPUT = OUTPUT_DIR / "left_upper_arm_test.png"
CANVAS = (1024, 1536)

PROMPT = """
Use the provided Bobo the Bear master image as the ONLY character reference.

Create ONLY Bobo's LEFT UPPER ARM as an animation-rig asset.
Preserve Bobo's exact established design: brown bear fur, same yellow shirt,
same preschool proportions, same rendering style, same lighting and colors.

The output must be a clean, complete left upper-arm layer, including the full
shoulder connection hidden behind the torso and the full elbow-side connection
hidden behind the lower arm. Reconstruct any occluded pixels naturally so the
layer can rotate around a true shoulder/elbow joint without exposing holes.

Do NOT redesign Bobo. Do NOT change body proportions. Do NOT include the head,
torso, right arm, hands, legs, feet, eyes, mouth, background, text, shadows
belonging to other body parts, or any extra objects.

Place the isolated arm centered on a solid pure cyan (#00FFFF) background.
The cyan background will be removed automatically after generation.
Keep the arm large, fully visible, uncropped, and anatomically connected.
"""

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing master reference: {INPUT}")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it as a GitHub Actions secret before running this test."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_bytes = INPUT.read_bytes()

    client = genai.Client(api_key=api_key)
    interaction = client.interactions.create(
        model=MODEL,
        input=[
            {"type": "image", "mime_type": "image/png",
             "data": base64.b64encode(image_bytes).decode("utf-8")},
            {"type": "text", "text": PROMPT},
        ],
        response_format={
            "type": "image",
            "mime_type": "image/png",
            "aspect_ratio": "2:3",
            "image_size": "1K",
        },
        generation_config={"thinking_level": "high"},
    )

    if not interaction.output_image or not interaction.output_image.data:
        raise RuntimeError("Gemini returned no image output.")

    generated = Image.open(
        io.BytesIO(base64.b64decode(interaction.output_image.data))
    ).convert("RGBA")

    # Chroma-key only the connected cyan background from the image edges.
    px = generated.load()
    w, h = generated.size

    def is_cyan(r, g, b):
        return g > 205 and b > 205 and r < 80

    from collections import deque
    q = deque()
    seen = set()

    for x in range(w):
        q.append((x, 0))
        q.append((x, h - 1))
    for y in range(h):
        q.append((0, y))
        q.append((w - 1, y))

    while q:
        x, y = q.popleft()
        if (x, y) in seen or x < 0 or y < 0 or x >= w or y >= h:
            continue
        seen.add((x, y))
        r, g, b, a = px[x, y]
        if not is_cyan(r, g, b):
            continue
        px[x, y] = (r, g, b, 0)
        q.extend(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))

    # Normalize to the exact master canvas without stretching.
    canvas = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    generated.thumbnail(CANVAS, Image.Resampling.LANCZOS)
    x = (CANVAS[0] - generated.width) // 2
    y = (CANVAS[1] - generated.height) // 2
    canvas.alpha_composite(generated, (x, y))
    canvas.save(OUTPUT)

    alpha = canvas.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        raise RuntimeError("Validation failed: generated arm has no visible pixels.")

    if bbox.width < 80 or bbox.height < 120:
        raise RuntimeError(
            f"Validation failed: visible arm bbox is too small: {bbox.width}x{bbox.height}"
        )

    print(f"Generated: {OUTPUT}")
    print(f"Canvas: {canvas.size}")
    print(f"Visible bbox: {bbox}")

if __name__ == "__main__":
    main()
