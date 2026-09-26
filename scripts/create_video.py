import json
import math
import os
import subprocess
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
SCENES = os.path.join(OUT, "DDK001_scenes.json")

W, H = 1280, 720
FPS = 24
DURATION = 12

def load_rgba(path):
    return Image.open(path).convert("RGBA")

def cover(img, size):
    img = img.copy()
    scale = max(size[0] / img.width, size[1] / img.height)
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    left = (img.width - size[0]) // 2
    top = (img.height - size[1]) // 2
    return img.crop((left, top, left + size[0], top + size[1]))

def fit_character(img, max_h):
    scale = max_h / img.height
    return img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)

def make_frame(bg, bobo, mimi, t):
    frame = bg.copy()

    # Gentle preschool-style floating motion.
    bob1 = int(6 * math.sin(t * math.pi * 2 / 2.8))
    bob2 = int(5 * math.sin(t * math.pi * 2 / 3.2 + 1.0))

    b = fit_character(bobo, 500)
    m = fit_character(mimi, 420)

    bx = 105 + int(18 * math.sin(t * math.pi / 6))
    by = H - b.height - 35 + bob1
    mx = W - m.width - 120 + int(15 * math.sin(t * math.pi / 5 + 1.2))
    my = H - m.height - 38 + bob2

    frame.alpha_composite(b, (bx, by))
    frame.alpha_composite(m, (mx, my))
    return frame

def main():
    with open(SCENES, "r", encoding="utf-8") as f:
        scenes = json.load(f)

    # generate_scenes.py stores the scene list inside a top-level "scenes" object.
    # Accept both that format and a raw list so the video step is robust.
    if isinstance(scenes, dict):
        scenes = scenes.get("scenes", [])

    if not isinstance(scenes, list) or not scenes:
        raise RuntimeError("No scenes found in DDK001_scenes.json")

    bg = cover(load_rgba(os.path.join(ROOT, "assets", "backgrounds", "happy_meadow.png")), (W, H))
    bobo = load_rgba(os.path.join(ROOT, "assets", "characters", "bobo.png"))
    mimi = load_rgba(os.path.join(ROOT, "assets", "characters", "mimi.png"))

    frames_dir = os.path.join(OUT, "test_frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Build a short test from the first scene only.
    scene = scenes[0]
    duration = min(DURATION, max(6, int(scene.get("duration_seconds", DURATION))))
    total = duration * FPS

    for i in range(total):
        t = i / FPS
        frame = make_frame(bg, bobo, mimi, t)
        frame.save(os.path.join(frames_dir, f"frame_{i:05d}.png"), optimize=True)

    mp4 = os.path.join(OUT, "DDK001_test_video.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "frame_%05d.png"),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        mp4,
    ]
    subprocess.run(cmd, check=True)

    print(f"Created {mp4}")

if __name__ == "__main__":
    main()
