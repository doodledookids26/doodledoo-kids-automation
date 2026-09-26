import math
import os
import subprocess
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
BG_PATH = ROOT / "assets" / "backgrounds" / "happy_meadow.png"
BOBO_PATH = ROOT / "assets" / "characters" / "bobo.png"
MIMI_PATH = ROOT / "assets" / "characters" / "mimi.png"

W, H = 1280, 720
FPS = 24
DURATION = 12


def cover(img, size):
    scale = max(size[0] / img.width, size[1] / img.height)
    img = img.resize(
        (int(img.width * scale), int(img.height * scale)),
        Image.Resampling.LANCZOS,
    )
    left = (img.width - size[0]) // 2
    top = (img.height - size[1]) // 2
    return img.crop((left, top, left + size[0], top + size[1]))


def resize_height(img, height):
    scale = height / img.height
    return img.resize(
        (int(img.width * scale), int(img.height * scale)),
        Image.Resampling.LANCZOS,
    )


def character_pose(img, t, base_height, phase):
    # Simulates a light 2D rig: breathing, body sway and small vertical motion.
    # The source artwork remains intact; motion is applied to the whole pose
    # to avoid the rigid "sticker sliding" look of the first test.
    pose = resize_height(img, base_height)
    breathe = 1.0 + 0.012 * math.sin(t * math.pi * 2 / 2.1 + phase)
    pose = pose.resize(
        (int(pose.width * breathe), int(pose.height * breathe)),
        Image.Resampling.BICUBIC,
    )
    return pose


def shadow_layer(size, cx, cy, rx, ry, opacity=90):
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    pixels = layer.load()
    for y in range(max(0, int(cy - ry)), min(size[1], int(cy + ry) + 1)):
        for x in range(max(0, int(cx - rx)), min(size[0], int(cx + rx) + 1)):
            dx = (x - cx) / rx
            dy = (y - cy) / ry
            d = dx * dx + dy * dy
            if d <= 1:
                pixels[x, y] = (0, 0, 0, int(opacity * (1 - d) ** 0.7))
    return layer


def make_frame(bg, bobo_src, mimi_src, t):
    # Very slow camera drift rather than moving the entire background abruptly.
    zoom = 1.0 + 0.025 * (t / DURATION)
    cam = bg.resize((int(W * zoom), int(H * zoom)), Image.Resampling.BICUBIC)
    left = (cam.width - W) // 2
    top = (cam.height - H) // 2
    frame = cam.crop((left, top, left + W, top + H)).convert("RGBA")

    bobo = character_pose(bobo_src, t, 500, 0.0)
    mimi = character_pose(mimi_src, t, 420, 1.4)

    # Characters ease into their positions during the opening seconds.
    ease = min(1.0, max(0.0, t / 2.0))
    ease = ease * ease * (3 - 2 * ease)

    bx_target = 120 + int(18 * math.sin(t * math.pi / 3.0))
    by_target = H - bobo.height - 48 + int(5 * math.sin(t * math.pi * 2 / 2.8))
    bx = int(-bobo.width * (1 - ease) + bx_target * ease)

    mx_target = W - mimi.width - 135 + int(14 * math.sin(t * math.pi / 3.4 + 1.0))
    my_target = H - mimi.height - 50 + int(5 * math.sin(t * math.pi * 2 / 3.2 + 1.0))
    mx = int(W + (mx_target - W) * ease)
    
    # Grounded soft shadows help separate characters from the background.
    frame.alpha_composite(
        shadow_layer((W, H), bx + bobo.width // 2, H - 43, 100, 16, 55),
        (0, 0),
    )
    frame.alpha_composite(
        shadow_layer((W, H), mx + mimi.width // 2, H - 43, 85, 14, 50),
        (0, 0),
    )

    frame.alpha_composite(bobo, (bx, by_target))
    frame.alpha_composite(mimi, (mx, my_target))
    return frame


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = OUT / "animation_test_frames_v2"
    frames.mkdir(parents=True, exist_ok=True)

    bg = cover(Image.open(BG_PATH).convert("RGBA"), (W, H))
    bobo = Image.open(BOBO_PATH).convert("RGBA")
    mimi = Image.open(MIMI_PATH).convert("RGBA")

    total = DURATION * FPS
    for i in range(total):
        t = i / FPS
        make_frame(bg, bobo, mimi, t).save(
            frames / f"frame_{i:05d}.png",
            optimize=True,
        )

    output = OUT / "DDK001_animation_test_v2.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", str(frames / "frame_%05d.png"),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    print(f"Created animation test: {output}")


if __name__ == "__main__":
    main()
