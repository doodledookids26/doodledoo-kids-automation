from pathlib import Path
import math
import subprocess
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RIG = ROOT / "output" / "bobo_rig"
OUT_DIR = ROOT / "output"
FRAMES = OUT_DIR / "bobo_animation_frames"
VIDEO = OUT_DIR / "bobo_animation_test.mp4"

W, H = 1024, 1536
FPS = 24
SECONDS = 8

def load(name):
    return Image.open(RIG / name).convert("RGBA")

base = load("base.png")
head = load("head.png")
left_arm = load("left_arm.png")
right_arm = load("right_arm.png")
left_leg = load("left_leg.png")
right_leg = load("right_leg.png")

def ease(t):
    return t * t * (3.0 - 2.0 * t)

def rotate_layer(layer, angle, center):
    return layer.rotate(angle, resample=Image.Resampling.BICUBIC, center=center)

def frame_character(frame_no):
    t = frame_no / FPS
    cycle = t % 4.0

    # Natural idle breathing and gentle body sway.
    breath = 1.0 + 0.012 * math.sin(2 * math.pi * t / 2.4)
    sway = 2.5 * math.sin(2 * math.pi * t / 2.8)

    # Four-second animation cycle:
    # 0-1 idle, 1-2 wave, 2-3 walk, 3-4 happy bounce.
    wave = 0.0
    walk = 0.0
    bounce = 0.0

    if 1.0 <= cycle < 2.0:
        p = (cycle - 1.0)
        wave = 24.0 * math.sin(2 * math.pi * p * 2.0)
    elif 2.0 <= cycle < 3.0:
        p = cycle - 2.0
        walk = 13.0 * math.sin(2 * math.pi * p * 2.0)
    elif 3.0 <= cycle < 4.0:
        p = cycle - 3.0
        bounce = -12.0 * abs(math.sin(math.pi * p))

    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    # Legs swing from the upper-leg area.
    ll = rotate_layer(left_leg, walk, (360, 1200))
    rl = rotate_layer(right_leg, -walk, (650, 1200))

    # Arms: right arm performs a friendly wave.
    la = rotate_layer(left_arm, -walk * 0.35, (300, 820))
    ra = rotate_layer(right_arm, wave - walk * 0.35, (735, 820))

    # Head gives a tiny responsive tilt.
    hd = rotate_layer(head, sway * 0.22, (510, 430))

    # Apply a subtle vertical bounce by shifting the whole character.
    yoff = int(bounce)
    xoff = int(sway)

    for layer in (ll, rl, base, la, ra, hd):
        shifted = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        shifted.alpha_composite(layer, (xoff, yoff))
        canvas.alpha_composite(shifted)

    return canvas

FRAMES.mkdir(parents=True, exist_ok=True)
total = FPS * SECONDS

for i in range(total):
    frame_character(i).save(FRAMES / f"frame_{i:04d}.png")

cmd = [
    "ffmpeg", "-y",
    "-framerate", str(FPS),
    "-i", str(FRAMES / "frame_%04d.png"),
    "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black",
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    str(VIDEO),
]
subprocess.run(cmd, check=True)

print(f"Created {VIDEO}")
