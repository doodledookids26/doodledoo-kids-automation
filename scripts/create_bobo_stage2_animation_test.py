from pathlib import Path
import math
import subprocess
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RIG = ROOT / "output" / "bobo_rig_stage2"
OUT_DIR = ROOT / "output"
FRAMES = OUT_DIR / "bobo_stage2_animation_frames"
VIDEO = OUT_DIR / "bobo_stage2_animation_test.mp4"
BG_PATH = ROOT / "assets" / "backgrounds" / "happy_meadow.png"

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

if BG_PATH.exists():
    bg_source = Image.open(BG_PATH).convert("RGB")
else:
    bg_source = Image.new("RGB", (1280, 720), (230, 245, 255))

def fit_background(size):
    bw, bh = bg_source.size
    tw, th = size
    scale = max(tw / bw, th / bh)
    resized = bg_source.resize((int(bw * scale), int(bh * scale)), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - tw) // 2)
    top = max(0, (resized.height - th) // 2)
    return resized.crop((left, top, left + tw, top + th)).convert("RGBA")

def ease(t):
    return t * t * (3.0 - 2.0 * t)

def rotate_layer(layer, angle, center):
    return layer.rotate(angle, resample=Image.Resampling.BICUBIC, center=center)

def frame_character(frame_no):
    t = frame_no / FPS
    cycle = t % 4.0

    # Stage-2 is deliberately conservative. Small motions first; no large limb swings.
    sway = 2.0 * math.sin(2 * math.pi * t / 2.8)
    wave = 0.0
    walk = 0.0
    bounce = 0.0

    if 1.0 <= cycle < 2.0:
        p = cycle - 1.0
        wave = 8.0 * math.sin(2 * math.pi * p * 2.0)
    elif 2.0 <= cycle < 3.0:
        p = cycle - 2.0
        walk = 5.0 * math.sin(2 * math.pi * p * 2.0)
    elif 3.0 <= cycle < 4.0:
        p = cycle - 3.0
        bounce = -5.0 * abs(math.sin(math.pi * p))

    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    # Stage-2 uses six pieces, so movement stays small to avoid exposing hidden joint seams.
    ll = rotate_layer(left_leg, walk, (360, 1200))
    rl = rotate_layer(right_leg, -walk, (650, 1200))
    la = rotate_layer(left_arm, -walk * 0.25, (300, 820))
    ra = rotate_layer(right_arm, wave - walk * 0.25, (735, 820))
    hd = rotate_layer(head, sway * 0.18, (510, 430))

    yoff = int(round(bounce))
    xoff = int(round(sway))

    for layer in (ll, rl, base, la, ra, hd):
        shifted = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        shifted.alpha_composite(layer, (xoff, yoff))
        canvas.alpha_composite(shifted)

    return canvas

FRAMES.mkdir(parents=True, exist_ok=True)
total = FPS * SECONDS

# Render frames with Bobo over a clean 16:9 Happy Meadow presentation.
presentation_w, presentation_h = 1280, 720
background = fit_background((presentation_w, presentation_h))

for i in range(total):
    character = frame_character(i)
    # Fit Bobo to the same visual scale used by the earlier animation test.
    scale = min(600 / character.width, 700 / character.height)
    new_size = (int(character.width * scale), int(character.height * scale))
    character = character.resize(new_size, Image.Resampling.LANCZOS)

    frame = background.copy()
    x = (presentation_w - character.width) // 2
    y = presentation_h - character.height - 5
    frame.alpha_composite(character, (x, y))
    frame.convert("RGB").save(FRAMES / f"frame_{i:04d}.png", quality=95)

cmd = [
    "ffmpeg", "-y",
    "-framerate", str(FPS),
    "-i", str(FRAMES / "frame_%04d.png"),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    str(VIDEO),
]
subprocess.run(cmd, check=True)

print(f"Created {VIDEO}")
