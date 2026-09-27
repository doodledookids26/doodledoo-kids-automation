import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets" / "characters" / "bobo.png"
PLAN = ROOT / "output" / "DDK001_bobo_animation_plan.json"
OUT = ROOT / "output" / "DDK001_bobo_smooth_deformation_test.mp4"

with PLAN.open("r", encoding="utf-8") as f:
    plan = json.load(f)

src = cv2.imread(str(SOURCE), cv2.IMREAD_UNCHANGED)
if src is None or src.shape[2] != 4:
    raise RuntimeError("Bobo source must be an RGBA PNG")

h, w = src.shape[:2]
fps = int(plan["fps"])
frames = fps * 8

# Full-image smooth deformation. Every output pixel samples the original
# image, so there are no triangle rasterization holes or cutout seams.
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

def gaussian(cx, cy, sx, sy):
    return np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) * 0.5)

def deform(t):
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)

    # Gentle breathing keeps the test visibly alive without moving the
    # character's silhouette aggressively.
    breathe = math.sin(t * math.pi * 2 / 2.4)
    body = gaussian(510, 930, 360, 650)
    dx += 2.0 * breathe * body
    dy += -2.5 * breathe * body

    # Wave section: smooth localized deformation around the left arm.
    if 1.0 <= t < 4.5:
        p = t - 1.0
        wave = math.sin(p * math.pi * 2 / 0.9)
        arm = gaussian(275, 930, 260, 380)
        dx += 10.0 * wave * arm
        dy += -22.0 * abs(wave) * arm

        # Slight shoulder follow-through reduces the "paper cutout" look.
        shoulder = gaussian(350, 760, 300, 260)
        dx += 5.0 * wave * shoulder
        dy += -6.0 * abs(wave) * shoulder

    # Bounce section: deform the whole image continuously.
    elif t >= 5.0:
        p = t - 5.0
        bounce = abs(math.sin(p * math.pi * 2 / 1.2))
        dy += -12.0 * bounce
        squash = bounce * 0.018
        body = gaussian(510, 1030, 430, 650)
        dy += squash * (yy - 1030.0) * body

    map_x = xx - dx
    map_y = yy - dy
    return map_x, map_y

writer = cv2.VideoWriter(
    str(OUT),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h),
)
if not writer.isOpened():
    raise RuntimeError("Could not open output video")

for i in range(frames):
    t = i / fps
    map_x, map_y = deform(t)

    warped = cv2.remap(
        src,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    rgb = warped[:, :, :3].astype(np.float32)
    bg = np.full_like(rgb, 255)
    final = (rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)
    writer.write(final)

writer.release()
print(f"Created {OUT}")
