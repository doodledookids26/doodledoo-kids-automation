import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "output" / "DDK001_bobo_animation_plan.json"
BOBO = ROOT / "assets" / "characters" / "bobo.png"
OUT = ROOT / "output" / "DDK001_bobo_pose_test.mp4"

with PLAN.open("r", encoding="utf-8") as f:
    plan = json.load(f)

img = cv2.imread(str(BOBO), cv2.IMREAD_UNCHANGED)
if img is None or img.shape[2] != 4:
    raise RuntimeError("Bobo PNG must be a readable RGBA image")

h, w = img.shape[:2]
fps = int(plan["fps"])
duration = 8
frames = fps * duration
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUT), fourcc, fps, (w, h))
if not writer.isOpened():
    raise RuntimeError("Could not open MP4 writer")

def affine(src, angle=0, dx=0, dy=0, scale=1.0):
    center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    m[0, 2] += dx
    m[1, 2] += dy
    return cv2.warpAffine(src, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0, 0))

for i in range(frames):
    t = i / fps
    pose = "idle"
    if t >= 2 and t < 6:
        pose = "wave"
    elif t >= 6:
        pose = "bounce"

    phase = (t - 2) if pose == "wave" else 0
    bounce = 0
    angle = 0
    if pose == "idle":
        bounce = int(3 * math.sin(t * math.pi * 2 / 2))
        angle = 0.8 * math.sin(t * math.pi * 2 / 4)
    elif pose == "wave":
        angle = 2.5 * math.sin(phase * math.pi * 2 / 2)
        bounce = int(4 * math.sin(phase * math.pi * 2 / 1.5))
    else:
        bounce = int(10 * abs(math.sin((t - 6) * math.pi * 2 / 1.5)))
        angle = 1.5 * math.sin((t - 6) * math.pi * 2 / 1.5)

    frame = affine(img, angle=angle, dy=-bounce)
    bgr = frame[:, :, :3].copy()
    alpha = frame[:, :, 3:4].astype(np.float32) / 255.0
    bg = np.full_like(bgr, 255)
    out = (bgr * alpha + bg * (1 - alpha)).astype(np.uint8)
    writer.write(out)

writer.release()
print(f"Created {OUT}")
