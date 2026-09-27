import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RIG = ROOT / "output" / "bobo_rig_stage2"
PLAN = ROOT / "output" / "DDK001_bobo_animation_plan.json"
OUT = ROOT / "output" / "DDK001_bobo_articulated_test.mp4"

with PLAN.open("r", encoding="utf-8") as f:
    plan = json.load(f)

required = ["base.png", "head.png", "left_arm.png", "right_arm.png", "left_leg.png", "right_leg.png"]
for name in required:
    if not (RIG / name).exists():
        raise FileNotFoundError(f"Missing rig layer: {name}")

layers = {name[:-4]: cv2.imread(str(RIG / name), cv2.IMREAD_UNCHANGED) for name in required}
h, w = layers["base.png"].shape[:2]
fps = int(plan["fps"])
frames = fps * 8

writer = cv2.VideoWriter(
    str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)
if not writer.isOpened():
    raise RuntimeError("Could not open output video")

# Approximate joints from the approved Bobo rig blueprint.
joints = {
    "left_shoulder": (300, 720),
    "right_shoulder": (735, 720),
    "left_hip": (365, 1160),
    "right_hip": (650, 1160),
}

def rotate_layer(layer, pivot, angle):
    px, py = pivot
    m = cv2.getRotationMatrix2D((px, py), angle, 1.0)
    return cv2.warpAffine(
        layer, m, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )

def composite_rgba(canvas, layer):
    a = layer[:, :, 3:4].astype(np.float32) / 255.0
    b = canvas[:, :, :3].astype(np.float32)
    l = layer[:, :, :3].astype(np.float32)
    out = l * a + b * (1.0 - a)
    alpha = np.maximum(canvas[:, :, 3:4], layer[:, :, 3:4])
    return np.dstack([out.astype(np.uint8), alpha])

for i in range(frames):
    t = i / fps
    canvas = layers["base.png"].copy()

    left_arm = 0.0
    right_arm = 0.0
    left_leg = 0.0
    right_leg = 0.0
    dy = 0

    if 1.0 <= t < 4.5:
        # Simple readable wave: the whole left arm pivots at the shoulder.
        p = t - 1.0
        left_arm = -35 + 18 * math.sin(p * math.pi * 2 / 0.9)
    elif t >= 5.0:
        p = t - 5.0
        bounce = abs(math.sin(p * math.pi * 2 / 1.2))
        dy = int(-8 * bounce)
        left_leg = 5 * math.sin(p * math.pi * 2 / 1.2)
        right_leg = -5 * math.sin(p * math.pi * 2 / 1.2)

    canvas = composite_rgba(canvas, rotate_layer(layers["left_leg"], joints["left_hip"], left_leg))
    canvas = composite_rgba(canvas, rotate_layer(layers["right_leg"], joints["right_hip"], right_leg))
    canvas = composite_rgba(canvas, rotate_layer(layers["left_arm"], joints["left_shoulder"], left_arm))
    canvas = composite_rgba(canvas, rotate_layer(layers["right_arm"], joints["right_shoulder"], right_arm))
    canvas = composite_rgba(canvas, layers["head"])

    # Global vertical motion is applied only to the finished character.
    if dy:
        m = np.float32([[1, 0, 0], [0, 1, dy]])
        canvas = cv2.warpAffine(canvas, m, (w, h), borderMode=cv2.BORDER_CONSTANT)

    rgb = canvas[:, :, :3]
    alpha = canvas[:, :, 3:4].astype(np.float32) / 255.0
    bg = np.full_like(rgb, 255)
    final = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
    writer.write(final)

writer.release()
print(f"Created {OUT}")
