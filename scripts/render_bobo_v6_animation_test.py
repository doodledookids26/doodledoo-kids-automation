from pathlib import Path
import json
import math

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RIG = ROOT / "output" / "bobo_rig_v6"
OUT = ROOT / "output" / "bobo_v6_animation_test.mp4"
MANIFEST = RIG / "rig_manifest.json"

FPS = 24
SEGMENTS = [
    ("idle", 3.0),
    ("wave", 3.0),
    ("walk", 4.0),
    ("run", 3.0),
    ("jump", 3.0),
    ("dance", 4.0),
]

with MANIFEST.open("r", encoding="utf-8") as f:
    manifest = json.load(f)

W, H = manifest["canvas"]
layers = {}
for name in manifest["layers"]:
    path = RIG / name
    if not path.exists():
        raise FileNotFoundError(path)
    layers[name] = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if layers[name] is None or layers[name].shape[2] != 4:
        raise RuntimeError(f"Invalid RGBA layer: {name}")

parent = manifest["hierarchy"]
pivots = {k: tuple(v) for k, v in manifest["pivots"].items()}

children = {}
for name, p in parent.items():
    children.setdefault(p, []).append(name)

def T(x, y):
    return np.array([[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]], dtype=np.float64)

def R(angle):
    a = math.radians(angle)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

def TR(p, angle=0.0, dx=0.0, dy=0.0):
    return T(dx, dy) @ T(p[0], p[1]) @ R(angle) @ T(-p[0], -p[1])

def rgba_affine(layer, matrix):
    return cv2.warpAffine(
        layer, matrix[:2, :], (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

def composite(canvas, layer):
    a = layer[:, :, 3:4].astype(np.float32) / 255.0
    rgb = layer[:, :, :3].astype(np.float32)
    dst = canvas[:, :, :3].astype(np.float32)
    out = rgb * a + dst * (1.0 - a)
    alpha = np.maximum(canvas[:, :, 3:4], layer[:, :, 3:4])
    return np.dstack([out.astype(np.uint8), alpha])

def pose(t, mode):
    a = {}
    root_angle = 0.0
    root_dx = 0.0
    root_dy = 0.0

    if mode == "idle":
        root_dy = -3.0 * (0.5 + 0.5 * math.sin(2 * math.pi * t / 2.0))
        root_angle = 0.8 * math.sin(2 * math.pi * t / 4.0)
        a["head.png"] = 1.5 * math.sin(2 * math.pi * t / 3.0)

    elif mode == "wave":
        p = t
        root_dy = -3.0 * abs(math.sin(math.pi * p / 1.5))
        a["left_upper_arm.png"] = -38 + 7 * math.sin(2 * math.pi * p / 0.7)
        a["left_lower_arm.png"] = -22 + 18 * math.sin(2 * math.pi * p / 0.7)
        a["left_hand.png"] = 8 * math.sin(2 * math.pi * p / 0.7)
        a["head.png"] = 2.0 * math.sin(2 * math.pi * p / 2.0)

    elif mode in ("walk", "run"):
        speed = 1.15 if mode == "walk" else 0.72
        amp = 24 if mode == "walk" else 34
        phase = 2 * math.pi * t / speed
        a["left_upper_leg.png"] = amp * math.sin(phase)
        a["right_upper_leg.png"] = -amp * math.sin(phase)
        a["left_lower_leg.png"] = -0.55 * amp * max(0.0, -math.sin(phase))
        a["right_lower_leg.png"] = -0.55 * amp * max(0.0, math.sin(phase))
        a["left_foot.png"] = -0.35 * amp * math.sin(phase)
        a["right_foot.png"] = 0.35 * amp * math.sin(phase)
        a["left_upper_arm.png"] = -0.8 * amp * math.sin(phase)
        a["right_upper_arm.png"] = 0.8 * amp * math.sin(phase)
        a["left_lower_arm.png"] = 0.35 * amp * math.sin(phase)
        a["right_lower_arm.png"] = -0.35 * amp * math.sin(phase)
        a["left_hand.png"] = 0.15 * amp * math.sin(phase)
        a["right_hand.png"] = -0.15 * amp * math.sin(phase)
        root_dy = -5.0 * abs(math.sin(phase)) if mode == "walk" else -10.0 * abs(math.sin(phase))
        root_angle = 1.2 * math.sin(phase)

    elif mode == "jump":
        p = t / 3.0
        y = math.sin(math.pi * p)
        root_dy = -95.0 * y
        squash = 8.0 * math.sin(math.pi * p)
        a["left_upper_leg.png"] = squash
        a["right_upper_leg.png"] = -squash
        a["left_lower_leg.png"] = -8.0 * y
        a["right_lower_leg.png"] = 8.0 * y
        a["left_upper_arm.png"] = -18.0 * y
        a["right_upper_arm.png"] = 18.0 * y
        a["left_lower_arm.png"] = -10.0 * y
        a["right_lower_arm.png"] = 10.0 * y
        a["head.png"] = 3.0 * math.sin(2 * math.pi * p)

    elif mode == "dance":
        phase = 2 * math.pi * t / 1.2
        root_dy = -10.0 * abs(math.sin(phase))
        root_angle = 5.0 * math.sin(phase)
        a["left_upper_arm.png"] = -28.0 + 20.0 * math.sin(phase)
        a["right_upper_arm.png"] = 28.0 - 20.0 * math.sin(phase)
        a["left_lower_arm.png"] = 18.0 * math.sin(phase)
        a["right_lower_arm.png"] = -18.0 * math.sin(phase)
        a["left_upper_leg.png"] = 12.0 * math.sin(phase)
        a["right_upper_leg.png"] = -12.0 * math.sin(phase)
        a["left_lower_leg.png"] = -8.0 * math.sin(phase)
        a["right_lower_leg.png"] = 8.0 * math.sin(phase)
        a["head.png"] = -4.0 * math.sin(phase)

    return a, root_angle, root_dx, root_dy

# OpenCV writes BGR frames.
writer = cv2.VideoWriter(
    str(OUT),
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (W, H),
)
if not writer.isOpened():
    raise RuntimeError("Could not open animation output")

root_static = layers["base.png"]
frame_count = 0

for mode, duration in SEGMENTS:
    frames = int(round(duration * FPS))
    for i in range(frames):
        t = i / FPS
        angles, root_angle, root_dx, root_dy = pose(t, mode)

        local = {name: np.eye(3, dtype=np.float64) for name in parent}
        local["torso.png"] = TR(
            pivots["torso.png"], root_angle, root_dx, root_dy
        )

        for name, angle in angles.items():
            if name in local:
                local[name] = TR(pivots[name], angle)

        world = {}
        for name in manifest["layers"]:
            if name == "base.png":
                continue
            p = parent.get(name)
            world[name] = local.get(name, np.eye(3, dtype=np.float64))
            if p:
                world[name] = world[p] @ world[name]

        canvas = root_static.copy()
        for name in manifest["layers"]:
            if name == "base.png":
                continue
            canvas = composite(canvas, rgba_affine(layers[name], world[name]))

        rgb = canvas[:, :, :3]
        alpha = canvas[:, :, 3:4].astype(np.float32) / 255.0
        bg = np.full_like(rgb, 255)
        frame = (rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)
        writer.write(frame)
        frame_count += 1

writer.release()
print(f"Created {OUT} with {frame_count} frames at {FPS} FPS")
