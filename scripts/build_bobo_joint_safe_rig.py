from pathlib import Path
import json
import math
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "assets" / "bobo.png"
V5 = ROOT / "artifacts" / "bobo-v5-rig-layers"
OUT = ROOT / "artifacts" / "bobo-v6-joint-safe-rig"
OUT.mkdir(parents=True, exist_ok=True)

spec = json.loads((ROOT / "config" / "bobo_rig_spec.json").read_text())
joint = {k: tuple(v) for k, v in spec["joint_targets"].items()}
W, H = spec["canvas"]["width"], spec["canvas"]["height"]
src = np.array(Image.open(SRC).convert("RGBA"))
xx, yy = np.meshgrid(np.arange(W), np.arange(H))

# Load V5 layers.
layers = {}
for p in V5.glob("*.png"):
    layers[p.name] = np.array(Image.open(p).convert("RGBA"))

# Derive the static shorts layer from the blue shorts region in the source.
hsv = cv2.cvtColor(src[:, :, :3], cv2.COLOR_RGB2HSV)
blue = (
    (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 140) &
    (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 40) &
    (src[:, :, 3] > 16)
)
kernel = np.ones((5, 5), np.uint8)
blue = cv2.morphologyEx(blue.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1).astype(bool)
shorts = np.zeros_like(src)
shorts[:, :, :3] = src[:, :, :3]
shorts[:, :, 3] = (blue.astype(np.uint8) * 255)
Image.fromarray(shorts).save(OUT / "hip_shorts.png")

# Remove shorts pixels from upper-leg layers so shorts belong to the hip parent.
for name in ("left_upper_leg.png", "right_upper_leg.png"):
    a = layers[name][:, :, 3]
    layers[name][:, :, 3] = np.where(blue, 0, a).astype(np.uint8)

parent_for_overlap = {
    "left_upper_arm.png": "torso.png",
    "right_upper_arm.png": "torso.png",
    "left_lower_arm.png": "left_upper_arm.png",
    "right_lower_arm.png": "right_upper_arm.png",
    "left_lower_leg.png": "left_upper_leg.png",
    "right_lower_leg.png": "right_upper_leg.png",
    "left_foot.png": "left_lower_leg.png",
    "right_foot.png": "right_lower_leg.png",
}
joint_for_layer = {
    "left_upper_arm.png": "left_shoulder",
    "right_upper_arm.png": "right_shoulder",
    "left_lower_arm.png": "left_elbow",
    "right_lower_arm.png": "right_elbow",
    "left_lower_leg.png": "left_knee",
    "right_lower_leg.png": "right_knee",
    "left_foot.png": "left_ankle",
    "right_foot.png": "right_ankle",
}
radius = {
    "left_upper_arm.png": 32,
    "right_upper_arm.png": 32,
    "left_lower_arm.png": 28,
    "right_lower_arm.png": 28,
    "left_lower_leg.png": 30,
    "right_lower_leg.png": 30,
    "left_foot.png": 24,
    "right_foot.png": 24,
}

def add_directed_overlap(name):
    im = layers[name]
    visible = im[:, :, 3] > 16
    px, py = joint[joint_for_layer[name]]
    r = radius[name]
    disk = ((xx - px) ** 2 + (yy - py) ** 2 <= r * r)

    if parent_for_overlap[name] == "hip_shorts.png":
        parent_alpha = shorts[:, :, 3] > 16
    else:
        parent_alpha = layers[parent_for_overlap[name]][:, :, 3] > 16

    # Only add hidden overlap where the original source is transparent.
    # Never let an articulation patch overwrite source-visible pixels in the
    # reconstructed rest pose; those pixels must remain exact.
    source_visible = src[:, :, 3] > 16
    target = visible | (disk & parent_alpha & (~source_visible))

    # Propagate the nearest visible pixel's RGB into the small hidden region.
    _, indices = cv2.distanceTransformWithLabels(
        (~visible).astype(np.uint8), cv2.DIST_L2, 5,
        labelType=cv2.DIST_LABEL_PIXEL
    )
    visible_points = np.flatnonzero(visible)
    if len(visible_points) == 0:
        raise RuntimeError(f"No visible pixels in {name}")
    labels = np.clip(indices.astype(np.int64) - 1, 0, len(visible_points) - 1)
    flat_rgb = im[:, :, :3].reshape(-1, 3)
    source_flat = flat_rgb[visible_points]
    rgb = source_flat[labels]

    # Preserve every original visible pixel exactly. Only synthesize RGB
    # inside the newly added hidden overlap region.
    out_rgb = im[:, :, :3].copy()
    hidden_fill = target & (~visible)
    out_rgb[hidden_fill] = rgb[hidden_fill]
    out_alpha = target.astype(np.uint8) * 255
    out = np.dstack([out_rgb, out_alpha])
    Image.fromarray(out.astype(np.uint8)).save(OUT / name)
    layers[name] = out.astype(np.uint8)

for name in parent_for_overlap:
    add_directed_overlap(name)

# Copy static layers.
for name in ("base.png", "torso.png", "left_ear.png", "right_ear.png", "eyes.png", "mouth.png"):
    Image.fromarray(layers[name]).save(OUT / name)
