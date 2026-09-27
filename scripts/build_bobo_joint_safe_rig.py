from pathlib import Path
import json
import math

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "output" / "bobo_rig_v5"
OUT = ROOT / "output" / "bobo_rig_v6"
SOURCE = ROOT / "assets" / "characters" / "bobo.png"
SPEC = ROOT / "config" / "bobo_rig_spec.json"

OUT.mkdir(parents=True, exist_ok=True)

W, H = 1024, 1536
img = Image.open(SOURCE).convert("RGBA")
src = np.array(img)
if (W, H) != img.size:
    raise ValueError(f"Expected Bobo source {W}x{H}, got {img.size}")

required_v5 = [
    "base.png", "head.png", "left_ear.png", "right_ear.png", "eyes.png", "mouth.png",
    "torso.png",
    "left_upper_arm.png", "left_lower_arm.png", "left_hand.png",
    "right_upper_arm.png", "right_lower_arm.png", "right_hand.png",
    "left_upper_leg.png", "left_lower_leg.png", "left_foot.png",
    "right_upper_leg.png", "right_lower_leg.png", "right_foot.png",
]
for name in required_v5:
    if not (V5 / name).exists():
        raise FileNotFoundError(f"Missing V5 layer: {V5 / name}")

joint = json.loads(SPEC.read_text(encoding="utf-8"))["joint_targets_1024x1536"]

# The V5 extraction correctly separates the visible artwork. V6 adds the
# small, directed overlaps required for rotation and moves the blue shorts
# into a static clothing layer so the leg bones contain only leg artwork.
hsv = cv2.cvtColor(src[:, :, :3], cv2.COLOR_RGB2HSV)
yy, xx = np.indices((H, W))

shorts = (
    (hsv[:, :, 0] >= 98) & (hsv[:, :, 0] <= 115) &
    (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 50) &
    (yy > 1050) & (yy < 1380) & (xx > 150) & (xx < 870)
).astype(np.uint8) * 255
shorts = cv2.morphologyEx(shorts, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
shorts = cv2.dilate(shorts, np.ones((3, 3), np.uint8), iterations=1)

# Static hip clothing helper. It stays attached to the torso/hip and masks
# the hidden upper-leg roots during animation.
shorts_rgba = np.dstack([src[:, :, :3], shorts])
Image.fromarray(shorts_rgba.astype(np.uint8)).save(OUT / "hip_shorts.png")

layers = {}
for name in required_v5:
    layers[name] = np.array(Image.open(V5 / name).convert("RGBA"))

# Upper-leg V5 masks include the shorts. Remove that clothing from the
# articulated leg pieces before adding the directed hidden overlap.
for name in ("left_upper_leg.png", "right_upper_leg.png"):
    a = layers[name][:, :, 3]
    a = np.where(shorts > 16, 0, a).astype(np.uint8)
    layers[name][:, :, 3] = a

parent_for_overlap = {
    "head.png": "torso.png",
    "left_upper_arm.png": "torso.png",
    "left_lower_arm.png": "left_upper_arm.png",
    "left_hand.png": "left_lower_arm.png",
    "right_upper_arm.png": "torso.png",
    "right_lower_arm.png": "right_upper_arm.png",
    "right_hand.png": "right_lower_arm.png",
    "left_upper_leg.png": "hip_shorts.png",
    "left_lower_leg.png": "left_upper_leg.png",
    "left_foot.png": "left_lower_leg.png",
    "right_upper_leg.png": "hip_shorts.png",
    "right_lower_leg.png": "right_upper_leg.png",
    "right_foot.png": "right_lower_leg.png",
}

joint_for_layer = {
    "head.png": "neck",
    "left_upper_arm.png": "left_shoulder",
    "left_lower_arm.png": "left_elbow",
    "left_hand.png": "left_wrist",
    "right_upper_arm.png": "right_shoulder",
    "right_lower_arm.png": "right_elbow",
    "right_hand.png": "right_wrist",
    "left_upper_leg.png": "left_hip",
    "left_lower_leg.png": "left_knee",
    "left_foot.png": "left_ankle",
    "right_upper_leg.png": "right_hip",
    "right_lower_leg.png": "right_knee",
    "right_foot.png": "right_ankle",
}

# Small overlap is enough because the parent is rendered over the joint.
# Larger radial dilation caused visible halos, so V6 deliberately extends
# only into the parent layer near the actual joint.
radius = {
    "head.png": 18,
    "left_upper_arm.png": 24, "left_lower_arm.png": 22, "left_hand.png": 18,
    "right_upper_arm.png": 24, "right_lower_arm.png": 22, "right_hand.png": 18,
    "left_upper_leg.png": 10, "left_lower_leg.png": 22, "left_foot.png": 18,
    "right_upper_leg.png": 10, "right_lower_leg.png": 22, "right_foot.png": 18,
}

def add_directed_overlap(name):
    im = layers[name]
    visible = im[:, :, 3] > 16
    px, py = joint[joint_for_layer[name]]
    r = radius[name]
    disk = ((xx - px) ** 2 + (yy - py) ** 2 <= r * r)

    if parent_for_overlap[name] == "hip_shorts.png":
        parent_alpha = shorts > 16
    else:
        parent_alpha = layers[parent_for_overlap[name]][:, :, 3] > 16

    # Only add hidden overlap where the original source is transparent.\n    # Never let an articulation patch overwrite source-visible pixels in the\n    # reconstructed rest pose; those pixels must remain exact.\n    source_visible = src[:, :, 3] > 16\n    target = visible | (disk & parent_alpha & (~source_visible))\n
    # Propagate the nearest visible pixel's RGB into the small hidden region.
    # This produces a texture-consistent continuation without inventing a
    # broad halo outside the parent overlap.
    _, indices = cv2.distanceTransformWithLabels(
        (~visible).astype(np.uint8), cv2.DIST_L2, 5,
        labelType=cv2.DIST_LABEL_PIXEL
    )
    # OpenCV's pixel labels are 1-based and map to flattened visible pixels.
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
    out_alpha = (target.astype(np.uint8) * 255)
    out = np.dstack([out_rgb, out_alpha])
    Image.fromarray(out.astype(np.uint8)).save(OUT / name)
    layers[name] = out.astype(np.uint8)

for name in parent_for_overlap:
    add_directed_overlap(name)

# Copy static layers.
for name in ("base.png", "torso.png", "left_ear.png", "right_ear.png", "eyes.png", "mouth.png"):
    Image.fromarray(layers[name]).save(OUT / name)

layer_order = [
    "base.png",
    "torso.png",
    "hip_shorts.png",
    "left_upper_leg.png", "right_upper_leg.png",
    "left_lower_leg.png", "right_lower_leg.png",
    "left_foot.png", "right_foot.png",
    "left_upper_arm.png", "right_upper_arm.png",
    "left_lower_arm.png", "right_lower_arm.png",
    "left_hand.png", "right_hand.png",
    "head.png",
    "left_ear.png", "right_ear.png",
    "eyes.png", "mouth.png",
]

hierarchy = {
    "torso.png": None,
    "hip_shorts.png": "torso.png",
    "left_upper_leg.png": "hip_shorts.png",
    "left_lower_leg.png": "left_upper_leg.png",
    "left_foot.png": "left_lower_leg.png",
    "right_upper_leg.png": "hip_shorts.png",
    "right_lower_leg.png": "right_upper_leg.png",
    "right_foot.png": "right_lower_leg.png",
    "left_upper_arm.png": "torso.png",
    "left_lower_arm.png": "left_upper_arm.png",
    "left_hand.png": "left_lower_arm.png",
    "right_upper_arm.png": "torso.png",
    "right_lower_arm.png": "right_upper_arm.png",
    "right_hand.png": "right_lower_arm.png",
    "head.png": "torso.png",
    "left_ear.png": "head.png",
    "right_ear.png": "head.png",
    "eyes.png": "head.png",
    "mouth.png": "head.png",
}

pivots = {
    "torso.png": [510, 900],
    "hip_shorts.png": [510, 1160],
    "left_upper_leg.png": joint["left_hip"],
    "left_lower_leg.png": joint["left_knee"],
    "left_foot.png": joint["left_ankle"],
    "right_upper_leg.png": joint["right_hip"],
    "right_lower_leg.png": joint["right_knee"],
    "right_foot.png": joint["right_ankle"],
    "left_upper_arm.png": joint["left_shoulder"],
    "left_lower_arm.png": joint["left_elbow"],
    "left_hand.png": joint["left_wrist"],
    "right_upper_arm.png": joint["right_shoulder"],
    "right_lower_arm.png": joint["right_elbow"],
    "right_hand.png": joint["right_wrist"],
    "head.png": joint["neck"],
    "left_ear.png": joint["neck"],
    "right_ear.png": joint["neck"],
    "eyes.png": joint["neck"],
    "mouth.png": joint["neck"],
}

# Rest-pose reconstruction from the V6 hierarchy.
preview = Image.new("RGBA", (W, H), (0, 0, 0, 0))
for name in layer_order:
    preview.alpha_composite(Image.open(OUT / name))
preview.save(OUT / "reconstructed_preview.png")

src16 = src.astype(np.int16)
rec16 = np.array(preview).astype(np.int16)
visible = src[:, :, 3] > 16
intersection = visible & (rec16[:, :, 3] > 16)
coverage = float(intersection[visible].mean()) if np.any(visible) else 0.0
rgb_error = float(np.abs(src16[:, :, :3][intersection] - rec16[:, :, :3][intersection]).mean()) if np.any(intersection) else 999.0
alpha_error = float(np.abs(src16[:, :, 3] - rec16[:, :, 3])[visible].mean()) if np.any(visible) else 999.0

manifest = {
    "character": "Bobo the Bear",
    "rig_type": "automated_2d_joint_rig_v6",
    "source": "assets/characters/bobo.png",
    "canvas": [W, H],
    "layers": layer_order,
    "hierarchy": hierarchy,
    "pivots": pivots,
    "joint_targets": joint,
    "controls": {
        "global_root": "torso.png",
        "head_tilt": "head.png",
        "left_arm_chain": ["left_upper_arm.png", "left_lower_arm.png", "left_hand.png"],
        "right_arm_chain": ["right_upper_arm.png", "right_lower_arm.png", "right_hand.png"],
        "left_leg_chain": ["left_upper_leg.png", "left_lower_leg.png", "left_foot.png"],
        "right_leg_chain": ["right_upper_leg.png", "right_lower_leg.png", "right_foot.png"],
        "face": ["eyes.png", "mouth.png"],
    },
    "reconstruction_check": {
        "visible_coverage": round(coverage, 6),
        "visible_rgb_error": round(rgb_error, 4),
        "visible_alpha_error": round(alpha_error, 4),
    },
    "quality_gate": {
        "rest_pose_reconstructed": coverage >= 0.995 and rgb_error < 3.0 and alpha_error < 3.0,
        "visual_motion_review_required": True,
    },
}

(OUT / "rig_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
if not manifest["quality_gate"]["rest_pose_reconstructed"]:
    raise SystemExit("V6 REST-POSE QUALITY GATE FAILED")
print(json.dumps(manifest["reconstruction_check"]))
