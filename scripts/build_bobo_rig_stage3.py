from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets" / "characters" / "bobo.png"
OUT = ROOT / "output" / "bobo_rig_stage3"
OUT.mkdir(parents=True, exist_ok=True)

img = Image.open(SOURCE).convert("RGBA")
W, H = img.size
if (W, H) != (1024, 1536):
    raise ValueError(f"Expected Bobo source 1024x1536, got {W}x{H}")

alpha = np.array(img.getchannel("A"), dtype=np.uint8)
rgb = np.array(img.convert("RGB"))

def poly(points):
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).polygon(points, fill=255)
    return np.array(m, dtype=np.uint8)

def ellipse(box):
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).ellipse(box, fill=255)
    return np.array(m, dtype=np.uint8)

def soften(mask, radius=1.0):
    return np.array(Image.fromarray(mask, "L").filter(ImageFilter.GaussianBlur(radius)), dtype=np.uint8)

def source_mask(mask):
    return np.minimum(mask, alpha)

# Stage 3 is a controlled multi-joint experiment.
# Segments overlap at joints to reduce gaps during small rotations.
masks = {
    "head.png": ellipse((105, 45, 925, 720)),
    "torso.png": poly([
        (275, 620), (745, 620), (790, 760), (775, 950),
        (735, 1125), (690, 1195), (335, 1195), (285, 1110),
        (250, 930), (245, 760)
    ]),
    "left_upper_arm.png": poly([
        (245, 675), (370, 660), (405, 750), (385, 850),
        (365, 920), (330, 965), (250, 940), (205, 865),
        (215, 770)
    ]),
    "left_lower_arm.png": poly([
        (300, 880), (370, 900), (365, 990), (345, 1080),
        (325, 1160), (285, 1210), (215, 1195), (165, 1145),
        (175, 1040), (205, 950), (245, 900)
    ]),
    "right_upper_arm.png": poly([
        (650, 665), (770, 680), (805, 770), (820, 850),
        (800, 925), (735, 965), (675, 920), (650, 850),
        (625, 750)
    ]),
    "right_lower_arm.png": poly([
        (690, 880), (760, 900), (800, 960), (830, 1040),
        (850, 1140), (810, 1205), (745, 1210), (700, 1160),
        (680, 1060), (660, 970)
    ]),
    "left_upper_leg.png": poly([
        (235, 1135), (500, 1135), (500, 1295), (470, 1360),
        (300, 1360), (235, 1300)
    ]),
    "left_lower_leg.png": poly([
        (275, 1300), (480, 1300), (475, 1435), (470, 1510),
        (225, 1510), (230, 1430)
    ]),
    "right_upper_leg.png": poly([
        (515, 1135), (795, 1135), (795, 1300), (735, 1360),
        (545, 1360), (515, 1295)
    ]),
    "right_lower_leg.png": poly([
        (545, 1300), (775, 1300), (805, 1430), (815, 1510),
        (565, 1510), (555, 1435)
    ])
}

layers = {}
movable = np.zeros((H, W), dtype=np.uint8)

for name, raw in masks.items():
    m = source_mask(soften(raw, 0.8))
    m = cv2.dilate(m, np.ones((5, 5), np.uint8), iterations=1)
    layers[name] = m
    movable = np.maximum(movable, m)

# Reconstruct the static base underneath all movable pieces.
hole = (movable > 24).astype(np.uint8) * 255
hole = cv2.dilate(hole, np.ones((7, 7), np.uint8), iterations=1)
filled = cv2.inpaint(rgb, hole, 7, cv2.INPAINT_TELEA)

base_alpha = alpha.copy()
base_alpha[movable > 24] = 0
base = Image.fromarray(filled).convert("RGBA")
base.putalpha(Image.fromarray(base_alpha, "L"))
base.save(OUT / "base.png")

for name, mask in layers.items():
    layer = img.copy()
    layer.putalpha(Image.fromarray(mask, "L"))
    layer.save(OUT / name)

# Reconstruction preview.
order = [
    "base.png",
    "left_upper_leg.png", "right_upper_leg.png",
    "left_lower_leg.png", "right_lower_leg.png",
    "torso.png",
    "left_upper_arm.png", "right_upper_arm.png",
    "left_lower_arm.png", "right_lower_arm.png",
    "head.png"
]
preview = Image.new("RGBA", (W, H), (0, 0, 0, 0))
for name in order:
    preview.alpha_composite(Image.open(OUT / name))
preview.save(OUT / "reconstructed_preview.png")

src = np.array(img).astype(np.int16)
rec = np.array(preview).astype(np.int16)
rgb_error = float(np.abs(src[:, :, :3] - rec[:, :, :3]).mean())
alpha_error = float(np.abs(src[:, :, 3] - rec[:, :, 3]).mean())

manifest = {
    "character": "Bobo the Bear",
    "source_reference": "../../assets/characters/bobo.png",
    "canvas": {"width": W, "height": H, "transparent_background": True},
    "rig_type": "automated_2d_cutout_stage3_multi_joint_experiment",
    "status": "experimental",
    "layers": ["base.png", "head.png", "torso.png",
               "left_upper_arm.png", "left_lower_arm.png",
               "right_upper_arm.png", "right_lower_arm.png",
               "left_upper_leg.png", "left_lower_leg.png",
               "right_upper_leg.png", "right_lower_leg.png"],
    "joint_targets": {
        "left_shoulder": [300, 720],
        "left_elbow": [300, 920],
        "right_shoulder": [735, 720],
        "right_elbow": [735, 920],
        "left_hip": [365, 1160],
        "left_knee": [355, 1330],
        "right_hip": [650, 1160],
        "right_knee": [650, 1330]
    },
    "known_limits": [
        "This is an automated experiment, not the final production rig.",
        "Hidden joint geometry is reconstructed only by conservative inpainting.",
        "Hands remain attached to lower-arm pieces and feet remain attached to lower-leg pieces.",
        "Visual approval is required before production animation."
    ],
    "reconstruction_check": {
        "mean_rgb_error": round(rgb_error, 4),
        "mean_alpha_error": round(alpha_error, 4)
    }
}

(OUT / "rig_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Stage-3 Bobo rig written to {OUT}")
print(f"Mean RGB reconstruction error: {rgb_error:.4f}")
print(f"Mean alpha reconstruction error: {alpha_error:.4f}")
