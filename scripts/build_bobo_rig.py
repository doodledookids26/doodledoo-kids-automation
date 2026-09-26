from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets" / "characters" / "bobo.png"
OUT = ROOT / "output" / "bobo_rig"
OUT.mkdir(parents=True, exist_ok=True)

if not SOURCE.exists():
    raise FileNotFoundError(f"Missing Bobo source: {SOURCE}")

img = Image.open(SOURCE).convert("RGBA")
width, height = img.size
if (width, height) != (1024, 1536):
    raise ValueError(f"Expected Bobo source 1024x1536, got {width}x{height}")

alpha = img.getchannel("A")

def polygon_mask(points):
    m = Image.new("L", (width, height), 0)
    ImageDraw.Draw(m).polygon(points, fill=255)
    return m.filter(ImageFilter.GaussianBlur(2))

def ellipse_mask(box):
    m = Image.new("L", (width, height), 0)
    ImageDraw.Draw(m).ellipse(box, fill=255)
    return m.filter(ImageFilter.GaussianBlur(2))

def intersect(mask):
    return Image.fromarray(
        np.minimum(np.array(mask, dtype=np.uint8),
                   np.array(alpha, dtype=np.uint8)),
        "L",
    )

def layer_from_mask(mask):
    layer = img.copy()
    layer.putalpha(mask)
    return layer

# Conservative masks around the visible movable regions.
masks = {
    "head.png": ellipse_mask((100, 60, 920, 790)),
    "left_arm.png": polygon_mask([
        (220, 780), (340, 760), (410, 870), (365, 1020),
        (350, 1160), (300, 1210), (220, 1190), (160, 1120),
        (175, 980)
    ]),
    "right_arm.png": polygon_mask([
        (620, 780), (800, 760), (850, 900), (875, 1060),
        (860, 1160), (800, 1210), (735, 1190), (700, 1080),
        (650, 960)
    ]),
    "left_leg.png": polygon_mask([
        (245, 1180), (470, 1180), (450, 1495), (235, 1495)
    ]),
    "right_leg.png": polygon_mask([
        (550, 1180), (800, 1180), (820, 1495), (565, 1495)
    ]),
}

masks = {name: intersect(mask) for name, mask in masks.items()}

# Remove all movable regions from the base and reconstruct the exposed
# background/cloth/fur areas with OpenCV inpainting.
movable = np.zeros((height, width), dtype=np.uint8)
for mask in masks.values():
    movable = np.maximum(movable, np.array(mask, dtype=np.uint8))

base_alpha = np.array(alpha, dtype=np.uint8).copy()
base_alpha[movable > 20] = 0

rgb = np.array(img.convert("RGB"))
hole = (movable > 20).astype(np.uint8) * 255
hole = cv2.dilate(hole, np.ones((9, 9), np.uint8), iterations=1)
filled = cv2.inpaint(rgb, hole, 9, cv2.INPAINT_TELEA)

base = Image.fromarray(filled).convert("RGBA")
base.putalpha(Image.fromarray(base_alpha, "L"))
base.save(OUT / "base.png")

for filename, mask in masks.items():
    layer_from_mask(mask).save(OUT / filename)

# Reconstruct a preview to verify that the layers still reproduce the source.
preview = Image.new("RGBA", (width, height), (0, 0, 0, 0))
for filename in ["base.png", "left_leg.png", "right_leg.png",
                 "left_arm.png", "right_arm.png", "head.png"]:
    preview.alpha_composite(Image.open(OUT / filename))
preview.save(OUT / "reconstructed_preview.png")

diff = np.abs(
    np.array(img).astype(np.int16) -
    np.array(preview).astype(np.int16)
)
mean_rgb_error = float(diff[:, :, :3].mean())
alpha_error = float(diff[:, :, 3].mean())

manifest = {
    "character": "Bobo the Bear",
    "source_reference": "../../assets/characters/bobo.png",
    "canvas": {"width": width, "height": height, "transparent_background": True},
    "rig_type": "automated_2d_cutout_stage1",
    "status": "generated",
    "layers": [
        "base.png",
        "head.png",
        "left_arm.png",
        "right_arm.png",
        "left_leg.png",
        "right_leg.png"
    ],
    "animation_capabilities": [
        "idle",
        "head_turn",
        "look_left_right",
        "wave",
        "walk",
        "jump",
        "simple_dance"
    ],
    "limitations": [
        "Face remains part of the head layer in stage 1.",
        "Arms and legs are single rigid pieces in stage 1.",
        "Stage 2 can subdivide limbs and separate eyes/mouth."
    ],
    "reconstruction_check": {
        "mean_rgb_error": round(mean_rgb_error, 4),
        "mean_alpha_error": round(alpha_error, 4),
        "source_reconstructed": mean_rgb_error < 2.0 and alpha_error < 3.0
    }
}

(OUT / "rig_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(f"Bobo rig generated in {OUT}")
print(f"Reconstruction mean RGB error: {mean_rgb_error:.4f}")
print(f"Reconstruction mean alpha error: {alpha_error:.4f}")
print(f"Source reconstructed: {manifest['reconstruction_check']['source_reconstructed']}")
