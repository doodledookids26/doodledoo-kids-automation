from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets" / "characters" / "bobo.png"
OUT = ROOT / "output" / "bobo_rig_stage2"
OUT.mkdir(parents=True, exist_ok=True)

img = Image.open(SOURCE).convert("RGBA")
width, height = img.size
if (width, height) != (1024, 1536):
    raise ValueError(f"Expected Bobo source 1024x1536, got {width}x{height}")

alpha = np.array(img.getchannel("A"), dtype=np.uint8)

def poly(points):
    m = Image.new("L", (width, height), 0)
    ImageDraw.Draw(m).polygon(points, fill=255)
    return np.array(m, dtype=np.uint8)

def ellipse(box):
    m = Image.new("L", (width, height), 0)
    ImageDraw.Draw(m).ellipse(box, fill=255)
    return np.array(m, dtype=np.uint8)

def soften(mask, radius=1.2):
    pil = Image.fromarray(mask, "L").filter(ImageFilter.GaussianBlur(radius))
    return np.array(pil, dtype=np.uint8)

def source_mask(mask):
    return np.minimum(mask, alpha)

# Stage 2 intentionally uses six large, seam-tolerant pieces rather than
# tiny rigid segments.  The masks are deliberately conservative around
# clothing boundaries so the yellow shirt and blue shorts remain intact.
masks = {
    "head.png": ellipse((105, 45, 925, 720)),
    "left_arm.png": poly([
        (285, 665), (365, 675), (405, 760), (385, 850),
        (360, 955), (345, 1060), (320, 1150), (285, 1205),
        (215, 1200), (165, 1150), (170, 1060), (190, 960),
        (205, 865), (225, 770), (250, 700)
    ]),
    "right_arm.png": poly([
        (650, 670), (735, 665), (770, 735), (795, 825),
        (815, 930), (835, 1040), (850, 1140), (815, 1200),
        (745, 1205), (700, 1150), (680, 1050), (660, 950),
        (640, 850), (625, 760)
    ]),
    "left_leg.png": poly([
        (235, 1170), (500, 1170), (495, 1310), (480, 1430),
        (470, 1510), (225, 1510), (230, 1430), (235, 1320)
    ]),
    "right_leg.png": poly([
        (525, 1170), (795, 1170), (800, 1320), (805, 1430),
        (815, 1510), (565, 1510), (555, 1430), (540, 1310)
    ]),
}

layers = {}
movable = np.zeros((height, width), dtype=np.uint8)

for name, raw in masks.items():
    m = source_mask(soften(raw, 0.8))
    # Small dilation preserves antialiased fur edges and prevents a hard
    # one-pixel seam when the layer is transformed.
    m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)
    layers[name] = m
    movable = np.maximum(movable, m)

# Reconstruct the base where movable pieces were removed.  Inpainting is
# limited to a small neighborhood; this is only a seam-preparation step,
# not an attempt to hallucinate large hidden body regions.
rgb = np.array(img.convert("RGB"))
hole = (movable > 24).astype(np.uint8) * 255
hole = cv2.dilate(hole, np.ones((5, 5), np.uint8), iterations=1)
filled = cv2.inpaint(rgb, hole, 5, cv2.INPAINT_TELEA)

base_alpha = alpha.copy()
base_alpha[movable > 24] = 0
base = Image.fromarray(filled).convert("RGBA")
base.putalpha(Image.fromarray(base_alpha, "L"))
base.save(OUT / "base.png")

for name, mask in layers.items():
    layer = img.copy()
    layer.putalpha(Image.fromarray(mask, "L"))
    layer.save(OUT / name)

# Preserve the original visual stacking order for the reconstruction check.
preview = Image.new("RGBA", (width, height), (0, 0, 0, 0))
for name in ["base.png", "left_leg.png", "right_leg.png",
             "left_arm.png", "right_arm.png", "head.png"]:
    preview.alpha_composite(Image.open(OUT / name))
preview.save(OUT / "reconstructed_preview.png")

src = np.array(img).astype(np.int16)
rec = np.array(preview).astype(np.int16)
rgb_error = float(np.abs(src[:, :, :3] - rec[:, :, :3]).mean())
alpha_error = float(np.abs(src[:, :, 3] - rec[:, :, 3]).mean())

manifest = {
    "character": "Bobo the Bear",
    "source_reference": "../../assets/characters/bobo.png",
    "canvas": {"width": width, "height": height, "transparent_background": True},
    "rig_type": "automated_2d_cutout_stage2_six_piece",
    "status": "experimental",
    "layers": [
        "base.png", "head.png",
        "left_arm.png", "right_arm.png",
        "left_leg.png", "right_leg.png"
    ],
    "design_goal": "large seam-tolerant pieces with intact hands and feet",
    "known_limits": [
        "Hidden joint geometry is not fully reconstructed.",
        "Arms and legs are single pieces in this stage.",
        "Visual approval is required before animation use."
    ],
    "reconstruction_check": {
        "mean_rgb_error": round(rgb_error, 4),
        "mean_alpha_error": round(alpha_error, 4)
    }
}

(OUT / "rig_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(f"Stage-2 Bobo rig written to {OUT}")
print(f"Mean RGB reconstruction error: {rgb_error:.4f}")
print(f"Mean alpha reconstruction error: {alpha_error:.4f}")
