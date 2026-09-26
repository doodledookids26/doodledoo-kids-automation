from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets" / "characters" / "bobo.png"
SPEC = ROOT / "config" / "bobo_rig_spec.json"
OUT = ROOT / "output" / "bobo_rig_blueprint"
OUT.mkdir(parents=True, exist_ok=True)

img = Image.open(SOURCE).convert("RGBA")
spec = json.loads(SPEC.read_text(encoding="utf-8"))

canvas = img.copy()
draw = ImageDraw.Draw(canvas)

joints = spec["joint_targets_1024x1536"]
radius = 13

# Draw the kinematic skeleton over the original Bobo artwork.
for name, (x, y) in joints.items():
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(0, 255, 0, 220))
    draw.text((x+18, y-10), name, fill=(255, 255, 255, 255))

connections = [
    ("neck", "left_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("neck", "right_shoulder"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("neck", "left_hip"), ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("neck", "right_hip"), ("right_hip", "right_knee"),
    ("right_knee", "right_ankle")
]

for a, b in connections:
    x1, y1 = joints[a]
    x2, y2 = joints[b]
    draw.line((x1, y1, x2, y2), fill=(0, 220, 255, 220), width=8)

canvas.save(OUT / "bobo_joint_blueprint.png")
(OUT / "bobo_rig_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

print(f"Created {OUT / 'bobo_joint_blueprint.png'}")
