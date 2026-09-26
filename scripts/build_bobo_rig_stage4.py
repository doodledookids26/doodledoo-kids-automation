from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets" / "characters" / "bobo.png"
OUT = ROOT / "output" / "bobo_rig_stage4"
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

def mask_from_shape(shape):
    return np.minimum(shape, alpha)

def expand(mask, px=10):
    return cv2.dilate(mask, np.ones((px, px), np.uint8), iterations=1)

def feather(mask, radius=1.0):
    return np.array(Image.fromarray(mask, "L").filter(ImageFilter.GaussianBlur(radius)), dtype=np.uint8)

# Stage 4: joint-safe puppet experiment.
# Pieces overlap around joints intentionally. Distal hand/foot pieces remain
# attached to their forearm/lower-leg so they can be rotated without cutting
# off the original silhouette.
shapes = {
    "head.png": ellipse((105,45,925,720)),
    "torso.png": poly([(270,620),(755,620),(790,760),(775,955),(735,1135),(690,1200),
                       (335,1200),(285,1120),(250,930),(245,760)]),

    "left_upper_arm.png": poly([(245,660),(375,655),(420,735),(405,820),(375,895),
                                (335,945),(245,925),(195,850),(210,745)]),
    "left_lower_arm.png": poly([(245,875),(350,875),(385,940),(370,1030),(350,1120),
                                (330,1190),(270,1220),(195,1185),(165,1120),(180,1020),(205,935)]),
    "left_hand.png": ellipse((155,1110,345,1245)),

    "right_upper_arm.png": poly([(645,655),(775,670),(810,750),(825,835),(805,910),
                                 (755,955),(675,925),(640,845),(625,750)]),
    "right_lower_arm.png": poly([(675,875),(770,875),(815,945),(840,1030),(860,1135),
                                 (825,1210),(750,1220),(700,1170),(680,1070),(655,970)]),
    "right_hand.png": ellipse((735,1110,875,1245)),

    "left_upper_leg.png": poly([(225,1125),(505,1125),(505,1295),(470,1370),(295,1370),(225,1300)]),
    "left_lower_leg.png": poly([(270,1290),(485,1290),(485,1435),(475,1520),(215,1520),(220,1430)]),
    "left_foot.png": ellipse((205,1430,490,1535)),

    "right_upper_leg.png": poly([(510,1125),(805,1125),(805,1300),(735,1370),(545,1370),(510,1295)]),
    "right_lower_leg.png": poly([(540,1290),(785,1290),(810,1430),(820,1520),(555,1520),(550,1435)]),
    "right_foot.png": ellipse((545,1430,830,1535))
}

# Prevent very small pieces from becoming empty while keeping antialiased edges.
layers = {}
movable = np.zeros((H,W), np.uint8)
for name, shape in shapes.items():
    m = feather(mask_from_shape(shape), 0.8)
    m = expand(m, 7)
    layers[name] = m
    movable = np.maximum(movable, m)

# Static base: remove all puppet pieces and reconstruct only the exposed area.
hole = (movable > 24).astype(np.uint8) * 255
hole = cv2.dilate(hole, np.ones((9,9),np.uint8), iterations=1)
filled = cv2.inpaint(rgb, hole, 9, cv2.INPAINT_TELEA)
base_alpha = alpha.copy()
base_alpha[movable > 24] = 0
base = Image.fromarray(filled).convert("RGBA")
base.putalpha(Image.fromarray(base_alpha,"L"))
base.save(OUT/"base.png")

for name, m in layers.items():
    p = img.copy()
    p.putalpha(Image.fromarray(m,"L"))
    p.save(OUT/name)

# Rebuild the original pose for a visual reconstruction check.
order = [
    "base.png",
    "left_upper_leg.png","right_upper_leg.png",
    "left_lower_leg.png","right_lower_leg.png",
    "left_foot.png","right_foot.png",
    "torso.png",
    "left_upper_arm.png","right_upper_arm.png",
    "left_lower_arm.png","right_lower_arm.png",
    "left_hand.png","right_hand.png",
    "head.png"
]
preview = Image.new("RGBA",(W,H),(0,0,0,0))
for name in order:
    preview.alpha_composite(Image.open(OUT/name))
preview.save(OUT/"reconstructed_preview.png")

src=np.array(img).astype(np.int16)
rec=np.array(preview).astype(np.int16)
manifest={
    "character":"Bobo the Bear",
    "source_reference":"../../assets/characters/bobo.png",
    "canvas":{"width":W,"height":H,"transparent_background":True},
    "rig_type":"automated_2d_joint_safe_stage4_experiment",
    "status":"experimental",
    "layers":order,
    "joint_targets":{
        "left_shoulder":[300,720],"left_elbow":[300,920],"left_wrist":[275,1160],
        "right_shoulder":[735,720],"right_elbow":[735,920],"right_wrist":[800,1160],
        "left_hip":[365,1160],"left_knee":[365,1330],"left_ankle":[355,1450],
        "right_hip":[650,1160],"right_knee":[650,1330],"right_ankle":[685,1450]
    },
    "notes":[
        "Large overlaps are intentional to reduce visible gaps during rotation.",
        "Hands and feet are explicit pieces for controlled articulation.",
        "This remains an experiment; hidden joint reconstruction is limited by the flattened source."
    ],
    "reconstruction_check":{
        "mean_rgb_error":round(float(np.abs(src[:,:,:3]-rec[:,:,:3]).mean()),4),
        "mean_alpha_error":round(float(np.abs(src[:,:,3]-rec[:,:,3]).mean()),4)
    }
}
(OUT/"rig_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
print(f"Stage-4 rig written to {OUT}")
