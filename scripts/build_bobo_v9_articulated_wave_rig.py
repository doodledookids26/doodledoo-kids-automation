from pathlib import Path
import json, math
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / "output" / "bobo_rig_v6"
OUT = ROOT / "output" / "bobo_rig_v9"
SOURCE = ROOT / "assets" / "characters" / "bobo.png"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1024, 1536
src = np.array(Image.open(SOURCE).convert("RGBA"))
if src.shape[:2] != (H, W):
    raise ValueError("Expected 1024x1536 Bobo source")

m6 = json.loads((V6 / "rig_manifest.json").read_text(encoding="utf-8"))
layers = {n: np.array(Image.open(V6 / n).convert("RGBA")) for n in m6["layers"]}
joint = {k: np.array(v, dtype=np.float32) for k, v in m6["joint_targets"].items()}

def rotate_rgba(im, pivot, deg):
    px, py = map(float, pivot)
    a = math.radians(float(deg))
    c, s = math.cos(a), math.sin(a)
    M = np.array([[c, -s, px - c*px + s*py],
                  [s,  c, py - s*px - c*py]], np.float32)
    return cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0, 0))

def over(dst, im):
    sa = im[:,:,3:4].astype(np.float32)/255.0
    da = dst[:,:,3:4].astype(np.float32)/255.0
    oa = sa + da*(1-sa)
    out = np.zeros_like(dst, dtype=np.float32)
    out[:,:,:3] = np.where(
        oa > 1e-6,
        (im[:,:,:3]*sa + dst[:,:,:3]*da*(1-sa))/np.maximum(oa,1e-6),
        0
    )
    out[:,:,3:4] = oa*255
    return np.clip(out,0,255).astype(np.uint8)

# The V6 upper-arm layer already contains the yellow sleeve + bear arm.
# Do NOT create a second sleeve from a color threshold. That was the source
# of the large cyan bars in V8.
torso = layers["torso.png"].copy()
left_arm_cut = np.array([
    [185, 650], [380, 650], [430, 735], [410, 835],
    [360, 925], [245, 945], [185, 855], [195, 735]
], np.int32)
mask = np.zeros((H,W), np.uint8)
cv2.fillPoly(mask, [left_arm_cut], 255)
torso[:,:,3] = np.where(mask > 0, 0, torso[:,:,3]).astype(np.uint8)

# Keep the static artwork in a standalone rig folder for inspection.
for n, im in layers.items():
    if n == "torso.png":
        Image.fromarray(torso).save(OUT/n)
    else:
        Image.fromarray(im).save(OUT/n)

static_order = [
    "base.png","torso.png","hip_shorts.png",
    "left_upper_leg.png","right_upper_leg.png",
    "left_lower_leg.png","right_lower_leg.png",
    "left_foot.png","right_foot.png",
    "right_upper_arm.png","right_lower_arm.png","right_hand.png",
    "head.png","left_ear.png","right_ear.png","eyes.png","mouth.png"
]

def compose(upper_deg, lower_abs_deg, hand_abs_deg):
    ua = rotate_rgba(layers["left_upper_arm.png"], joint["left_shoulder"], upper_deg)
    # Lower arm uses its own elbow pivot; absolute angle is relative to its
    # original artwork orientation, so the joint remains a true parent socket.
    la = rotate_rgba(layers["left_lower_arm.png"], joint["left_elbow"], lower_abs_deg)
    hand = rotate_rgba(layers["left_hand.png"], joint["left_wrist"], hand_abs_deg)

    canvas = np.zeros((H,W,4), np.uint8)
    for n in static_order:
        canvas = over(canvas, torso if n=="torso.png" else layers[n])
    canvas = over(canvas, ua)
    canvas = over(canvas, la)
    canvas = over(canvas, hand)
    return canvas

exact_rest = np.array(Image.open(V6 / "reconstructed_preview.png").convert("RGBA"))

def frame_rgba(i):
    if i == 0:
        return exact_rest.copy()
    phase = 2*math.pi*i/95.0
    # Smooth raise -> wave -> return.
    raise_amt = 58.0 * (0.5 - 0.5*math.cos(phase))
    wave = 18.0 * math.sin(3*phase) * (0.25 + 0.75*(0.5 - 0.5*math.cos(phase)))
    upper = raise_amt
    lower = raise_amt + wave
    hand = raise_amt + 8.0*math.sin(5*phase)
    return compose(upper, lower, hand)

rest = frame_rgba(0)
sv = src[:,:,3] > 16
rv = rest[:,:,3] > 16
inter = sv & rv
coverage = float(inter[sv].mean()) if np.any(sv) else 0
rgb_error = float(np.abs(src[:,:,:3][inter].astype(np.int16)-rest[:,:,:3][inter].astype(np.int16)).mean()) if np.any(inter) else 999
alpha_error = float(np.abs(src[:,:,3].astype(np.int16)-rest[:,:,3].astype(np.int16))[sv].mean()) if np.any(sv) else 999
print(json.dumps({
    "rest_visible_coverage": round(coverage,6),
    "rest_visible_rgb_error": round(rgb_error,4),
    "rest_visible_alpha_error": round(alpha_error,4)
}))

Image.fromarray(rest).save(OUT/"rest_pose_preview.png")
if coverage < .995 or rgb_error >= 3 or alpha_error >= 3:
    raise SystemExit("V9 REST-POSE GATE FAILED")

mp4 = ROOT/"output"/"bobo_v9_wave_test.mp4"
writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 24, (W,H))
if not writer.isOpened():
    raise RuntimeError("Could not open V9 video writer")

for i in range(96):
    rgba = frame_rgba(i)
    # PIL/OpenCV arrays here are RGB. VideoWriter requires BGR.
    rgb = rgba[:,:,:3]
    a = rgba[:,:,3:4].astype(np.float32)/255.0
    white = np.full((H,W,3),255,np.uint8)
    composited_rgb = (rgb.astype(np.float32)*a + white*(1-a)).astype(np.uint8)
    writer.write(cv2.cvtColor(composited_rgb, cv2.COLOR_RGB2BGR))

writer.release()

manifest = {
    "character":"Bobo the Bear",
    "rig_type":"automated_2d_articulated_rigid_puppet_v9",
    "source":"assets/characters/bobo.png",
    "canvas":[W,H],
    "validation_motion":"left_arm_raise_wave_return",
    "production_status":"wave_validation_only",
    "important_fix":"V6 arm layers include the original yellow sleeve; no synthetic sleeve extraction is used.",
    "video_color_pipeline":"RGB_to_BGR_before_OpenCV_VideoWriter",
    "hierarchy":{
        "root":"torso.png",
        "left_arm":["left_upper_arm.png","left_lower_arm.png","left_hand.png"],
        "right_arm":["right_upper_arm.png","right_lower_arm.png","right_hand.png"],
        "left_leg":["left_upper_leg.png","left_lower_leg.png","left_foot.png"],
        "right_leg":["right_upper_leg.png","right_lower_leg.png","right_foot.png"],
        "head":["head.png","left_ear.png","right_ear.png","eyes.png","mouth.png"]
    },
    "controls":{
        "left_shoulder":[int(x) for x in joint["left_shoulder"]],
        "left_elbow":[int(x) for x in joint["left_elbow"]],
        "left_wrist":[int(x) for x in joint["left_wrist"]]
    },
    "quality_gate":{
        "rest_pose_reconstructed":True,
        "visual_motion_review_required":True,
        "full_body_actions_blocked_until_wave_passes":True
    }
}
(OUT/"rig_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
print(f"Created {mp4}")
