from pathlib import Path
import json, math
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / "output" / "bobo_rig_v6"
OUT = ROOT / "output" / "bobo_rig_v8"
SOURCE = ROOT / "assets" / "characters" / "bobo.png"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1024, 1536
src = np.array(Image.open(SOURCE).convert("RGBA"))
if src.shape[:2] != (H, W):
    raise ValueError(f"Expected Bobo source {W}x{H}, got {src.shape[1]}x{src.shape[0]}")

manifest6 = json.loads((V6 / "rig_manifest.json").read_text(encoding="utf-8"))
order6 = manifest6["layers"]
layers = {name: np.array(Image.open(V6 / name).convert("RGBA")) for name in order6}
joint = {k: np.array(v, dtype=np.float32) for k, v in manifest6["joint_targets"].items()}

# V8 changes the deformation model. Limbs are not rotated as whole rectangular
# cutouts. A strip/mesh map follows the source limb centerline and transports
# the texture into the posed centerline. The shoulder sleeve is its own textured
# mesh region, so the shirt does not stay rigid while the arm moves.

def rgba_over(dst, src_im):
    sa = src_im[:, :, 3:4].astype(np.float32) / 255.0
    da = dst[:, :, 3:4].astype(np.float32) / 255.0
    oa = sa + da * (1.0 - sa)
    out = np.zeros_like(dst, dtype=np.float32)
    safe = oa > 1e-6
    out[:, :, :3] = np.where(
        safe,
        (src_im[:, :, :3] * sa + dst[:, :, :3] * da * (1.0 - sa)) / np.maximum(oa, 1e-6),
        0.0,
    )
    out[:, :, 3:4] = oa * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)

def rot_point(p, center, deg):
    a = math.radians(float(deg))
    c, s = math.cos(a), math.sin(a)
    q = p - center
    return center + np.array([c*q[0] - s*q[1], s*q[0] + c*q[1]], dtype=np.float32)

def segment_warp(im, src_a, src_b, dst_a, dst_b, pad=60):
    """Texture-preserving strip mesh warp for one articulated segment."""
    src_a, src_b = map(lambda x: np.asarray(x, np.float32), (src_a, src_b))
    dst_a, dst_b = map(lambda x: np.asarray(x, np.float32), (dst_a, dst_b))
    sv = src_b - src_a
    dv = dst_b - dst_a
    sl = float(np.linalg.norm(sv))
    dl = float(np.linalg.norm(dv))
    if sl < 1 or dl < 1:
        return im.copy()

    su = sv / sl
    du = dv / dl
    sn = np.array([-su[1], su[0]], dtype=np.float32)
    dn = np.array([-du[1], du[0]], dtype=np.float32)

    ys, xs = np.where(im[:, :, 3] > 8)
    if len(xs) == 0:
        return im.copy()

    # Width of the source artwork around the segment.
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    rel = pts - src_a
    t0 = np.clip((rel @ su) / sl, 0.0, 1.0)
    normal = rel - t0[:, None] * sl * su[None, :]
    half_width = float(np.percentile(np.linalg.norm(normal, axis=1), 98))
    half_width = max(18.0, min(180.0, half_width + 8.0))

    x0 = max(0, int(min(dst_a[0], dst_b[0]) - half_width - pad))
    x1 = min(W - 1, int(max(dst_a[0], dst_b[0]) + half_width + pad))
    y0 = max(0, int(min(dst_a[1], dst_b[1]) - half_width - pad))
    y1 = min(H - 1, int(max(dst_a[1], dst_b[1]) + half_width + pad))

    gy, gx = np.mgrid[y0:y1+1, x0:x1+1].astype(np.float32)
    p = np.stack([gx - dst_a[0], gy - dst_a[1]], axis=-1)
    t = np.clip((p[..., 0]*du[0] + p[..., 1]*du[1]) / dl, 0.0, 1.0)
    along = t * dl
    n = p[..., 0]*dn[0] + p[..., 1]*dn[1]

    # Smooth endpoint blending keeps the joint from shearing abruptly.
    edge = np.clip(np.minimum(t, 1.0-t) * 8.0, 0.0, 1.0)
    src_xy = src_a[None,None,:] + t[...,None] * sv[None,None,:] + n[...,None] * sn[None,None,:]

    map_x = src_xy[...,0].astype(np.float32)
    map_y = src_xy[...,1].astype(np.float32)
    warped = cv2.remap(im, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    dist = np.abs(n)
    support = np.clip((half_width + 4.0 - dist) / 8.0, 0.0, 1.0)
    # Preserve source alpha shape; the support mask only limits the destination.
    warped[:, :, 3] = np.clip(warped[:, :, 3].astype(np.float32) * support, 0, 255).astype(np.uint8)

    out = np.zeros_like(im)
    out[y0:y1+1, x0:x1+1] = warped
    return out

def rigid_warp(im, pivot, deg):
    a = math.radians(float(deg))
    c, s = math.cos(a), math.sin(a)
    M = np.array([[c, -s, pivot[0] - c*pivot[0] + s*pivot[1]],
                  [s,  c, pivot[1] - s*pivot[0] - c*pivot[1]]], np.float32)
    return cv2.warpAffine(im, M, (W,H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

# Extract the two yellow sleeve regions from the original artwork. The connected
# components are spatially constrained to the shoulder zones, avoiding the body
# of the shirt. The static torso is cut by exactly the same mask.
hsv = cv2.cvtColor(src[:, :, :3], cv2.COLOR_RGB2HSV)
yellow = (
    (hsv[:,:,0] >= 15) & (hsv[:,:,0] <= 35) &
    (hsv[:,:,1] > 70) & (hsv[:,:,2] > 70) &
    (src[:,:,3] > 8)
)

def shoulder_sleeve(side):
    if side == "left":
        x0,x1,y0,y1 = 190, 455, 675, 950
        shoulder = tuple(joint["left_shoulder"].astype(int))
    else:
        x0,x1,y0,y1 = 555, 860, 675, 950
        shoulder = tuple(joint["right_shoulder"].astype(int))
    roi = (yellow[y0:y1, x0:x1]).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(roi, 8)
    if n <= 1:
        raise RuntimeError(f"Could not find {side} sleeve component")
    best = 1
    best_score = -1e9
    for i in range(1,n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 500:
            continue
        cx,cy = cents[i][0]+x0, cents[i][1]+y0
        d = math.hypot(cx-shoulder[0], cy-shoulder[1])
        score = area - 250*d
        if score > best_score:
            best_score, best = score, i
    mask = lab == best
    full = np.zeros((H,W), np.uint8)
    full[y0:y1,x0:x1] = (mask*255).astype(np.uint8)
    full = cv2.dilate(full, np.ones((3,3),np.uint8), 1)
    rgba = src.copy()
    rgba[:,:,3] = np.where(full>0, src[:,:,3], 0).astype(np.uint8)
    return rgba, full>0

left_sleeve, left_sleeve_mask = shoulder_sleeve("left")
right_sleeve, right_sleeve_mask = shoulder_sleeve("right")

# Static torso with sleeve pixels removed. This prevents a rigid yellow sleeve from
# being left behind when the arm is raised.
torso = layers["torso.png"].copy()
torso[:,:,3] = np.where(left_sleeve_mask | right_sleeve_mask, 0, torso[:,:,3]).astype(np.uint8)

# Copy all non-deforming layers first.
static_names = [
    "base.png","torso.png","hip_shorts.png",
    "left_upper_leg.png","right_upper_leg.png","left_lower_leg.png","right_lower_leg.png",
    "left_foot.png","right_foot.png",
    "head.png","left_ear.png","right_ear.png","eyes.png","mouth.png",
]
for name in static_names:
    if name == "torso.png":
        Image.fromarray(torso).save(OUT / name)
    else:
        Image.fromarray(layers[name]).save(OUT / name)

# Wave pose: left arm raises as one chain, then bends at the elbow.
def make_frame(frame_no):
    t = frame_no / 24.0
    phase = 2.0 * math.pi * t / 2.0
    motion = 0.5 - 0.5*math.cos(phase)
    upper_deg = -48.0 * motion
    elbow_rel = 30.0 * math.sin(phase)
    hand_deg = 12.0 * math.sin(2.0*phase)

    sh = joint["left_shoulder"]
    el = joint["left_elbow"]
    wr = joint["left_wrist"]

    posed_el = rot_point(el, sh, upper_deg)
    posed_wr = rot_point(wr, el, upper_deg + elbow_rel)

    # Upper arm follows shoulder->elbow.
    ua = segment_warp(layers["left_upper_arm.png"], sh, el, sh, posed_el)
    la = segment_warp(layers["left_lower_arm.png"], el, wr, posed_el, posed_wr)
    hand = rigid_warp(layers["left_hand.png"], posed_wr, hand_deg)

    # Sleeve follows the upper-arm mesh, using the same shoulder/elbow path.
    sleeve = segment_warp(left_sleeve, sh, el, sh, posed_el)

    # Right arm remains in a stable rest pose.
    rua = layers["right_upper_arm.png"]
    rla = layers["right_lower_arm.png"]
    rhand = layers["right_hand.png"]

    canvas = np.zeros((H,W,4),np.uint8)
    for name in ["base.png","torso.png","hip_shorts.png",
                 "left_upper_leg.png","right_upper_leg.png","left_lower_leg.png","right_lower_leg.png",
                 "left_foot.png","right_foot.png",
                 "right_upper_arm.png","left_upper_arm.png",
                 "right_lower_arm.png","left_lower_arm.png",
                 "right_hand.png","left_hand.png",
                 "head.png","left_ear.png","right_ear.png","eyes.png","mouth.png"]:
        im = torso if name=="torso.png" else (
            ua if name=="left_upper_arm.png" else
            la if name=="left_lower_arm.png" else
            hand if name=="left_hand.png" else
            layers[name]
        )
        canvas = rgba_over(canvas, im)

    # Sleeves sit over the top of the brown upper arm at the shoulder seam.
    canvas = rgba_over(canvas, sleeve)
    canvas = rgba_over(canvas, right_sleeve)
    return canvas

out_mp4 = ROOT / "output" / "bobo_v8_mesh_wave_test.mp4"
writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 24, (W,H))
if not writer.isOpened():
    raise RuntimeError("Could not open V8 output")

# Rest-pose reconstruction is measured before animation.
rest = make_frame(0)
src16 = src.astype(np.int16)
rest16 = rest.astype(np.int16)
visible = src[:,:,3] > 16
inter = visible & (rest[:,:,3] > 16)
coverage = float(inter[visible].mean()) if np.any(visible) else 0.0
rgb_error = float(np.abs(src16[:,:,:3][inter]-rest16[:,:,:3][inter]).mean()) if np.any(inter) else 999.0
alpha_error = float(np.abs(src16[:,:,3]-rest16[:,:,3])[visible].mean()) if np.any(visible) else 999.0

Image.fromarray(rest).save(OUT / "rest_pose_preview.png")
print(json.dumps({
    "rest_visible_coverage": round(coverage,6),
    "rest_visible_rgb_error": round(rgb_error,4),
    "rest_visible_alpha_error": round(alpha_error,4)
}))

if coverage < 0.995 or rgb_error >= 3.0 or alpha_error >= 3.0:
    raise SystemExit("V8 REST-POSE RECONSTRUCTION GATE FAILED")

for frame_no in range(96):
    canvas = make_frame(frame_no)
    bg = np.full((H,W,3),255,np.uint8)
    a = canvas[:,:,3:4].astype(np.float32)/255.0
    frame = (canvas[:,:,:3].astype(np.float32)*a + bg*(1-a)).astype(np.uint8)
    writer.write(frame)

writer.release()

manifest = {
    "character":"Bobo the Bear",
    "rig_type":"automated_2d_mesh_strip_deformation_v8",
    "source":"assets/characters/bobo.png",
    "canvas":[W,H],
    "deformation":"centerline strip mesh with articulated shoulder sleeve",
    "validation_motion":"left_arm_wave_only",
    "production_status":"mesh_validation_only",
    "hierarchy":{
        "root":"torso.png",
        "left_arm":["left_upper_arm.png","left_lower_arm.png","left_hand.png"],
        "right_arm":["right_upper_arm.png","right_lower_arm.png","right_hand.png"],
        "left_leg":["left_upper_leg.png","left_lower_leg.png","left_foot.png"],
        "right_leg":["right_upper_leg.png","right_lower_leg.png","right_foot.png"],
        "head":["head.png","left_ear.png","right_ear.png","eyes.png","mouth.png"]
    },
    "controls":{
        "left_shoulder":[int(x) for x in sh],
        "left_elbow_rest":[int(x) for x in el],
        "left_wrist_rest":[int(x) for x in wr]
    },
    "quality_gate":{
        "rest_pose_reconstructed":True,
        "visual_motion_review_required":True,
        "walk_run_jump_dance_blocked_until_wave_passes":True
    }
}
(OUT/"rig_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
print(f"Created {out_mp4} with 96 frames at 24 FPS")
