from pathlib import Path
import json, math
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "output" / "bobo_rig_v6"
OUT = ROOT / "output" / "bobo_rig_v7"
OUT.mkdir(parents=True, exist_ok=True)

manifest = json.loads((SRC / "rig_manifest.json").read_text(encoding="utf-8"))
W, H = manifest["canvas"]
order = manifest["layers"]
piv = {k: tuple(v) for k, v in manifest["pivots"].items()}
parent = manifest["hierarchy"]
layers = {n: cv2.imread(str(SRC / n), cv2.IMREAD_UNCHANGED) for n in order}

# V7 deliberately adds a small articulated "socket" around each joint.
# The socket is textured from the existing limb artwork and moves with that limb.
# This is intended to prevent exposed background when the limb rotates.
sockets = {
    "left_upper_arm.png": ((300, 720), 34),
    "right_upper_arm.png": ((735, 720), 34),
    "left_lower_arm.png": ((300, 920), 30),
    "right_lower_arm.png": ((735, 920), 30),
    "left_hand.png": ((275, 1160), 24),
    "right_hand.png": ((800, 1160), 24),
    "left_upper_leg.png": ((365, 1160), 24),
    "right_upper_leg.png": ((650, 1160), 24),
    "left_lower_leg.png": ((365, 1330), 28),
    "right_lower_leg.png": ((650, 1330), 28),
    "left_foot.png": ((355, 1460), 24),
    "right_foot.png": ((685, 1460), 24),
    "head.png": ((510, 650), 22),
}

for name, ((x, y), radius) in sockets.items():
    im = layers[name]
    alpha = im[:, :, 3]
    disk = np.zeros((H, W), np.uint8)
    cv2.circle(disk, (x, y), radius, 255, -1)
    dist = cv2.distanceTransform((alpha == 0).astype(np.uint8), cv2.DIST_L2, 5)
    add = (disk > 0) & (alpha == 0) & (dist <= radius + 10)
    if np.any(add):
        _, labels = cv2.distanceTransformWithLabels(
            (alpha == 0).astype(np.uint8), cv2.DIST_L2, 5,
            labelType=cv2.DIST_LABEL_PIXEL
        )
        visible = np.flatnonzero(alpha > 16)
        labels = np.clip(labels.astype(np.int64) - 1, 0, len(visible) - 1)
        rgb = im[:, :, :3].reshape(-1, 3)[visible][labels]
        im[add, :3] = rgb[add]
        im[add, 3] = 255
    layers[name] = im

for name, im in layers.items():
    cv2.imwrite(str(OUT / name), im)

v7 = dict(manifest)
v7["rig_type"] = "automated_2d_joint_rig_v7_socketed"
v7["deformation"] = "hierarchical_affine_with_articulated_joint_sockets"
v7["production_status"] = "wave_validation_only"
v7["validation_motion"] = "wave"
v7["validation_rules"] = {
    "max_upper_arm_degrees": 42,
    "max_lower_arm_degrees": 45,
    "visual_review_required": True,
}
(OUT / "rig_manifest.json").write_text(json.dumps(v7, indent=2), encoding="utf-8")

def T(x, y):
    return np.array([[1., 0., x], [0., 1., y], [0., 0., 1.]], dtype=np.float64)

def R(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.],
                     [math.sin(a), math.cos(a), 0.],
                     [0., 0., 1.]], dtype=np.float64)

def TR(p, deg):
    return T(*p) @ R(deg) @ T(-p[0], -p[1])

def warp(im, matrix):
    return cv2.warpAffine(
        im, matrix[:2], (W, H), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )

def composite(dst, src):
    a = src[:, :, 3:4].astype(np.float32) / 255.
    rgb = src[:, :, :3].astype(np.float32)
    base = dst[:, :, :3].astype(np.float32)
    out = rgb * a + base * (1. - a)
    alpha = np.maximum(dst[:, :, 3:4], src[:, :, 3:4])
    return np.dstack([out.astype(np.uint8), alpha])

# One motion only. No Walk/Run/Jump/Dance are generated until Wave is visually clean.
out_mp4 = ROOT / "output" / "bobo_v7_wave_test.mp4"
writer = cv2.VideoWriter(
    str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 24, (W, H)
)
if not writer.isOpened():
    raise RuntimeError("Could not open V7 wave output")

for frame_no in range(96):
    t = frame_no / 24.
    phase = 2. * math.pi * t / 1.0
    upper = -32. + 9. * math.sin(phase)
    lower = -12. + 28. * math.sin(phase)
    hand = 8. * math.sin(phase)

    local = {n: np.eye(3, dtype=np.float64) for n in order}
    local["left_upper_arm.png"] = TR(piv["left_upper_arm.png"], upper)
    local["left_lower_arm.png"] = TR(piv["left_lower_arm.png"], lower)
    local["left_hand.png"] = TR(piv["left_hand.png"], hand)
    local["right_upper_arm.png"] = TR(piv["right_upper_arm.png"], 2.)
    local["right_lower_arm.png"] = TR(piv["right_lower_arm.png"], 1.)

    world = {}
    for name in order:
        if name == "base.png":
            continue
        p = parent.get(name)
        world[name] = local[name] if not p else world[p] @ local[name]

    canvas = layers["base.png"].copy()
    for name in order:
        if name == "base.png":
            continue
        canvas = composite(canvas, warp(layers[name], world[name]))

    bg = np.full((H, W, 3), 255, np.uint8)
    a = canvas[:, :, 3:4].astype(np.float32) / 255.
    frame = (canvas[:, :, :3] * a + bg * (1. - a)).astype(np.uint8)
    writer.write(frame)

writer.release()
print(f"Created {out_mp4} with 96 frames at 24 FPS")
