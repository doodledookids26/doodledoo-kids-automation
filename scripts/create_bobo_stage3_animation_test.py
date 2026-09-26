from pathlib import Path
import math
import subprocess
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RIG = ROOT / "output" / "bobo_rig_stage3"
OUT = ROOT / "output"
FRAMES = OUT / "bobo_stage3_animation_frames"
VIDEO = OUT / "bobo_stage3_animation_test.mp4"
BG = ROOT / "assets" / "backgrounds" / "happy_meadow.png"

W, H = 1024, 1536
FPS, SECONDS = 24, 8
PW, PH = 1280, 720

def load(name):
    return Image.open(RIG / name).convert("RGBA")

layers = {n: load(n) for n in [
    "base.png","head.png","torso.png",
    "left_upper_arm.png","left_lower_arm.png",
    "right_upper_arm.png","right_lower_arm.png",
    "left_upper_leg.png","left_lower_leg.png",
    "right_upper_leg.png","right_lower_leg.png"
]}

bg = Image.open(BG).convert("RGB")
scale = max(PW / bg.width, PH / bg.height)
bg = bg.resize((int(bg.width*scale), int(bg.height*scale)), Image.Resampling.LANCZOS)
bg = bg.crop(((bg.width-PW)//2, (bg.height-PH)//2, (bg.width-PW)//2+PW, (bg.height-PH)//2+PH)).convert("RGBA")

def rot(layer, angle, center):
    return layer.rotate(angle, resample=Image.Resampling.BICUBIC, center=center)

def build(frame):
    t = frame / FPS
    cycle = t % 4.0
    arm_wave = 0.0
    leg_walk = 0.0
    bounce = 0.0

    if 1 <= cycle < 2:
        arm_wave = 14 * math.sin(2*math.pi*(cycle-1)*2)
    elif 2 <= cycle < 3:
        leg_walk = 9 * math.sin(2*math.pi*(cycle-2)*2)
    elif 3 <= cycle < 4:
        bounce = -8 * abs(math.sin(math.pi*(cycle-3)))

    c = Image.new("RGBA", (W,H), (0,0,0,0))
    c.alpha_composite(layers["base.png"])

    # Legs rotate around knee/hip regions with deliberately modest motion.
    c.alpha_composite(rot(layers["left_upper_leg.png"], leg_walk*0.45, (365,1160)))
    c.alpha_composite(rot(layers["right_upper_leg.png"], -leg_walk*0.45, (650,1160)))
    c.alpha_composite(rot(layers["left_lower_leg.png"], -leg_walk*0.65, (355,1330)))
    c.alpha_composite(rot(layers["right_lower_leg.png"], leg_walk*0.65, (650,1330)))

    c.alpha_composite(layers["torso.png"])

    # Arms: upper segments move from shoulders; lower segments make a smaller elbow motion.
    c.alpha_composite(rot(layers["left_upper_arm.png"], -leg_walk*0.25, (300,720)))
    c.alpha_composite(rot(layers["right_upper_arm.png"], leg_walk*0.25, (735,720)))
    c.alpha_composite(rot(layers["left_lower_arm.png"], -leg_walk*0.2, (300,920)))
    c.alpha_composite(rot(layers["right_lower_arm.png"], arm_wave, (735,920)))

    # Head tilt is intentionally tiny.
    c.alpha_composite(rot(layers["head.png"], 2.5*math.sin(2*math.pi*t/2.8), (510,430)))

    if bounce:
        moved = Image.new("RGBA",(W,H),(0,0,0,0))
        moved.alpha_composite(c,(0,int(bounce)))
        c=moved
    return c

FRAMES.mkdir(parents=True, exist_ok=True)
for i in range(FPS*SECONDS):
    character=build(i)
    s=min(600/character.width,700/character.height)
    character=character.resize((int(character.width*s),int(character.height*s)),Image.Resampling.LANCZOS)
    frame=bg.copy()
    frame.alpha_composite(character,((PW-character.width)//2,PH-character.height-5))
    frame.convert("RGB").save(FRAMES/f"frame_{i:04d}.png",quality=95)

subprocess.run([
    "ffmpeg","-y","-framerate",str(FPS),"-i",str(FRAMES/"frame_%04d.png"),
    "-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",str(VIDEO)
],check=True)
print(f"Created {VIDEO}")
