from pathlib import Path
import math, subprocess
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
RIG=ROOT/"output"/"bobo_rig_stage4"
OUT=ROOT/"output"
FRAMES=OUT/"bobo_stage4_animation_frames"
VIDEO=OUT/"bobo_stage4_animation_test.mp4"
BG=ROOT/"assets"/"backgrounds"/"happy_meadow.png"
W,H=1024,1536
FPS,SECONDS=24,10
PW,PH=1280,720

names=["base.png","head.png","torso.png",
"left_upper_arm.png","left_lower_arm.png","left_hand.png",
"right_upper_arm.png","right_lower_arm.png","right_hand.png",
"left_upper_leg.png","left_lower_leg.png","left_foot.png",
"right_upper_leg.png","right_lower_leg.png","right_foot.png"]
L={n:Image.open(RIG/n).convert("RGBA") for n in names}

bg=Image.open(BG).convert("RGB")
s=max(PW/bg.width,PH/bg.height)
bg=bg.resize((int(bg.width*s),int(bg.height*s)),Image.Resampling.LANCZOS)
bg=bg.crop(((bg.width-PW)//2,(bg.height-PH)//2,(bg.width-PW)//2+PW,(bg.height-PH)//2+PH)).convert("RGBA")

def rot(n,a,c):
    return L[n].rotate(a,resample=Image.Resampling.BICUBIC,center=c)

def pose(i):
    t=i/FPS
    c4=t%4
    wave=0; walk=0; bounce=0
    if 1<=c4<2: wave=28*math.sin(2*math.pi*(c4-1)*2)
    elif 2<=c4<3: walk=16*math.sin(2*math.pi*(c4-2)*2)
    elif 3<=c4<4: bounce=-10*abs(math.sin(math.pi*(c4-3)))
    c=Image.new("RGBA",(W,H),(0,0,0,0))
    c.alpha_composite(L["base.png"])
    # Legs: visible knee articulation.
    c.alpha_composite(rot("left_upper_leg.png",walk*0.55,(365,1160)))
    c.alpha_composite(rot("right_upper_leg.png",-walk*0.55,(650,1160)))
    c.alpha_composite(rot("left_lower_leg.png",-walk*0.8,(365,1330)))
    c.alpha_composite(rot("right_lower_leg.png",walk*0.8,(650,1330)))
    c.alpha_composite(rot("left_foot.png",-walk*0.3,(355,1460)))
    c.alpha_composite(rot("right_foot.png",walk*0.3,(685,1460)))
    c.alpha_composite(L["torso.png"])
    # Arms: clear shoulder + elbow movement.
    c.alpha_composite(rot("left_upper_arm.png",-walk*0.35,(300,720)))
    c.alpha_composite(rot("left_lower_arm.png",-walk*0.45,(300,920)))
    c.alpha_composite(rot("left_hand.png",-walk*0.25,(275,1160)))
    c.alpha_composite(rot("right_upper_arm.png",walk*0.35,(735,720)))
    c.alpha_composite(rot("right_lower_arm.png",wave,(735,920)))
    c.alpha_composite(rot("right_hand.png",wave*1.15,(800,1160)))
    c.alpha_composite(rot("head.png",2.8*math.sin(2*math.pi*t/2.8),(510,430)))
    if bounce:
        moved=Image.new("RGBA",(W,H),(0,0,0,0)); moved.alpha_composite(c,(0,int(bounce))); c=moved
    return c

FRAMES.mkdir(parents=True,exist_ok=True)
for i in range(FPS*SECONDS):
    ch=pose(i)
    sc=min(600/ch.width,700/ch.height)
    ch=ch.resize((int(ch.width*sc),int(ch.height*sc)),Image.Resampling.LANCZOS)
    f=bg.copy(); f.alpha_composite(ch,((PW-ch.width)//2,PH-ch.height-5))
    f.convert("RGB").save(FRAMES/f"frame_{i:04d}.png",quality=95)

subprocess.run(["ffmpeg","-y","-framerate",str(FPS),"-i",str(FRAMES/"frame_%04d.png"),
"-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",str(VIDEO)],check=True)
print(f"Created {VIDEO}")
