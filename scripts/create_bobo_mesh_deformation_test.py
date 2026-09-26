from pathlib import Path
import json, math, subprocess
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"assets"/"characters"/"bobo.png"
SPEC=ROOT/"config"/"bobo_rig_spec.json"
OUT=ROOT/"output"/"bobo_mesh_deformation_test"
FRAMES=OUT/"frames"
VIDEO=OUT/"bobo_mesh_deformation_test.mp4"
BG=ROOT/"assets"/"backgrounds"/"happy_meadow.png"
W,H=1024,1536
FPS=24
SECONDS=10
PW,PH=1280,720

src_rgba=np.array(Image.open(SOURCE).convert("RGBA"))
src_bgr=cv2.cvtColor(src_rgba[:,:,:3],cv2.COLOR_RGB2BGR)
src_a=src_rgba[:,:,3]
spec=json.loads(SPEC.read_text(encoding="utf-8"))
J=spec["joint_targets_1024x1536"]

# A compact mesh follows Bobo's silhouette and the approved skeleton.
# Boundary points stay fixed while joint points move. This deforms one
# continuous artwork instead of rotating separately cut limb images.
points=[
 ("p0",(110,60)),("p1",(510,45)),("p2",(915,60)),
 ("p3",(130,400)),("p4",(510,430)),("p5",(890,400)),
 ("p6",(245,650)),("p7",(510,650)),("p8",(775,650)),
 ("p9",(220,930)),("p10",(510,930)),("p11",(800,930)),
 ("p12",(225,1180)),("p13",(510,1180)),("p14",(805,1180)),
 ("p15",(225,1505)),("p16",(510,1505)),("p17",(815,1505)),
 ("ls",tuple(J["left_shoulder"])),("le",tuple(J["left_elbow"])),("lw",tuple(J["left_wrist"])),
 ("rs",tuple(J["right_shoulder"])),("re",tuple(J["right_elbow"])),("rw",tuple(J["right_wrist"])),
 ("lh",tuple(J["left_hip"])),("lk",tuple(J["left_knee"])),("la",tuple(J["left_ankle"])),
 ("rh",tuple(J["right_hip"])),("rk",tuple(J["right_knee"])),("ra",tuple(J["right_ankle"])),
 ("neck",tuple(J["neck"]))
]
name_to_i={n:i for i,(n,_) in enumerate(points)}
src_pts=np.float32([p for _,p in points])

# Fixed topology. Each triangle is a small local patch of the continuous image.
T=[
 ("p0","p1","p4"),("p0","p4","p3"),("p1","p2","p5"),("p1","p5","p4"),
 ("p3","p4","p7"),("p3","p7","p6"),("p4","p5","p8"),("p4","p8","p7"),
 ("p6","p7","ls"),("p6","ls","le"),("p6","le","p9"),("p6","p9","p12"),
 ("p7","ls","neck"),("p7","neck","rs"),("p7","rs","p8"),
 ("p8","rs","re"),("p8","re","p11"),("p8","p11","p14"),
 ("ls","le","lw"),("le","p9","lw"),("p9","p10","lw"),("p10","p11","lw"),
 ("rs","rw","re"),("re","rw","p11"),("p10","rw","p11"),
 ("p9","p10","lh"),("p9","lh","p12"),("p10","lh","rh"),("p10","rh","p13"),
 ("p11","rh","p14"),("p12","lh","lk"),("p12","lk","p15"),
 ("p13","lk","la"),("p13","la","p16"),("p13","rh","rk"),("p13","rk","ra"),
 ("p14","rk","p17"),("p15","lk","la"),("p15","la","p16"),
 ("p16","la","ra"),("p16","ra","p17"),("p16","rk","p17")
]
triangles=[tuple(name_to_i[n] for n in t) for t in T]

def render_mesh(dst_pts):
    out=np.zeros_like(src_rgba)
    coverage=np.zeros((H,W),np.uint8)
    for tri in triangles:
        si=np.float32([src_pts[i] for i in tri])
        di=np.float32([dst_pts[i] for i in tri])
        x0,y0=np.floor(np.minimum(si[:,0].min(),di[:,0].min())-3).astype(int), np.floor(np.minimum(si[:,1].min(),di[:,1].min())-3).astype(int)
        x1,y1=np.ceil(np.maximum(si[:,0].max(),di[:,0].max())+3).astype(int), np.ceil(np.maximum(si[:,1].max(),di[:,1].max())+3).astype(int)
        x0=max(0,int(x0)); y0=max(0,int(y0)); x1=min(W,int(x1)); y1=min(H,int(y1))
        if x1<=x0 or y1<=y0: continue
        src_crop=src_bgr[y0:y1,x0:x1]
        src_mask=np.zeros((y1-y0,x1-x0),np.uint8)
        cv2.fillConvexPoly(src_mask,np.int32(si-np.array([x0,y0])),255)
        dst_mask=np.zeros((H,W),np.uint8)
        cv2.fillConvexPoly(dst_mask,np.int32(di),255)
        M=cv2.getAffineTransform(si-np.array([x0,y0],np.float32),di)
        warped=cv2.warpAffine(src_crop,M,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        amask=cv2.warpAffine(src_mask,M,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        warped_a=cv2.warpAffine(src_a[y0:y1,x0:x1],M,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        mask=(amask.astype(np.float32)/255.0)
        # Alpha-composite triangle patch onto the continuous canvas.
        for c in range(3):
            out[:,:,c]=(warped[:,:,c]*mask + out[:,:,c]*(1-mask)).astype(np.uint8)
        out[:,:,3]=np.maximum(out[:,:,3],warped_a)
        coverage=np.maximum(coverage,amask)
    out[:,:,:3]=cv2.cvtColor(out[:,:,:3],cv2.COLOR_BGR2RGB)
    return out

def points_for(t):
    d=src_pts.copy()
    cycle=t%4.0
    if 1<=cycle<2:
        p=cycle-1
        a=26*math.sin(2*math.pi*p*2)
        # Clear, visible wave: shoulder stays mostly stable, elbow/wrist arc.
        d[name_to_i["re"],1]+=a*0.55
        d[name_to_i["rw"],0]+=a*0.85
        d[name_to_i["rw"],1]-=abs(a)*0.65
    elif 2<=cycle<3:
        p=cycle-2
        w=22*math.sin(2*math.pi*p*2)
        d[name_to_i["lk"],0]+=w
        d[name_to_i["rk"],0]-=w
        d[name_to_i["la"],0]+=w*0.7
        d[name_to_i["ra"],0]-=w*0.7
    elif 3<=cycle<4:
        b=-14*abs(math.sin(math.pi*(cycle-3)))
        for n in ["neck","ls","rs","lh","rh","lk","rk","la","ra"]:
            d[name_to_i[n],1]+=b
    return d

bg=np.array(Image.open(BG).convert("RGB"))
s=max(PW/bg.shape[1],PH/bg.shape[0])
bg=cv2.resize(bg,(int(bg.shape[1]*s),int(bg.shape[0]*s)),interpolation=cv2.INTER_LANCZOS4)
y=(bg.shape[0]-PH)//2; x=(bg.shape[1]-PW)//2
bg=bg[y:y+PH,x:x+PW]

FRAMES.mkdir(parents=True,exist_ok=True)
for i in range(FPS*SECONDS):
    t=i/FPS
    warped=render_mesh(points_for(t))
    # Scale character into presentation frame.
    rgba=Image.fromarray(warped).resize((600,900),Image.Resampling.LANCZOS)
    canvas=Image.fromarray(bg).convert("RGBA")
    canvas.alpha_composite(rgba,((PW-600)//2,PH-900-5))
    canvas.convert("RGB").save(FRAMES/f"frame_{i:04d}.png",quality=95)

subprocess.run(["ffmpeg","-y","-framerate",str(FPS),"-i",str(FRAMES/"frame_%04d.png"),
                "-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",str(VIDEO)],check=True)

(OUT/"mesh_manifest.json").write_text(json.dumps({
 "method":"piecewise_affine_mesh_deformation",
 "source":"assets/characters/bobo.png",
 "movement_tests":["idle","large_wave","walk","bounce"],
 "note":"Experimental continuous-image deformation. No separate limb cutout rotation."
},indent=2),encoding="utf-8")
print(f"Created {VIDEO}")
