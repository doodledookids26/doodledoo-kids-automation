from pathlib import Path
import json, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'assets/characters/bobo.png'
OUT=ROOT/'output/bobo_rig_v5'
OUT.mkdir(parents=True,exist_ok=True)
img=Image.open(SRC).convert('RGBA')
W,H=img.size
if (W,H)!=(1024,1536): raise ValueError(f'Expected 1024x1536, got {W}x{H}')
a=np.array(img.getchannel('A'),dtype=np.uint8); rgb=np.array(img.convert('RGB'))

def poly(pts):
    m=Image.new('L',(W,H),0); ImageDraw.Draw(m).polygon(pts,fill=255); return np.array(m)
def ell(box):
    m=Image.new('L',(W,H),0); ImageDraw.Draw(m).ellipse(box,fill=255); return np.array(m)
def refine(m,dil=6,blur=0.7):
    m=np.minimum(m,a); m=cv2.dilate(m,np.ones((dil,dil),np.uint8),iterations=1)
    if blur: m=np.array(Image.fromarray(m,'L').filter(ImageFilter.GaussianBlur(blur)))
    return m

shapes={
'head.png':ell((95,35,930,735)),
'left_ear.png':ell((145,80,390,365)),'right_ear.png':ell((625,80,875,365)),
'left_upper_arm.png':poly([(235,650),(380,650),(430,735),(405,835),(355,925),(245,940),(185,855),(195,740)]),
'left_lower_arm.png':poly([(205,850),(350,875),(395,955),(375,1060),(350,1160),(320,1215),(245,1225),(170,1160),(175,1060),(190,950)]),
'left_hand.png':ell((150,1110,345,1250)),
'right_upper_arm.png':poly([(640,650),(790,655),(830,740),(835,835),(785,935),(675,940),(620,840),(615,735)]),
'right_lower_arm.png':poly([(675,855),(810,875),(850,950),(865,1060),(875,1155),(825,1225),(750,1215),(700,1160),(680,1060),(650,950)]),
'right_hand.png':ell((735,1110,900,1250)),
'left_upper_leg.png':poly([(220,1110),(510,1110),(505,1300),(470,1380),(285,1380),(220,1300)]),
'left_lower_leg.png':poly([(250,1280),(490,1280),(485,1445),(475,1525),(205,1525),(220,1435)]),
'left_foot.png':ell((190,1420,500,1535)),
'right_upper_leg.png':poly([(500,1110),(810,1110),(810,1300),(735,1380),(545,1380),(500,1300)]),
'right_lower_leg.png':poly([(535,1280),(800,1280),(815,1435),(830,1525),(545,1525),(540,1445)]),
'right_foot.png':ell((535,1420,850,1535)),
'torso.png':poly([(255,610),(770,610),(810,760),(795,950),(760,1140),(700,1210),(330,1210),(270,1130),(235,950),(225,760)])
}
head_region=ell((120,150,900,700))
hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV); yy,xx=np.indices((H,W))
white=(hsv[:,:,1]<70)&(hsv[:,:,2]>150)
pink=((hsv[:,:,0]<15)|(hsv[:,:,0]>165))&(hsv[:,:,1]>70)&(hsv[:,:,2]>70)
face=(head_region>0)&(a>0)
cand=((white)&(yy>300)&(yy<650)&(xx>230)&(xx<790)).astype(np.uint8)*255
n,lab,stats,_=cv2.connectedComponentsWithStats(cand,8); eye=np.zeros((H,W),np.uint8)
for i in range(1,n):
    x,y,w,h,area=stats[i]
    if 150<area<30000 and h>8 and w>8: eye[lab==i]=255
eye=np.minimum(eye,face.astype(np.uint8)*255)
mouth=(pink&(yy>500)&(yy<760)&(xx>300)&(xx<720)).astype(np.uint8)*255
shapes['eyes.png']=eye; shapes['mouth.png']=mouth

layers={k:refine(v,7) for k,v in shapes.items()}
movable=np.zeros((H,W),np.uint8)
for m in layers.values(): movable=np.maximum(movable,m)
hole=cv2.dilate((movable>24).astype(np.uint8)*255,np.ones((13,13),np.uint8),iterations=1)
filled=cv2.inpaint(rgb,hole,11,cv2.INPAINT_TELEA)
base_alpha=a.copy(); base_alpha[movable>24]=0
base=Image.fromarray(filled).convert('RGBA'); base.putalpha(Image.fromarray(base_alpha,'L')); base.save(OUT/'base.png')
for name,m in layers.items():
    p=img.copy(); p.putalpha(Image.fromarray(m,'L')); p.save(OUT/name)

order=['base.png','left_upper_leg.png','right_upper_leg.png','left_lower_leg.png','right_lower_leg.png','left_foot.png','right_foot.png','torso.png','left_upper_arm.png','right_upper_arm.png','left_lower_arm.png','right_lower_arm.png','left_hand.png','right_hand.png','head.png','left_ear.png','right_ear.png','eyes.png','mouth.png']
prev=Image.new('RGBA',(W,H),(0,0,0,0))
for n in order: prev.alpha_composite(Image.open(OUT/n))
prev.save(OUT/'reconstructed_preview.png')
src=np.array(img).astype(np.int16)
rec=np.array(prev).astype(np.int16)

# Compare only pixels that are actually visible in the original artwork.
# Hidden areas are intentionally reconstructed/inpainted and therefore
# should not be required to have the original RGB values.
src_visible=src[:,:,3]>16
rec_visible=rec[:,:,3]>16
intersection=src_visible & rec_visible
coverage=float(rec_visible[src_visible].mean()) if np.any(src_visible) else 0.0
rgb_err=float(np.abs(src[:,:,:3][intersection]-rec[:,:,:3][intersection]).mean()) if np.any(intersection) else 999.0
alpha_err=float(np.abs(src[:,:,3].astype(np.int16)-rec[:,:,3].astype(np.int16))[src_visible].mean()) if np.any(src_visible) else 999.0

spec=json.loads((ROOT/'config/bobo_rig_spec.json').read_text())
manifest={'character':'Bobo the Bear','rig_type':'automated_2d_joint_rig_v5','source':'assets/characters/bobo.png','canvas':[W,H],'layers':order,'joint_targets':spec['joint_targets_1024x1536'],'reconstruction_check':{'visible_coverage':round(coverage,6),'visible_rgb_error':round(rgb_err,4),'visible_alpha_error':round(alpha_err,4)},'quality_gate':{'source_reconstructed':coverage>=0.995 and rgb_err<3.0 and alpha_err<3.0,'visual_review_required':True}}
(OUT/'rig_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest['reconstruction_check']))

if not manifest['quality_gate']['source_reconstructed']: raise SystemExit('QUALITY_GATE_FAILED')
