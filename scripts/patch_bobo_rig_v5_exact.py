from pathlib import Path
p=Path("scripts/build_bobo_rig_v5.py")
s=p.read_text()
a=s.index("def refine(")
b=s.index("\n\nshapes=",a)
s=s[:a]+'''def refine(m,dil=0,blur=0):
    return np.minimum(m,a)
'''+s[b:]
s=s.replace("layers={k:refine(v,7) for k,v in shapes.items()}","layers={k:refine(v) for k,v in shapes.items()}")
s=s.replace("base_alpha=a.copy(); base_alpha[movable>24]=0","base_alpha=a.copy(); base_alpha[movable>0]=0")
p.write_text(s)
print("patched")
