from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'/'Bobo_Synfig_Automation_Test.sif'
ASSET='assets/characters/bobo.png'

# Minimal Synfig project scaffold. The image is deliberately kept as a normal
# raster layer for the first automation checkpoint; deformation integration is
# added only after this file is accepted by the installed Synfig version.
root=ET.Element('canvas', {'version':'1.0','width':'1024','height':'1536','fps':'24.0','begin-time':'0s','end-time':'4s'})
ET.SubElement(root,'name').text='Bobo Synfig Automation Test'
ET.SubElement(root,'layer', {'type':'import','active':'true','version':'0.1','desc':'Bobo source artwork','z_depth':'0'}).append(ET.Element('param',{'name':'filename','value':ASSET}))
ET.indent(root, space='  ')
OUT.parent.mkdir(parents=True,exist_ok=True)
ET.ElementTree(root).write(OUT,encoding='utf-8',xml_declaration=True)
print(OUT)
