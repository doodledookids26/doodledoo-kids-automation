from pathlib import Path
import gzip
import math
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "assets" / "characters" / "bobo_synfig_template.sifz"
DEFAULT_OUTPUT = ROOT / "output" / "Bobo_Synfig_AutoRig_Arm_Test.sifz"

# Coordinates are based on the 1024x1536 Bobo master artwork.
VIEW = (-2.546166, 3.818316, 2.546166, -3.818316)

ROOT_GUID = "A10000000000000000000000000000001"
UA_R = "A10000000000000000000000000000002"
LA_R = "A10000000000000000000000000000003"
UA_P = "A10000000000000000000000000000004"
LA_P = "A10000000000000000000000000000005"

def px_to_world(px, py):
    x0, y0, x1, y1 = VIEW
    return (
        x0 + (x1 - x0) * px / 1024.0,
        y0 + (y1 - y0) * py / 1536.0,
    )

def distance(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])

def degrees(a, b):
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))

def real(value, static=False):
    node = ET.Element("real", {"value": f"{value:.10f}"})
    if static:
        node.set("static", "true")
    return node

def integer(value, static=False):
    node = ET.Element("integer", {"value": str(value)})
    if static:
        node.set("static", "true")
    return node

def string(value):
    node = ET.Element("string")
    node.text = value
    return node

def vector(x, y):
    node = ET.Element("vector")
    ET.SubElement(node, "x").text = f"{x:.10f}"
    ET.SubElement(node, "y").text = f"{y:.10f}"
    return node

def angle(value):
    return ET.Element("angle", {"value": f"{value:.6f}"})

def add_param(parent, name, child):
    node = ET.SubElement(parent, "param", {"name": name})
    node.append(child)

def add_animatable(parent, name, child):
    node = ET.SubElement(parent, name)
    node.append(child)

def make_bone(name, guid, parent_guid, origin, bone_angle, length):
    bone = ET.Element("bone", {"type": "bone_object", "guid": guid})
    add_animatable(bone, "name", string(name))
    parent = ET.SubElement(bone, "parent")
    parent.append(ET.Element(
        "bone_valuenode",
        {"type": "bone_object", "guid": parent_guid},
    ))
    add_animatable(bone, "origin", vector(*origin))
    add_animatable(bone, "angle", angle(bone_angle))
    add_animatable(bone, "scalelx", real(1.0))
    add_animatable(bone, "width", real(0.60))
    add_animatable(bone, "scalex", real(1.0))
    add_animatable(bone, "tipwidth", real(0.60))
    add_animatable(bone, "bone_depth", real(0.0))
    add_animatable(bone, "length", real(length))
    return bone

def build(template, output):
    with gzip.open(template, "rt", encoding="utf-8") as stream:
        root = ET.fromstring(stream.read())

    # Remove prior generated test sections.
    for child in list(root):
        if child.tag == "bones":
            root.remove(child)
        elif child.tag == "layer" and child.attrib.get("type") == "skeleton_deformation":
            root.remove(child)

    shoulder = px_to_world(300, 720)
    elbow = px_to_world(300, 920)
    wrist = px_to_world(275, 1160)

    upper_length = distance(shoulder, elbow)
    lower_length = distance(elbow, wrist)
    upper_angle = degrees(shoulder, elbow)
    lower_relative_angle = degrees(elbow, wrist) - upper_angle

    bones = ET.Element("bones")
    bones.append(ET.Element("bone_root", {
        "type": "bone_object",
        "guid": ROOT_GUID,
    }))

    bones.append(make_bone(
        "Bobo_Left_Upper_Arm_Rest", UA_R, ROOT_GUID,
        shoulder, upper_angle, upper_length,
    ))
    bones.append(make_bone(
        "Bobo_Left_Lower_Arm_Rest", LA_R, UA_R,
        (upper_length, 0.0), lower_relative_angle, lower_length,
    ))
    bones.append(make_bone(
        "Bobo_Left_Upper_Arm_Pose", UA_P, ROOT_GUID,
        shoulder, upper_angle + 25.0, upper_length,
    ))
    bones.append(make_bone(
        "Bobo_Left_Lower_Arm_Pose", LA_P, UA_P,
        (upper_length, 0.0), lower_relative_angle - 10.0, lower_length,
    ))

    layer = ET.Element("layer", {
        "type": "skeleton_deformation",
        "active": "true",
        "exclude_from_rendering": "false",
        "version": "0.2",
        "desc": "Bobo Automated Arm Rig Test",
    })

    add_param(layer, "z_depth", real(0.0))
    add_param(layer, "amount", real(1.0))
    add_param(layer, "blend_method", integer(0, static=True))

    bones_param = ET.SubElement(layer, "param", {"name": "bones"})
    static_list = ET.SubElement(
        bones_param, "static_list",
        {"type": "pair_bone_object_bone_object"},
    )

    for rest_guid, pose_guid in ((UA_R, UA_P), (LA_R, LA_P)):
        entry = ET.SubElement(static_list, "entry")
        pair = ET.SubElement(
            entry, "composite",
            {"type": "pair_bone_object_bone_object"},
        )
        first = ET.SubElement(pair, "first")
        first.append(ET.Element(
            "bone_valuenode",
            {"type": "bone_object", "guid": rest_guid},
        ))
        second = ET.SubElement(pair, "second")
        second.append(ET.Element(
            "bone_valuenode",
            {"type": "bone_object", "guid": pose_guid},
        ))

    add_param(layer, "point1", vector(*VIEW[:2]))
    add_param(layer, "point2", vector(*VIEW[2:]))
    add_param(layer, "x_subdivisions", integer(48))
    add_param(layer, "y_subdivisions", integer(48))

    first_layer = next(
        i for i, child in enumerate(list(root))
        if child.tag == "layer"
    )
    root.insert(first_layer, layer)
    root.insert(0, bones)

    ET.indent(root, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)

    sif_path = output.with_suffix(".sif")
    ET.ElementTree(root).write(
        sif_path, encoding="utf-8", xml_declaration=True
    )

    with sif_path.open("rb") as source, gzip.open(output, "wb") as target:
        target.write(source.read())

    print(f"Created {output}")

if __name__ == "__main__":
    import sys
    template = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TEMPLATE
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT
    build(template, output)
