# Synfig automation research notes

## Verified serialization facts

These notes are derived from the Synfig source at the time of research.

- Canvas serialization writes a top-level `<bones>` section when the canvas bone map is non-empty.
- Each bone is emitted as a `<value_node>` containing a bone value node.
- `ValueNode_Bone` serializes as `<bone type="bone_object">...`.
- `ValueNode_Bone` fields are:
  - name
  - parent
  - origin
  - angle
  - scalelx
  - width
  - scalex
  - tipwidth
  - bone_depth
  - length
- The BonePair used by Skeleton Deformation is `std::pair<Bone, Bone>`.
- Synfig's TypePair naming rule makes the BonePair type name:
  `pair_bone_object_bone_object`.
- Pair values serialize as:
  - `<first><value>...</value></first>`
  - `<second><value>...</value></second>`
- Skeleton Deformation layer type is `skeleton_deformation`, version `0.2`.
- Its critical parameters include `bones`, `point1`, `point2`, `x_subdivisions`, and `y_subdivisions`.
- The `bones` parameter is a list of BonePair values. The first bone in each pair is the rest pose and the second is the animated pose.

## Important limitation

The exact serialized root-bone/reference structure still needs to be validated against a real Synfig-saved skeleton-deformation file. Do not generate guessed XML and treat it as valid.

## Source files

- synfig-core/src/synfig/savecanvas.cpp
- synfig-core/src/synfig/loadcanvas.cpp
- synfig-core/src/synfig/layers/layer_skeletondeformation.cpp
- synfig-core/src/synfig/layers/layer_skeletondeformation.h
- synfig-core/src/synfig/valuenodes/valuenode_bone.cpp
- synfig-core/src/synfig/valuenodes/valuenode_bone.h
- synfig-core/src/synfig/pair.h
