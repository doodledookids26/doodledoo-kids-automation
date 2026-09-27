# Synfig automation research

## Verified serialization facts

The automation work is based on Synfig source code, the Synfig SIF format documentation, and a real serialized Skeleton Deformation example published in Synfig support material.

- Canvas-level `<bones>` contains a mandatory `<bone_root>` and bone definitions.
- Bone definitions use `<bone type="bone_object" guid="...">`.
- Bone parents use `<bone_valuenode type="bone_object" guid="..."/>`.
- Skeleton Deformation uses a static list with type `pair_bone_object_bone_object`.
- Each pair has `first` = rest-position bone and `second` = pose bone.
- Skeleton Deformation parameters include `bones`, `point1`, `point2`, `x_subdivisions`, and `y_subdivisions`.
- The deformation layer type is `skeleton_deformation`.
- The current generator deliberately creates only a small two-bone arm validation test.

## Validation status

The XML generator is source-derived and structurally consistent with the documented Synfig format, but the generated test still needs to be opened by Synfig itself. It must not be considered production-ready until Synfig accepts it and the rendered character visibly deforms.

## Production goal

Once the arm test is accepted, the same generator can be expanded to the full Bobo hierarchy and animation presets without requiring manual bone drawing.

## Source files

- `synfig-core/src/synfig/savecanvas.cpp`
- `synfig-core/src/synfig/loadcanvas.cpp`
- `synfig-core/src/synfig/layers/layer_skeletondeformation.cpp`
- `synfig-core/src/synfig/valuenodes/valuenode_bone.cpp`
- `synfig-core/src/synfig/valuenodes/valuenode_composite.cpp`
- `synfig-core/src/synfig/pair.h`
