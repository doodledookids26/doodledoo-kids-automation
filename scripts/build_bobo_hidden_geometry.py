#!/usr/bin/env python3
"""
scripts/build_bobo_hidden_geometry.py
--------------------------------------
Reconstructs hidden artwork behind Bobo's right shoulder/arm.
Uses Navier-Stokes inpainting + texture continuation on the torso layer.
"""

import os
import cv2
import numpy as np
from PIL import Image

def reconstruct_hidden_torso_and_sleeve():
    source_path = "assets/characters/bobo.png"
    out_dir = "output/bobo_rig_v10_assets"
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(source_path):
        print(f"[WARN] Source image {source_path} not found. Creating placeholder.")
        return out_dir

    img = Image.open(source_path).convert("RGBA")
    arr = np.array(img)
    h, w, c = arr.shape

    # Occlusion region for occluded torso under right arm
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[680, 710], [780, 710], [820, 1000], [680, 1000]], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)

    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    # Inpaint RGB channels across occluded boundary
    inpainted_rgb = cv2.inpaint(rgb, mask, inpaintRadius=7, flags=cv2.INPAINT_NS)

    # Fill alpha in occluded zone
    reconstructed_alpha = alpha.copy()
    occluded_pixels = (mask > 0) & (alpha < 200)
    reconstructed_alpha[occluded_pixels] = 255

    reconstructed_torso = np.dstack((inpainted_rgb, reconstructed_alpha))
    unmodified_mask = (mask == 0)
    reconstructed_torso[unmodified_mask] = arr[unmodified_mask]

    out_path = os.path.join(out_dir, "torso_reconstructed.png")
    Image.fromarray(reconstructed_torso).save(out_path)
    print(f"[Step 1] Successfully generated hidden geometry: {out_path}")
    return out_path

if __name__ == "__main__":
    reconstruct_hidden_torso_and_sleeve()
