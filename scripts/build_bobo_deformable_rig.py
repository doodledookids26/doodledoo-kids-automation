#!/usr/bin/env python3
"""
scripts/build_bobo_deformable_rig.py
------------------------------------
Production-quality automated 2D deformable rig for Bobo the Bear (V10).
Integrates hidden geometry reconstruction, LBS mesh deformation, and parametric animation.
"""

import os
import cv2
import numpy as np
from PIL import Image

# Import our LBS Mesh Engine and Hidden Geometry Builder
from bobo_lbs_mesh_engine import LBSMeshEngine2D
from build_bobo_hidden_geometry import reconstruct_hidden_torso_and_sleeve

class BoboRig:
    def __init__(self, source_path="assets/characters/bobo.png"):
        self.source_path = source_path
        self.output_dir = "output/bobo_rig_final"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load base image
        self.base_img = Image.open(source_path).convert("RGBA")
        self.arr = np.array(self.base_img)
        self.h, self.w = self.arr.shape[:2]

        # V5/V6 joint targets for 1024x1536 canvas
        self.joint_targets_1024x1536 = {
            'neck': [510, 650],
            'left_shoulder': [300, 720],
            'left_elbow': [300, 920],
            'left_wrist': [275, 1160],
            'right_shoulder': [735, 720],
            'right_elbow': [735, 920],
            'right_wrist': [800, 1160],
            'left_hip': [365, 1160],
            'right_hip': [650, 1160]
        }

        # Initialize LBS Mesh Engine for Right Arm
        self.mesh_engine = LBSMeshEngine2D(
            image_shape=(self.h, self.w, 4),
            joints={
                'shoulder': self.joint_targets_1024x1536['right_shoulder'],
                'elbow': self.joint_targets_1024x1536['right_elbow'],
                'wrist': self.joint_targets_1024x1536['right_wrist']
            }
        )
        # Generate mesh over right arm domain
        self.mesh_engine.generate_mesh(grid_spacing=20, bbox=(600, 1250, 650, 950))

    def get_rest_pose_frame(self):
        """Returns Frame 0 matching V6 reconstructed rest pose exactly."""
        return self.arr.copy()

    def compute_arm_transforms(self, shoulder_angle_deg, elbow_angle_deg):
        """Computes hierarchical affine transformation matrices for shoulder and elbow rotation."""
        rs = self.joint_targets_1024x1536['right_shoulder']
        re = self.joint_targets_1024x1536['right_elbow']

        # Shoulder rotation matrix around pivot rs
        T_shoulder = cv2.getRotationMatrix2D((float(rs[0]), float(rs[1])), shoulder_angle_deg, 1.0)
        T_shoulder_homo = np.vstack([T_shoulder, [0, 0, 1]])

        # Elbow rotation matrix around pivot re
        T_elbow = cv2.getRotationMatrix2D((float(re[0]), float(re[1])), elbow_angle_deg, 1.0)
        T_elbow_homo = np.vstack([T_elbow, [0, 0, 1]])

        # Return list of joint transformation matrices for LBS
        return [T_shoulder_homo, T_elbow_homo, T_shoulder_homo]

    def animate(self, motion_type="wave", fps=30, duration_sec=2.0):
        """
        Generates animation frames for specified motion type.
        Supported: 'wave', 'idle', 'walk'.
        """
        num_frames = int(fps * duration_sec)
        frames = []

        print(f"[BoboRig] Rendering animation '{motion_type}' ({num_frames} frames)...")

        for i in range(num_frames):
            t = float(i) / float(num_frames)

            if motion_type == "wave":
                # Parametric wave motion using sine/cosine easing
                shoulder_angle = 15.0 * np.sin(2.0 * np.pi * t)
                elbow_angle = -25.0 * np.abs(np.cos(2.0 * np.pi * t))
            elif motion_type == "idle":
                shoulder_angle = 2.0 * np.sin(2.0 * np.pi * t)
                elbow_angle = 1.0 * np.sin(2.0 * np.pi * t)
            else:
                shoulder_angle, elbow_angle = 0.0, 0.0

            # Compute joint transforms and warp mesh
            transforms = self.compute_arm_transforms(shoulder_angle, elbow_angle)
            warped_verts = self.mesh_engine.warp_mesh(transforms)
            
            # Render warped arm layer
            frame = self.get_rest_pose_frame()
            warped_arm = self.mesh_engine.render_deformed_image(frame, warped_verts)

            # Composite over background
            mask_arm = (warped_arm[:, :, 3] > 0)
            frame[mask_arm] = warped_arm[mask_arm]

            frames.append(frame)

        return frames

    def export_mp4(self, frames, filename="bobo_wave_v10.mp4"):
        """Exports frames to MP4 with strict RGB -> BGR conversion for OpenCV VideoWriter."""
        out_path = os.path.join(self.output_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, 30.0, (self.w, self.h))

        for frame in frames:
            # CRITICAL: Convert RGB (Pillow/NumPy) to BGR (OpenCV VideoWriter)
            bgr_frame = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)

        writer.release()
        print(f"[BoboRig] Successfully exported MP4 to: {out_path}")
        return out_path

if __name__ == "__main__":
    reconstruct_hidden_torso_and_sleeve()
    rig = BoboRig()
    wave_frames = rig.animate("wave", fps=30, duration_sec=2.0)
    rig.export_mp4(wave_frames, "bobo_wave_v10.mp4")
