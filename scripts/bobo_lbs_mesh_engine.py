#!/usr/bin/env python3
"""
scripts/bobo_lbs_mesh_engine.py
-------------------------------
2D Linear Blend Skinning (LBS) Mesh Deformation Engine for Bobo the Bear.
Generates a 2D Delaunay mesh, calculates joint weights, and warps local geometry smoothly.
"""

import numpy as np
import cv2
from scipy.spatial import Delaunay

class LBSMeshEngine2D:
    def __init__(self, image_shape, joints):
        """
        image_shape: (height, width, channels)
        joints: dict of joint coordinates e.g. {'shoulder': [735, 720], 'elbow': [735, 920], 'wrist': [800, 1160]}
        """
        self.h, self.w = image_shape[:2]
        self.joints = {k: np.array(v, dtype=np.float32) for k, v in joints.items()}
        self.vertices = None
        self.triangles = None
        self.weights = None

    def generate_mesh(self, grid_spacing=20, bbox=None):
        """Generates a 2D mesh across the bounding box domain or full image."""
        if bbox is None:
            ymin, ymax, xmin, xmax = 0, self.h, 0, self.w
        else:
            ymin, ymax, xmin, xmax = bbox

        y_coords = np.arange(ymin, ymax, grid_spacing)
        x_coords = np.arange(xmin, xmax, grid_spacing)
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        
        verts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        
        # Perform 2D Delaunay Triangulation
        tri = Delaunay(verts)
        self.vertices = verts.astype(np.float32)
        self.triangles = tri.simplices
        self._compute_skin_weights()

    def _compute_skin_weights(self, sigma=120.0):
        """Computes smooth Gaussian distance-based skinning weights per vertex."""
        num_verts = len(self.vertices)
        joint_names = list(self.joints.keys())
        num_joints = len(joint_names)
        
        dists = np.zeros((num_verts, num_joints), dtype=np.float32)
        
        for j_idx, name in enumerate(joint_names):
            joint_pos = self.joints[name]
            # Euclidean distance from vertex to joint
            dists[:, j_idx] = np.linalg.norm(self.vertices - joint_pos, axis=1)

        # Gaussian kernel weight calculation
        raw_weights = np.exp(- (dists ** 2) / (2.0 * (sigma ** 2)))
        
        # Normalize weights so sum(w_i) == 1.0 per vertex
        weight_sum = np.sum(raw_weights, axis=1, keepdims=True)
        weight_sum[weight_sum == 0] = 1.0
        self.weights = raw_weights / weight_sum

    def warp_mesh(self, transform_matrices):
        """
        Warps mesh vertices using Linear Blend Skinning (LBS).
        transform_matrices: list of 3x3 affine matrices corresponding to joint order.
        """
        num_verts = len(self.vertices)
        warped_vertices = np.zeros_like(self.vertices)

        # Convert vertices to homogeneous coordinates (N, 3)
        ones = np.ones((num_verts, 1), dtype=np.float32)
        homo_verts = np.hstack([self.vertices, ones])

        for j_idx, T in enumerate(transform_matrices):
            w = self.weights[:, j_idx:j_idx+1]  # (N, 1)
            # Transform vertices by joint matrix T
            transformed = (T @ homo_verts.T).T[:, :2]  # (N, 2)
            warped_vertices += w * transformed

        return warped_vertices

    def render_deformed_image(self, source_rgba, warped_vertices):
        """Renders warped mesh triangles onto output RGBA canvas via affine piecewise warping."""
        output_rgba = np.zeros_like(source_rgba)

        for tri in self.triangles:
            src_pts = self.vertices[tri].astype(np.float32)
            dst_pts = warped_vertices[tri].astype(np.float32)

            # Draw triangle via affine transform mapping
            self._warp_triangle(source_rgba, output_rgba, src_pts, dst_pts)

        return output_rgba

    def _warp_triangle(self, src_img, dst_img, src_pts, dst_pts):
        """Helper to warp a single triangular mesh element from src_img to dst_img."""
        r1 = cv2.boundingRect(src_pts)
        r2 = cv2.boundingRect(dst_pts)

        # Offset points by bounding box top-left
        src_rect_pts = src_pts - np.array([r1[0], r1[1]], dtype=np.float32)
        dst_rect_pts = dst_pts - np.array([r2[0], r2[1]], dtype=np.float32)

        if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
            return

        # Get affine transform matrix for triangle
        M = cv2.getAffineTransform(src_rect_pts, dst_rect_pts)

        # Crop patches
        src_crop = src_img[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
        if src_crop.shape[0] == 0 or src_crop.shape[1] == 0:
            return

        # Warp cropped patch
        warped_crop = cv2.warpAffine(
            src_crop, M, (r2[2], r2[3]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        # Apply triangular mask to prevent boundary bleeding
        mask = np.zeros((r2[3], r2[2]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_rect_pts.astype(np.int32), 255)

        # Blend warped triangle into destination frame
        dst_patch = dst_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
        if dst_patch.shape[:2] == warped_crop.shape[:2]:
            mask_3d = mask[:, :, np.newaxis] / 255.0
            dst_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = (
                dst_patch * (1.0 - mask_3d) + warped_crop * mask_3d
            ).astype(np.uint8)

if __name__ == "__main__":
    joints_1024x1536 = {
        'shoulder': [735, 720],
        'elbow': [735, 920],
        'wrist': [800, 1160]
    }
    engine = LBSMeshEngine2D((1536, 1024, 4), joints_1024x1536)
    engine.generate_mesh(grid_spacing=25, bbox=(600, 1250, 650, 900))
    print(f"[Step 2] Initialized LBS Engine successfully with {len(engine.vertices)} vertices and {len(engine.triangles)} triangles.")
