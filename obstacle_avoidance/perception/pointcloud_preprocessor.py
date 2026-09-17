# -*- coding: utf-8 -*-
"""
Point Cloud Preprocessor for RealSense RGB-D observations.
Performs:
1. Fast vectorized back-projection from depth image to camera 3D points.
2. Extrinsic transformation from Camera frame to Franka Base frame.
3. Workspace ROI bounding box filtering.
4. Statistical outlier noise removal.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

DEFAULT_WORKSPACE_BOUNDS = {
    "x_min": 0.20, "x_max": 0.80,
    "y_min": -0.45, "y_max": 0.45,
    "z_min": -0.05, "z_max": 0.65
}


def depth_to_pointcloud(
    depth_image: np.ndarray,
    intrinsics: Dict[str, float],
    depth_scale: float = 0.001,
    min_depth: float = 0.15,
    max_depth: float = 2.00,
    stride: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized back-projection from 2D depth image to 3D camera coordinates.
    
    Args:
        depth_image: (H, W) uint16 or float array (depth in raw units, e.g. mm)
        intrinsics: dict with keys 'fx', 'fy', 'cx', 'cy'
        depth_scale: conversion factor to meters (default 0.001 for mm -> m)
        min_depth: minimum valid depth in meters
        max_depth: maximum valid depth in meters
        stride: pixel sub-sampling stride to accelerate processing
        
    Returns:
        points_cam: (N, 3) float64 array of 3D points in camera frame [X_c, Y_c, Z_c]
        pixel_coords: (N, 2) int array of corresponding [u, v] image pixels
    """
    depth = np.asarray(depth_image, dtype=np.float32)
    H, W = depth.shape[:2]
    
    # Subsample grid
    v_idx, u_idx = np.mgrid[0:H:stride, 0:W:stride]
    sub_depth = depth[v_idx, u_idx] * depth_scale
    
    # Filter valid depth range
    valid_mask = np.isfinite(sub_depth) & (sub_depth >= min_depth) & (sub_depth <= max_depth)
    
    z = sub_depth[valid_mask]
    u = u_idx[valid_mask]
    v = v_idx[valid_mask]
    
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    points_cam = np.column_stack((x, y, z)).astype(np.float64)
    pixel_coords = np.column_stack((u, v)).astype(np.int32)
    
    return points_cam, pixel_coords


def transform_points_to_base(
    points_cam: np.ndarray,
    camera_to_base: np.ndarray
) -> np.ndarray:
    """
    Transforms 3D points from Camera frame to Robot Base frame using 4x4 homogeneous matrix.
    P_base = R * P_cam + T
    """
    if len(points_cam) == 0:
        return np.empty((0, 3), dtype=np.float64)
        
    T = np.asarray(camera_to_base, dtype=np.float64)
    assert T.shape == (4, 4), f"camera_to_base must be 4x4, got {T.shape}"
    
    R = T[:3, :3]
    t = T[:3, 3]
    
    # (N, 3) @ (3, 3).T + (3,)
    points_base = points_cam @ R.T + t
    return points_base


def filter_workspace_roi(
    points: np.ndarray,
    bounds: Optional[Dict[str, float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters points within Cartesian bounding box ROI.
    Returns:
        filtered_points: (M, 3)
        in_roi_mask: (N,) bool mask
    """
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64), np.zeros(0, dtype=bool)
        
    b = bounds or DEFAULT_WORKSPACE_BOUNDS
    mask = (
        (points[:, 0] >= b["x_min"]) & (points[:, 0] <= b["x_max"]) &
        (points[:, 1] >= b["y_min"]) & (points[:, 1] <= b["y_max"]) &
        (points[:, 2] >= b["z_min"]) & (points[:, 2] <= b["z_max"])
    )
    return points[mask], mask


def remove_statistical_outliers(
    points: np.ndarray,
    voxel_size: float = 0.008,
    min_neighbors_per_voxel: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast voxel-grid density filtering to eliminate airborne flying noise pixels.
    Very fast pure-NumPy implementation (no heavy Open3D dependency required).
    """
    if len(points) < 10:
        return points, np.ones(len(points), dtype=bool)
        
    # Discretize points into voxel coordinates
    voxel_coords = np.floor(points / voxel_size).astype(np.int32)
    
    # Find unique voxels and their counts
    unique_voxels, inverse_indices, counts = np.unique(
        voxel_coords, axis=0, return_inverse=True, return_counts=True
    )
    
    # Points belonging to dense voxels are kept
    point_density = counts[inverse_indices]
    keep_mask = point_density >= min_neighbors_per_voxel
    
    return points[keep_mask], keep_mask
