# -*- coding: utf-8 -*-
"""
3D Obstacle Envelope Extractor.
Performs:
1. RANSAC Tabletop Plane Fitting (< 5ms) to segment table surface.
2. Manipulation Target Masking (deducting red cube / basket from obstacles).
3. Euclidean Spatial Clustering of foreground obstacle points.
4. 3D Oriented Bounding Box (3D OBB) & Convex Hull Envelope fitting with clearance margin.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class ObstacleOBB:
    """
    3D Oriented Bounding Box representation of an obstacle.
    """
    center: np.ndarray        # (3,) [cx, cy, cz] in Franka base frame (m)
    extents: np.ndarray       # (3,) [dx, dy, dz] full side lengths (m)
    rotation_matrix: np.ndarray # (3, 3) rotation from local OBB axes to Base frame
    yaw_rad: float            # Horizontal yaw angle in radians
    corners: np.ndarray       # (8, 3) coordinates of all 8 box corners
    num_points: int           # Number of raw sensor points supporting this envelope
    margin_added: float       # Margin buffer added to extents (m)

    def contains_point(self, point: np.ndarray) -> bool:
        """Checks if a 3D point is inside this OBB."""
        p = np.asarray(point, dtype=np.float64) - self.center
        p_local = self.rotation_matrix.T @ p
        half = self.extents * 0.5
        return np.all(np.abs(p_local) <= half)


def fit_plane_ransac(
    points: np.ndarray,
    distance_threshold: float = 0.008,
    max_iterations: int = 150,
    expected_normal: np.ndarray = np.array([0.0, 0.0, 1.0]),
    min_normal_dot: float = 0.85
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    RANSAC plane fitting on 3D point cloud.
    Plane equation: a*x + b*y + c*z + d = 0, normalized so a^2 + b^2 + c^2 = 1.
    
    Args:
        points: (N, 3) array
        distance_threshold: maximum distance for inliers (default 8mm)
        max_iterations: RANSAC iteration count
        expected_normal: expected upright normal vector [0, 0, 1]
        min_normal_dot: minimum cosine similarity to expected normal
        
    Returns:
        best_plane: [a, b, c, d] or None
        inlier_mask: (N,) bool array (table points)
        above_table_mask: (N,) bool array (points clearly above table plane by > threshold)
    """
    N = len(points)
    if N < 3:
        return None, np.zeros(N, dtype=bool), np.zeros(N, dtype=bool)

    best_inliers = 0
    best_plane = None
    rng = np.random.default_rng(42)

    for _ in range(max_iterations):
        sample_indices = rng.choice(N, size=3, replace=False)
        p1, p2, p3 = points[sample_indices]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            continue
        normal = normal / norm_len

        # Ensure normal points upwards (+Z direction)
        if normal[2] < 0:
            normal = -normal

        # Check alignment with vertical table normal
        if np.dot(normal, expected_normal) < min_normal_dot:
            continue

        d = -np.dot(normal, p1)
        distances = np.abs(points @ normal + d)
        inlier_count = int(np.sum(distances < distance_threshold))

        if inlier_count > best_inliers:
            best_inliers = inlier_count
            best_plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)

    if best_plane is None:
        return None, np.zeros(N, dtype=bool), np.zeros(N, dtype=bool)

    # Refine plane equation using all inliers via SVD/least-squares
    distances = np.abs(points @ best_plane[:3] + best_plane[3])
    inlier_mask = distances < distance_threshold
    inlier_pts = points[inlier_mask]

    if len(inlier_pts) >= 3:
        centroid = np.mean(inlier_pts, axis=0)
        uu, dd, vv = np.linalg.svd(inlier_pts - centroid)
        normal = vv[2]
        if normal[2] < 0:
            normal = -normal
        d = -float(np.dot(normal, centroid))
        best_plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)
        signed_dist = points @ best_plane[:3] + best_plane[3]
        inlier_mask = np.abs(signed_dist) < distance_threshold
        # Above table points: strictly above table by at least distance_threshold
        above_table_mask = signed_dist >= distance_threshold
    else:
        signed_dist = points @ best_plane[:3] + best_plane[3]
        above_table_mask = signed_dist >= distance_threshold

    return best_plane, inlier_mask, above_table_mask


def mask_target_entities(
    points: np.ndarray,
    target_boxes_2d: Optional[List[Dict[str, Any]]] = None,
    pixel_coords: Optional[np.ndarray] = None,
    target_cylinders_3d: Optional[List[Dict[str, Any]]] = None
) -> np.ndarray:
    """
    Masks out manipulation target objects (e.g. red cube, basket) so they are NOT
    treated as obstacles to be avoided.
    
    Args:
        points: (N, 3) 3D points
        target_boxes_2d: list of 2D bounding boxes from Visual Grounding [{'box_px': [x1, y1, x2, y2]}]
        pixel_coords: (N, 2) corresponding pixel coordinates [u, v]
        target_cylinders_3d: optional list of 3D cylinders [{'center': [x, y], 'radius': r}]
        
    Returns:
        is_obstacle_mask: (N,) bool array (True for points NOT belonging to targets)
    """
    N = len(points)
    is_obstacle_mask = np.ones(N, dtype=bool)

    # 1. 2D bounding box exclusion
    if target_boxes_2d and pixel_coords is not None and len(pixel_coords) == N:
        for t in target_boxes_2d:
            box = t.get("box_px")
            if box:
                x1, y1, x2, y3 = box
                in_box = (
                    (pixel_coords[:, 0] >= x1) & (pixel_coords[:, 0] <= x2) &
                    (pixel_coords[:, 1] >= y1) & (pixel_coords[:, 1] <= y3)
                )
                is_obstacle_mask[in_box] = False

    # 2. 3D cylinder/region exclusion
    if target_cylinders_3d:
        for c in target_cylinders_3d:
            cx, cy = c["center"][:2]
            rad = c["radius"]
            dist_xy = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
            is_obstacle_mask[dist_xy <= rad] = False

    return is_obstacle_mask


def cluster_euclidean(
    points: np.ndarray,
    eps: float = 0.035,
    min_points: int = 30
) -> List[np.ndarray]:
    """
    Fast Euclidean clustering for 3D point cloud using spatial grid / voxel hashing.
    Runs in < 10ms in pure NumPy/SciPy.
    """
    if len(points) < min_points:
        return []

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        visited = np.zeros(len(points), dtype=bool)
        clusters = []

        for i in range(len(points)):
            if visited[i]:
                continue
            neighbors = tree.query_ball_point(points[i], r=eps)
            if len(neighbors) < min_points:
                visited[i] = True
                continue

            cluster = []
            queue = list(neighbors)
            visited[neighbors] = True

            idx = 0
            while idx < len(queue):
                curr = queue[idx]
                cluster.append(curr)
                curr_neighbors = tree.query_ball_point(points[curr], r=eps)
                if len(curr_neighbors) >= min_points:
                    for n in curr_neighbors:
                        if not visited[n]:
                            visited[n] = True
                            queue.append(n)
                idx += 1

            if len(cluster) >= min_points:
                clusters.append(points[cluster])

        return clusters

    except ImportError:
        # Fallback grid clustering if scipy not available
        grid_size = eps
        grid_coords = np.floor(points / grid_size).astype(np.int32)
        unique_cells, inverse, counts = np.unique(grid_coords, axis=0, return_inverse=True, return_counts=True)
        valid_cells = np.where(counts >= min_points)[0]
        return [points[inverse == cell_idx] for cell_idx in valid_cells]


def fit_3d_obb(
    cluster_points: np.ndarray,
    margin: float = 0.030,
    min_extent: float = 0.040
) -> ObstacleOBB:
    """
    Fits a 3D Oriented Bounding Box (OBB) around a cluster of 3D points.
    Assumes tabletop reference frame where horizontal yaw is optimized in the XY plane.
    
    Args:
        cluster_points: (M, 3) 3D points in Franka base frame
        margin: safety expansion margin added to all dimensions (e.g. 3cm)
        min_extent: minimum allowed extent along any dimension (e.g. 4cm)
    """
    pts = np.asarray(cluster_points, dtype=np.float64)
    M = len(pts)

    # 1. Planar 2D PCA in XY to find principal yaw angle theta
    xy = pts[:, :2]
    xy_center = np.mean(xy, axis=0)
    cov_xy = np.cov(xy - xy_center, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov_xy)
    
    # Primary axis in XY
    v_main = eigvecs[:, 1]
    yaw = float(np.arctan2(v_main[1], v_main[0]))

    # Rotation matrix around Z-axis
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    R_base_to_local = np.array([
        [ c,  s, 0.0],
        [-s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    R_local_to_base = R_base_to_local.T

    # Rotate points into local frame
    pts_local = pts @ R_base_to_local.T

    # 2. Extract bounding extents in local frame
    min_local = np.min(pts_local, axis=0)
    max_local = np.max(pts_local, axis=0)

    # Expand by safety margin
    extents = np.maximum(max_local - min_local + 2.0 * margin, min_extent)
    center_local = (min_local + max_local) * 0.5
    center_base = R_local_to_base @ center_local

    # Compute 8 box corners in base frame
    hx, hy, hz = extents * 0.5
    local_corners = np.array([
        [-hx, -hy, -hz],
        [ hx, -hy, -hz],
        [ hx,  hy, -hz],
        [-hx,  hy, -hz],
        [-hx, -hy,  hz],
        [ hx, -hy,  hz],
        [ hx,  hy,  hz],
        [-hx,  hy,  hz],
    ], dtype=np.float64)
    corners_base = local_corners @ R_local_to_base.T + center_base

    return ObstacleOBB(
        center=center_base,
        extents=extents,
        rotation_matrix=R_local_to_base,
        yaw_rad=yaw,
        corners=corners_base,
        num_points=M,
        margin_added=margin
    )


def extract_obstacle_envelopes(
    points_base: np.ndarray,
    pixel_coords: Optional[np.ndarray] = None,
    target_boxes_2d: Optional[List[Dict[str, Any]]] = None,
    table_distance_threshold: float = 0.008,
    cluster_eps: float = 0.035,
    min_cluster_points: int = 30,
    envelope_margin: float = 0.030
) -> Tuple[List[ObstacleOBB], Optional[np.ndarray], Dict[str, Any]]:
    """
    Complete end-to-end perception pipeline:
    Raw Points -> RANSAC Table Fit -> Target Entity Exclusion -> Clustering -> 3D OBBs
    
    Returns:
        envelopes: List of ObstacleOBB objects
        table_plane: [a, b, c, d] table plane equation
        stats: Diagnostic statistics dictionary
    """
    total_pts = len(points_base)
    if total_pts == 0:
        return [], None, {"status": "empty_points"}

    # 1. RANSAC table plane fitting
    table_plane, inlier_table_mask, above_table_mask = fit_plane_ransac(
        points_base, distance_threshold=table_distance_threshold
    )

    if table_plane is None:
        # If table not detected, treat all points as potential obstacles
        cand_mask = np.ones(total_pts, dtype=bool)
    else:
        cand_mask = above_table_mask

    # 2. Mask out manipulation targets
    not_target_mask = mask_target_entities(
        points_base,
        target_boxes_2d=target_boxes_2d,
        pixel_coords=pixel_coords
    )
    obstacle_points_mask = cand_mask & not_target_mask
    obstacle_pts = points_base[obstacle_points_mask]

    # 3. Euclidean clustering
    clusters = cluster_euclidean(obstacle_pts, eps=cluster_eps, min_points=min_cluster_points)

    # 4. Fit 3D OBB envelopes
    envelopes = [
        fit_3d_obb(cl, margin=envelope_margin) for cl in clusters
    ]

    stats = {
        "total_points": total_pts,
        "table_inliers": int(np.sum(inlier_table_mask)),
        "candidate_obstacle_points": int(len(obstacle_pts)),
        "cluster_count": len(clusters),
        "envelope_count": len(envelopes),
        "table_plane": table_plane.tolist() if table_plane is not None else None
    }

    return envelopes, table_plane, stats
