"""
src/analysis/vessel_topology.py
================================
Pillar 4 — 3D Vessel Topology Mapping

Extracts the retinal vessel tree from a segmentation mask and converts
it to a graph structure suitable for interactive 3D visualization.

Pipeline:
  1. Skeletonize vessel mask to 1-pixel-wide skeleton
  2. Detect branch/junction points
  3. Trace individual vessel segments between junctions
  4. Build graph: nodes = junctions, edges = vessel segments
  5. Compute per-edge properties (length, width, curvature)
  6. Output JSON-serializable graph for WebGL rendering
"""

import cv2
import numpy as np
from typing import Optional


# ---- Skeleton Processing -------------------------------------

def extract_skeleton(vessel_mask: np.ndarray) -> np.ndarray:
    """
    Thin the binary vessel mask to a 1-pixel-wide skeleton.

    params: vessel_mask — (H, W) bool or uint8 mask
    returns: skeleton — (H, W) bool array
    """
    binary = (vessel_mask > 0).astype(np.uint8) * 255

    # Try OpenCV's ximgproc thinning first (faster, better)
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
        skeleton = cv2.ximgproc.thinning(binary)
    else:
        # Fallback: iterative morphological thinning
        skeleton = _morphological_thin(binary)

    return skeleton > 0


def _morphological_thin(binary: np.ndarray) -> np.ndarray:
    """Fallback skeletonization using Zhang-Suen-like iterative thinning."""
    skel = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = binary.copy()
    max_iter = 500

    for _ in range(max_iter):
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        diff = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, diff)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    return skel


# ---- Junction / Branch Point Detection -------------------------

def find_junction_points(skeleton: np.ndarray) -> list:
    """
    Find branch points (junctions) in a skeleton image.
    A pixel is a junction if it has 3 or more skeleton neighbors.

    params: skeleton — (H, W) bool array
    returns: list of (y, x) tuples for junction points
    """
    skel_uint8 = skeleton.astype(np.uint8)
    h, w = skel_uint8.shape
    junctions = []

    # Count neighbors using convolution
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    neighbor_count = cv2.filter2D(skel_uint8, -1, kernel)

    # Junction = skeleton pixel with 3+ neighbors
    junction_mask = (skel_uint8 > 0) & (neighbor_count >= 3)

    # Cluster nearby junctions (within 5 pixels) into single points
    junction_coords = np.argwhere(junction_mask)
    if len(junction_coords) == 0:
        return []

    # Use simple clustering: merge points within 5px of each other
    clustered = _cluster_points(junction_coords, min_dist=5)
    return [(int(y), int(x)) for y, x in clustered]


def find_endpoint_points(skeleton: np.ndarray) -> list:
    """
    Find endpoints in a skeleton image.
    A pixel is an endpoint if it has exactly 1 skeleton neighbor.
    """
    skel_uint8 = skeleton.astype(np.uint8)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    neighbor_count = cv2.filter2D(skel_uint8, -1, kernel)
    endpoint_mask = (skel_uint8 > 0) & (neighbor_count == 1)

    coords = np.argwhere(endpoint_mask)
    return [(int(y), int(x)) for y, x in coords]


def _cluster_points(points: np.ndarray, min_dist: int = 5) -> list:
    """Cluster nearby points by merging those within min_dist pixels."""
    if len(points) == 0:
        return []

    clustered = []
    used = set()

    for i, p1 in enumerate(points):
        if i in used:
            continue
        cluster = [p1]
        used.add(i)

        for j, p2 in enumerate(points):
            if j in used:
                continue
            if np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) < min_dist:
                cluster.append(p2)
                used.add(j)

        # Use centroid of cluster
        centroid = np.mean(cluster, axis=0)
        clustered.append(centroid)

    return clustered


# ---- Segment Tracing ------------------------------------------

def trace_vessel_segments(
    skeleton: np.ndarray,
    junctions: list,
    endpoints: list,
    width_map: Optional[np.ndarray] = None,
) -> list:
    """
    Trace individual vessel segments between junctions/endpoints.

    Each segment is a path of skeleton pixels between two key points.
    Returns a list of segment dictionaries with path coordinates and properties.

    params:
        skeleton — (H, W) bool skeleton
        junctions — list of (y, x) junction points
        endpoints — list of (y, x) endpoint points
        width_map — optional (H, W) distance transform for vessel width
    returns: list of segment dicts
    """
    h, w = skeleton.shape
    skel_copy = skeleton.copy()
    segments = []

    # Create a map of key points
    key_points = set()
    for y, x in junctions + endpoints:
        key_points.add((int(y), int(x)))

    # Remove junction pixels from skeleton to split into segments
    junction_mask = np.zeros((h, w), dtype=bool)
    for y, x in junctions:
        y, x = int(y), int(x)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    junction_mask[ny, nx] = True

    # Skeleton with junctions removed → disconnected segments
    segment_mask = skel_copy & ~junction_mask

    # Label connected components (each is a vessel segment)
    segment_uint8 = segment_mask.astype(np.uint8) * 255
    num_labels, labels = cv2.connectedComponents(segment_uint8, connectivity=8)

    for label_id in range(1, min(num_labels, 200)):  # Cap at 200 segments
        component_pixels = np.argwhere(labels == label_id)
        if len(component_pixels) < 5:  # Skip tiny fragments
            continue

        # Order pixels along the path (approximate by sorting)
        # Use distance from first pixel for ordering
        start = component_pixels[0]
        distances = np.sqrt(np.sum((component_pixels - start)**2, axis=1))
        sorted_indices = np.argsort(distances)
        path = component_pixels[sorted_indices]

        # Compute segment properties
        segment_length = float(np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))))

        # Average vessel width along segment
        avg_width = 2.0
        if width_map is not None and len(path) > 0:
            widths = [width_map[py, px] for py, px in path if 0 <= py < h and 0 <= px < w]
            avg_width = float(np.mean(widths)) * 2.0 if widths else 2.0

        # Compute curvature (tortuosity of this segment)
        if len(path) >= 2:
            straight_dist = np.sqrt((path[0][0] - path[-1][0])**2 + (path[0][1] - path[-1][1])**2)
            tortuosity = segment_length / (straight_dist + 1e-8)
        else:
            tortuosity = 1.0

        # Subsample path for JSON output (every 5th point)
        subsampled = path[::max(1, len(path) // 20)].tolist()

        segments.append({
            "id": label_id,
            "path": [[int(p[1]), int(p[0])] for p in subsampled],  # [x, y] format
            "start": [int(path[0][1]), int(path[0][0])],
            "end": [int(path[-1][1]), int(path[-1][0])],
            "length": round(segment_length, 1),
            "avg_width": round(avg_width, 1),
            "tortuosity": round(tortuosity, 3),
            "pixel_count": len(path),
        })

    return segments


# ---- Graph Construction ---------------------------------------

def build_vessel_graph(
    junctions: list,
    endpoints: list,
    segments: list,
    image_height: int,
    image_width: int,
) -> dict:
    """
    Build a JSON-serializable graph structure for WebGL rendering.

    Nodes = junctions + endpoints
    Edges = vessel segments connecting nodes

    params:
        junctions — list of (y, x)
        endpoints — list of (y, x)
        segments — list of segment dicts
        image_height, image_width — for normalization
    returns: dict with nodes and edges arrays
    """
    nodes = []

    # Add junction nodes
    for i, (y, x) in enumerate(junctions):
        nodes.append({
            "id": f"j_{i}",
            "type": "junction",
            "x": round(x / image_width, 4),    # Normalized to [0, 1]
            "y": round(y / image_height, 4),
            "z": round(0.05 * np.random.randn(), 3),  # Small Z for 3D effect
            "connections": 0,  # Updated below
            "size": 4,
        })

    # Add endpoint nodes
    for i, (y, x) in enumerate(endpoints[:100]):  # Cap endpoints
        nodes.append({
            "id": f"e_{i}",
            "type": "endpoint",
            "x": round(x / image_width, 4),
            "y": round(y / image_height, 4),
            "z": round(0.03 * np.random.randn(), 3),
            "connections": 1,
            "size": 2,
        })

    # Build edges from segments
    edges = []
    for seg in segments:
        # Classify vessel type by width (wider → artery, thinner → vein)
        vessel_type = "artery" if seg["avg_width"] > 3.0 else "vein"

        edges.append({
            "id": f"seg_{seg['id']}",
            "source_point": seg["start"],  # [x, y]
            "target_point": seg["end"],
            "path": seg["path"],           # Array of [x, y] points
            "vessel_type": vessel_type,
            "width": seg["avg_width"],
            "length": seg["length"],
            "tortuosity": seg["tortuosity"],
            "color": [220, 60, 60] if vessel_type == "artery" else [60, 100, 220],
        })

    # Compute topology metrics
    total_length = sum(s["length"] for s in segments)
    avg_tortuosity = np.mean([s["tortuosity"] for s in segments]) if segments else 1.0

    return {
        "nodes": nodes,
        "edges": edges,
        "topology_metrics": {
            "junction_count": len(junctions),
            "endpoint_count": len(endpoints),
            "segment_count": len(segments),
            "total_vessel_length_px": round(total_length, 1),
            "mean_tortuosity": round(float(avg_tortuosity), 3),
            "mean_vessel_width": round(float(np.mean([s["avg_width"] for s in segments])), 1) if segments else 0,
            "branching_density": round(len(junctions) / (total_length + 1e-8) * 1000, 2),
        },
        "image_dimensions": {"width": image_width, "height": image_height},
    }


# ---- Master Topology Function --------------------------------

def extract_vessel_topology(
    image_rgb: np.ndarray,
    vessel_mask: Optional[np.ndarray] = None,
    width_map: Optional[np.ndarray] = None,
) -> dict:
    """
    Extract the complete vessel topology from a retinal image.

    If vessel_mask is not provided, it will be computed using the
    anomaly_detector's vessel segmentation.

    params:
        image_rgb — (H, W, 3) float32 in [0, 1]
        vessel_mask — optional pre-computed vessel mask
        width_map — optional distance transform for vessel widths
    returns: JSON-serializable graph structure
    """
    h, w = image_rgb.shape[:2]

    # Get vessel mask if not provided
    if vessel_mask is None:
        from src.analysis.anomaly_detector import preprocess_for_detection, segment_vessels
        prep = preprocess_for_detection(image_rgb)
        vessel_result = segment_vessels(prep["green_clahe"], prep["fov_mask"])
        vessel_mask = vessel_result["vessel_mask"]
        width_map = vessel_result["width_map"]

    # Step 1: Skeletonize
    skeleton = extract_skeleton(vessel_mask)

    # Step 2: Find key points
    junctions = find_junction_points(skeleton)
    endpoints = find_endpoint_points(skeleton)

    # Step 3: Trace segments
    segments = trace_vessel_segments(skeleton, junctions, endpoints, width_map)

    # Step 4: Build graph
    graph = build_vessel_graph(junctions, endpoints, segments, h, w)

    return graph
