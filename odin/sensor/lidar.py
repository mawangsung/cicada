"""LiDAR point cloud input — stub for Metin (qubit 2) sensor interface.

Supported formats: .pcd, .las, .ply
Maps point cloud metadata → hexagram index → Qubit state for Metin.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger("odin.lidar")

SUPPORTED_FORMATS = {".pcd", ".las", ".ply"}


class LiDARInput:
    """Load LiDAR point cloud files and encode them as a Metin qubit state."""

    def load(self, path: str) -> dict:
        """Dispatch to format-specific loader."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"LiDAR file not found: {path}")
        suffix = p.suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format {suffix!r}. Supported: {SUPPORTED_FORMATS}")
        logger.info("Loading LiDAR file: %s", path)
        if suffix == ".pcd":
            return self.load_pcd(path)
        elif suffix == ".las":
            return self.load_las(path)
        elif suffix == ".ply":
            return self.load_ply(path)

    def load_pcd(self, path: str) -> dict:
        """Load .pcd point cloud via open3d."""
        try:
            import open3d as o3d
            pc = o3d.io.read_point_cloud(path)
            pts = len(pc.points)
            bounds = pc.get_axis_aligned_bounding_box()
            info = {
                "format": "pcd",
                "path": path,
                "num_points": pts,
                "has_normals": pc.has_normals(),
                "has_colors": pc.has_colors(),
                "bounds_min": list(bounds.min_bound),
                "bounds_max": list(bounds.max_bound),
            }
            logger.info("PCD loaded: %d points", pts)
            return info
        except ImportError:
            logger.warning("open3d not installed — returning stub metadata.")
            return self._stub_meta(path, "pcd")

    def load_las(self, path: str) -> dict:
        """Load .las LiDAR file via laspy."""
        try:
            import laspy
            las = laspy.read(path)
            pts = len(las.points)
            info = {
                "format": "las",
                "path": path,
                "num_points": pts,
                "version": str(las.header.version),
                "point_format": int(las.point_format.id),
            }
            logger.info("LAS loaded: %d points", pts)
            return info
        except ImportError:
            logger.warning("laspy not installed — returning stub metadata.")
            return self._stub_meta(path, "las")

    def load_ply(self, path: str) -> dict:
        """Load .ply point cloud via open3d."""
        try:
            import open3d as o3d
            pc = o3d.io.read_point_cloud(path)
            pts = len(pc.points)
            info = {
                "format": "ply",
                "path": path,
                "num_points": pts,
                "has_normals": pc.has_normals(),
            }
            logger.info("PLY loaded: %d points", pts)
            return info
        except ImportError:
            logger.warning("open3d not installed — returning stub metadata.")
            return self._stub_meta(path, "ply")

    def to_qubit_encoding(self, point_cloud_dict: dict) -> "Qubit":
        """Encode point cloud metadata as a Metin qubit state.

        Maps num_points % 64 → hexagram index → Qubit.
        This is the bridge between classical LiDAR data and the quantum register.
        """
        from ..state.qubit import Qubit
        num_points = point_cloud_dict.get("num_points", 1)
        hexagram_number = (num_points % 64) + 1  # 1-64
        logger.info(
            "Encoding %d points → hexagram %d → Metin qubit",
            num_points, hexagram_number,
        )
        return Qubit.from_hexagram(hexagram_number)

    @staticmethod
    def _stub_meta(path: str, fmt: str) -> dict:
        """Return deterministic stub metadata based on file size."""
        size = os.path.getsize(path) if os.path.exists(path) else 1024
        return {
            "format": fmt,
            "path": path,
            "num_points": size // 16,  # rough estimate: 16 bytes/point
            "stub": True,
        }
