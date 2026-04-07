"""Dashcam / blackbox video input — stub for Metin (qubit 2) sensor interface.

Supported formats: .mp4, .avi, .mov, .mkv
Maps video metadata → hexagram index → Qubit state for Metin.
"""

import logging
from pathlib import Path

logger = logging.getLogger("odin.dashcam")

SUPPORTED_FORMATS = {".mp4", ".avi", ".mov", ".mkv"}


class DashcamInput:
    """Load dashcam / blackbox video files and encode them as a Metin qubit state."""

    def load_video(self, path: str) -> dict:
        """Load video metadata via OpenCV.

        Returns dict with fps, frame_count, resolution, duration_s.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        if p.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format {p.suffix!r}. Supported: {SUPPORTED_FORMATS}"
            )
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            duration_s = frame_count / fps if fps > 0 else 0.0
            info = {
                "format": p.suffix.lower().lstrip("."),
                "path": path,
                "fps": fps,
                "frame_count": frame_count,
                "resolution": (width, height),
                "duration_s": duration_s,
            }
            logger.info(
                "Video loaded: %d frames @ %.1f fps (%.1fs)", frame_count, fps, duration_s
            )
            return info
        except ImportError:
            logger.warning("opencv-python not installed — returning stub metadata.")
            return self._stub_meta(path)

    def extract_frame(self, path: str, timestamp_s: float):
        """Extract a single frame at timestamp_s as a numpy BGR ndarray."""
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp_s * fps))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise RuntimeError(f"Could not read frame at t={timestamp_s}s from {path}")
            return frame
        except ImportError:
            logger.warning("opencv-python not installed — cannot extract frame.")
            return None

    def to_qubit_encoding(self, video_dict: dict) -> "Qubit":
        """Encode video metadata as a Metin qubit state.

        Maps frame_count % 64 → hexagram index → Qubit.
        This is the bridge between classical dashcam data and the quantum register.
        """
        from ..state.qubit import Qubit
        frame_count = video_dict.get("frame_count", 1)
        hexagram_number = (frame_count % 64) + 1  # 1-64
        logger.info(
            "Encoding %d frames → hexagram %d → Metin qubit",
            frame_count, hexagram_number,
        )
        return Qubit.from_hexagram(hexagram_number)

    @staticmethod
    def _stub_meta(path: str) -> dict:
        return {
            "format": Path(path).suffix.lower().lstrip("."),
            "path": path,
            "fps": 30.0,
            "frame_count": 900,
            "resolution": (1920, 1080),
            "duration_s": 30.0,
            "stub": True,
        }
