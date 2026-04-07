"""HuggingFace dataset bridge for ODIN circuit results and sensor files."""

import json
import os
import datetime
import tempfile
from pathlib import Path
from typing import Optional


_LIDAR_EXTS = {".pcd", ".las", ".ply"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
SENSOR_EXTS = _LIDAR_EXTS | _VIDEO_EXTS


class HFDatasetBridge:
    """Bridge between ODIN quantum objects and HuggingFace dataset repos.

    Token is read from the ``token`` argument or the ``HF_TOKEN`` env var.
    """

    def __init__(self, token: Optional[str] = None):
        self._token = token or os.environ.get("HF_TOKEN")

    def _api(self, token: Optional[str] = None):
        from huggingface_hub import HfApi
        return HfApi(token=token or self._token)

    # ── Circuit result upload ────────────────────────────────────────────────

    def push_circuit_result(
        self,
        register,
        rune_sequence: list[str],
        dataset_repo: str,
        token: Optional[str] = None,
    ) -> str:
        """Serialize register state + rune sequence to JSON and upload to dataset repo.

        Returns the path_in_repo of the uploaded file.
        """
        api = self._api(token)
        timestamp = datetime.datetime.utcnow().isoformat(timespec="seconds")
        payload = {
            "timestamp": timestamp,
            "rune_sequence": rune_sequence,
            "circuit_result": register.to_dict(),
        }
        json_bytes = json.dumps(payload, indent=2).encode("utf-8")
        filename = f"results/{timestamp.replace(':', '-')}.json"

        api.upload_file(
            path_or_fileobj=json_bytes,
            path_in_repo=filename,
            repo_id=dataset_repo,
            repo_type="dataset",
            commit_message=f"ODIN circuit result: {' '.join(rune_sequence)}",
        )
        return filename

    # ── Sensor file access ────────────────────────────────────────────────────

    def pull_file(
        self,
        repo_id: str,
        filename: str,
        token: Optional[str] = None,
        local_dir: Optional[str] = None,
    ) -> str:
        """Download a sensor file from a HuggingFace dataset repo.

        Returns the absolute local path, ready for LiDARInput.load() or
        DashcamInput.load_video().
        """
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token or self._token,
            local_dir=local_dir or tempfile.mkdtemp(),
        )
        return local_path

    def list_sensor_files(
        self,
        repo_id: str,
        token: Optional[str] = None,
    ) -> list[str]:
        """Return filenames in the dataset repo that match known sensor extensions."""
        api = self._api(token)
        all_files = list(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
        return [f for f in all_files if Path(f).suffix.lower() in SENSOR_EXTS]
