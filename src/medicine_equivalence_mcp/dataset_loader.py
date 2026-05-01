"""Dataset bootstrap helpers for local development and Render deployments."""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DATASETS_ROOT = PROJECT_ROOT / "Datasets"
RENDER_DATASETS_ROOT = Path("/tmp/datasets")
READY_MARKER_NAME = ".ready"


def _download_archive(data_url: str, destination: Path) -> None:
    """Stream a dataset archive to disk without loading it fully into memory."""
    request = Request(
        data_url,
        headers={
            "User-Agent": "Mozilla/5.0",
        },
    )
    with urlopen(request, timeout=120) as response:
        with destination.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)


def _safe_extract_zip(zip_path: Path, destination: Path) -> None:
    """Extract a zip archive while blocking path traversal entries."""
    with zipfile.ZipFile(zip_path, "r") as archive:
        destination_root = destination.resolve()
        for member in archive.infolist():
            target_path = (destination / member.filename).resolve()
            if not str(target_path).startswith(str(destination_root)):
                raise RuntimeError(f"Unsafe zip entry: {member.filename}")
        archive.extractall(destination)


def _resolve_extracted_datasets_root(data_dir: Path) -> Path:
    """Return the extracted dataset root, tolerating one nested parent folder.

    Some archives are packaged as:
    - Italy/
    - France/
    - UK/

    while others include an extra top-level folder:
    - Datasets/Italy/
    - Datasets/France/
    - Datasets/UK/
    """
    nested_root = data_dir / "Datasets"
    if nested_root.is_dir():
        return nested_root
    return data_dir


def ensure_datasets_ready(data_url: str, data_dir: Path = RENDER_DATASETS_ROOT) -> Path:
    """Download and extract datasets once for the current container lifecycle."""
    ready_marker = data_dir / READY_MARKER_NAME
    if ready_marker.exists():
        return _resolve_extracted_datasets_root(data_dir)

    if not data_url:
        raise RuntimeError("DATA_URL is not set")

    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    fd, temp_zip_str = tempfile.mkstemp(suffix=".zip", dir="/tmp")
    os.close(fd)
    temp_zip_path = Path(temp_zip_str)

    try:
        _download_archive(data_url, temp_zip_path)
        _safe_extract_zip(temp_zip_path, data_dir)
        ready_marker.write_text("ok", encoding="utf-8")
        return _resolve_extracted_datasets_root(data_dir)
    finally:
        if temp_zip_path.exists():
            temp_zip_path.unlink()


def get_datasets_root() -> Path:
    """Resolve the active dataset root for local runs or Render deployments."""
    explicit_root = os.getenv("DATASETS_ROOT")
    if explicit_root:
        return Path(explicit_root)

    data_url = os.getenv("DATA_URL")
    if data_url:
        return ensure_datasets_ready(data_url, RENDER_DATASETS_ROOT)

    return LOCAL_DATASETS_ROOT
