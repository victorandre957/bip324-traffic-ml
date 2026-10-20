"""Immutable, atomic output directories and reproducibility metadata."""

from __future__ import annotations

from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


@contextmanager
def new_directory(destination: Path):
    """Publish only completed artifacts. Never replace an existing output directory.

    The exclusive lock serializes our writers; interrupted temporary directories are
    not considered datasets. Only this call's own temporary files are cleaned up.
    """
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name(f".{destination.name}.lock")
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    temporary = None
    try:
        os.write(fd, f"pid={os.getpid()}\n".encode())
        if destination.exists():
            raise FileExistsError(f"output exists; use a new directory: {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
        yield temporary
        if destination.exists():
            raise FileExistsError(f"output appeared during processing: {destination}")
        temporary.rename(destination)
        temporary = None
    finally:
        os.close(fd)
        if temporary is not None:
            shutil.rmtree(temporary)
        lock.unlink()


def file_identity(path: Path, *, checksum: bool = False) -> dict:
    stat = path.stat()
    result = {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if checksum:
        with path.open("rb") as stream:
            result["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def environment() -> dict:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("bip324-traffic-ml", "numpy", "pandas", "scikit-learn", "xgboost", "torch"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            pass
    return result
