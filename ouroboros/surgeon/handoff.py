"""The file handoff between the laptop controller and the VM round loop.

Both sides share one root: a gs:// prefix in Stage B, a local directory in tests. The VM writes each round's
scores there and the laptop writes Beam's candidates, so the HF token never reaches the VM. Every write lands whole:
a GCS object appears only once its upload completes, and LocalHandoff writes a temp file and renames it. A reader
therefore sees a file in full or not at all, and a writer orders its files so the one a reader polls for comes last.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

TIMEOUT_S = 120.0  # a hung gcloud call fails after this, so it cannot stall the controller or trip the VM's idle kill


class Handoff(ABC):
    """Whole-object reads and writes under one root, by relative path."""

    @abstractmethod
    def read(self, rel: str) -> bytes | None:
        """The object's bytes, or None when it does not exist."""

    @abstractmethod
    def write(self, rel: str, data: bytes) -> None:
        """Store the whole object; readers never see a partial file."""

    def exists(self, rel: str) -> bool:
        return self.read(rel) is not None


class LocalHandoff(Handoff):
    """A directory standing in for the bucket (the GCS simulator in tests)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, rel: str) -> Path:
        return self.root / rel

    def read(self, rel: str) -> bytes | None:
        try:
            return self.path(rel).read_bytes()
        except FileNotFoundError:
            return None

    def write(self, rel: str, data: bytes) -> None:
        dest = self.path(rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".partial-", dir=dest.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def exists(self, rel: str) -> bool:
        return self.path(rel).is_file()

    def __repr__(self) -> str:
        return f"LocalHandoff({str(self.root)!r})"


class GCSHandoff(Handoff):
    """A gs:// prefix through the gcloud CLI: `storage cat` to read, `storage cp -` to write from stdin."""

    def __init__(self, url: str):
        if not url.startswith("gs://"):
            raise ValueError(f"{url}: GCSHandoff needs a gs:// URL")
        self.url = url.rstrip("/")
        self.gcloud = shutil.which("gcloud")
        if self.gcloud is None:
            raise FileNotFoundError("a gs:// handoff needs the gcloud CLI")

    def _url(self, rel: str) -> str:
        return f"{self.url}/{rel.lstrip('/')}"

    def _gcloud(self, args: list[str], **kw) -> subprocess.CompletedProcess:
        """One gcloud call under TIMEOUT_S; a hang becomes RuntimeError like any other storage failure."""
        try:
            return subprocess.run([self.gcloud, "storage", *args], capture_output=True, timeout=TIMEOUT_S, **kw)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"gcloud storage {' '.join(args)}: no answer in {TIMEOUT_S:g} s") from None

    def read(self, rel: str) -> bytes | None:
        url = self._url(rel)
        proc = self._gcloud(["cat", url])
        if proc.returncode == 0:
            return proc.stdout
        err = proc.stderr.decode("utf-8", "replace")
        if "matched no objects" in err or "No URLs matched" in err:
            return None
        raise RuntimeError(f"gcloud storage cat {url}: {err.strip()}")

    def write(self, rel: str, data: bytes) -> None:
        url = self._url(rel)
        proc = self._gcloud(["cp", "-", url], input=data)
        if proc.returncode != 0:
            raise RuntimeError(f"gcloud storage cp - {url}: {proc.stderr.decode('utf-8', 'replace').strip()}")

    def exists(self, rel: str) -> bool:
        url = self._url(rel)
        proc = self._gcloud(["ls", url], text=True)
        if proc.returncode == 0:
            return bool(proc.stdout.strip())
        if "matched no objects" in proc.stderr or "No URLs matched" in proc.stderr:
            return False
        raise RuntimeError(f"gcloud storage ls {url}: {proc.stderr.strip()}")

    def __repr__(self) -> str:
        return f"GCSHandoff({self.url!r})"


def open_handoff(location: str | Path) -> Handoff:
    """GCSHandoff for a gs:// URL, LocalHandoff for a directory."""
    return GCSHandoff(str(location)) if str(location).startswith("gs://") else LocalHandoff(location)


def dumps(obj: dict) -> bytes:
    """The Stage B JSON convention: indent 1, a trailing newline, UTF-8."""
    return (json.dumps(obj, indent=1) + "\n").encode("utf-8")


def read_json(h: Handoff, rel: str) -> dict | None:
    data = h.read(rel)
    return None if data is None else json.loads(data.decode("utf-8"))


def write_json(h: Handoff, rel: str, obj: dict) -> None:
    h.write(rel, dumps(obj))
