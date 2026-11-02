"""Per-layer Beam weight loading from a local directory or GCS, dequantized to BF16.

A WeightSource parses model.safetensors.index.json once and each shard header once, on first use.
A load plans the byte spans it needs, coalesces neighbouring spans per shard, reads them as parallel
chunks into one host buffer, then moves each tensor to the device and dequantizes FP8 weights there.
stream() prefetches into two host buffers, sized for the largest requested layer rounded up to
STREAM_GRANULE, page-locked when the target is CUDA, and kept by the source for later streams.

Keys are checkpoint names relative to the layer ("self_attn.q_proj.weight",
"mlp.experts.17.down_proj.weight", "mlp.gate.e_score_correction_bias") or, for globals, relative to
"model." ("embed_tokens.weight", "norm.weight", "lm_head.weight"). Each "x.weight_scale" is consumed
by its "x.weight" and never returned.

Bench a source from the command line: python -m ouroboros.harness.weights LOCATION LAYER [LAYER ...]
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit

import numpy as np
import torch

from ouroboros.harness import fp8

INDEX_FILE = "model.safetensors.index.json"
SCALE_SUFFIX = "_scale"
# transformers keeps the router in FP32 (BeamPreTrainedModel._keep_in_fp32_modules_strict).
KEEP_FP32 = ("mlp.gate.weight", "mlp.gate.e_score_correction_bias")

_LAYER = re.compile(r"model\.layers\.(\d+)\.")
_DTYPES = {"F8_E4M3": torch.float8_e4m3fn, "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
_ALIGN = 64
STREAM_GRANULE = 64 << 20  # stream() buffer sizes are multiples of this

Keys =dict[str, tuple[str, str | None]]  # key -> (checkpoint name, its weight_scale name or None)


@dataclass(frozen=True)
class TensorInfo:
    shard: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    start: int  # absolute byte offset in the shard file
    nbytes: int
    data_start: int  # where the data section begins; offsets relative to it are aligned


@dataclass(frozen=True)
class _Read:
    shard: str
    start: int  # file offset
    offset: int  # host buffer offset
    nbytes: int


def parse_header(shard: str, read: Callable[[int, int], bytes]) -> dict[str, TensorInfo]:
    """Parse a safetensors header with two range reads: 8-byte little-endian length, then JSON."""
    (n,) = struct.unpack("<Q", read(0, 8))
    header = json.loads(read(8, n))
    base = 8 + n
    return {
        name: TensorInfo(shard, _DTYPES[m["dtype"]], tuple(m["shape"]), base + m["data_offsets"][0],
                         m["data_offsets"][1] - m["data_offsets"][0], base)
        for name, m in header.items() if name != "__metadata__"
    }


def claim(names: Iterable[str], prefix: str) -> Keys:
    """Pair each weight with its weight_scale and strip `prefix`. Every name is claimed exactly once."""
    names = set(names)
    scales = {n for n in names if n.endswith(".weight" + SCALE_SUFFIX)}
    keys = {}
    for name in sorted(names - scales):
        scale = name + SCALE_SUFFIX
        keys[name.removeprefix(prefix)] = (name, scale if scale in scales else None)
    orphans = scales - {s for _, s in keys.values()}
    if orphans:
        raise KeyError(f"weight_scale without a weight: {sorted(orphans)[:5]}")
    return keys


def names_of(keys: Keys) -> list[str]:
    return [n for pair in keys.values() for n in pair if n is not None]


def _plan(infos: dict[str, TensorInfo], merge_gap: int, chunk_bytes: int) -> tuple[list[_Read], dict[str, int], int]:
    """Coalesce tensors into runs per shard, lay the runs out in one buffer, split runs into chunks."""
    by_shard: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for name, info in infos.items():
        by_shard[info.shard].append((info.start, name))
    reads, where, size = [], {}, 0
    for shard, items in sorted(by_shard.items()):
        runs: list[list] = []  # [start, end, names]
        for start, name in sorted(items):
            end = start + infos[name].nbytes
            if runs and start - runs[-1][1] <= merge_gap:
                runs[-1][1] = max(runs[-1][1], end)
                runs[-1][2].append(name)
            else:
                runs.append([start, end, [name]])
        for start, end, run_names in runs:
            # Keep buffer offsets congruent to data-section offsets, which writers align per dtype.
            # Absolute file offsets are not enough: some writers leave the header unpadded.
            size += (start - infos[run_names[0]].data_start - size) % _ALIGN
            for name in run_names:
                where[name] = size + infos[name].start - start
            for s in range(start, end, chunk_bytes):
                reads.append(_Read(shard, s, size + s - start, min(chunk_bytes, end - s)))
            size += end - start
    return reads, where, size


def host_buffer(nbytes: int, pin: bool = False) -> torch.Tensor:
    """A 64-byte aligned uint8 host buffer, page-locked at its exact size when `pin` is set and CUDA is available.

    torch's pinned allocator rounds every block up to a power of two, so a 10.2 GB MoE layer would lock
    16 GiB. Registering a plain allocation locks only what is used. The registration ends when the
    last view of the buffer is freed.
    """
    if not (pin and nbytes and torch.cuda.is_available()):
        return torch.empty(nbytes, dtype=torch.uint8)
    owner = np.empty(nbytes + _ALIGN, dtype=np.uint8)
    ptr = owner.ctypes.data
    cudart = torch.cuda.cudart()
    err = int(cudart.cudaHostRegister(ptr, owner.nbytes, 0))
    if err:
        raise RuntimeError(f"cudaHostRegister of {owner.nbytes} bytes failed with CUDA error {err}")
    weakref.finalize(owner, cudart.cudaHostUnregister, ptr).atexit = False
    skip = -ptr % _ALIGN
    return torch.frombuffer(owner, dtype=torch.uint8)[skip:skip + nbytes]


def _keep_fp32(name: str) -> bool:
    return name.endswith(KEEP_FP32)


def materialize(raw: dict[str, torch.Tensor], keys: Keys, device: torch.device,
                dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    """Move raw host tensors to `device`; FP8 weights are dequantized, the router stays FP32, the rest becomes `dtype`.

    The results never alias `raw`, so its host buffer can be reused once this returns.
    """
    non_blocking = device.type == "cuda"
    out = {}
    for key, (name, scale) in keys.items():
        t = raw[name]
        if scale is not None:
            out[key] = fp8.dequantize(t.to(device, non_blocking=non_blocking),
                                      raw[scale].to(device, non_blocking=non_blocking), dtype)
        elif t.dtype == fp8.FP8_DTYPE:
            raise ValueError(f"{name} is FP8 but has no weight_scale")
        else:
            target = torch.float32 if _keep_fp32(name) else dtype
            out[key] = t.to(device=device, dtype=target, non_blocking=non_blocking, copy=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return out


class WeightSource(ABC):
    """A sharded safetensors checkpoint read through byte ranges."""

    def __init__(self, index: dict, *, workers: int, chunk_bytes: int, merge_gap: int):
        self.weight_map: dict[str, str] = index["weight_map"]
        self.chunk_bytes = chunk_bytes
        self.merge_gap = merge_gap
        self._headers: dict[str, dict[str, TensorInfo]] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(workers, thread_name_prefix="weights")
        self._buffers: list[torch.Tensor | None] = [None, None]  # stream() prefetch buffers, kept between streams
        layers: dict[int, list[str]] = defaultdict(list)
        self._global_names: list[str] = []
        for name in self.weight_map:
            m = _LAYER.match(name)
            (layers[int(m.group(1))] if m else self._global_names).append(name)
        self.num_layers = max(layers) + 1 if layers else 0
        if sorted(layers) != list(range(self.num_layers)):
            raise ValueError("layer indices in the index are not contiguous from 0")
        self._layer_names = dict(layers)

    @abstractmethod
    def read_into(self, shard: str, start: int, out: memoryview) -> None:
        """Fill `out` with the shard's bytes from `start`, or raise."""

    def close(self) -> None:
        self._pool.shutdown()
        self._buffers = [None, None]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def layer_keys(self, i: int) -> Keys:
        return claim(self._layer_names[i], f"model.layers.{i}.")

    def global_keys(self) -> Keys:
        return claim(self._global_names, "model.")

    def load_layer(self, i: int, device: str | torch.device = "cpu",
                   dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
        return self._load(self.layer_keys(i), torch.device(device), dtype)

    def load_globals(self, device: str | torch.device = "cpu",
                     dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
        return self._load(self.global_keys(), torch.device(device), dtype)

    def load_tensors(self, names: Iterable[str], device: str | torch.device = "cpu",
                     dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
        """Load full checkpoint names; a weight's weight_scale is fetched with it."""
        names = set(names)
        names |= {n + SCALE_SUFFIX for n in names if n + SCALE_SUFFIX in self.weight_map}
        return self._load(claim(names, ""), torch.device(device), dtype)

    def stream(self, layers: Sequence[int], device: str | torch.device = "cpu",
               dtype: torch.dtype = torch.bfloat16) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
        """Yield (i, tensors) in order, fetching layer i+1 to host while the caller computes layer i.

        At most two layers are in flight, in two host buffers sized for the largest requested layer
        (rounded up to STREAM_GRANULE) and kept for the next stream (page-locked for a CUDA target).
        Drop the previous layer's tensors before asking for the next one to hold a single layer on the device.
        """
        device = torch.device(device)
        pin = device.type == "cuda"
        keys = [self.layer_keys(i) for i in layers]
        names = [names_of(k) for k in keys]
        self.headers({self.weight_map[n] for ns in names for n in ns})  # one parallel pass over the shards
        # Real MoE plans span 2.9 MB around 10.18 GB. Sizing for the largest layer, rounded up to the
        # granule, gives every MoE layer one buffer size, so no later stream re-locks 10 GB.
        size = -(-max(map(self.plan_bytes, names), default=0) // STREAM_GRANULE) * STREAM_GRANULE
        with self._lock:  # a concurrent stream gets fresh buffers
            buffers, self._buffers = self._buffers, [None, None]

        def fetch(k: int) -> dict[str, torch.Tensor]:
            j = k % 2
            if buffers[j] is None or buffers[j].numel() < size or (pin and not buffers[j].is_pinned()):
                buffers[j] = None  # free the old buffer before locking its replacement
                buffers[j] = host_buffer(size, pin)
            return self.fetch(names[k], buffers[j], pin)[0]

        try:
            with ThreadPoolExecutor(1, thread_name_prefix="prefetch") as ahead:
                pending = ahead.submit(fetch, 0) if layers else None
                for k, i in enumerate(layers):
                    raw = pending.result()
                    if k + 1 < len(layers):
                        pending = ahead.submit(fetch, k + 1)
                    yield i, materialize(raw, keys[k], device, dtype)
        except GeneratorExit:  # the caller stopped early; the prefetcher has joined, so both buffers are idle
            self._buffers = buffers
            raise
        # Kept only after a clean pass: after an error a device copy may still be reading a buffer.
        self._buffers = buffers

    def headers(self, shards: Iterable[str]) -> dict[str, dict[str, TensorInfo]]:
        """Shard headers, parsed in parallel on first use and cached."""
        shards = set(shards)
        with self._lock:
            missing = sorted(shards - self._headers.keys())
        parsed = list(self._pool.map(self._parse_header, missing))
        with self._lock:
            self._headers.update(zip(missing, parsed))
            return {s: self._headers[s] for s in shards}

    def infos(self, names: Iterable[str]) -> dict[str, TensorInfo]:
        names = list(names)
        headers = self.headers(self.weight_map[n] for n in names)
        return {n: headers[self.weight_map[n]][n] for n in names}

    def plan_bytes(self, names: Iterable[str]) -> int:
        """Size of the host buffer that fetch(names) fills."""
        return _plan(self.infos(names), self.merge_gap, self.chunk_bytes)[2]

    def fetch(self, names: Iterable[str], buffer: torch.Tensor | None = None,
              pin: bool = False) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Read raw tensors into a host buffer, reusing `buffer` when it is large enough.

        Returns typed views into the buffer (FP8 stays FP8) and the buffer itself. No read is still
        writing into the buffer once this returns or raises.
        """
        infos = self.infos(names)
        reads, where, size = _plan(infos, self.merge_gap, self.chunk_bytes)
        if buffer is None or buffer.numel() < size:
            buffer = host_buffer(size, pin)
        view = memoryview(buffer.numpy())
        futures = [self._pool.submit(self.read_into, r.shard, r.start, view[r.offset:r.offset + r.nbytes])
                   for r in reads]
        wait(futures, return_when=FIRST_EXCEPTION)
        wait([f for f in futures if not f.cancel()])  # after a failure, drop queued reads and let running ones end
        for f in futures:
            if not f.cancelled():
                f.result()
        raw = {n: buffer[where[n]:where[n] + i.nbytes].view(i.dtype).view(i.shape) for n, i in infos.items()}
        return raw, buffer

    def _load(self, keys: Keys, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        # One-off loads stay pageable: locking a fresh buffer costs about as much as the copy it speeds up.
        raw, _ = self.fetch(names_of(keys))
        return materialize(raw, keys, device, dtype)

    def _parse_header(self, shard: str) -> dict[str, TensorInfo]:
        def read(start: int, n: int) -> bytes:
            buf = bytearray(n)
            self.read_into(shard, start, memoryview(buf))
            return bytes(buf)
        return parse_header(shard, read)


class LocalDir(WeightSource):
    """A checkpoint directory on local disk."""

    def __init__(self, path: str | Path, *, workers: int = 8, chunk_bytes: int = 64 << 20, merge_gap: int = 1 << 20):
        self.path = Path(path)
        index = json.loads((self.path / INDEX_FILE).read_text(encoding="utf-8"))
        super().__init__(index, workers=workers, chunk_bytes=chunk_bytes, merge_gap=merge_gap)

    def read_into(self, shard: str, start: int, out: memoryview) -> None:
        _read_file(self.path / shard, start, out)


def _read_file(path: Path, start: int, out: memoryview) -> None:
    with open(path, "rb", buffering=0) as f:
        f.seek(start)
        n = 0
        while n < len(out):
            k = f.readinto(out[n:])
            if not k:
                raise EOFError(f"{path.name}: short read at byte {start + n}")
            n += k


class GCSError(Exception):
    """A GCS request failed in a way a retry cannot fix."""


_RETRY_STATUS = {408, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class Retry:
    """Truncated exponential backoff with full jitter, given up when the next sleep would pass `deadline`.

    The defaults follow google-cloud-storage's policy (1 s first ceiling, doubling, 120 s deadline)
    with the 32 s cap that the GCS retry guidance suggests.
    """

    first: float = 1.0
    cap: float = 32.0
    deadline: float = 120.0

    def delays(self) -> Iterator[float]:
        ceiling = min(self.first, self.cap)
        while True:
            yield random.uniform(0.0, ceiling)
            ceiling = min(2 * ceiling, self.cap)


def google_token() -> Callable[[], str]:
    """Application-default credentials; on a GCE VM they come from the metadata server."""
    import google.auth
    from google.auth.transport.requests import Request

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/devstorage.read_only"])
    lock = threading.Lock()

    def token() -> str:
        with lock:
            if not creds.valid:
                creds.refresh(Request())
            return creds.token

    return token


class GCS(WeightSource):
    """A checkpoint under gs://bucket/prefix, read with ranged GETs on the JSON API.

    Each GET runs in its own curl process, which writes the range to a file in tmpfs; the worker
    thread then reads that file into the host buffer in one call. Decrypting TLS in Python threads
    takes the GIL back for every 16 KB record and stalls near 1 GB/s per process (measured:
    in-memory decryption with 8 threads ran at 0.39 GB/s, with 4 processes at 3.8 GB/s). curl
    decrypts outside the interpreter, so the threads here mostly wait.
    """

    def __init__(self, url: str, *, token: Callable[[], str] | None = None,
                 endpoint: str = "https://storage.googleapis.com", workers: int = 32,
                 chunk_bytes: int = 32 << 20, merge_gap: int = 1 << 20, retry: Retry = Retry(),
                 timeout: float = 60.0, curl: str = "curl"):
        parts = urlsplit(url)
        if parts.scheme != "gs" or not parts.netloc:
            raise ValueError(f"expected gs://bucket/prefix, got {url!r}")
        self.bucket = parts.netloc
        self.prefix = parts.path.strip("/") + "/" if parts.path.strip("/") else ""
        self.endpoint = endpoint.rstrip("/")
        self.curl = shutil.which(curl)
        if self.curl is None:
            raise FileNotFoundError(f"curl not found: {curl!r}")
        self._token = token or google_token()
        self.retry = retry
        self.timeout = timeout
        self._tmp = Path(tempfile.mkdtemp(prefix="ouroboros-gcs-", dir="/dev/shm" if Path("/dev/shm").is_dir() else None))
        self._ids = itertools.count()
        try:
            index = json.loads(self._get(INDEX_FILE, 0, None))
            super().__init__(index, workers=workers, chunk_bytes=chunk_bytes, merge_gap=merge_gap)
        except BaseException:
            shutil.rmtree(self._tmp, ignore_errors=True)
            raise

    def close(self) -> None:
        super().close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def read_into(self, shard: str, start: int, out: memoryview) -> None:
        self._get(shard, start, out)

    def _get(self, obj: str, start: int, out: memoryview | None) -> bytes | None:
        """GET the whole object (out=None) or exactly len(out) bytes from `start` into `out`."""
        url = f"{self.endpoint}/storage/v1/b/{self.bucket}/o/{quote(self.prefix + obj, safe='')}?alt=media"
        where = f"gs://{self.bucket}/{self.prefix}{obj}"
        span = None if out is None else (start, start + len(out) - 1)
        want = 200 if out is None else 206
        began = time.monotonic()
        for attempt, delay in enumerate(self.retry.delays(), 1):
            path = self._tmp / str(next(self._ids))
            try:
                code, status, problem = self._curl(url, span, path)
                if code == 0 and status == want:
                    if out is None:
                        return path.read_bytes()
                    size = path.stat().st_size
                    if size == len(out):
                        _read_file(path, 0, out)
                        return None
                    problem = f"range of {len(out)} bytes returned {size}"
                elif status not in _RETRY_STATUS and status not in (0, want):
                    raise GCSError(f"{where}: HTTP {status} {problem}".rstrip())
            finally:
                path.unlink(missing_ok=True)
            elapsed = time.monotonic() - began
            if elapsed + delay > self.retry.deadline:
                raise GCSError(f"{where}: gave up after {attempt} attempts in {elapsed:.1f} s: {problem}")
            time.sleep(delay)

    def _curl(self, url: str, span: tuple[int, int] | None, path: Path) -> tuple[int, int, str]:
        """One GET into `path`. Returns (curl exit code, HTTP status or 0, curl's message).

        The token goes in through stdin so it never appears in a process listing.
        """
        t = f"{self.timeout:g}"
        cmd = [self.curl, "--silent", "--show-error", "--fail", "--http1.1", "--header", "@-",
               "--write-out", "%{stderr}%{http_code}", "--output", str(path),
               "--connect-timeout", t, "--speed-time", t, "--speed-limit", "1"]
        if span is not None:
            cmd += ["--range", f"{span[0]}-{span[1]}"]
        proc = subprocess.run([*cmd, url], input=f"Authorization: Bearer {self._token()}\n".encode(),
                              capture_output=True)
        err = proc.stderr.decode(errors="replace").strip()
        m = re.search(r"(\d{3})$", err)
        message = (err[:m.start()] if m else err).strip()
        if proc.returncode and not message:  # curl died before it could say why, e.g. it failed to start
            message = f"curl exited with status {proc.returncode}"
        return proc.returncode, int(m.group(1)) if m else 0, message


def open_source(location: str | Path, **kwargs) -> WeightSource:
    """GCS for a gs:// URL, LocalDir otherwise."""
    return GCS(str(location), **kwargs) if str(location).startswith("gs://") else LocalDir(location, **kwargs)


def to_transformers(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fuse a layer's per-expert tensors into transformers' BeamExperts layout.

    gate_up_proj is (E, 2I, H) with gate rows first; down_proj is (E, H, I). Other keys pass through.
    """
    out = {k: v for k, v in tensors.items() if not k.startswith("mlp.experts.")}
    n = sum(1 for k in tensors if k.startswith("mlp.experts.") and k.endswith(".down_proj.weight"))
    if n:
        gate, down = tensors["mlp.experts.0.gate_proj.weight"], tensors["mlp.experts.0.down_proj.weight"]
        gate_up = gate.new_empty(n, 2 * gate.shape[0], gate.shape[1])
        out["mlp.experts.down_proj"] = down.new_empty(n, *down.shape)
        for e in range(n):
            gate_up[e, : gate.shape[0]] = tensors[f"mlp.experts.{e}.gate_proj.weight"]
            gate_up[e, gate.shape[0]:] = tensors[f"mlp.experts.{e}.up_proj.weight"]
            out["mlp.experts.down_proj"][e] = tensors[f"mlp.experts.{e}.down_proj.weight"]
        out["mlp.experts.gate_up_proj"] = gate_up
    return out


def _bench(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Time host fetches of whole layers (no dequantization).")
    p.add_argument("location", help="checkpoint directory or gs://bucket/prefix")
    p.add_argument("layers", type=int, nargs="+")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--chunk-mb", type=int, default=32)
    args = p.parse_args(argv)
    pin = torch.cuda.is_available()
    with open_source(args.location, workers=args.workers, chunk_bytes=args.chunk_mb << 20) as src:
        layers = [(i, names_of(src.layer_keys(i))) for i in args.layers]
        size = max(src.plan_bytes(names) for _, names in layers)  # also parses headers outside the timed region
        t = time.perf_counter()
        buffer = host_buffer(size, pin)
        print(f"host buffer: {size / 1e9:.2f} GB, pinned={buffer.is_pinned()}, "
              f"made in {time.perf_counter() - t:.2f} s", flush=True)
        for i, names in layers:
            t = time.perf_counter()
            raw, buffer = src.fetch(names, buffer, pin)
            dt = time.perf_counter() - t
            gb = sum(v.nbytes for v in raw.values()) / 1e9
            print(f"layer {i}: {gb:.2f} GB in {dt:.2f} s = {gb / dt:.2f} GB/s", flush=True)


if __name__ == "__main__":
    _bench()
