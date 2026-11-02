"""GCS source against a fake JSON API that serves a local checkpoint through range reads."""

import itertools
import random
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
import torch

from ouroboros.harness import tinyckpt
from ouroboros.harness.weights import GCS, GCSError, Retry, names_of, open_source

BUCKET = "fake-bucket"
PREFIX = "weights/beam-fp8/7449ff0f/"
TOKEN = "t0k3n"


class FakeGCS(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, root: Path):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.root = root
        self.lock = threading.Lock()
        self.ranges: list[tuple[str, int, int]] = []
        self.fail_next = 0  # answer 503
        self.truncate_next = 0  # promise the whole range, send half, hang up

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _reply(self, status: int, body: bytes = b"", headers: dict | None = None):
        self.send_response(status)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        server: FakeGCS = self.server
        with server.lock:
            if server.fail_next:
                server.fail_next -= 1
                return self._reply(503, b"try again")
        url = urlsplit(self.path)
        m = re.fullmatch(rf"/storage/v1/b/{BUCKET}/o/([^/]+)", url.path)
        if not m or url.query != "alt=media":
            return self._reply(400)
        if self.headers.get("Authorization") != f"Bearer {TOKEN}":
            return self._reply(401)
        name = unquote(m.group(1))
        path = server.root / name.removeprefix(PREFIX)
        if not name.startswith(PREFIX) or not path.is_file():
            return self._reply(404)
        rng = self.headers.get("Range")
        if rng is None:
            return self._reply(200, path.read_bytes())
        a, b = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", rng).groups())
        with server.lock:
            server.ranges.append((name, a, b))
            truncate = server.truncate_next > 0
            server.truncate_next -= truncate
        with open(path, "rb") as f:
            f.seek(a)
            data = f.read(b + 1 - a)
        if truncate:
            self.send_response(206)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data[: len(data) // 2])
            self.close_connection = True
            return
        self._reply(206, data, {"Content-Range": f"bytes {a}-{b}/{path.stat().st_size}"})


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    return tinyckpt.build_checkpoint(tmp_path_factory.mktemp("tiny"))


@pytest.fixture
def fake(ckpt):
    server = FakeGCS(ckpt[0])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


FAST = Retry(first=0.01, cap=0.05, deadline=10.0)


def _gcs(fake, **kwargs) -> GCS:
    kwargs = {"token": lambda: TOKEN, "endpoint": fake.endpoint, "workers": 4, "retry": FAST, **kwargs}
    return open_source(f"gs://{BUCKET}/{PREFIX}", **kwargs)


def _layer_reference(reference, i):
    p = f"model.layers.{i}."
    return {k[len(p):]: v for k, v in reference.items() if k.startswith(p)}


def _assert_same(got: dict, want: dict):
    assert got.keys() == want.keys()
    for k, v in want.items():
        assert got[k].dtype == v.dtype and torch.equal(got[k], v), k


def test_layers_and_globals_match_reference(ckpt, fake):
    _, reference = ckpt
    with _gcs(fake, chunk_bytes=8192, merge_gap=0) as src:
        assert isinstance(src, GCS) and src.num_layers == 6
        src.headers(set(src.weight_map.values()))
        fake.ranges.clear()
        for i in range(src.num_layers):
            _assert_same(src.load_layer(i), _layer_reference(reference, i))
        # Only the needed byte spans travel, in chunks no larger than chunk_bytes.
        layer_names = [n for i in range(src.num_layers) for n in names_of(src.layer_keys(i))]
        assert sum(b - a + 1 for _, a, b in fake.ranges) == sum(i.nbytes for i in src.infos(layer_names).values())
        assert max(b - a + 1 for _, a, b in fake.ranges) <= 8192
        assert not any(src._tmp.iterdir())  # each range file is gone once read
    assert not src._tmp.exists()
    with _gcs(fake) as src:
        _assert_same(src.load_globals(), {
            "embed_tokens.weight": reference["model.embed_tokens.weight"],
            "norm.weight": reference["model.norm.weight"],
            "lm_head.weight": reference["lm_head.weight"],
        })


def test_stream_over_gcs(ckpt, fake):
    _, reference = ckpt
    with _gcs(fake) as src:
        for i, tensors in src.stream([2, 3, 4]):
            _assert_same(tensors, _layer_reference(reference, i))


def test_retries_transient_errors(ckpt, fake):
    _, reference = ckpt
    with _gcs(fake) as src:
        fake.fail_next = 3
        _assert_same(src.load_layer(5), _layer_reference(reference, 5))
        assert fake.fail_next == 0
        fake.truncate_next = 3
        _assert_same(src.load_layer(4), _layer_reference(reference, 4))
        assert fake.truncate_next == 0


def test_permanent_errors_raise(fake):
    with pytest.raises(GCSError, match="HTTP 401"):
        _gcs(fake, token=lambda: "wrong")
    with pytest.raises(GCSError, match="HTTP 404"):
        open_source(f"gs://{BUCKET}/elsewhere", token=lambda: TOKEN, endpoint=fake.endpoint, retry=FAST)


def test_transient_errors_retry_until_the_deadline(fake):
    fake.fail_next = 10**6  # a 503 storm that outlasts the deadline
    began = time.monotonic()
    with pytest.raises(GCSError, match=r"gave up after \d+ attempts in [\d.]+ s: .*503"):
        _gcs(fake, retry=Retry(first=0.05, cap=0.1, deadline=1.0))
    assert 10**6 - fake.fail_next >= 2  # it retried
    assert time.monotonic() - began < 1.0 + 5.0  # the deadline, plus one request in flight


def test_backoff_doubles_to_its_cap_with_full_jitter(monkeypatch):
    # Defaults follow google-cloud-storage: first ceiling 1 s, doubling, 120 s deadline.
    assert (Retry().first, Retry().deadline) == (1.0, 120.0)
    ceilings = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 32.0, 32.0]
    delays = list(itertools.islice(Retry().delays(), len(ceilings)))
    assert all(0.0 <= d <= c for d, c in zip(delays, ceilings))
    monkeypatch.setattr(random, "uniform", lambda a, b: b)
    assert list(itertools.islice(Retry().delays(), len(ceilings))) == ceilings


def test_token_stays_off_the_command_line(fake, monkeypatch):
    commands = []
    run = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: commands.append(cmd) or run(cmd, **kw))
    with _gcs(fake) as src:
        src.load_layer(0)
    assert len(commands) > 2 and not any(TOKEN in arg for cmd in commands for arg in cmd)


def test_rejects_non_gcs_urls_and_missing_curl(fake):
    with pytest.raises(ValueError):
        GCS("s3://bucket/x", token=lambda: TOKEN)
    with pytest.raises(FileNotFoundError):
        _gcs(fake, curl="no-such-curl")
