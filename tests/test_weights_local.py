import gc
import json
import struct
import threading
import time
import weakref

import pytest
import torch

from ouroboros.harness import fp8, tinyckpt, weights
from ouroboros.harness.weights import INDEX_FILE, LocalDir, names_of, open_source, to_transformers


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    return tinyckpt.build_checkpoint(tmp_path_factory.mktemp("tiny"))


def _layer_reference(reference, i):
    p = f"model.layers.{i}."
    return {k[len(p):]: v for k, v in reference.items() if k.startswith(p)}


def _assert_same(got: dict, want: dict):
    assert got.keys() == want.keys()
    for k, v in want.items():
        assert got[k].dtype == v.dtype and got[k].shape == v.shape and torch.equal(got[k], v), k


def test_load_layer_and_globals(ckpt):
    path, reference = ckpt
    with LocalDir(path) as src:
        assert src.num_layers == 6
        for i in range(src.num_layers):
            got = src.load_layer(i)
            _assert_same(got, _layer_reference(reference, i))
        moe = src.load_layer(2)
        assert moe["mlp.gate.weight"].dtype == torch.float32
        assert moe["mlp.gate.e_score_correction_bias"].dtype == torch.float32
        assert moe["mlp.experts.15.down_proj.weight"].dtype == torch.bfloat16
        _assert_same(src.load_globals(), {
            "embed_tokens.weight": reference["model.embed_tokens.weight"],
            "norm.weight": reference["model.norm.weight"],
            "lm_head.weight": reference["lm_head.weight"],
        })


def test_load_tensors_pairs_scales(ckpt):
    path, reference = ckpt
    names = ["model.layers.3.mlp.experts.7.up_proj.weight", "model.layers.1.self_attn.gate_bias"]
    with open_source(path) as src:
        _assert_same(src.load_tensors(names), {n: reference[n] for n in names})
        raw, buffer = src.fetch([names[0] + "_scale"], pin=True)  # pinning falls back to pageable without CUDA
        assert raw[names[0] + "_scale"].shape == (32, 1) and not buffer.is_pinned()


def test_bench_cli(ckpt, capsys):
    weights._bench([str(ckpt[0]), "2", "5", "--workers", "2"])
    lines = capsys.readouterr().out.splitlines()
    # The buffer is made once, before and outside the per-layer timings.
    assert [line.split(":")[0] for line in lines] == ["host buffer", "layer 2", "layer 5"]
    assert "pinned=False" in lines[0] and all("GB/s" in line for line in lines[1:])


class CountingDir(LocalDir):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reads: list[tuple[str, int, int]] = []
        self.count_lock = threading.Lock()

    def read_into(self, shard, start, out):
        with self.count_lock:
            self.reads.append((shard, start, len(out)))
        super().read_into(shard, start, out)


def test_small_chunks_and_no_merging(ckpt):
    path, reference = ckpt
    with CountingDir(path, workers=4, chunk_bytes=1000, merge_gap=0) as src:
        src.headers(set(src.weight_map.values()))
        src.reads.clear()
        got = src.load_layer(4)
        _assert_same(got, _layer_reference(reference, 4))
        assert max(n for _, _, n in src.reads) <= 1000
        infos = src.infos(names_of(src.layer_keys(4)))
        assert sum(n for _, _, n in src.reads) == sum(i.nbytes for i in infos.values())
        # The layer straddles shards: the router shard plus at least one other.
        assert len({s for s, _, _ in src.reads}) >= 2


def test_headers_parsed_once(ckpt):
    path, _ = ckpt
    with CountingDir(path) as src:
        src.load_layer(2)
        first = len(src.reads)
        src.reads.clear()
        src.load_layer(2)
        assert first - len(src.reads) == 2 * len({src.weight_map[n] for n in names_of(src.layer_keys(2))})


def test_unpadded_header(tmp_path):
    """The BF16 Beam repo leaves headers unpadded, so absolute tensor offsets are odd."""
    a = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    b = torch.tensor([1.5, -2.0], dtype=torch.float32)
    header = {"a": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [8, 20]},
              "b": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (1 - len(text) % 2)  # odd header length
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(text)) + text + b.numpy().tobytes() + a.view(torch.int16).numpy().tobytes())
    (tmp_path / INDEX_FILE).write_text(json.dumps({"weight_map": {"a": "model.safetensors", "b": "model.safetensors"}}))
    with LocalDir(tmp_path) as src:
        got = src.load_tensors(["a", "b"], dtype=torch.bfloat16)
    assert torch.equal(got["a"], a) and torch.equal(got["b"], b.to(torch.bfloat16))


class EventDir(LocalDir):
    """Records which layer each fetch serves, to check the prefetch depth."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started: list[int] = []
        self.layer_of = {n: i for i in range(self.num_layers) for n in names_of(self.layer_keys(i))}

    def fetch(self, names, buffer=None, pin=False):
        names = list(names)
        self.started.append(self.layer_of[names[0]])
        return super().fetch(names, buffer, pin)


def test_stream_prefetches_one_layer_ahead(ckpt):
    path, reference = ckpt
    order = [0, 1, 2, 3, 4, 5]
    with EventDir(path) as src:
        seen = []
        for k, (i, tensors) in enumerate(src.stream(order)):
            assert src.started[-1] <= order[min(k + 1, len(order) - 1)]  # never two layers ahead
            _assert_same(tensors, _layer_reference(reference, i))
            seen.append(i)
        assert seen == order and src.started == order


def test_stream_results_survive_buffer_reuse(ckpt):
    path, reference = ckpt
    with LocalDir(path) as src:
        kept = dict(src.stream([2, 3, 4, 5]))
    for i, tensors in kept.items():
        _assert_same(tensors, _layer_reference(reference, i))


def test_stream_keeps_its_buffers(ckpt, monkeypatch):
    path, reference = ckpt
    made = []
    real = weights.host_buffer
    monkeypatch.setattr(weights, "host_buffer", lambda n, pin=False: made.append(n) or real(n, pin))
    with LocalDir(path) as src:
        sizes = [src.plan_bytes(names_of(src.layer_keys(i))) for i in range(6)]
        list(src.stream([sizes.index(min(sizes))]))
        list(src.stream(range(6)))
        # Layers of different sizes round to one granule, so a stream with a larger layer reuses the buffer.
        assert max(sizes) < weights.STREAM_GRANULE and made == [weights.STREAM_GRANULE] * 2
        for i, tensors in src.stream(range(6)):
            _assert_same(tensors, _layer_reference(reference, i))
        assert len(made) == 2  # later passes allocate nothing
        assert all(b is not None for b in src._buffers)
    assert all(b is None for b in src._buffers)


def test_stream_frees_a_buffer_before_replacing_it(ckpt, monkeypatch):
    path, _ = ckpt
    monkeypatch.setattr(weights, "STREAM_GRANULE", 1)  # exact sizes, so a larger layer forces a new buffer
    with LocalDir(path) as src:
        sizes = [src.plan_bytes(names_of(src.layer_keys(i))) for i in range(6)]
        small = min(range(6), key=sizes.__getitem__)
        list(src.stream([small]))
        old = weakref.ref(src._buffers[0])
        alive_at_alloc = []
        real = weights.host_buffer
        monkeypatch.setattr(weights, "host_buffer",
                            lambda n, pin=False: alive_at_alloc.append(old() is not None) or real(n, pin))
        list(src.stream(range(6)))
        assert alive_at_alloc == [False, False] and old() is None
        assert [b.numel() for b in src._buffers] == [max(sizes)] * 2


class FlakyDir(LocalDir):
    """The first read of an armed fetch fails at once; every other read lands 0.2 s later."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.armed, self.failed, self.started = False, False, 0
        self.flaky_lock = threading.Lock()

    def read_into(self, shard, start, out):
        if self.armed:
            with self.flaky_lock:
                first, self.failed = not self.failed, True
                self.started += 1
            if first:
                raise OSError("simulated read failure")
            time.sleep(0.2)
        super().read_into(shard, start, out)


def test_failed_fetch_leaves_no_read_running(ckpt):
    path, _ = ckpt
    with FlakyDir(path, workers=4, chunk_bytes=4096, merge_gap=0) as src:
        names = names_of(src.layer_keys(3))
        _, buffer = src.fetch(names)
        n_reads = -(-src.plan_bytes(names) // 4096)
        src.armed = True
        with pytest.raises(OSError, match="simulated"):
            src.fetch(names, buffer)
        buffer.fill_(0xAB)  # the caller reuses the buffer at once
        time.sleep(0.5)
        assert bool((buffer == 0xAB).all())  # nothing wrote into it after fetch raised
        assert src.started < n_reads  # queued reads were cancelled


def test_stream_keeps_buffers_after_early_stop_but_not_after_error(ckpt):
    path, _ = ckpt
    with FlakyDir(path, workers=4) as src:
        for _ in src.stream(range(6)):
            break
        assert src._buffers[0] is not None
        src.armed = True
        with pytest.raises(OSError, match="simulated"):
            list(src.stream(range(6)))
        assert src._buffers == [None, None]


class FakeCudart:
    """Stands in for torch.cuda.cudart() and records the page-locked ranges."""

    def __init__(self):
        self.locked: dict[int, int] = {}

    def cudaHostRegister(self, ptr, size, flags):
        self.locked[ptr] = size
        return 0

    def cudaHostUnregister(self, ptr):
        del self.locked[ptr]
        return 0


@pytest.fixture
def fake_cuda(monkeypatch):
    fake = FakeCudart()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: fake)
    return fake


def test_pinned_buffer_is_exact_and_released(fake_cuda):
    buf = weights.host_buffer(10_000_001, pin=True)
    ((ptr, size),) = fake_cuda.locked.items()
    assert buf.numel() == 10_000_001 and buf.data_ptr() % 64 == 0
    assert size == 10_000_001 + 64 and ptr <= buf.data_ptr() < ptr + 64
    view = buf[128:256].view(torch.float32)
    del buf
    gc.collect()
    assert fake_cuda.locked  # a live view keeps the registration
    view.fill_(1.0)
    del view
    gc.collect()
    assert not fake_cuda.locked
    assert weights.host_buffer(0, pin=True).numel() == 0 and weights.host_buffer(8).numel() == 8
    assert not fake_cuda.locked


def test_pinned_fetch(ckpt, fake_cuda):
    path, reference = ckpt
    name = "model.layers.3.mlp.experts.7.up_proj.weight"
    with LocalDir(path) as src:
        raw, _ = src.fetch([name, name + "_scale"], pin=True)
        assert len(fake_cuda.locked) == 1
        assert torch.equal(fp8.dequantize(raw[name], raw[name + "_scale"]), reference[name])


def test_reference_model_holds_reference(ckpt):
    path, reference = ckpt
    config = tinyckpt.tiny_config()
    model = tinyckpt.reference_model(config, reference)
    state = model.state_dict()
    with LocalDir(path) as src:
        for i in (1, 2):
            fused = to_transformers(src.load_layer(i))
            for k, v in fused.items():
                assert torch.equal(state[f"model.layers.{i}.{k}"], v), k
    ids = torch.tensor([[151643, 872, 1023, 4, 99, 17, 300, 12, 8, 9, 10, 11]])
    with torch.no_grad():
        logits = model(ids).logits
    assert logits.shape == (1, ids.shape[1], config.vocab_size) and torch.isfinite(logits).all()
