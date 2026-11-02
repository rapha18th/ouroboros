"""Opt-in network check of the FP8 convention on real Beam tensors, by HTTP range reads only.

Set OUROBOROS_NET_TESTS=1 to run; add -s to see the error report. Reads under 100 MB from the Hub.
"""

import json
import os
import re
import threading

import pytest
import requests
import torch
from huggingface_hub import get_token

from ouroboros.harness import fp8, tinyckpt
from ouroboros.harness.weights import INDEX_FILE, LocalDir, WeightSource

pytestmark = pytest.mark.skipif(os.environ.get("OUROBOROS_NET_TESTS") != "1", reason="set OUROBOROS_NET_TESTS=1")

FP8_REPO = ("reflection/Beam-1.0-501B-A23B-FP8", "7449ff0f358983375e15ec9624b315de58ce8489")
BF16_REPO = ("reflection/Beam-1.0-501B-A23B", "24595dc986af66da3e40229d46aa5f6d1fd2d0c7")
QUANTIZED = ["model.layers.0.self_attn.k_proj.weight", "model.layers.3.mlp.experts.17.down_proj.weight"]
PLAIN = ["model.layers.0.self_attn.gate_bias", "model.layers.0.input_layernorm.weight",
         "model.layers.3.mlp.gate.weight", "model.layers.3.mlp.gate.e_score_correction_bias"]


class Hub(WeightSource):
    """Hub files through ranged GETs; counts the bytes it reads."""

    def __init__(self, repo: str, revision: str):
        self.base = f"https://huggingface.co/{repo}/resolve/{revision}/"
        token = get_token()
        self.auth = {"Authorization": f"Bearer {token}"} if token else {}
        self.bytes = 0
        self.count_lock = threading.Lock()
        index = requests.get(self.base + INDEX_FILE, headers=self.auth, timeout=120).json()
        super().__init__(index, workers=8, chunk_bytes=16 << 20, merge_gap=0)

    def read_into(self, shard, start, out):
        r = requests.get(self.base + shard, headers={**self.auth, "Range": f"bytes={start}-{start + len(out) - 1}"},
                         timeout=300)
        assert r.status_code == 206 and len(r.content) == len(out), (shard, r.status_code)
        out[:] = r.content
        with self.count_lock:
            self.bytes += len(out)


def _pattern(name: str) -> str:
    return re.sub(r"experts\.\d+\.", "experts.E.", re.sub(r"layers\.\d+\.", "layers.N.", name))


def test_dequantized_fp8_matches_bf16(tmp_path):
    with Hub(*FP8_REPO) as fp8_src, Hub(*BF16_REPO) as bf16_src:
        raw, _ = fp8_src.fetch(QUANTIZED + [n + "_scale" for n in QUANTIZED] + PLAIN)
        ref, _ = bf16_src.fetch(QUANTIZED + PLAIN)
        loaded = fp8_src.load_tensors(QUANTIZED)
        report = {}
        for name in QUANTIZED:
            q, scale, w = raw[name], raw[name + "_scale"], ref[name].float()
            assert q.dtype == torch.float8_e4m3fn and scale.dtype == torch.bfloat16
            assert scale.shape == (q.shape[0], 1) and w.shape == q.shape
            deq = fp8.dequantize(q, scale).float()
            assert torch.equal(loaded[name].float(), deq)
            err = (deq - w).abs()
            nz = w != 0
            normal = w.abs() >= scale.float() * 2**-6
            divided = (q.float() / scale.float() - w).abs()
            requant, requant_scale = fp8.quantize(ref[name])
            same_codes = requant.view(torch.uint8) == q.view(torch.uint8)
            report[name] = {
                "shape": list(q.shape),
                "max_abs_err": err.max().item(),
                "max_rel_err": (err[nz] / w.abs()[nz]).max().item(),
                "median_rel_err": (err[nz] / w.abs()[nz]).median().item(),
                # Excluding FP8 subnormals (|w / scale| < 2^-6), where relative error grows to 1.
                "max_rel_err_fp8_normal_range": (err[normal] / w.abs()[normal]).max().item(),
                "max_err_over_row_absmax": (err / w.abs().amax(1, keepdim=True)).max().item(),
                "exact_fraction": (err == 0).float().mean().item(),
                "divide_max_abs_err": divided.max().item(),
                "scale_equals_bf16_rowmax_over_448": torch.equal(requant_scale, scale),
                "requantized_fp8_match_fraction": same_codes.double().mean().item(),
            }
            bound = w.abs() * (2**-4 + 2**-7) + scale.float() * 2**-9
            assert (err <= bound).all(), name
            assert divided.max() > 100 * err.max()
            # Our quantizer reproduces the published codes and scales bit for bit.
            assert torch.equal(requant_scale, scale) and same_codes.all()
        for name in PLAIN:
            a, b = raw[name], ref[name]
            report[name] = {"fp8_repo": str(a.dtype), "bf16_repo": str(b.dtype),
                            "max_abs_diff": (a.float() - b.float()).abs().max().item()}
            assert torch.equal(a.float(), b.float()), name

        # The tiny checkpoint uses the real dtype and scale layout for every name pattern seen here.
        real = {}
        for header in fp8_src.headers({fp8_src.weight_map[n] for n in QUANTIZED}).values():
            for n, info in header.items():
                real.setdefault(_pattern(n), (info.dtype, info.shape[1:] if n.endswith("_scale") else None))
        ckpt, _ = tinyckpt.build_checkpoint(tmp_path)
        with LocalDir(ckpt) as tiny:
            tiny_infos = tiny.infos(tiny.weight_map)
        tiny_real = {}
        for n, info in tiny_infos.items():
            tiny_real.setdefault(_pattern(n), (info.dtype, info.shape[1:] if n.endswith("_scale") else None))
        assert {p: tiny_real[p] for p in real} == real

        total = fp8_src.bytes + bf16_src.bytes
        report["bytes_read"] = total
        print("\n" + json.dumps(report, indent=2))
        assert total < 200e6
