"""Name coverage: every tensor in the real FP8 index maps to exactly one load_layer or load_globals key."""

import json
import re
from pathlib import Path

import pytest
from transformers.models.beam.modeling_beam import BeamPreTrainedModel

from ouroboros.harness import tinyckpt
from ouroboros.harness.weights import INDEX_FILE, KEEP_FP32, WeightSource, claim, names_of

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "beam"
REAL_INDEX = json.loads((VENDOR / INDEX_FILE).read_text(encoding="utf-8"))
REAL_CONFIG = json.loads((VENDOR / "config.json").read_text(encoding="utf-8"))

ATTENTION = {f"self_attn.{p}.weight" for p in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj")}
UNQUANTIZED = {"self_attn.q_norm.weight", "self_attn.k_norm.weight", "self_attn.gate_bias"} | {
    f"{n}_layernorm.weight" for n in ("input", "post_attention", "pre_feedforward", "post_feedforward")
}


def expected_layer(sparse: bool, num_experts: int) -> tuple[set[str], set[str]]:
    """(quantized keys, unquantized keys) of one decoder layer."""
    if not sparse:
        return ATTENTION | {f"mlp.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")}, UNQUANTIZED
    experts = {f"mlp.experts.{e}.{p}.weight" for e in range(num_experts) for p in ("gate_proj", "up_proj", "down_proj")}
    return ATTENTION | experts, UNQUANTIZED | {"mlp.gate.weight", "mlp.gate.e_score_correction_bias"}


class IndexOnly(WeightSource):
    def read_into(self, shard, start, out):
        raise AssertionError("index-only source")


def test_real_index_coverage():
    src = IndexOnly(REAL_INDEX, workers=1, chunk_bytes=1 << 20, merge_gap=0)
    assert src.num_layers == REAL_CONFIG["num_hidden_layers"] == 52
    claimed = []
    kinds = {"dense": 0, "sparse": 0}
    for i in range(src.num_layers):
        kind = REAL_CONFIG["mlp_layer_types"][i]
        kinds[kind] += 1
        quantized, plain = expected_layer(kind == "sparse", REAL_CONFIG["num_experts"])
        keys = src.layer_keys(i)
        assert {k for k, (_, s) in keys.items() if s} == quantized, i
        assert {k for k, (_, s) in keys.items() if s is None} == plain, i
        claimed += names_of(keys)
    assert kinds == {"dense": 3, "sparse": 49}
    globals_ = src.global_keys()
    assert globals_ == {
        "embed_tokens.weight": ("model.embed_tokens.weight", None),
        "norm.weight": ("model.norm.weight", None),
        "lm_head.weight": ("lm_head.weight", None),
    }
    claimed += names_of(globals_)
    assert len(claimed) == len(set(claimed)) == len(REAL_INDEX["weight_map"])
    assert set(claimed) == set(REAL_INDEX["weight_map"])


def test_orphan_scale_is_unclaimed():
    with pytest.raises(KeyError):
        claim(["a.weight", "b.weight_scale"], "")


def test_keep_fp32_matches_transformers():
    strict = BeamPreTrainedModel._keep_in_fp32_modules_strict
    for name in REAL_INDEX["weight_map"]:
        assert name.endswith(KEEP_FP32) == any(p in name for p in strict), name


def _pattern(name: str) -> str:
    return re.sub(r"experts\.\d+\.", "experts.E.", re.sub(r"layers\.\d+\.", "layers.N.", name))


def _last_shard(weight_map: dict[str, str]) -> tuple[set[str], set[int]]:
    """Name patterns in the last shard, and the layers with a tensor there."""
    last = max(weight_map.values())
    names = [n for n, f in weight_map.items() if f == last]
    return {_pattern(n) for n in names}, {int(n.split(".")[2]) for n in names if n.startswith("model.layers.")}


def test_tiny_checkpoint_uses_real_names(tmp_path):
    ckpt, reference = tinyckpt.build_checkpoint(tmp_path)
    tiny = json.loads((ckpt / INDEX_FILE).read_text(encoding="utf-8"))["weight_map"]
    assert {_pattern(n) for n in tiny} == {_pattern(n) for n in REAL_INDEX["weight_map"]}
    assert set(reference) == {n for n in tiny if not n.endswith("weight_scale")}
    assert len(set(tiny.values())) == 4
    # The last shard holds the same kinds of tensor (the real one adds expert filler), and every layer reads it.
    tiny_kinds, tiny_layers = _last_shard(tiny)
    real_kinds, real_layers = _last_shard(REAL_INDEX["weight_map"])
    assert {p for p in real_kinds if ".experts." not in p} == tiny_kinds
    assert tiny_layers == set(range(6)) and real_layers == set(range(52))
    config = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    assert config["quantization_config"] == REAL_CONFIG["quantization_config"]
    assert {"dense", "sparse"} == set(config["mlp_layer_types"])
    assert {"sliding_attention", "full_attention"} == set(config["layer_types"])
