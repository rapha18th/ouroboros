"""A tiny random Beam checkpoint in the real FP8 format, for streaming and end-to-end tests.

Same tensor names, dtypes, quantization layout (FP8 weight + BF16 (out, 1) weight_scale on every
Linear) and ignore list (lm_head, embed_tokens BF16; mlp.gate FP32) as reflection/Beam-1.0-501B-A23B-FP8.
Like the real checkpoint, the last shard holds every norm, gate_bias, k/v projection and router, so
each layer reads it plus the shards holding its large weights, and layers straddle shard boundaries.

    ckpt_dir, reference = build_checkpoint(tmp_path)
    model = reference_model(tiny_config(), reference)   # transformers Beam holding the same values
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import BeamConfig, BeamForCausalLM

from ouroboros.harness import fp8
from ouroboros.harness.weights import INDEX_FILE, SCALE_SUFFIX, to_transformers

VENDOR_CONFIG = Path(__file__).resolve().parents[2] / "vendor" / "beam" / "config.json"
LAST_SHARD = re.compile(r"norm\.weight$|\.gate_bias$|\.mlp\.gate\.|\.self_attn\.[kv]_proj\.")


def tiny_config() -> BeamConfig:
    """6 layers: dense+sliding, dense+full, three sparse+sliding, sparse+full. Real vocab and quantization config."""
    real = json.loads(VENDOR_CONFIG.read_text(encoding="utf-8"))
    return BeamConfig(
        vocab_size=real["vocab_size"],
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=16,
        num_experts_per_tok=4,
        n_group=4,
        topk_group=2,
        sliding_window=8,
        max_position_embeddings=4096,
        layer_types=["sliding_attention", "full_attention", "sliding_attention",
                     "sliding_attention", "sliding_attention", "full_attention"],
        mlp_layer_types=["dense", "dense", "sparse", "sparse", "sparse", "sparse"],
        rope_parameters=real["rope_parameters"],
        routed_scaling_factor=real["routed_scaling_factor"],
        final_logit_softcapping=real["final_logit_softcapping"],
        rms_norm_eps=real["rms_norm_eps"],
        norm_topk_prob=real["norm_topk_prob"],
        quantization_config=real["quantization_config"],
    )


def _specs(c: BeamConfig) -> dict[str, tuple[tuple[int, ...], str]]:
    """Checkpoint name -> (shape, kind), in checkpoint order."""
    h, d, heads, kv = c.hidden_size, c.head_dim, c.num_attention_heads, c.num_key_value_heads
    specs = {"model.embed_tokens.weight": ((c.vocab_size, h), "embed")}
    for i in range(c.num_hidden_layers):
        p = f"model.layers.{i}."
        for norm in ("input", "post_attention", "pre_feedforward", "post_feedforward"):
            specs[f"{p}{norm}_layernorm.weight"] = ((h,), "norm")
        a = p + "self_attn."
        specs[a + "q_proj.weight"] = ((heads * d, h), "linear")
        specs[a + "k_proj.weight"] = ((kv * d, h), "linear")
        specs[a + "v_proj.weight"] = ((kv * d, h), "linear")
        specs[a + "gate_proj.weight"] = ((heads * d, h), "linear")
        specs[a + "o_proj.weight"] = ((h, heads * d), "linear")
        specs[a + "gate_bias"] = ((heads,), "gate_bias")
        specs[a + "q_norm.weight"] = ((d,), "norm")
        specs[a + "k_norm.weight"] = ((d,), "norm")
        if c.mlp_layer_types[i] == "dense":
            mlps = [(p + "mlp.", c.intermediate_size)]
        else:
            specs[p + "mlp.gate.weight"] = ((c.num_experts, h), "router")
            specs[p + "mlp.gate.e_score_correction_bias"] = ((c.num_experts,), "bias")
            mlps = [(f"{p}mlp.experts.{e}.", c.moe_intermediate_size) for e in range(c.num_experts)]
        for m, inter in mlps:
            specs[m + "gate_proj.weight"] = ((inter, h), "linear")
            specs[m + "up_proj.weight"] = ((inter, h), "linear")
            specs[m + "down_proj.weight"] = ((h, inter), "linear")
    specs["model.norm.weight"] = ((h,), "norm")
    specs["lm_head.weight"] = ((c.vocab_size, h), "lm_head")
    return specs


def _random(shape: tuple[int, ...], kind: str, g: torch.Generator) -> torch.Tensor:
    x = torch.randn(shape, generator=g)
    if kind == "norm":
        return (1 + 0.1 * x).to(torch.bfloat16)
    if kind == "gate_bias":
        return (0.5 * x).to(torch.bfloat16)
    if kind == "bias":
        return 0.05 * x
    if kind == "embed":
        return x.to(torch.bfloat16)
    scaled = x * shape[-1] ** -0.5  # linear, router, lm_head
    return scaled if kind == "router" else scaled.to(torch.bfloat16)


def build_checkpoint(out_dir: str | Path, config: BeamConfig | None = None, seed: int = 0,
                     num_shards: int = 4) -> tuple[Path, dict[str, torch.Tensor]]:
    """Write a random FP8 checkpoint to `out_dir` and return (out_dir, reference).

    `reference` maps every checkpoint name except the weight_scales to the values a loader must
    return: dequantized BF16 for FP8 weights, FP32 for the router, BF16 for everything else.
    """
    config = config or tiny_config()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    stored, reference = {}, {}
    for name, (shape, kind) in _specs(config).items():
        value = _random(shape, kind, g)
        if kind == "linear":
            q, scale = fp8.quantize(value)
            stored[name], stored[name + SCALE_SUFFIX] = q, scale
            reference[name] = fp8.dequantize(q, scale)
        else:
            stored[name] = reference[name] = value

    # The small tensors go to the last shard; the rest is split by tensor count so layers straddle shards.
    last = [n for n in stored if LAST_SHARD.search(n)]
    rest = [n for n in stored if not LAST_SHARD.search(n)]
    per = -(-len(rest) // (num_shards - 1))
    groups = [rest[k:k + per] for k in range(0, len(rest), per)] + [last]
    files = [f"model-{k + 1:05d}-of-{len(groups):05d}.safetensors" for k in range(len(groups))]
    weight_map = {}
    for file, group in zip(files, groups):
        save_file({n: stored[n].contiguous() for n in group}, str(out / file), metadata={"format": "pt"})
        weight_map.update({n: file for n in group})
    index = {"metadata": {"total_size": sum(t.nbytes for t in stored.values())}, "weight_map": weight_map}
    (out / INDEX_FILE).write_text(json.dumps(index, indent=2), encoding="utf-8")
    config.to_json_file(str(out / "config.json"))
    return out, reference


def reference_model(config: BeamConfig, reference: dict[str, torch.Tensor]) -> BeamForCausalLM:
    """A BF16 transformers Beam (router in FP32) holding exactly the reference values."""
    model = BeamForCausalLM._from_config(config, dtype=torch.bfloat16).eval()
    for layer in model.model.layers:
        if config.mlp_layer_types[layer.self_attn.layer_idx] == "sparse":
            layer.mlp.gate.float()
    state = {k: v for k, v in reference.items() if not k.startswith("model.layers.")}
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}."
        layer = {k[len(p):]: v for k, v in reference.items() if k.startswith(p)}
        state.update({p + k: v for k, v in to_transformers(layer).items()})
    model.load_state_dict(state, strict=True)
    return model
