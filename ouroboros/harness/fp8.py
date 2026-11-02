"""compressed-tensors FP8 per-channel weights (Beam FP8_DYNAMIC) to and from BF16.

Convention, from compressed-tensors `forward_helpers._quantize_dequantize` and
`utils.helpers.calculate_qparams` (symmetric float, strategy "channel"):

    scale = max(|w[row, :]|) / 448        shape (out_features, 1), stored in the weight's dtype (BF16)
    q     = float8_e4m3fn(clamp(w / scale, -448, 448))   the division runs in BF16
    w     = q * scale                     dequantization multiplies

Zero scales are replaced by the dtype's eps to avoid division by zero. Checked on real Beam
tensors (tests/test_weights_hub.py): quantize() reproduces the published FP8 codes and scales exactly.
"""

import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)  # 448.0


def dequantize(weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """FP8 weight (out, in) times per-output-channel scale (out, 1), rounded once to `dtype`."""
    if weight.dtype != FP8_DTYPE:
        raise TypeError(f"expected {FP8_DTYPE}, got {weight.dtype}")
    if scale.shape != (weight.shape[0], 1):
        raise ValueError(f"scale shape {tuple(scale.shape)} does not match weight rows {weight.shape[0]}")
    # The product of an e4m3 and a bf16 value is exact in FP32, so this matches compressed-tensors bit for bit.
    return (weight.to(torch.float32) * scale.to(torch.float32)).to(dtype)


def quantize(weight: torch.Tensor, scale_dtype: torch.dtype = torch.bfloat16) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight to (float8_e4m3fn weight, per-output-channel scale of shape (out, 1))."""
    if weight.ndim != 2:
        raise ValueError(f"expected a 2D weight, got shape {tuple(weight.shape)}")
    w = weight.to(scale_dtype)
    scale = (w.to(torch.float32).abs().amax(dim=1, keepdim=True) / FP8_MAX).to(scale_dtype)
    scale = torch.where(scale == 0, torch.finfo(scale_dtype).eps, scale)
    q = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return q, scale
