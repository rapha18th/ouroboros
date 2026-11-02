import pytest
import torch

from ouroboros.harness import fp8


def _weight(rows: int = 48, cols: int = 96, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # Rows of very different magnitude, so per-channel scales matter.
    return (torch.randn(rows, cols, generator=g) * torch.logspace(-3, 1, rows)[:, None]).to(torch.bfloat16)


def test_layout_and_scale_convention():
    w = _weight()
    q, scale = fp8.quantize(w)
    assert q.dtype == torch.float8_e4m3fn and q.shape == w.shape
    assert scale.dtype == torch.bfloat16 and scale.shape == (w.shape[0], 1)
    expected = (w.float().abs().amax(dim=1, keepdim=True) / 448).to(torch.bfloat16)
    assert torch.equal(scale, expected)
    # Each row's largest magnitude lands at the top of the FP8 range.
    assert (q.float().abs().amax(dim=1) >= 416).all()


def test_dequantize_multiplies_like_compressed_tensors():
    w = _weight(seed=1)
    q, scale = fp8.quantize(w)
    deq = fp8.dequantize(q, scale)
    assert deq.dtype == torch.bfloat16
    # compressed-tensors: q.to(scale.dtype) * scale. Bit-identical to the FP32 product rounded once.
    assert torch.equal(deq, q.to(torch.bfloat16) * scale)
    assert torch.equal(fp8.dequantize(q, scale, torch.float32), q.float() * scale.float())


def test_round_trip_error_bound():
    w = _weight(seed=2).float()
    q, scale = fp8.quantize(w.to(torch.bfloat16))
    err = (fp8.dequantize(q, scale).float() - w).abs()
    # e4m3 keeps 3 mantissa bits (relative error 2^-4), subnormals add an absolute floor, BF16 adds 2^-8.
    bound = w.abs() * (2**-4 + 2**-7) + scale.float() * 2**-9
    assert (err <= bound).all()
    assert (err / w.abs().clamp_min(1e-30)).median() < 0.04


def test_zero_rows_and_saturation():
    w = torch.zeros(4, 8, dtype=torch.bfloat16)
    w[1] = 1e4
    w[2, 0] = -3.0
    q, scale = fp8.quantize(w)
    deq = fp8.dequantize(q, scale)
    assert torch.isfinite(deq).all() and torch.isfinite(q.float()).all()
    assert (scale > 0).all()
    assert torch.equal(deq[0], w[0]) and torch.equal(deq[3], w[3])
    # Values at the row maximum saturate at +-448 and come back within the BF16 rounding of the scale.
    torch.testing.assert_close(deq[1:3].float(), w[1:3].float(), rtol=2**-7, atol=0)


def test_dequantize_rejects_bad_inputs():
    q, scale = fp8.quantize(_weight())
    with pytest.raises(TypeError):
        fp8.dequantize(q.to(torch.bfloat16), scale)
    with pytest.raises(ValueError):
        fp8.dequantize(q, scale[:-1])
    with pytest.raises(ValueError):
        fp8.quantize(torch.zeros(3))
