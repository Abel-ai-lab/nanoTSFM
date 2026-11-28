"""Invert a weighted CDF with nine monotone quantiles per component."""

import torch


def _cdf_deciles(knots, values, *, right):
    """CDF or its left limit in decile units, including point masses."""
    count = torch.searchsorted(knots, values, right=right)
    upper = count.clamp(1, 8)
    low = knots.gather(-1, upper - 1)
    high = knots.gather(-1, upper)
    width = high - low
    ratio = (values - low) / torch.where(width > 0, width, 1)
    interior = upper + ratio.clamp(0, 1)
    return torch.where(count == 0, 0, torch.where(count == 9, 10, interior))


def pool_quantiles(raw, phase, phase_weight):
    """Mix two CDFs at a scalar weight and recover their 0.1 through 0.9 quantiles.

    Use a linear quantile function between reported levels and put the
    unreported lower and upper tails at the respective endpoints. Inputs have
    matching shapes, finite values and quantiles already in ascending order.
    """
    if raw.shape != phase.shape or raw.shape[-1:] != (9,):
        raise ValueError("Expected two matching tensors of nine ordered quantiles")
    if phase_weight == 0:
        return raw
    if phase_weight == 1:
        return phase
    if not 0 < phase_weight < 1:
        raise ValueError("Phase weight must be between zero and one")

    left, right = raw.double().contiguous(), phase.double().contiguous()
    knots = torch.cat((left, right), dim=-1).sort(dim=-1).values
    limit = torch.lerp(_cdf_deciles(left, knots, right=True), _cdf_deciles(right, knots, right=True), phase_weight)
    levels = torch.arange(1, 10, dtype=limit.dtype, device=limit.device).expand_as(left)
    upper = torch.searchsorted(limit, levels - 8 * torch.finfo(limit.dtype).eps)
    lower = (upper - 1).clamp_min(0)
    lo, hi = knots.gather(-1, lower), knots.gather(-1, upper)
    f_lo = limit.gather(-1, lower)
    f_hi = torch.lerp(_cdf_deciles(left, hi, right=False), _cdf_deciles(right, hi, right=False), phase_weight)
    # A crossing inside a jump takes the knot itself. Otherwise invert the
    # linear segment between the two knots, including its upper left limit.
    interval = f_hi > levels
    rise = torch.where(interval & (upper > 0), f_hi - f_lo, 1)
    ratio = ((levels - f_lo) / rise).clamp(0, 1)
    pool = torch.where(interval & (upper > 0), torch.lerp(lo, hi, ratio), hi)
    return pool.to(raw.dtype)
