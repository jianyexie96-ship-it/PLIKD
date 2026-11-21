import math

import torch


def _positive_scalar(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive finite scalar") from error
    if isinstance(value, bool) or not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return result


def _prepare(cost, row_marginal, col_marginal, epsilon, max_iter, tol, strict):
    if (
        not isinstance(cost, torch.Tensor)
        or not cost.is_floating_point()
        or cost.ndim < 2
    ):
        raise ValueError("Cost must be a floating tensor with at least two dimensions")
    if not cost.numel() or not torch.isfinite(cost).all():
        raise ValueError("Cost must be nonempty and finite")
    epsilon = _positive_scalar(epsilon, "epsilon")
    tol = _positive_scalar(tol, "tol")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not isinstance(strict, bool):
        raise TypeError("strict must be a boolean")
    work_dtype = torch.float32 if cost.device.type == "mps" else torch.float64
    work = cost.detach().to(work_dtype)
    marginals = []
    for value, size, name in (
        (row_marginal, cost.shape[-2], "row_marginal"),
        (col_marginal, cost.shape[-1], "col_marginal"),
    ):
        shape = cost.shape[:-2] + (size,)
        try:
            marginal = (
                torch.full(shape, 1 / size, dtype=work_dtype, device=cost.device)
                if value is None
                else torch.broadcast_to(
                    torch.as_tensor(
                        value, dtype=work_dtype, device=cost.device
                    ).detach(),
                    shape,
                )
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError(f"{name} cannot broadcast to {shape}") from error
        if not torch.isfinite(marginal).all() or torch.any(marginal <= 0):
            raise ValueError(f"{name} must contain positive finite probability masses")
        if not torch.allclose(
            marginal.sum(-1),
            torch.ones(cost.shape[:-2], dtype=work_dtype, device=cost.device),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError(f"{name} must sum to one")
        marginals.append(marginal)
    return work, marginals[0], marginals[1], epsilon, tol


def _threshold(offsets, mass):
    ordered = offsets.sort(dim=-1).values
    count = torch.arange(
        1, offsets.shape[-1] + 1, dtype=offsets.dtype, device=offsets.device
    )
    thresholds = (mass.unsqueeze(-1) + ordered.cumsum(-1)) / count
    active = (thresholds > ordered).sum(-1).sub(1).clamp_min(0)
    return thresholds.gather(-1, active.unsqueeze(-1)).squeeze(-1)


def _residuals(plan, rows, columns):
    return (plan.sum(-1) - rows).abs().amax(-1), (plan.sum(-2) - columns).abs().amax(-1)


def _result(cost, plan, rows, columns, first, second, iterations, tol, strict, method):
    dtype = torch.float64 if cost.dtype == torch.float64 else torch.float32
    plan = plan.to(dtype).detach()
    row_residual, col_residual = _residuals(plan.to(rows.dtype), rows, columns)
    converged = bool(torch.all(torch.maximum(row_residual, col_residual) <= tol).item())
    info = {
        "converged": converged,
        "row_residual": row_residual.detach(),
        "col_residual": col_residual.detach(),
        "iterations": iterations,
        "row_dual": first.detach(),
        "col_dual": second.detach(),
        "method": method,
    }
    if strict and not converged:
        largest = torch.maximum(row_residual, col_residual).max().item()
        raise RuntimeError(
            f"{method} did not converge in {iterations} iterations; maximum marginal residual {largest:.6g} exceeds tol={tol:.6g}"
        )
    return plan, info


def sparse_transport(
    cost,
    row_marginal=None,
    col_marginal=None,
    epsilon=1.0,
    max_iter=100,
    tol=1e-6,
    strict=True,
):
    work, rows, columns, epsilon, tol = _prepare(
        cost, row_marginal, col_marginal, epsilon, max_iter, tol, strict
    )
    with torch.no_grad():
        offset = work.amin(dim=(-2, -1), keepdim=True)
        centered = work - offset
        first, second = torch.zeros_like(rows), torch.zeros_like(columns)
        for iteration in range(1, max_iter + 1):
            first = _threshold(centered - second.unsqueeze(-2), epsilon * rows)
            second = _threshold(
                centered.transpose(-2, -1) - first.unsqueeze(-2), epsilon * columns
            )
            gauge = first.mean(-1, keepdim=True)
            first = first - gauge
            second = second + gauge
            plan = (
                (first.unsqueeze(-1) + second.unsqueeze(-2) - centered) / epsilon
            ).clamp_min(0)
            row_residual, col_residual = _residuals(plan, rows, columns)
            if torch.all(torch.maximum(row_residual, col_residual) <= tol * 0.5):
                break
        return _result(
            cost,
            plan,
            rows,
            columns,
            first + offset.squeeze(-1),
            second,
            iteration,
            tol,
            strict,
            "squared_l2",
        )


def entropic_transport(
    cost,
    row_marginal=None,
    col_marginal=None,
    epsilon=1.0,
    max_iter=100,
    tol=1e-6,
    strict=True,
):
    work, rows, columns, epsilon, tol = _prepare(
        cost, row_marginal, col_marginal, epsilon, max_iter, tol, strict
    )
    with torch.no_grad():
        offset = work.amin(dim=(-2, -1), keepdim=True)
        log_kernel = -(work - offset) / epsilon
        first, second = torch.zeros_like(rows), torch.zeros_like(columns)
        log_rows, log_columns = rows.log(), columns.log()
        for iteration in range(1, max_iter + 1):
            first = log_rows - torch.logsumexp(
                log_kernel + second.unsqueeze(-2), dim=-1
            )
            second = log_columns - torch.logsumexp(
                log_kernel + first.unsqueeze(-1), dim=-2
            )
            gauge = first.mean(-1, keepdim=True)
            first = first - gauge
            second = second + gauge
            plan = (log_kernel + first.unsqueeze(-1) + second.unsqueeze(-2)).exp()
            row_residual, col_residual = _residuals(plan, rows, columns)
            if torch.all(torch.maximum(row_residual, col_residual) <= tol * 0.5):
                break
        return _result(
            cost,
            plan,
            rows,
            columns,
            epsilon * first + offset.squeeze(-1),
            epsilon * second,
            iteration,
            tol,
            strict,
            "entropic",
        )


def transport_distance(cost, plan, epsilon=1.0, regularized=False):
    if (
        not isinstance(cost, torch.Tensor)
        or not cost.is_floating_point()
        or cost.ndim < 2
        or not cost.numel()
        or not torch.isfinite(cost).all()
    ):
        raise ValueError("Cost must be a nonempty finite floating tensor")
    if (
        not isinstance(plan, torch.Tensor)
        or not plan.is_floating_point()
        or plan.shape[-2:] != cost.shape[-2:]
        or plan.device != cost.device
    ):
        raise ValueError(
            "Plan must be a floating tensor with matching support dimensions and device"
        )
    if not torch.isfinite(plan).all() or torch.any(plan < 0):
        raise ValueError("Plan must contain finite nonnegative masses")
    epsilon = _positive_scalar(epsilon, "epsilon")
    try:
        cost, fixed = torch.broadcast_tensors(cost, plan.detach())
    except RuntimeError as error:
        raise ValueError("Cost and plan batch dimensions cannot broadcast") from error
    distance = (cost * fixed).sum(dim=(-2, -1))
    if regularized:
        distance = distance + 0.5 * epsilon * fixed.square().sum(dim=(-2, -1))
    return distance
