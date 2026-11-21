import importlib

import pytest
import torch


def transport():
    return importlib.import_module("transport")


def test_zero_cost_uniform_rectangular_batched_transport():
    module = transport()
    cost = torch.zeros(2, 3, 4, 7, dtype=torch.float64)
    plan, info = module.sparse_transport(cost)
    assert plan.shape == cost.shape
    assert torch.allclose(plan, torch.full_like(plan, 1 / 28), atol=1e-10)
    assert info["converged"] is True
    assert info["row_residual"].shape == (2, 3)
    assert info["col_residual"].shape == (2, 3)
    assert info["iterations"] <= 100
    assert not plan.requires_grad


@pytest.mark.parametrize("epsilon,diagonal", [(1.0, 0.5), (4.0, 0.375)])
def test_sparse_transport_matches_analytic_two_by_two_solution(epsilon, diagonal):
    module = transport()
    cost = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    plan, info = module.sparse_transport(cost, epsilon=epsilon, tol=1e-12)
    expected = torch.tensor(
        [[diagonal, 0.5 - diagonal], [0.5 - diagonal, diagonal]], dtype=torch.float64
    )
    assert torch.allclose(plan, expected, atol=1e-10)
    assert not plan.requires_grad
    assert info["converged"]
    if epsilon == 1:
        assert (plan == 0).sum() == 2


def test_nonuniform_marginals_and_sparse_kkt_conditions():
    module = transport()
    cost = torch.zeros(2, 2, dtype=torch.float64)
    rows = torch.tensor([0.8, 0.2], dtype=torch.float64)
    columns = torch.tensor([0.6, 0.4], dtype=torch.float64)
    plan, info = module.sparse_transport(cost, rows, columns, tol=1e-12)
    expected = torch.tensor([[0.45, 0.35], [0.15, 0.05]], dtype=torch.float64)
    assert torch.allclose(plan, expected, atol=1e-10)
    assert torch.allclose(plan.sum(-1), rows, atol=1e-12)
    assert torch.allclose(plan.sum(-2), columns, atol=1e-12)
    recovered = (
        info["row_dual"].unsqueeze(-1) + info["col_dual"].unsqueeze(-2) - cost
    ).clamp_min(0)
    assert torch.allclose(plan, recovered, atol=1e-12)


def test_rectangular_sparse_solution_has_a_primal_dual_optimality_certificate():
    module = transport()
    cost = torch.tensor([[0.0, 2.0, 3.0], [2.0, 0.1, 0.4]], dtype=torch.float64)
    rows = torch.tensor([0.6, 0.4], dtype=torch.float64)
    columns = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    plan, info = module.sparse_transport(cost, rows, columns, max_iter=2000, tol=1e-11)
    slack = cost + plan - info["row_dual"][:, None] - info["col_dual"][None, :]
    assert torch.allclose(plan.sum(-1), rows, atol=1e-11)
    assert torch.allclose(plan.sum(-2), columns, atol=1e-11)
    assert (slack >= -1e-10).all()
    assert (plan * slack).abs().max() < 1e-10
    assert (plan == 0).any()


def test_marginals_broadcast_across_multiple_leading_dimensions():
    module = transport()
    generator = torch.Generator().manual_seed(17)
    cost = torch.rand(2, 3, 4, 5, generator=generator, dtype=torch.float64)
    rows = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    columns = torch.tensor([0.1, 0.2, 0.25, 0.25, 0.2], dtype=torch.float64)
    plan, info = module.sparse_transport(cost, rows, columns, max_iter=2000, tol=1e-10)
    assert torch.allclose(plan.sum(-1), rows.expand(2, 3, 4), atol=1e-10)
    assert torch.allclose(plan.sum(-2), columns.expand(2, 3, 5), atol=1e-10)
    assert info["row_residual"].max() <= 1e-10
    assert info["col_residual"].max() <= 1e-10


def test_sparse_plan_is_invariant_to_row_column_cost_potentials():
    module = transport()
    cost = torch.tensor([[0.2, 0.5, 0.9], [0.7, 0.1, 0.4]], dtype=torch.float64)
    row_shift = torch.tensor([2.0, -4.0], dtype=torch.float64)
    col_shift = torch.tensor([1.0, 5.0, -2.0], dtype=torch.float64)
    first, _ = module.sparse_transport(cost, max_iter=2000, tol=1e-11)
    shifted, _ = module.sparse_transport(
        cost + row_shift[:, None] + col_shift[None, :], max_iter=2000, tol=1e-11
    )
    assert torch.allclose(first, shifted, atol=1e-10)


def test_strict_nonconvergence_errors_and_nonstrict_returns_diagnostics():
    module = transport()
    cost = torch.tensor(
        [[0.0, 2.0, 4.0], [1.0, 0.0, 1.0], [3.0, 2.0, 0.0]], dtype=torch.float64
    )
    rows = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float64)
    columns = torch.tensor([0.4, 0.35, 0.25], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="converg|residual"):
        module.sparse_transport(cost, rows, columns, max_iter=1, tol=1e-12)
    plan, info = module.sparse_transport(
        cost, rows, columns, max_iter=1, tol=1e-12, strict=False
    )
    assert info["converged"] is False
    assert info["iterations"] == 1
    assert max(info["row_residual"].item(), info["col_residual"].item()) > 1e-12
    assert torch.isfinite(plan).all()


@pytest.mark.parametrize(
    "invalid",
    [
        "nan",
        "negative_mass",
        "zero_mass",
        "wrong_mass",
        "wrong_shape",
        "epsilon",
        "max_iter",
        "tol",
    ],
)
def test_sparse_transport_rejects_invalid_inputs(invalid):
    module = transport()
    cost = torch.zeros(2, 3)
    kwargs = {}
    if invalid == "nan":
        cost[0, 0] = torch.nan
    elif invalid == "negative_mass":
        kwargs["row_marginal"] = [-0.1, 1.1]
    elif invalid == "zero_mass":
        kwargs["row_marginal"] = [0.0, 1.0]
    elif invalid == "wrong_mass":
        kwargs["row_marginal"] = [0.2, 0.2]
    elif invalid == "wrong_shape":
        kwargs["row_marginal"] = [0.2, 0.3, 0.5]
    elif invalid == "epsilon":
        kwargs["epsilon"] = 0
    elif invalid == "max_iter":
        kwargs["max_iter"] = 0
    else:
        kwargs["tol"] = -1
    with pytest.raises(ValueError):
        module.sparse_transport(cost, **kwargs)


def test_regularized_optimal_value_gradient_matches_finite_differences():
    module = transport()
    cost = torch.tensor(
        [[0.1, 0.8], [0.7, 0.2]], dtype=torch.float64, requires_grad=True
    )
    plan, _ = module.sparse_transport(cost, epsilon=4.0, tol=1e-12)
    value = module.transport_distance(cost, plan, epsilon=4.0, regularized=True)
    value.backward()
    assert torch.allclose(cost.grad, plan, atol=1e-12)
    delta = 1e-5
    finite_difference = torch.zeros_like(cost)
    for i in range(2):
        for j in range(2):
            positive, negative = cost.detach().clone(), cost.detach().clone()
            positive[i, j] += delta
            negative[i, j] -= delta
            positive_plan, _ = module.sparse_transport(positive, epsilon=4.0, tol=1e-12)
            negative_plan, _ = module.sparse_transport(negative, epsilon=4.0, tol=1e-12)
            positive_value = module.transport_distance(
                positive, positive_plan, epsilon=4.0, regularized=True
            )
            negative_value = module.transport_distance(
                negative, negative_plan, epsilon=4.0, regularized=True
            )
            finite_difference[i, j] = (positive_value - negative_value) / (2 * delta)
    assert torch.allclose(cost.grad, finite_difference, atol=1e-8)


def test_fixed_plan_distance_preserves_cost_gradient_and_stops_plan_gradient():
    module = transport()
    cost = torch.rand(2, 2, 3, dtype=torch.float64, requires_grad=True)
    plan = torch.full((2, 2, 3), 1 / 6, dtype=torch.float64, requires_grad=True)
    distances = module.transport_distance(cost, plan)
    assert distances.shape == (2,)
    distances.sum().backward()
    assert torch.allclose(cost.grad, plan.detach())
    assert plan.grad is None


def test_log_domain_entropic_transport_matches_analytic_solution():
    module = transport()
    epsilon = 0.5
    cost = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    plan, info = module.entropic_transport(cost, epsilon=epsilon, tol=1e-12)
    diagonal = 0.5 / (1 + torch.exp(torch.tensor(-1 / epsilon, dtype=torch.float64)))
    expected = torch.tensor(
        [[diagonal, 0.5 - diagonal], [0.5 - diagonal, diagonal]], dtype=torch.float64
    )
    assert torch.allclose(plan, expected, atol=1e-12)
    assert (plan > 0).all()
    assert info["converged"]


def test_entropic_transport_handles_rectangular_masses_and_large_constant_costs():
    module = transport()
    cost = torch.tensor(
        [[[10000.0, 10001.0, 10002.0], [10003.0, 10000.0, 10001.0]]],
        dtype=torch.float64,
    )
    rows = torch.tensor([0.7, 0.3], dtype=torch.float64)
    columns = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    plan, info = module.entropic_transport(
        cost, rows, columns, epsilon=0.5, max_iter=2000, tol=1e-10
    )
    assert torch.isfinite(plan).all() and (plan > 0).all()
    assert torch.allclose(plan.sum(-1), rows[None], atol=1e-10)
    assert torch.allclose(plan.sum(-2), columns[None], atol=1e-10)
    assert info["converged"]


def test_half_precision_costs_keep_float32_transport_masses():
    module = transport()
    cost = torch.zeros(3, 7, dtype=torch.bfloat16)
    plan, info = module.sparse_transport(cost)
    assert plan.dtype == torch.float32
    assert info["converged"]
    assert torch.allclose(plan.sum(-1), torch.full((3,), 1 / 3), atol=1e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_transport_uses_supported_precision_and_preserves_cost_gradient():
    module = transport()
    cost = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device="mps", requires_grad=True)
    plan, info = module.sparse_transport(cost)
    assert plan.device.type == "mps" and plan.dtype == torch.float32
    assert info["converged"]
    assert torch.allclose(plan.cpu(), torch.tensor([[0.5, 0.0], [0.0, 0.5]]), atol=1e-6)
    module.transport_distance(cost, plan).backward()
    assert torch.allclose(cost.grad, plan)
    entropic, info = module.entropic_transport(cost.detach(), epsilon=0.5)
    assert info["converged"] and entropic.device.type == "mps"
