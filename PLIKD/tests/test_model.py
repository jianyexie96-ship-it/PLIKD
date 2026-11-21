import unittest

import torch
from torch import nn

import model as module


class TinyTransformer(nn.Module):
    def __init__(self, width, causal=False):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, 2, dropout=0)
        self.norm = nn.LayerNorm(width)
        self.causal = causal

    def forward(self, x):
        mask = (
            torch.ones(x.shape[0], x.shape[0], device=x.device, dtype=torch.bool).triu(
                1
            )
            if self.causal
            else None
        )
        return self.norm(x + self.attn(x, x, x, attn_mask=mask, need_weights=False)[0])


class TinyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 2, stride=2, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(8) * 0.05)
        self.positional_embedding = nn.Parameter(torch.randn(5, 8) * 0.05)
        self.ln_pre = nn.LayerNorm(8)
        self.transformer = TinyTransformer(8)
        self.ln_post = nn.LayerNorm(8)
        self.proj = nn.Parameter(torch.eye(8))


class TinyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = TinyVisual()
        self.token_embedding = nn.Embedding(64, 8)
        self.positional_embedding = nn.Parameter(torch.randn(24, 8) * 0.05)
        self.transformer = TinyTransformer(8, causal=True)
        self.ln_final = nn.LayerNorm(8)
        self.text_projection = nn.Parameter(torch.eye(8))
        self.context_length = 24


def tokenize(texts):
    if isinstance(texts, str):
        texts = [texts]
    vocabulary = {"cat": 3, "dog": 4, "red": 5, "bird": 6, ".": 2}
    result = torch.zeros(len(texts), 24, dtype=torch.long)
    for index, text in enumerate(texts):
        ids = (
            [62]
            + [vocabulary.get(word, 7) for word in text.replace(".", " .").split()]
            + [63]
        )
        if len(ids) > 24:
            raise ValueError("Too many tokens")
        result[index, : len(ids)] = torch.tensor(ids)
    return result


def configuration(**changes):
    config = {
        "model": {
            "visual_prompt_length": 2,
            "context_tokens": 3,
            "projector_hidden": 12,
            "text_support": "token",
            "text_prompts": 3,
            "class_chunk_size": 1,
            "activation_checkpointing": False,
            "classname_period": True,
        },
        "loss": {
            "temperature": 0.2,
            "knowledge_weight": 0.7,
            "visual_weight": 0.3,
            "classification_weight": 0.8,
            "classification": "ce",
        },
        "transport": {
            "solver": "sparse",
            "epsilon": 1.0,
            "max_iter": 100,
            "tol": 1e-6,
            "strict": True,
            "regularized": False,
        },
    }
    for key, value in changes.items():
        config[key].update(value)
    return config


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)

    def make_model(self, **changes):
        return module.PLIKD(
            TinyCLIP(), tokenize, ["cat", "dog"], configuration(**changes)
        )

    def test_local_features_and_class_logits_have_declared_shapes(self):
        model = self.make_model()
        result = model(torch.randn(2, 3, 4, 4))
        self.assertEqual(result["logits"].shape, (2, 2))
        self.assertEqual(result["z_r"].shape, (2, 8))
        self.assertEqual(result["visual_global"].shape, (2, 8))
        self.assertEqual(result["visual_local"].shape, (2, 4, 8))
        self.assertTrue(torch.isfinite(result["logits"]).all())
        self.assertEqual(result["losses"], {})

    def test_losses_reach_all_three_learned_components_but_not_clip(self):
        model = self.make_model()
        result = model(
            torch.randn(2, 3, 4, 4),
            knowledge_features=torch.randn(2, 8),
            labels=torch.tensor([0, 1]),
        )
        losses = result["losses"]
        self.assertTrue(
            torch.allclose(
                losses["total"],
                0.7 * losses["knowledge"]
                + 0.3 * losses["visual"]
                + 0.8 * losses["classification"],
            )
        )
        losses["total"].backward()
        for prefix in ("visual_prompt", "context", "projector"):
            gradients = [
                parameter.grad
                for name, parameter in model.named_parameters()
                if name.startswith(prefix)
            ]
            self.assertTrue(
                any(
                    gradient is not None and gradient.abs().sum() > 0
                    for gradient in gradients
                ),
                prefix,
            )
        self.assertTrue(
            all(
                not parameter.requires_grad and parameter.grad is None
                for parameter in model.clip_model.parameters()
            )
        )

    def test_novel_classes_use_shared_context_and_dynamic_candidates(self):
        model = self.make_model().eval()
        result = model(torch.randn(1, 3, 4, 4), class_names=["red bird", "cat", "bird"])
        self.assertEqual(result["logits"].shape, (1, 3))
        self.assertEqual(model.class_names, ["cat", "dog"])

    def test_duplicate_candidate_names_retain_distinct_logit_columns(self):
        model = module.PLIKD(
            TinyCLIP(), tokenize, ["cat", "cat"], configuration()
        ).eval()
        result = model(torch.randn(2, 3, 4, 4), class_names=["dog", "dog"])
        self.assertEqual(model.class_names, ["cat", "cat"])
        self.assertEqual(result["logits"].shape, (2, 2))
        self.assertTrue(torch.allclose(result["logits"][:, 0], result["logits"][:, 1]))

    def test_token_support_excludes_context_sot_padding_and_period(self):
        model = self.make_model()
        self.assertEqual(model.support_sizes(["cat", "red bird"]), [2, 3])

    def test_ensemble_support_has_multiple_eot_features(self):
        model = self.make_model(model={"text_support": "ensemble"})
        self.assertEqual(model.support_sizes(["cat", "red bird"]), [3, 3])
        result = model(torch.randn(2, 3, 4, 4))
        self.assertEqual(result["logits"].shape, (2, 2))

    def test_single_eot_ensemble_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "two|nondegenerate"):
            self.make_model(model={"text_support": "ensemble", "text_prompts": 1})

    def test_checkpointed_class_chunks_preserve_logits_and_gradients(self):
        first = self.make_model(model={"class_chunk_size": 2})
        second = self.make_model(
            model={"class_chunk_size": 1, "activation_checkpointing": True}
        )
        second.load_state_dict(first.state_dict())
        images = torch.randn(2, 3, 4, 4)
        teacher = torch.randn(2, 8)
        labels = torch.tensor([0, 1])
        left = first(images, knowledge_features=teacher, labels=labels)
        right = second(images, knowledge_features=teacher, labels=labels)
        self.assertTrue(torch.allclose(left["logits"], right["logits"], atol=1e-5))
        left["losses"]["total"].backward()
        right["losses"]["total"].backward()
        self.assertTrue(
            torch.allclose(first.context.grad, second.context.grad, atol=2e-5)
        )

    def test_trainable_state_excludes_frozen_clip_and_validates_shapes(self):
        model = self.make_model()
        state = model.trainable_state_dict()
        self.assertFalse(any(name.startswith("clip_model.") for name in state))
        self.assertEqual(
            set(state),
            {
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            },
        )
        model.load_trainable_state_dict(state)
        broken = dict(state)
        broken["visual_prompt"] = torch.zeros(1)
        with self.assertRaises(ValueError):
            model.load_trainable_state_dict(broken)

    def test_literal_classification_loss_is_negative_correct_probability(self):
        model = self.make_model(loss={"classification": "literal_eq26"}).eval()
        labels = torch.tensor([0, 1])
        result = model(torch.randn(2, 3, 4, 4), labels=labels)
        expected = -result["logits"].softmax(-1)[torch.arange(2), labels].mean()
        self.assertTrue(torch.allclose(result["losses"]["classification"], expected))

    def test_clip_remains_in_evaluation_mode_when_prompts_are_training(self):
        model = self.make_model()
        model.train()
        self.assertFalse(model.clip_model.training)
        self.assertTrue(model.training)

    def test_frozen_fingerprint_supports_native_clip_scalar_parameters(self):
        clip = TinyCLIP()
        clip.logit_scale = nn.Parameter(torch.tensor(2.0))
        model = module.PLIKD(clip, tokenize, ["cat", "dog"], configuration())
        self.assertEqual(len(model.frozen_fingerprint()), 64)

    def test_bundled_configuration_uses_regularized_method_distance(self):
        from pathlib import Path

        import yaml

        config = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text())
        self.assertIs(config["transport"]["regularized"], True)

    def test_default_sparse_score_matches_eq24_and_preserves_transport_only_option(
        self,
    ):
        from unittest.mock import patch

        model = self.make_model(model={"class_chunk_size": 2})
        model.transport_settings.pop("regularized")
        model.transport_settings["epsilon"] = 2.0
        z_r = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True
        )
        local = torch.zeros(1, 2, 8)
        local[:, :, 0] = 0.5
        local[:, :, 1] = 3**0.5 / 2
        plan = torch.tensor(
            [[[[0.25, 0.25], [0.25, 0.25]], [[0.5, 0.0], [0.0, 0.5]]]],
            requires_grad=True,
        )

        def support(z, specifications):
            return z[:, None, None].expand(len(z), len(specifications), 2, 8)

        def solver(cost, **settings):
            self.assertTrue(torch.allclose(cost, torch.full_like(cost, 0.5)))
            return plan, {
                "row_residual": torch.zeros(1, 2),
                "col_residual": torch.zeros(1, 2),
                "iterations": 1,
            }

        with (
            patch.object(model, "_support", side_effect=support),
            patch.object(module, "sparse_transport", side_effect=solver),
        ):
            default, _, _ = model._logits(z_r, local, ["cat", "dog"])
            model.transport_settings["regularized"] = True
            explicit, _, _ = model._logits(z_r, local, ["cat", "dog"])
            model.transport_settings["regularized"] = False
            transport_only, _, _ = model._logits(z_r, local, ["cat", "dog"])
        penalty = 0.5 * 2.0 * plan.detach().square().sum(dim=(-2, -1))
        expected = (1.0 - 0.5 - penalty) / model.temperature
        self.assertTrue(torch.allclose(default, explicit))
        self.assertTrue(torch.allclose(default, expected))
        self.assertTrue(
            torch.allclose(transport_only, torch.full((1, 2), 0.5 / model.temperature))
        )
        self.assertGreater(default[0, 0], default[0, 1])
        default.sum().backward()
        self.assertIsNone(plan.grad)
        self.assertIsNotNone(z_r.grad)
        self.assertGreater(z_r.grad.abs().sum(), 0)

    def test_regularized_entropic_score_uses_entropy_instead_of_squared_norm(self):
        first = self.make_model(
            transport={"solver": "entropic", "regularized": False, "return_plans": True}
        )
        second = self.make_model(
            transport={"solver": "entropic", "regularized": True, "return_plans": True}
        )
        second.load_state_dict(first.state_dict())
        images = torch.randn(2, 3, 4, 4)
        base = first(images)
        regularized = second(images)
        expected = base["logits"].clone()
        for group in base["plans"]:
            plan = group["plan"]
            entropy_term = torch.xlogy(plan, plan).sum(dim=(-1, -2))
            expected[:, group["class_indices"]] -= entropy_term / first.temperature
        self.assertTrue(torch.allclose(regularized["logits"], expected, atol=1e-5))

    def test_transport_diagnostics_are_always_per_image_without_exporting_plans(self):
        model = self.make_model(transport={"strict": False, "max_iter": 1})
        result = model(torch.randn(2, 3, 4, 4), class_names=["cat", "red bird", "dog"])
        diagnostics = result["transport_diagnostics"]
        for key in (
            "row_residual",
            "col_residual",
            "converged_plans",
            "total_plans",
            "iterations",
        ):
            self.assertEqual(diagnostics[key].shape, (2,))
            self.assertFalse(diagnostics[key].requires_grad)
        self.assertTrue(torch.equal(diagnostics["total_plans"], torch.full((2,), 3)))
        self.assertTrue(
            torch.all(diagnostics["converged_plans"] <= diagnostics["total_plans"])
        )
        self.assertTrue(torch.all(diagnostics["iterations"] == 1))
        self.assertNotIn("plans", result)

    def test_transport_diagnostics_match_each_group_marginal_residuals(self):
        model = self.make_model(
            transport={"strict": False, "max_iter": 1, "return_plans": True}
        )
        result = model(torch.randn(2, 3, 4, 4), class_names=["cat", "red bird", "dog"])
        expected_rows = torch.zeros(2)
        expected_columns = torch.zeros(2)
        expected_converged = torch.zeros(2, dtype=torch.long)
        for group in result["plans"]:
            info = group["diagnostics"]
            expected_rows = torch.maximum(
                expected_rows, info["row_residual"].amax(-1).float()
            )
            expected_columns = torch.maximum(
                expected_columns, info["col_residual"].amax(-1).float()
            )
            expected_converged += (
                torch.maximum(info["row_residual"], info["col_residual"]) <= 1e-6
            ).sum(-1)
        actual = result["transport_diagnostics"]
        self.assertTrue(torch.allclose(actual["row_residual"], expected_rows))
        self.assertTrue(torch.allclose(actual["col_residual"], expected_columns))
        self.assertTrue(torch.equal(actual["converged_plans"], expected_converged))

    def test_omitted_strict_flag_reports_finite_budget_nonconvergence(self):
        config = configuration()
        del config["transport"]["strict"]
        config["transport"].update({"max_iter": 1, "tol": 1e-20})
        model = module.PLIKD(TinyCLIP(), tokenize, ["cat", "dog"], config)
        result = model(torch.randn(2, 3, 4, 4))
        self.assertTrue(torch.isfinite(result["logits"]).all())
        self.assertTrue(
            torch.any(result["transport_diagnostics"]["converged_plans"] < 2)
        )


if __name__ == "__main__":
    unittest.main()
