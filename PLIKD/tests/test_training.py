import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_model import TinyCLIP, configuration, tokenize

import model as model_module
import train as training


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = model_module.PLIKD(
            TinyCLIP(), tokenize, ["cat", "dog"], configuration()
        )
        self.config = configuration()
        self.config["training"] = {
            "learning_rate": 0.002,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "epochs": 5,
            "warmup_epochs": 1,
            "warmup_lr": 1e-5,
            "seed": 1,
        }

    def test_optimizer_contains_only_learnable_prompt_and_projector_parameters(self):
        optimizer = training.build_optimizer(self.model, self.config)
        expected = {
            id(parameter)
            for parameter in self.model.parameters()
            if parameter.requires_grad
        }
        actual = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(actual, expected)
        self.assertEqual(optimizer.defaults["momentum"], 0.9)

    def test_epoch_schedule_has_constant_warmup_and_cosine_decay(self):
        values = [
            training.epoch_learning_rate(epoch, 0.002, 5, 1, 1e-5) for epoch in range(5)
        ]
        self.assertEqual(values[0], 1e-5)
        self.assertAlmostEqual(values[1], 0.002)
        self.assertEqual(values[1:], sorted(values[1:], reverse=True))
        self.assertAlmostEqual(values[-1], 0.0)

    def test_single_epoch_budget_can_consist_of_warmup(self):
        self.assertEqual(training.epoch_learning_rate(0, 0.002, 1, 1, 1e-5), 1e-5)

    def test_checkpoint_roundtrip_saves_adapter_not_frozen_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"
            optimizer = training.build_optimizer(self.model, self.config)
            scheduler = training.build_scheduler(optimizer, self.config)
            expected = self.model.context.detach().clone()
            training.save_checkpoint(
                path,
                self.model,
                self.config,
                optimizer,
                scheduler,
                epoch=2,
                best_value=0.4,
                history=[{"epoch": 2}],
            )
            with torch.no_grad():
                self.model.context.add_(1)
            payload = training.load_checkpoint(
                path, self.model, optimizer, scheduler, expected_config=self.config
            )
            self.assertEqual(payload["epoch"], 2)
            self.assertTrue(torch.equal(self.model.context, expected))
            self.assertFalse(
                any(name.startswith("clip_model.") for name in payload["model"])
            )
            self.assertEqual(payload["class_names"], ["cat", "dog"])
            self.assertEqual(len(payload["frozen_clip_sha256"]), 64)

    def test_checkpoint_rejects_different_frozen_clip_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"
            training.save_checkpoint(path, self.model, self.config)
            changed = model_module.PLIKD(
                TinyCLIP(), tokenize, ["cat", "dog"], configuration()
            )
            with self.assertRaisesRegex(ValueError, "CLIP|fingerprint"):
                training.load_checkpoint(path, changed)

    def test_resume_rejects_changed_training_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"
            training.save_checkpoint(path, self.model, self.config)
            changed = {
                **self.config,
                "training": {**self.config["training"], "learning_rate": 0.1},
            }
            with self.assertRaisesRegex(ValueError, "config|protocol"):
                training.load_checkpoint(path, self.model, expected_config=changed)

    def test_loader_epoch_seed_matches_restarted_epoch_without_training(self):
        def loader():
            return torch.utils.data.DataLoader(
                torch.arange(32),
                batch_size=4,
                shuffle=True,
                num_workers=0,
                generator=torch.Generator().manual_seed(9),
            )

        continued = loader()
        training.seed_loader_epoch(continued, 9, 0)
        first = torch.cat(list(continued))
        training.seed_loader_epoch(continued, 9, 1)
        second = torch.cat(list(continued))
        restarted = loader()
        training.seed_loader_epoch(restarted, 9, 1)
        resumed = torch.cat(list(restarted))
        self.assertTrue(torch.equal(second, resumed))
        self.assertFalse(torch.equal(first, second))

    def test_loader_epoch_seed_also_sets_distributed_sampler_epoch(self):
        sampler = SimpleNamespace(set_epoch=lambda epoch: calls.append(epoch))
        calls = []
        loader = SimpleNamespace(
            generator=torch.Generator(), sampler=sampler, persistent_workers=False
        )
        self.assertEqual(training.seed_loader_epoch(loader, 4, 3), 7)
        self.assertEqual(calls, [3])

    def test_epoch_timer_is_optional_and_does_not_sync_when_disabled(self):
        with patch.object(training, "synchronize_device") as synchronize:
            timer = training.EpochTimer(torch.device("cpu"), enabled=False)
            timer.start()
            self.assertIsNone(timer.stop())
            synchronize.assert_not_called()

    def test_epoch_timer_synchronizes_and_labels_only_training_batch_scope(self):
        with (
            patch.object(training, "synchronize_device") as synchronize,
            patch.object(training.time, "perf_counter", side_effect=[10.0, 12.5]),
        ):
            timer = training.EpochTimer(torch.device("cpu"), enabled=True)
            timer.start()
            result = timer.stop()
            self.assertEqual(result["train_seconds_per_epoch"], 2.5)
            self.assertEqual(result["scope"], "epoch_train_batches")
            self.assertEqual(result["device"], "cpu")
            self.assertFalse(result["includes_validation"])
            self.assertFalse(result["includes_knowledge_extraction"])
            self.assertEqual(synchronize.call_count, 2)

    def test_training_transport_summary_preserves_maxima_and_counts(self):
        first = {
            "row_residual": torch.tensor([0.001, 0.002]),
            "col_residual": torch.tensor([0.004, 0.003]),
            "converged_plans": torch.tensor([1, 2]),
            "total_plans": torch.tensor([2, 2]),
            "iterations": torch.tensor([100, 20]),
        }
        second = {
            "row_residual": torch.tensor([0.005]),
            "col_residual": torch.tensor([0.001]),
            "converged_plans": torch.tensor([1]),
            "total_plans": torch.tensor([3]),
            "iterations": torch.tensor([50]),
        }
        result = training.accumulate_transport_diagnostics(None, first)
        result = training.accumulate_transport_diagnostics(result, second)
        self.assertAlmostEqual(result["row_residual"], 0.005)
        self.assertAlmostEqual(result["col_residual"], 0.004)
        self.assertEqual(result["converged_plans"], 4)
        self.assertEqual(result["total_plans"], 7)
        self.assertEqual(result["iterations"], 100)
        self.assertEqual(result["examples"], 3)

    def test_mps_rng_state_is_captured_and_restored_without_model_execution(self):
        state = torch.tensor([1, 2, 3], dtype=torch.uint8)
        with (
            patch.object(torch.backends.mps, "is_available", return_value=True),
            patch.object(torch.mps, "get_rng_state", return_value=state),
            patch.object(torch.mps, "set_rng_state") as restore,
        ):
            saved = training._rng_state()
            self.assertTrue(torch.equal(saved["mps"], state))
            training._restore_rng(saved)
            restore.assert_called_once_with(state)

    def test_mps_rng_restore_rejects_unavailable_backend(self):
        state = training._rng_state()
        state["mps"] = torch.tensor([1, 2, 3], dtype=torch.uint8)
        with (
            patch.object(torch.backends.mps, "is_available", return_value=False),
            self.assertRaisesRegex(ValueError, "MPS"),
        ):
            training._restore_rng(state)

    def test_timer_device_sync_dispatches_to_cuda_and_mps_apis(self):
        with patch.object(torch.cuda, "synchronize") as cuda_sync:
            training.synchronize_device(torch.device("cuda:0"))
            cuda_sync.assert_called_once_with(torch.device("cuda:0"))
        with patch.object(torch.mps, "synchronize") as mps_sync:
            training.synchronize_device(torch.device("mps"))
            mps_sync.assert_called_once_with()

    def test_loader_reseed_rejects_persistent_augmentation_workers(self):
        loader = SimpleNamespace(
            generator=torch.Generator(), sampler=None, persistent_workers=True
        )
        with self.assertRaisesRegex(ValueError, "persistent_workers"):
            training.seed_loader_epoch(loader, 4, 3)


if __name__ == "__main__":
    unittest.main()
