"""Synthetic/source checks only: no real cache, fit, or validation inference."""
import ast
import inspect
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import final_month_neural as subject


class FinalMonthNeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.shared, cls.chunked, cls.physical, cls.tabm, _ = subject._modules()

    def test_actual_architectures_have_expected_outputs(self):
        cards, bins, plain = [3, 5], [torch.tensor([-1., 0., 1., 2.])] * 3, 5
        state = dict(cards=cards, bins=bins)
        for family in subject.FAMILIES:
            width = plain + (12 if family == "public_tabm" else 0)
            model = subject._make(family, state, width, self.shared, self.physical, self.tabm, "cpu").eval()
            tab, track = self.physical.parent.synthetic(cards, bins, width, 6, "cpu")
            with torch.no_grad():
                result = model(*tab) if family == "calls_ple" else model(*tab, *track)
            self.assertEqual(result.shape, (6, 32 if family == "public_tabm" else 1))
            self.assertTrue(torch.isfinite(result).all())

    def test_partial_microbatch_preserves_headwise_loss_gradients(self):
        # No optimizer step: validate the accumulation arithmetic independently.
        full, split = torch.nn.Linear(5, 32), torch.nn.Linear(5, 32)
        split.load_state_dict(full.state_dict())
        x, y = torch.arange(35.).reshape(7, 5) / 17, torch.linspace(-1, 1, 7)
        subject._loss(full(x), y).backward()
        for i in range(0, 7, 3):
            subject._loss(split(x[i:i+3]), y[i:i+3]).mul(len(x[i:i+3]) / 7).backward()
        for a, b in zip(full.parameters(), split.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-7, rtol=1e-6)
        self.assertEqual(subject._loss(torch.tensor([[-1., 1.]]), torch.zeros(1)).item(), 1.)

    def test_encoder_keeps_held_categories_and_values_out_of_state(self):
        train = pd.DataFrame(dict(category=pd.Categorical(["train"] * 40), value=np.arange(40.)))
        held = pd.DataFrame(dict(category=pd.Categorical(["unseen"]), value=[1e9]))
        encoder = self.shared.nnj.Encoder(train)
        before = dict(maps={k: dict(v) for k, v in encoder.maps.items()}, scale=dict(encoder.scale))
        categories, numeric = encoder.transform(held)
        self.assertEqual(categories[0, 0], 0)
        self.assertTrue(np.isfinite(numeric).all())
        self.assertEqual(encoder.maps, before["maps"])
        self.assertEqual(encoder.scale, before["scale"])
        self.assertEqual(encoder.scale["value"][0], 19.5)

    def test_module_has_no_execution_entrypoint_or_held_target_access(self):
        tree = ast.parse(Path(subject.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                self.assertNotIn(node.slice.value, ["raw", "yv", "held_labels"])
        self.assertNotIn("__main__", Path(subject.__file__).read_text())

    def test_source_frozen_recipe_matches_original_asserted_n09(self):
        tree = ast.parse(inspect.getsource(self.shared.recipe))
        assertion = next(n for n in ast.walk(tree) if isinstance(n, ast.Assert))
        specification = assertion.test.comparators[0]
        expected = {keyword.arg: ast.literal_eval(keyword.value) for keyword in specification.keywords}
        self.assertEqual(subject.N09_CONFIG, expected)

    def test_prediction_chunks_restore_backend_and_reconstruct_seconds(self):
        class NumericMean(torch.nn.Module):
            def forward(self, cat, g, plain):
                return g

        g = torch.linspace(-3, 3, 513).reshape(-1, 1)
        values = (torch.zeros((513, 1), dtype=torch.int64), g, torch.zeros((513, 1)))
        flags = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        actual = subject._prediction(NumericMean(), values, dict(sd=3., mu=2.), None, None)
        np.testing.assert_array_equal(actual, np.maximum(g.numpy().ravel().astype(float) * 3 + 2, 0))
        self.assertEqual(flags, (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark))


if __name__ == "__main__":
    unittest.main()
