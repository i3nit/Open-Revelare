"""Guard the audit against apparently good numbers caused by ties, wrong support or empty evidence."""

import unittest
import numpy as np
from detect import evidence
from evaluate import audit, top_weights


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.support = np.ones((20, 20), bool)
        upper = np.zeros_like(self.support)
        lower = upper.copy()
        upper[2:5, :] = True
        lower[15:17, :] = True
        self.regions = {"upper": upper, "lower": lower}

    def test_uniform_scores_are_chance_even_with_many_ties(self):
        report = audit(np.zeros((20, 20), np.float32), self.support, self.regions)
        for row in report["rank_budgets"]:
            self.assertAlmostEqual(row["enrichment"], 1, places=5)
            self.assertAlmostEqual(
                row["upper_selected_fraction"], row["budget"], places=5
            )
            self.assertAlmostEqual(
                row["lower_selected_fraction"], row["budget"], places=5
            )

    def test_localising_only_upper_band_does_not_hide_the_missing_lower_band(self):
        score = self.regions["upper"].astype(np.float32)
        report = audit(score, self.support, self.regions)["rank_budgets"][1]
        self.assertEqual(report["lower_selected_fraction"], 0)
        self.assertAlmostEqual(report["upper_selected_fraction"], 1, places=5)

    def test_evaluation_never_selects_outside_the_photographic_support(self):
        support = self.support.copy()
        support[:10] = False
        score = np.where(support, 1, 100).astype(np.float32)
        weights, _ = top_weights(score, support, 0.15)
        self.assertEqual(float(weights[~support].sum()), 0)
        self.assertAlmostEqual(float(weights.sum()), support.sum() * 0.15, places=4)

    def test_flat_source_does_not_create_periodicity_evidence(self):
        maps = evidence(np.full((64, 80, 3), 0.3, np.float32))
        self.assertLess(float(maps["periodicity"].max()), 1e-5)
        self.assertEqual(float(maps["phase_chromatic"].max()), 0)

    def test_colour_periodicity_is_evidence_for_a_pattern_not_its_physical_origin(self):
        y, x = np.mgrid[:96, :112]
        image = np.stack(
            [
                0.3 * np.exp(0.07 * np.sin(y * 2 * np.pi / 8 + c * 1.5))
                for c in range(3)
            ],
            axis=2,
        ).astype(np.float32)
        maps = evidence(image)
        self.assertGreater(float(maps["phase_chromatic"][24:-24, 24:-24].mean()), 0.5)
        # Exactly this RGB array could also depict a printed pattern; this test intentionally
        # verifies a periodic feature, never awards semantic Newton-ring classification.
        self.assertTrue(np.isfinite(maps["phase_chromatic"]).all())


if __name__ == "__main__":
    unittest.main()
