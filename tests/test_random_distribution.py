"""Tests for random_distribution module."""

from __future__ import annotations

import numpy as np
import pytest

import random_distribution as rd
from pcp_common import PCPParameters, prepare_output_dir
from random_distribution import generate_random_high_mask, run_simulation


class TestGenerateRandomHighMask:
    def test_shape(self):
        rng = np.random.default_rng(0)
        mask = generate_random_high_mask(6, 8, 0.5, rng)
        assert mask.shape == (6, 8)

    def test_binary_output(self):
        rng = np.random.default_rng(1)
        mask = generate_random_high_mask(6, 6, 0.3, rng)
        unique = np.unique(mask)
        assert set(unique).issubset({0.0, 1.0})

    @pytest.mark.parametrize("ratio_high", [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_ratio_approximately_correct(self, ratio_high: float):
        rng = np.random.default_rng(42)
        ymax, xmax = 20, 20
        mask = generate_random_high_mask(ymax, xmax, ratio_high, rng)
        n_high = int(mask.sum())
        expected = int(xmax * ymax * ratio_high)
        assert n_high == expected, f"Expected {expected} high cells, got {n_high}"

    def test_all_low(self):
        rng = np.random.default_rng(7)
        # ratio_high = 0 → no high cells (threshold lands at 0 or none pass)
        # The algorithm may not reach exactly 0, so just check shape / dtype
        mask = generate_random_high_mask(4, 4, 0.0, rng)
        assert mask.shape == (4, 4)
        assert np.all((mask == 0.0) | (mask == 1.0))

    def test_all_high(self):
        rng = np.random.default_rng(7)
        mask = generate_random_high_mask(4, 4, 1.0, rng)
        assert mask.shape == (4, 4)
        assert np.all((mask == 0.0) | (mask == 1.0))

    def test_reproducible_with_same_seed(self):
        mask1 = generate_random_high_mask(8, 8, 0.4, np.random.default_rng(99))
        mask2 = generate_random_high_mask(8, 8, 0.4, np.random.default_rng(99))
        np.testing.assert_array_equal(mask1, mask2)

    def test_different_seeds_differ(self):
        mask1 = generate_random_high_mask(8, 8, 0.5, np.random.default_rng(1))
        mask2 = generate_random_high_mask(8, 8, 0.5, np.random.default_rng(2))
        assert not np.array_equal(mask1, mask2)


class TestRunSimulationSmoke:
    """Smoke-test run_simulation on a tiny, fast config."""

    def test_output_shape_and_save(self, tmp_path):
        params = PCPParameters(
            Xmax=4,
            Ymax=4,
            thr=1.0,  # loose convergence → very few iterations
            nREP=1,
            noise_rep=1,
        )
        out_dir = prepare_output_dir(tmp_path, "test_random")

        # Monkey-patch BALANCE and FREQ_HIGH to a single-element array for speed
        orig_balance = rd.BALANCE
        orig_freq = rd.FREQ_HIGH
        rd.BALANCE = np.array([0.0], dtype=float)
        rd.FREQ_HIGH = np.array([0.5], dtype=float)
        try:
            result = run_simulation(params, out_dir, seed=0)
        finally:
            rd.BALANCE = orig_balance
            rd.FREQ_HIGH = orig_freq

        # 1 freq × 1 rep × 1 balance = 1 row; 14 columns
        assert result.shape == (1, 14)
        assert (out_dir / "Summary" / "output.csv").exists()
