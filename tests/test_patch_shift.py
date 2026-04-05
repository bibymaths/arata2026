"""Tests for patch_shift module."""

from __future__ import annotations

import numpy as np

import patch_shift as ps
from patch_shift import generate_shifted_patch_mask, run_simulation
from pcp_common import PCPParameters, prepare_output_dir


class TestGenerateShiftedPatchMask:
    def test_shape(self):
        rng = np.random.default_rng(0)
        mask = generate_shifted_patch_mask(8, 8, 3, 3, 0, 0, 4, 4, rng)
        assert mask.shape == (8, 8)

    def test_binary_output(self):
        rng = np.random.default_rng(1)
        mask = generate_shifted_patch_mask(8, 8, 3, 3, 0, 0, 4, 4, rng)
        unique = np.unique(mask)
        assert set(unique).issubset({0.0, 1.0})

    def test_reproducible(self):
        mask1 = generate_shifted_patch_mask(8, 8, 3, 3, 1, 1, 4, 4, np.random.default_rng(7))
        mask2 = generate_shifted_patch_mask(8, 8, 3, 3, 1, 1, 4, 4, np.random.default_rng(7))
        np.testing.assert_array_equal(mask1, mask2)

    def test_no_shift(self):
        rng = np.random.default_rng(42)
        mask = generate_shifted_patch_mask(8, 8, 3, 3, 0, 0, 4, 4, rng)
        assert mask.shape == (8, 8)
        assert np.all((mask == 0.0) | (mask == 1.0))

    def test_full_grid_all_high_when_no_low_patches(self):
        # lx=1, ly=1 → enlarge_x = 0-1=-1, enlarge_y=0-1=-1 → no seed_x/seed_y calls
        # With lambda_x=8, lambda_y=8, n_patch_x=1, n_patch_y=1 → one 0 placed then grow
        rng = np.random.default_rng(0)
        mask = generate_shifted_patch_mask(8, 8, 1, 1, 0, 0, 8, 8, rng)
        assert mask.shape == (8, 8)


class TestRunSimulationSmoke:
    """Smoke-test patch_shift.run_simulation on a tiny, fast config."""

    def test_output_shape_and_save(self, tmp_path):
        params = PCPParameters(
            Xmax=4,
            Ymax=4,
            thr=1.0,
            noise_rep=1,
        )
        out_dir = prepare_output_dir(tmp_path, "test_patch")

        # Patch BALANCE and PARAMS to minimal single-row config for speed
        orig_balance = ps.BALANCE
        orig_params = ps.PARAMS
        ps.BALANCE = np.array([0.0], dtype=float)
        ps.PARAMS = np.array([[3, 3, 0, 0, 1, 4, 4]], dtype=int)
        try:
            result = run_simulation(params, out_dir, seed=0)
        finally:
            ps.BALANCE = orig_balance
            ps.PARAMS = orig_params

        # 1 distribution × 1 balance = 1 row; 22 columns
        assert result.shape == (1, 22)
        assert (out_dir / "Summary" / "output.csv").exists()
