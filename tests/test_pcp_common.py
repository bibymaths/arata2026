"""Tests for pcp_common module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pcp_common import (
    PCPMetrics,
    PCPParameters,
    adj_amount,
    adj_conc,
    compute_metrics,
    grow_seed,
    initialize_unidirectional_state,
    non_local,
    prepare_output_dir,
    run_until_convergence,
    save_csv,
    seed_x,
    seed_y,
    sum2,
)

# ---------------------------------------------------------------------------
# PCPParameters
# ---------------------------------------------------------------------------


class TestPCPParameters:
    def test_defaults(self):
        p = PCPParameters()
        assert p.dt == 0.01
        assert p.Xmax == 24
        assert p.Ymax == 24
        assert p.Bonds == 6

    def test_cell_n(self):
        p = PCPParameters(Xmax=4, Ymax=6)
        assert p.cell_N == 24

    def test_custom_values(self):
        p = PCPParameters(dt=0.001, rho=0.5, mu=2.0)
        assert p.dt == 0.001
        assert p.rho == 0.5
        assert p.mu == 2.0


# ---------------------------------------------------------------------------
# PCPMetrics
# ---------------------------------------------------------------------------


class TestPCPMetrics:
    def test_construct(self):
        m = PCPMetrics(
            cv_all=0.1,
            cv_high=0.2,
            cv_low=0.3,
            mean_del_angle_all=0.4,
            mean_del_angle_high=0.5,
            mean_del_angle_low=0.6,
            mean_angle_all=0.7,
            mean_angle_high=0.8,
            mean_angle_low=0.9,
            mean_magnitude_all=1.0,
            mean_magnitude_high=1.1,
            mean_magnitude_low=1.2,
            n_high=10.0,
        )
        assert m.cv_all == 0.1
        assert m.n_high == 10.0


# ---------------------------------------------------------------------------
# sum2
# ---------------------------------------------------------------------------


class TestSum2:
    def test_zero(self):
        arr = np.zeros((3, 3))
        assert sum2(arr) == 0.0

    def test_ones(self):
        arr = np.ones((4, 5))
        assert sum2(arr) == 20.0

    def test_values(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert sum2(arr) == 10.0


# ---------------------------------------------------------------------------
# seed_x
# ---------------------------------------------------------------------------


class TestSeedX:
    def test_shape_preserved(self):
        mtemp = np.zeros((4, 4))
        result = seed_x(mtemp, 4, 4)
        assert result.shape == (4, 4)

    def test_all_zeros(self):
        mtemp = np.zeros((4, 4))
        result = seed_x(mtemp, 4, 4)
        np.testing.assert_array_equal(result, np.zeros((4, 4)))

    def test_all_ones(self):
        mtemp = np.ones((4, 4))
        result = seed_x(mtemp, 4, 4)
        np.testing.assert_array_equal(result, np.ones((4, 4)))


# ---------------------------------------------------------------------------
# seed_y
# ---------------------------------------------------------------------------


class TestSeedY:
    def test_shape_preserved(self):
        mtemp = np.zeros((4, 4))
        result = seed_y(mtemp, 4, 4)
        assert result.shape == (4, 4)

    def test_all_zeros(self):
        mtemp = np.zeros((4, 4))
        result = seed_y(mtemp, 4, 4)
        np.testing.assert_array_equal(result, np.zeros((4, 4)))

    def test_all_ones(self):
        mtemp = np.ones((4, 4))
        result = seed_y(mtemp, 4, 4)
        np.testing.assert_array_equal(result, np.ones((4, 4)))


# ---------------------------------------------------------------------------
# grow_seed
# ---------------------------------------------------------------------------


class TestGrowSeed:
    def test_shape_preserved(self):
        mtemp = np.zeros((4, 4))
        result = grow_seed(mtemp, 4, 4)
        assert result.shape == (4, 4)

    def test_all_ones_stays_ones(self):
        # A fully-filled grid: each cell plus its 6 neighbours sums to 7 → stays 1
        mtemp = np.ones((6, 6))
        result = grow_seed(mtemp, 6, 6)
        np.testing.assert_array_equal(result, np.ones((6, 6)))

    def test_all_zeros_stays_zeros(self):
        mtemp = np.zeros((6, 6))
        result = grow_seed(mtemp, 6, 6)
        np.testing.assert_array_equal(result, np.zeros((6, 6)))

    def test_output_binary(self):
        rng = np.random.default_rng(42)
        mtemp = (rng.random((8, 8)) > 0.5).astype(np.float64)
        result = grow_seed(mtemp, 8, 8)
        unique = np.unique(result)
        assert set(unique).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# adj_amount / adj_conc
# ---------------------------------------------------------------------------


class TestAdjAmount:
    def test_shape_preserved(self):
        mtemp = np.zeros((4, 4, 6))
        result = adj_amount(mtemp, 4, 4, 6)
        assert result.shape == (4, 4, 6)

    def test_alias(self):
        assert adj_conc is adj_amount

    def test_all_zeros(self):
        mtemp = np.zeros((4, 4, 6))
        result = adj_amount(mtemp, 4, 4, 6)
        np.testing.assert_array_equal(result, mtemp)


# ---------------------------------------------------------------------------
# non_local
# ---------------------------------------------------------------------------


class TestNonLocal:
    def test_shape_preserved(self):
        mtemp = np.ones((4, 4, 6))
        result = non_local(mtemp, 6, 1.0)
        assert result.shape == (4, 4, 6)

    def test_zero_da(self):
        # With d_a=0, non_local(x, 6, 0) == x (each element contributes 0*neighbours + itself)
        mtemp = np.arange(4 * 4 * 6, dtype=np.float64).reshape(4, 4, 6)
        result = non_local(mtemp, 6, 0.0)
        np.testing.assert_array_equal(result, mtemp)


# ---------------------------------------------------------------------------
# initialize_unidirectional_state
# ---------------------------------------------------------------------------


class TestInitializeUnidirectionalState:
    def setup_method(self):
        self.params = PCPParameters(Xmax=4, Ymax=4)
        self.rng = np.random.default_rng(0)

    def test_shapes(self):
        high = np.ones((4, 4))
        fzdmem, vanglmem, fzdint, vanglint = initialize_unidirectional_state(
            high, self.params, self.rng
        )
        assert fzdmem.shape == (4, 4, 6)
        assert vanglmem.shape == (4, 4, 6)
        assert fzdint.shape == (4, 4)
        assert vanglint.shape == (4, 4)

    def test_zero_high_gives_zero_state(self):
        high = np.zeros((4, 4))
        fzdmem, vanglmem, fzdint, vanglint = initialize_unidirectional_state(
            high, self.params, self.rng
        )
        np.testing.assert_array_equal(fzdmem, np.zeros((4, 4, 6)))
        np.testing.assert_array_equal(vanglmem, np.zeros((4, 4, 6)))
        np.testing.assert_array_equal(fzdint, np.zeros((4, 4)))
        np.testing.assert_array_equal(vanglint, np.zeros((4, 4)))

    def test_non_negative(self):
        high = np.ones((4, 4))
        fzdmem, vanglmem, fzdint, vanglint = initialize_unidirectional_state(
            high, self.params, self.rng
        )
        assert np.all(fzdmem >= 0)
        assert np.all(vanglmem >= 0)


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def setup_method(self):
        self.params = PCPParameters(Xmax=4, Ymax=4)

    def test_returns_pcpmetrics(self):
        fzdmem = np.ones((4, 4, 6))
        high_calc = np.ones((4, 4))
        result = compute_metrics(fzdmem, high_calc, self.params)
        assert isinstance(result, PCPMetrics)

    def test_cv_zero_for_uniform(self):
        # All cells point in same direction → CV should be 0
        fzdmem = np.zeros((4, 4, 6))
        fzdmem[:, :, 2] = 1.0  # All pointing in bond-2 direction
        high_calc = np.ones((4, 4))
        result = compute_metrics(fzdmem, high_calc, self.params)
        assert result.cv_all == pytest.approx(0.0, abs=1e-9)

    def test_n_high_all_high(self):
        fzdmem = np.ones((4, 4, 6))
        high_calc = np.ones((4, 4))
        result = compute_metrics(fzdmem, high_calc, self.params)
        assert result.n_high == 16.0

    def test_n_high_none_high(self):
        fzdmem = np.ones((4, 4, 6))
        high_calc = np.zeros((4, 4))
        result = compute_metrics(fzdmem, high_calc, self.params)
        assert result.n_high == 0.0


# ---------------------------------------------------------------------------
# run_until_convergence (small grid, few steps)
# ---------------------------------------------------------------------------


class TestRunUntilConvergence:
    def test_output_shapes_and_convergence(self):
        params = PCPParameters(Xmax=4, Ymax=4, thr=1.0)  # large thr → converges fast
        rng = np.random.default_rng(1)
        high = np.ones((4, 4))
        fzdmem, vanglmem, fzdint, vanglint = initialize_unidirectional_state(high, params, rng)
        fzdmem_out, vanglmem_out, fzdint_out, vanglint_out, t = run_until_convergence(
            fzdmem, vanglmem, fzdint, vanglint, params
        )
        assert fzdmem_out.shape == (4, 4, 6)
        assert vanglmem_out.shape == (4, 4, 6)
        assert fzdint_out.shape == (4, 4)
        assert vanglint_out.shape == (4, 4)
        assert t >= 1


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


class TestPrepareOutputDir:
    def test_creates_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = prepare_output_dir(Path(tmp), "myrun")
            assert (out / "Summary").exists()
            assert (out / "Data").exists()

    def test_copies_source_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "script.py"
            src.write_text("# hello")
            out = prepare_output_dir(Path(tmp), "myrun", source_script=src)
            assert (out / "script.py").exists()

    def test_no_source_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = prepare_output_dir(Path(tmp), "myrun", source_script=None)
            assert out.exists()


class TestSaveCsv:
    def test_saves_and_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            arr = np.array([[1.0, 2.0], [3.0, 4.0]])
            save_csv(path, arr)
            loaded = np.loadtxt(path, delimiter=",")
            np.testing.assert_allclose(loaded, arr)

    def test_single_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "row.csv"
            arr = np.array([[1.5, 2.5, 3.5]])
            save_csv(path, arr)
            loaded = np.loadtxt(path, delimiter=",")
            np.testing.assert_allclose(loaded, arr.squeeze())
