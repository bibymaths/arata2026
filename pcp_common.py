"""Common utilities for implementing the PCP Octave simulations in Python.

Credits
-------
Mathematical model and biological study:
    Masaki Arata, Hiroshi Koyama, Toshihiko Fujimori.
    "Directional alignment of different cell types organizes planar cell polarity"
    iScience 29, 114771 (2026). DOI: 10.1016/j.isci.2026.114771.

Notes
-----
This module preserves the original public API while providing optional
Numba acceleration for the numerical kernels. If Numba is unavailable,
the code falls back to pure NumPy/Python automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import shutil

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func
        return decorator


# -----------------------------------------------------------------------------
# Public dataclasses
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class PCPParameters:
    dt: float = 0.01
    rho: float = 0.1
    mu: float = 1.0
    phai: float = 1.0
    omega: float = 10.0
    Da: float = 1.0
    Di: float = 1.0
    Xmax: int = 24
    Ymax: int = 24
    Bonds: int = 6
    Total_protein: float = 2.0
    thr: float = 1e-4
    nREP: int = 10
    noise_rep: int = 1
    noise: float = 0.6e-10

    @property
    def cell_N(self) -> int:
        return self.Xmax * self.Ymax


@dataclass(slots=True)
class PCPMetrics:
    cv_all: float
    cv_high: float
    cv_low: float
    mean_del_angle_all: float
    mean_del_angle_high: float
    mean_del_angle_low: float
    mean_angle_all: float
    mean_angle_high: float
    mean_angle_low: float
    mean_magnitude_all: float
    mean_magnitude_high: float
    mean_magnitude_low: float
    n_high: float


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

EDGE_VAL = np.cos(np.pi / 6.0)
EDGE_ANGLES_X = np.array(
    [[[-EDGE_VAL, 0.0, EDGE_VAL, EDGE_VAL, 0.0, -EDGE_VAL]]],
    dtype=np.float64,
)
EDGE_ANGLES_Y = np.array(
    [[[0.5, 1.0, 0.5, -0.5, -1.0, -0.5]]],
    dtype=np.float64,
)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def sum2(arr: np.ndarray) -> float:
    return float(np.sum(arr))


@njit(cache=True, nogil=True)
def _sum2_jit(arr: np.ndarray) -> float:
    total = 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            total += arr[i, j]
    return total


# -----------------------------------------------------------------------------
# Periodic adjacency / neighborhood helpers
# -----------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def _seed_x_jit(mtemp: np.ndarray) -> np.ndarray:
    y_max, x_max = mtemp.shape
    out = np.empty((y_max, x_max), dtype=np.float64)
    for y in range(y_max):
        for x in range(x_max):
            src_x = x - 2
            if src_x < 0:
                src_x += x_max
            out[y, x] = 1.0 if (mtemp[y, x] + mtemp[y, src_x]) >= 2.0 else 0.0
    return out


def seed_x(mtemp: np.ndarray, x_max: int, y_max: int) -> np.ndarray:
    del x_max, y_max
    return _seed_x_jit(np.asarray(mtemp, dtype=np.float64))


@njit(cache=True, nogil=True)
def _seed_y_jit(mtemp: np.ndarray) -> np.ndarray:
    y_max, x_max = mtemp.shape
    out = np.empty((y_max, x_max), dtype=np.float64)
    for y in range(y_max):
        src_y = y - 2
        if src_y < 0:
            src_y += y_max
        for x in range(x_max):
            out[y, x] = 1.0 if (mtemp[y, x] + mtemp[src_y, x]) >= 2.0 else 0.0
    return out


def seed_y(mtemp: np.ndarray, x_max: int, y_max: int) -> np.ndarray:
    del x_max, y_max
    return _seed_y_jit(np.asarray(mtemp, dtype=np.float64))


@njit(cache=True, nogil=True)
def _grow_seed_jit(mtemp: np.ndarray) -> np.ndarray:
    y_max, x_max = mtemp.shape
    r = np.zeros((y_max, x_max), dtype=np.float64)

    for y in range(y_max):
        ym1 = y - 1 if y > 0 else y_max - 1
        yp1 = y + 1 if y < y_max - 1 else 0

        for x in range(x_max):
            xm1 = x - 1 if x > 0 else x_max - 1
            xp1 = x + 1 if x < x_max - 1 else 0

            total = 0.0

            # up/down
            total += mtemp[ym1, x]
            total += mtemp[yp1, x]

            # parity-dependent hex neighbors
            if x % 2 == 0:
                total += mtemp[y, xm1]
                total += mtemp[y, xp1]
                total += mtemp[ym1, xm1]
                total += mtemp[ym1, xp1]
            else:
                total += mtemp[y, xm1]
                total += mtemp[y, xp1]
                total += mtemp[yp1, xm1]
                total += mtemp[yp1, xp1]

            # original code uses (r + mtemp >= 7)
            out_val = total + mtemp[y, x]
            r[y, x] = 1.0 if out_val >= 7.0 else 0.0

    return r


def grow_seed(mtemp: np.ndarray, y_max: int, x_max: int) -> np.ndarray:
    del y_max, x_max
    return _grow_seed_jit(np.asarray(mtemp, dtype=np.float64))


@njit(cache=True, nogil=True)
def _adj_amount_jit(mtemp: np.ndarray) -> np.ndarray:
    y_max, x_max, bonds = mtemp.shape
    r = np.empty((y_max, x_max, bonds), dtype=np.float64)

    # start from copy
    for y in range(y_max):
        for x in range(x_max):
            for b in range(bonds):
                r[y, x, b] = mtemp[y, x, b]

    # bond 5 <- shifted bond 2 upward
    for y in range(y_max):
        src_y = y + 1 if y < y_max - 1 else 0
        for x in range(x_max):
            r[y, x, 4] = mtemp[src_y, x, 1]

    # bond 2 <- shifted bond 5 downward
    for y in range(y_max):
        src_y = y - 1 if y > 0 else y_max - 1
        for x in range(x_max):
            r[y, x, 1] = mtemp[src_y, x, 4]

    # bond 4 from bond 1 of left neighbor, with parity adjustment
    for y in range(y_max):
        ym1 = y - 1 if y > 0 else y_max - 1
        for x in range(x_max):
            xp1 = x + 1 if x < x_max - 1 else 0
            if x % 2 == 0:
                r[y, x, 3] = mtemp[y, xp1, 0]
            else:
                r[y, x, 3] = mtemp[ym1, xp1, 0]

    # bond 6 from bond 3 of right neighbor, with parity adjustment
    for y in range(y_max):
        ym1 = y - 1 if y > 0 else y_max - 1
        for x in range(x_max):
            xm1 = x - 1 if x > 0 else x_max - 1
            if x % 2 == 0:
                r[y, x, 5] = mtemp[y, xm1, 2]
            else:
                r[y, x, 5] = mtemp[ym1, xm1, 2]

    # bond 3 from bond 6 of left-ish neighbor, with parity adjustment
    for y in range(y_max):
        yp1 = y + 1 if y < y_max - 1 else 0
        for x in range(x_max):
            xp1 = x + 1 if x < x_max - 1 else 0
            if x % 2 == 0:
                r[y, x, 2] = mtemp[yp1, xp1, 5]
            else:
                r[y, x, 2] = mtemp[y, xp1, 5]

    # bond 1 from bond 4 of right-ish neighbor, with parity adjustment
    for y in range(y_max):
        yp1 = y + 1 if y < y_max - 1 else 0
        for x in range(x_max):
            xm1 = x - 1 if x > 0 else x_max - 1
            if x % 2 == 0:
                r[y, x, 0] = mtemp[yp1, xm1, 3]
            else:
                r[y, x, 0] = mtemp[y, xm1, 3]

    return r


def adj_amount(mtemp: np.ndarray, x_max: int, y_max: int, bonds: int) -> np.ndarray:
    del x_max, y_max, bonds
    return _adj_amount_jit(np.asarray(mtemp, dtype=np.float64))


adj_conc = adj_amount


@njit(cache=True, nogil=True)
def _non_local_jit(mtemp: np.ndarray, d_a: float) -> np.ndarray:
    y_max, x_max, bonds = mtemp.shape
    out = np.empty((y_max, x_max, bonds), dtype=np.float64)

    for y in range(y_max):
        for x in range(x_max):
            for b in range(bonds):
                left = b - 1 if b > 0 else bonds - 1
                right = b + 1 if b < bonds - 1 else 0
                out[y, x, b] = d_a * (mtemp[y, x, left] + mtemp[y, x, right]) + mtemp[y, x, b]

    return out


def non_local(mtemp: np.ndarray, bonds: int, d_a: float) -> np.ndarray:
    del bonds
    return _non_local_jit(np.asarray(mtemp, dtype=np.float64), float(d_a))


# -----------------------------------------------------------------------------
# Initial conditions / metrics
# -----------------------------------------------------------------------------

def initialize_unidirectional_state(
    high: np.ndarray,
    params: PCPParameters,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fzdmem = np.zeros((params.Ymax, params.Xmax, params.Bonds), dtype=np.float64)
    vanglmem = np.zeros_like(fzdmem)

    fzdmem[:, :, 2] = 0.6
    fzdmem[:, :, 3] = 0.6
    vanglmem[:, :, 0] = 0.6
    vanglmem[:, :, 5] = 0.6

    fzdmem += params.noise * rng.random(fzdmem.shape)
    vanglmem += params.noise * rng.random(vanglmem.shape)

    fzdint = high * (params.Total_protein - np.sum(fzdmem, axis=2))
    vanglint = high * (params.Total_protein - np.sum(vanglmem, axis=2))

    fzdmem = high[:, :, None] * fzdmem
    vanglmem = high[:, :, None] * vanglmem

    return fzdmem, vanglmem, fzdint, vanglint


@njit(cache=True, nogil=True)
def _compute_metrics_jit(
    fzdmem: np.ndarray,
    high_calc: np.ndarray,
    cell_n: int,
    divide_angles_as_in_script2: bool,
    normalize_magnitude_as_in_paper: bool,
) -> tuple[float, float, float, float, float, float, float, float, float, float, float, float, float]:
    y_max, x_max, bonds = fzdmem.shape

    allx = 0.0
    ally = 0.0
    highx = 0.0
    highy = 0.0
    lowx = 0.0
    lowy = 0.0
    n_high = 0.0

    sum_abs_angle_all = 0.0
    sum_abs_angle_high = 0.0
    sum_abs_angle_low = 0.0

    sum_mag_all = 0.0
    sum_mag_high = 0.0
    sum_mag_low = 0.0

    for y in range(y_max):
        for x in range(x_max):
            vx = 0.0
            vy = 0.0
            denom = 0.0
            for b in range(bonds):
                val = fzdmem[y, x, b]
                vx += EDGE_ANGLES_X[0, 0, b] * val
                vy += EDGE_ANGLES_Y[0, 0, b] * val
                denom += val

            angle = np.arctan2(vy, vx)
            mag = np.sqrt(vx * vx + vy * vy)
            if normalize_magnitude_as_in_paper:
                if denom == 0.0:
                    denom = 1.0
                mag = mag / denom

            c = np.cos(angle)
            s = np.sin(angle)

            allx += c
            ally += s
            sum_abs_angle_all += abs(angle)
            sum_mag_all += mag

            h = high_calc[y, x]
            l = abs(1.0 - h)
            n_high += h

            highx += c * h
            highy += s * h
            lowx += c * l
            lowy += s * l

            sum_abs_angle_high += abs(angle) * h
            sum_abs_angle_low += abs(angle) * l

            sum_mag_high += mag * h
            sum_mag_low += mag * l

    n_low = float(cell_n) - n_high

    mean_angle_all = np.arctan2(ally, allx)
    mean_angle_high = np.arctan2(highy, highx)
    mean_angle_low = np.nan if n_low == 0.0 else np.arctan2(lowy, lowx)

    if divide_angles_as_in_script2:
        mean_angle_all = mean_angle_all / float(cell_n)
        mean_angle_high = np.nan if n_high == 0.0 else mean_angle_high / n_high
        mean_angle_low = np.nan if n_low == 0.0 else mean_angle_low / n_low

    cv_all = 1.0 - np.sqrt(allx * allx + ally * ally) / float(cell_n)
    cv_high = np.nan if n_high == 0.0 else 1.0 - np.sqrt(highx * highx + highy * highy) / n_high
    cv_low = np.nan if n_low == 0.0 else 1.0 - np.sqrt(lowx * lowx + lowy * lowy) / n_low

    mean_del_angle_all = sum_abs_angle_all / float(cell_n)
    mean_del_angle_high = np.nan if n_high == 0.0 else sum_abs_angle_high / n_high
    mean_del_angle_low = np.nan if n_low == 0.0 else sum_abs_angle_low / n_low

    mean_magnitude_all = sum_mag_all / float(cell_n)
    mean_magnitude_high = np.nan if n_high == 0.0 else sum_mag_high / n_high
    mean_magnitude_low = np.nan if n_low == 0.0 else sum_mag_low / n_low

    return (
        cv_all,
        cv_high,
        cv_low,
        mean_del_angle_all,
        mean_del_angle_high,
        mean_del_angle_low,
        mean_angle_all,
        mean_angle_high,
        mean_angle_low,
        mean_magnitude_all,
        mean_magnitude_high,
        mean_magnitude_low,
        n_high,
    )


def compute_metrics(
    fzdmem: np.ndarray,
    high_calc: np.ndarray,
    params: PCPParameters,
    *,
    divide_angles_as_in_script2: bool = False,
    normalize_magnitude_as_in_paper: bool = False,
) -> PCPMetrics:
    vals = _compute_metrics_jit(
        np.asarray(fzdmem, dtype=np.float64),
        np.asarray(high_calc, dtype=np.float64),
        params.cell_N,
        divide_angles_as_in_script2,
        normalize_magnitude_as_in_paper,
    )
    return PCPMetrics(*vals)


# -----------------------------------------------------------------------------
# Dynamics
# -----------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def _adj_amount_inplace(src: np.ndarray, out: np.ndarray) -> None:
    y_max, x_max, bonds = src.shape

    for y in range(y_max):
        for x in range(x_max):
            for b in range(bonds):
                out[y, x, b] = src[y, x, b]

    # out[:, :, 4] = shifted src[:, :, 1] upward
    for y in range(y_max):
        src_y = y + 1 if y < y_max - 1 else 0
        for x in range(x_max):
            out[y, x, 4] = src[src_y, x, 1]

    # out[:, :, 1] = shifted src[:, :, 4] downward
    for y in range(y_max):
        src_y = y - 1 if y > 0 else y_max - 1
        for x in range(x_max):
            out[y, x, 1] = src[src_y, x, 4]

    # out[:, :, 3]
    for y in range(y_max):
        ym1 = y - 1 if y > 0 else y_max - 1
        for x in range(x_max):
            xp1 = x + 1 if x < x_max - 1 else 0
            if x % 2 == 0:
                out[y, x, 3] = src[y, xp1, 0]
            else:
                out[y, x, 3] = src[ym1, xp1, 0]

    # out[:, :, 5]
    for y in range(y_max):
        ym1 = y - 1 if y > 0 else y_max - 1
        for x in range(x_max):
            xm1 = x - 1 if x > 0 else x_max - 1
            if x % 2 == 0:
                out[y, x, 5] = src[y, xm1, 2]
            else:
                out[y, x, 5] = src[ym1, xm1, 2]

    # out[:, :, 2]
    for y in range(y_max):
        yp1 = y + 1 if y < y_max - 1 else 0
        for x in range(x_max):
            xp1 = x + 1 if x < x_max - 1 else 0
            if x % 2 == 0:
                out[y, x, 2] = src[yp1, xp1, 5]
            else:
                out[y, x, 2] = src[y, xp1, 5]

    # out[:, :, 0]
    for y in range(y_max):
        yp1 = y + 1 if y < y_max - 1 else 0
        for x in range(x_max):
            xm1 = x - 1 if x > 0 else x_max - 1
            if x % 2 == 0:
                out[y, x, 0] = src[yp1, xm1, 3]
            else:
                out[y, x, 0] = src[y, xm1, 3]


@njit(cache=True, nogil=True)
def _non_local_inplace(src: np.ndarray, d_a: float, out: np.ndarray) -> None:
    y_max, x_max, bonds = src.shape
    for y in range(y_max):
        for x in range(x_max):
            for b in range(bonds):
                left = b - 1 if b > 0 else bonds - 1
                right = b + 1 if b < bonds - 1 else 0
                out[y, x, b] = d_a * (src[y, x, left] + src[y, x, right]) + src[y, x, b]


@njit(cache=True, nogil=True)
def _run_until_convergence_jit(
    fzdmem: np.ndarray,
    vanglmem: np.ndarray,
    fzdint: np.ndarray,
    vanglint: np.ndarray,
    dt: float,
    rho: float,
    mu: float,
    phai: float,
    omega: float,
    d_a: float,
    d_i: float,
    thr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    y_max, x_max, bonds = fzdmem.shape

    # Preallocated work buffers
    fzdmem_adj = np.empty_like(fzdmem)
    vanglmem_adj = np.empty_like(vanglmem)

    tmp_mul_1 = np.empty_like(fzdmem)
    tmp_mul_2 = np.empty_like(fzdmem)

    nl_pos_fzd = np.empty_like(fzdmem)
    nl_neg_fzd = np.empty_like(fzdmem)
    nl_pos_vangl = np.empty_like(vanglmem)
    nl_neg_vangl = np.empty_like(vanglmem)

    del_fzdmem = np.empty_like(fzdmem)
    del_vanglmem = np.empty_like(vanglmem)
    del_fzdint = np.empty_like(fzdint)
    del_vanglint = np.empty_like(vanglint)

    convergence = 20000.0
    t = 0

    while convergence > thr:
        _adj_amount_inplace(fzdmem, fzdmem_adj)
        _adj_amount_inplace(vanglmem, vanglmem_adj)

        # tmp_mul_1 = fzdmem * vanglmem_adj
        # tmp_mul_2 = vanglmem * fzdmem_adj
        for y in range(y_max):
            for x in range(x_max):
                for b in range(bonds):
                    tmp_mul_1[y, x, b] = fzdmem[y, x, b] * vanglmem_adj[y, x, b]
                    tmp_mul_2[y, x, b] = vanglmem[y, x, b] * fzdmem_adj[y, x, b]

        _non_local_inplace(tmp_mul_1, d_a, nl_pos_fzd)   # nonLocal(fzdmem * vanglmemAdj, Da)
        _non_local_inplace(tmp_mul_2, d_i, nl_neg_fzd)   # nonLocal(vanglmem * fzdmemAdj, Di)
        _non_local_inplace(tmp_mul_2, d_a, nl_pos_vangl) # nonLocal(vanglmem * fzdmemAdj, Da)
        _non_local_inplace(tmp_mul_1, d_i, nl_neg_vangl) # nonLocal(fzdmem * vanglmemAdj, Di)

        convergence = 0.0

        for y in range(y_max):
            for x in range(x_max):
                sum_df = 0.0
                sum_dv = 0.0
                f_int = fzdint[y, x]
                v_int = vanglint[y, x]

                for b in range(bonds):
                    df = (rho + omega * nl_pos_fzd[y, x, b]) * f_int - (
                        mu + phai * nl_neg_fzd[y, x, b]
                    ) * fzdmem[y, x, b]

                    dv = (rho + omega * nl_pos_vangl[y, x, b]) * v_int - (
                        mu + phai * nl_neg_vangl[y, x, b]
                    ) * vanglmem[y, x, b]

                    del_fzdmem[y, x, b] = df
                    del_vanglmem[y, x, b] = dv
                    sum_df += df
                    sum_dv += dv

                del_fzdint[y, x] = -sum_df
                del_vanglint[y, x] = -sum_dv

                abs_dfint = abs(del_fzdint[y, x])
                if abs_dfint > convergence:
                    convergence = abs_dfint

        for y in range(y_max):
            for x in range(x_max):
                fzdint[y, x] += dt * del_fzdint[y, x]
                vanglint[y, x] += dt * del_vanglint[y, x]
                for b in range(bonds):
                    fzdmem[y, x, b] += dt * del_fzdmem[y, x, b]
                    vanglmem[y, x, b] += dt * del_vanglmem[y, x, b]

        t += 1

    return fzdmem, vanglmem, fzdint, vanglint, t


def run_until_convergence(
    fzdmem: np.ndarray,
    vanglmem: np.ndarray,
    fzdint: np.ndarray,
    vanglint: np.ndarray,
    params: PCPParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    return _run_until_convergence_jit(
        np.asarray(fzdmem, dtype=np.float64),
        np.asarray(vanglmem, dtype=np.float64),
        np.asarray(fzdint, dtype=np.float64),
        np.asarray(vanglint, dtype=np.float64),
        float(params.dt),
        float(params.rho),
        float(params.mu),
        float(params.phai),
        float(params.omega),
        float(params.Da),
        float(params.Di),
        float(params.thr),
    )


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def prepare_output_dir(base_dir: Path, dirname: str, source_script: Optional[Path] = None) -> Path:
    out_dir = base_dir / dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "Summary").mkdir(exist_ok=True)
    (out_dir / "Data").mkdir(exist_ok=True)

    if source_script is not None and source_script.exists():
        shutil.copy2(source_script, out_dir / source_script.name)

    return out_dir


def save_csv(path: Path, arr: np.ndarray) -> None:
    np.savetxt(path, arr, delimiter=",", fmt="%.10g")