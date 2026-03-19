"""Common utilities for implementing the PCP Octave simulations to Python.

Credits
-------
Mathematical model and biological study:
    Masaki Arata, Hiroshi Koyama, Toshihiko Fujimori.
    "Directional alignment of different cell types organizes planar cell polarity"
    iScience 29, 114771 (2026). DOI: 10.1016/j.isci.2026.114771.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Optional

import numpy as np


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


EDGE_VAL = np.cos(np.pi / 6.0)
EDGE_ANGLES_X = np.array([[[-EDGE_VAL, 0.0, EDGE_VAL, EDGE_VAL, 0.0, -EDGE_VAL]]], dtype=float)
EDGE_ANGLES_Y = np.array([[[0.5, 1.0, 0.5, -0.5, -1.0, -0.5]]], dtype=float)


def sum2(arr: np.ndarray) -> float:
    return float(np.sum(arr))


# --- Periodic adjacency / neighborhood helpers ---------------------------------

def seed_x(mtemp: np.ndarray, x_max: int, y_max: int) -> np.ndarray:
    del x_max, y_max
    shift = np.empty_like(mtemp)
    shift[:, 0:2] = mtemp[:, -2:]
    shift[:, 2:] = mtemp[:, :-2]
    return (mtemp + shift >= 2).astype(float)


def seed_y(mtemp: np.ndarray, x_max: int, y_max: int) -> np.ndarray:
    del x_max
    shift = np.empty_like(mtemp)
    shift[0:2, :] = mtemp[-2:, :]
    shift[2:, :] = mtemp[:-2, :]
    return (mtemp + shift >= 2).astype(float)


def grow_seed(mtemp: np.ndarray, y_max: int, x_max: int) -> np.ndarray:
    mleft = np.zeros((y_max, x_max), dtype=float)
    mright = np.zeros_like(mleft)
    mup = np.zeros_like(mleft)
    mdown = np.zeros_like(mleft)

    mdown[0, :] = mtemp[-1, :]
    mdown[1:, :] = mtemp[:-1, :]
    r = mdown.copy()

    mup[-1, :] = mtemp[0, :]
    mup[:-1, :] = mtemp[1:, :]
    r = r + mup

    mright[:, 0] = mtemp[:, -1]
    mright[:, 1:] = mtemp[:, :-1]
    r[:, 1::2] = r[:, 1::2] + mright[:, 1::2]
    mdown[0, :] = mright[-1, :]
    mdown[1:, :] = mright[:-1, :]
    r[:, 0::2] = r[:, 0::2] + mdown[:, 0::2]

    mleft[:, -1] = mtemp[:, 0]
    mleft[:, :-1] = mtemp[:, 1:]
    r[:, 1::2] = r[:, 1::2] + mleft[:, 1::2]
    mdown[0, :] = mleft[-1, :]
    mdown[1:, :] = mleft[:-1, :]
    r[:, 0::2] = r[:, 0::2] + mdown[:, 0::2]

    mright[:, 0] = mtemp[:, -1]
    mright[:, 1:] = mtemp[:, :-1]
    r[:, 0::2] = r[:, 0::2] + mright[:, 0::2]
    mup[-1, :] = mright[0, :]
    mup[:-1, :] = mright[1:, :]
    r[:, 1::2] = r[:, 1::2] + mup[:, 1::2]

    mleft[:, -1] = mtemp[:, 0]
    mleft[:, :-1] = mtemp[:, 1:]
    r[:, 0::2] = r[:, 0::2] + mleft[:, 0::2]
    mup[-1, :] = mleft[0, :]
    mup[:-1, :] = mleft[1:, :]
    r[:, 1::2] = r[:, 1::2] + mup[:, 1::2]

    return (r + mtemp >= 7).astype(float)


def adj_amount(mtemp: np.ndarray, x_max: int, y_max: int, bonds: int) -> np.ndarray:
    del bonds
    mleft = np.zeros((y_max, x_max), dtype=float)
    mright = np.zeros_like(mleft)
    mup = np.zeros_like(mleft)
    mdown = np.zeros_like(mleft)
    r = np.array(mtemp, copy=True)

    mup[-1, :] = mtemp[0, :, 1]
    mup[:-1, :] = mtemp[1:, :, 1]
    r[:, :, 4] = mup

    mdown[0, :] = mtemp[-1, :, 4]
    mdown[1:, :] = mtemp[:-1, :, 4]
    r[:, :, 1] = mdown

    mleft[:, -1] = mtemp[:, 0, 0]
    mleft[:, :-1] = mtemp[:, 1:, 0]
    r[:, 0::2, 3] = mleft[:, 0::2]
    mup[-1, :] = mleft[0, :]
    mup[:-1, :] = mleft[1:, :]
    r[:, 1::2, 3] = mup[:, 1::2]

    mright[:, 0] = mtemp[:, -1, 2]
    mright[:, 1:] = mtemp[:, :-1, 2]
    r[:, 0::2, 5] = mright[:, 0::2]
    mup[-1, :] = mright[0, :]
    mup[:-1, :] = mright[1:, :]
    r[:, 1::2, 5] = mup[:, 1::2]

    mleft[:, -1] = mtemp[:, 0, 5]
    mleft[:, :-1] = mtemp[:, 1:, 5]
    r[:, 1::2, 2] = mleft[:, 1::2]
    mdown[0, :] = mleft[-1, :]
    mdown[1:, :] = mleft[:-1, :]
    r[:, 0::2, 2] = mdown[:, 0::2]

    mright[:, 0] = mtemp[:, -1, 3]
    mright[:, 1:] = mtemp[:, :-1, 3]
    r[:, 1::2, 0] = mright[:, 1::2]
    mdown[0, :] = mright[-1, :]
    mdown[1:, :] = mright[:-1, :]
    r[:, 0::2, 0] = mdown[:, 0::2]

    return r


# Alias retained to mirror script 2 nomenclature.
adj_conc = adj_amount


def non_local(mtemp: np.ndarray, bonds: int, d_a: float) -> np.ndarray:
    mleft = np.array(mtemp, copy=True)
    mright = np.array(mtemp, copy=True)
    mleft[:, :, 0] = mtemp[:, :, bonds - 1]
    mleft[:, :, 1:bonds] = mtemp[:, :, 0:bonds - 1]
    mright[:, :, bonds - 1] = mtemp[:, :, 0]
    mright[:, :, 0:bonds - 1] = mtemp[:, :, 1:bonds]
    return d_a * (mleft + mright) + mtemp


# --- Initial conditions / metrics ------------------------------------------------

def initialize_unidirectional_state(
        high: np.ndarray,
        params: PCPParameters,
        rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fzdmem = np.zeros((params.Ymax, params.Xmax, params.Bonds), dtype=float)
    vanglmem = np.zeros_like(fzdmem)
    fzdmem[:, :, 2] = 0.6
    fzdmem[:, :, 3] = 0.6
    vanglmem[:, :, 0] = 0.6
    vanglmem[:, :, 5] = 0.6
    fzdmem = fzdmem + params.noise * rng.random(fzdmem.shape)
    vanglmem = vanglmem + params.noise * rng.random(vanglmem.shape)
    fzdint = high * (params.Total_protein - np.sum(fzdmem, axis=2))
    vanglint = high * (params.Total_protein - np.sum(vanglmem, axis=2))
    fzdmem = high[:, :, None] * fzdmem
    vanglmem = high[:, :, None] * vanglmem
    return fzdmem, vanglmem, fzdint, vanglint


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


def compute_metrics(
        fzdmem: np.ndarray,
        high_calc: np.ndarray,
        params: PCPParameters,
        *,
        divide_angles_as_in_script2: bool = False,
        normalize_magnitude_as_in_paper: bool = False,
) -> PCPMetrics:
    vx = np.sum(EDGE_ANGLES_X * fzdmem, axis=2)
    vy = np.sum(EDGE_ANGLES_Y * fzdmem, axis=2)
    angle = np.arctan2(vy, vx)

    if normalize_magnitude_as_in_paper:
        denom = np.sum(fzdmem, axis=2)
        denom = np.where(denom == 0.0, 1.0, denom)
        magnitude = np.sqrt(vx ** 2 + vy ** 2) / denom
    else:
        magnitude = np.sqrt(vx ** 2 + vy ** 2)

    allx = sum2(np.cos(angle))
    ally = sum2(np.sin(angle))
    highx = sum2(np.cos(angle) * high_calc)
    highy = sum2(np.sin(angle) * high_calc)
    low_mask = np.abs(1.0 - high_calc)
    lowx = sum2(np.cos(angle) * low_mask)
    lowy = sum2(np.sin(angle) * low_mask)
    n_high = sum2(high_calc)
    n_low = params.cell_N - n_high

    def safe_div(num: float, den: float) -> float:
        return float(num / den) if den != 0 else np.nan

    mean_angle_all = float(np.arctan2(ally, allx))
    mean_angle_high = float(np.arctan2(highy, highx))
    mean_angle_low = float(np.arctan2(lowy, lowx)) if n_low != 0 else np.nan

    if divide_angles_as_in_script2:
        mean_angle_all = safe_div(mean_angle_all, params.cell_N)
        mean_angle_high = safe_div(mean_angle_high, n_high)
        mean_angle_low = safe_div(mean_angle_low, n_low) if n_low != 0 else np.nan

    return PCPMetrics(
        cv_all=1.0 - np.sqrt(allx ** 2 + ally ** 2) / params.cell_N,
        cv_high=1.0 - np.sqrt(highx ** 2 + highy ** 2) / n_high if n_high != 0 else np.nan,
        cv_low=1.0 - np.sqrt(lowx ** 2 + lowy ** 2) / n_low if n_low != 0 else np.nan,
        mean_del_angle_all=sum2(np.abs(angle)) / params.cell_N,
        mean_del_angle_high=sum2(np.abs(angle) * high_calc) / n_high if n_high != 0 else np.nan,
        mean_del_angle_low=sum2(np.abs(angle) * low_mask) / n_low if n_low != 0 else np.nan,
        mean_angle_all=mean_angle_all,
        mean_angle_high=mean_angle_high,
        mean_angle_low=mean_angle_low,
        mean_magnitude_all=sum2(magnitude) / params.cell_N,
        mean_magnitude_high=sum2(magnitude * high_calc) / n_high if n_high != 0 else np.nan,
        mean_magnitude_low=sum2(magnitude * low_mask) / n_low if n_low != 0 else np.nan,
        n_high=n_high,
    )


# --- Dynamics -------------------------------------------------------------------

def run_until_convergence(
        fzdmem: np.ndarray,
        vanglmem: np.ndarray,
        fzdint: np.ndarray,
        vanglint: np.ndarray,
        params: PCPParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    convergence = 20000.0
    t = 0
    while convergence > params.thr:
        fzdmem_adj = adj_amount(fzdmem, params.Xmax, params.Ymax, params.Bonds)
        vanglmem_adj = adj_amount(vanglmem, params.Xmax, params.Ymax, params.Bonds)

        del_fzdmem = (params.rho + params.omega * non_local(fzdmem * vanglmem_adj, params.Bonds, params.Da)) * fzdint[
            :, :, None]
        del_fzdmem = del_fzdmem - (
                    params.mu + params.phai * non_local(vanglmem * fzdmem_adj, params.Bonds, params.Di)) * fzdmem

        del_vanglmem = (params.rho + params.omega * non_local(vanglmem * fzdmem_adj, params.Bonds, params.Da)) * \
                       vanglint[:, :, None]
        del_vanglmem = del_vanglmem - (
                    params.mu + params.phai * non_local(fzdmem * vanglmem_adj, params.Bonds, params.Di)) * vanglmem

        del_fzdint = -np.sum(del_fzdmem, axis=2)
        del_vanglint = -np.sum(del_vanglmem, axis=2)

        fzdmem = fzdmem + params.dt * del_fzdmem
        vanglmem = vanglmem + params.dt * del_vanglmem
        fzdint = fzdint + params.dt * del_fzdint
        vanglint = vanglint + params.dt * del_vanglint

        t += 1
        convergence = float(np.max(np.abs(del_fzdint)))

    return fzdmem, vanglmem, fzdint, vanglint, t


# --- I/O ------------------------------------------------------------------------

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
