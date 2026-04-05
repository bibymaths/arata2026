"""Patch-shifted low-cell cluster PCP simulation.

Credits
-------
Mathematical model and biological study:
    Masaki Arata, Hiroshi Koyama, Toshihiko Fujimori.
    "Directional alignment of different cell types organizes planar cell polarity"
    iScience 29, 114771 (2026). DOI: 10.1016/j.isci.2026.114771.

Original Octave source:
    User-provided script labelled here as "Script 1".
    The original script header does not state individual code authors.
"""

from __future__ import annotations
import logging
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pcp_common import (
    PCPParameters,
    compute_metrics,
    grow_seed,
    initialize_unidirectional_state,
    prepare_output_dir,
    run_until_convergence,
    save_csv,
    seed_x,
    seed_y,
    sum2,
)

BALANCE = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=float)
PARAMS = np.array([
    [3, 3, 0, 0, 1, 4, 4],
    [3, 3, 1, 1, 10, 4, 4],
    [3, 3, 2, 2, 10, 4, 4],
    [3, 3, 3, 3, 10, 4, 4],
    [5, 5, 0, 0, 1, 6, 6],
    [5, 5, 1, 1, 10, 6, 6],
    [5, 5, 2, 2, 10, 6, 6],
    [5, 5, 3, 3, 10, 6, 6],
], dtype=int)

logging.basicConfig(level=logging.INFO)

def generate_shifted_patch_mask(
        x_max: int,
        y_max: int,
        lx: int,
        ly: int,
        shift_x: int,
        shift_y: int,
        lambda_x: int,
        lambda_y: int,
        rng: np.random.Generator,
) -> np.ndarray:
    n_patch_x = x_max // lambda_x
    n_patch_y = y_max // lambda_y
    high = np.ones((y_max, x_max), dtype=float)

    for x in range(1, n_patch_x + 1):
        for y in range(1, n_patch_y + 1):
            rot_x = int(np.floor(rng.random() * (shift_x * 2 + 1))) - shift_x
            rot_y = int(np.floor(rng.random() * (shift_y * 2 + 1))) - shift_y
            x_l = 1 + (x - 1) * lambda_x + rot_x
            y_t = 1 + (y - 1) * lambda_y + rot_y

            if x_l < 1:
                x_l += x_max
            if y_t < 1:
                y_t += y_max
            if x_l > x_max:
                x_l -= x_max
            if y_t > y_max:
                y_t -= y_max
            high[y_t - 1, x_l - 1] = 0.0

    enlarge_x = int(np.floor(lx / 2)) - 1
    enlarge_y = int(np.floor(ly / 2)) - 1

    for _ in range(max(0, enlarge_x)):
        high = seed_x(high, x_max, y_max)
    for _ in range(max(0, enlarge_y)):
        high = seed_y(high, x_max, y_max)

    high = grow_seed(high, y_max, x_max)
    return high


def run_simulation(
        params: PCPParameters,
        out_dir: Path,
        seed: int = 0,
        *,
        paper_magnitude: bool = False,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nbalance = BALANCE.size
    output = np.zeros((int(np.sum(PARAMS[:, 4])) * nbalance, 22), dtype=float)
    count = 0

    logging.info(f"Running simulation with parameters: {params}")

    total = int(np.sum(PARAMS[:, 4])) * len(BALANCE) * params.noise_rep
    pbar = tqdm(total=total, desc="Simulations")

    for row in PARAMS:
        lx, ly, shift_x, shift_y, distribution, lambda_x, lambda_y = map(int, row)

        for rep1 in range(1, distribution + 1):
            high_calc = generate_shifted_patch_mask(
                params.Xmax,
                params.Ymax,
                lx,
                ly,
                shift_x,
                shift_y,
                lambda_x,
                lambda_y,
                rng,
            )

            for low_conc in BALANCE:
                high = high_calc + low_conc * np.abs(1.0 - high_calc)
                for rep2 in range(1, params.noise_rep + 1):
                    t0 = 0
                    fzdmem, vanglmem, fzdint, vanglint = initialize_unidirectional_state(high, params, rng)

                    fname = (
                        f"{lx}_Y{ly}_sX{shift_x}_sY{shift_y}_delX{lambda_x}_delY{lambda_y}"
                        f"_low{low_conc}_noiseRep{rep2}_rep{rep1}_t{t0}.csv"
                    )
                    save_csv(out_dir / "Data" / f"HIGH_patchX{fname}", high_calc)
                    save_csv(out_dir / "Data" / f"Fzdmem_patchX{fname}",
                             fzdmem.reshape(params.Ymax * params.Xmax, params.Bonds))
                    save_csv(out_dir / "Data" / f"Vanglmem_patchX{fname}",
                             vanglmem.reshape(params.Ymax * params.Xmax, params.Bonds))
                    save_csv(out_dir / "Data" / f"Fzdint_patchX{fname}", fzdint)
                    save_csv(out_dir / "Data" / f"Vanglint_patchX{fname}", vanglint)

                    fzdmem, vanglmem, fzdint, vanglint, t_final = run_until_convergence(
                        fzdmem, vanglmem, fzdint, vanglint, params
                    )

                    fname = (
                        f"{lx}_Y{ly}_sX{shift_x}_sY{shift_y}_delX{lambda_x}_delY{lambda_y}"
                        f"_low{low_conc}_noiseRep{rep2}_rep{rep1}_t{t_final}.csv"
                    )
                    save_csv(out_dir / "Data" / f"Fzdmem_patchX{fname}",
                             fzdmem.reshape(params.Ymax * params.Xmax, params.Bonds))
                    save_csv(out_dir / "Data" / f"Vanglmem_patchX{fname}",
                             vanglmem.reshape(params.Ymax * params.Xmax, params.Bonds))
                    save_csv(out_dir / "Data" / f"Fzdint_patchX{fname}", fzdint)
                    save_csv(out_dir / "Data" / f"Vanglint_patchX{fname}", vanglint)

                    metrics = compute_metrics(
                        fzdmem,
                        high_calc,
                        params,
                        divide_angles_as_in_script2=False,
                        normalize_magnitude_as_in_paper=paper_magnitude,
                    )

                    n_low = params.cell_N - metrics.n_high
                    output[count, 0] = metrics.cv_all
                    output[count, 1] = metrics.cv_high
                    output[count, 2] = metrics.cv_low
                    output[count, 3] = metrics.mean_del_angle_all
                    output[count, 4] = metrics.mean_del_angle_high
                    output[count, 5] = metrics.mean_del_angle_low
                    output[count, 6] = metrics.mean_angle_all
                    output[count, 7] = metrics.mean_angle_high
                    output[count, 8] = metrics.mean_angle_low
                    output[count, 9] = metrics.mean_magnitude_all
                    output[count, 10] = metrics.mean_magnitude_high
                    output[count, 11] = metrics.mean_magnitude_low
                    output[count, 12] = metrics.n_high
                    output[count, 13] = lx
                    output[count, 14] = ly
                    output[count, 15] = shift_x
                    output[count, 16] = shift_y
                    output[count, 17] = lambda_x
                    output[count, 18] = lambda_y
                    output[count, 19] = low_conc
                    output[count, 20] = rep1
                    output[count, 21] = rep2
                    _ = n_low
                    count += 1
                    pbar.update(1)

    pbar.close()
    save_csv(out_dir / "Summary" / "output.csv", output)

    logging.info(f"Saved output to {out_dir / 'Summary' / 'output.csv'}")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path.cwd())
    parser.add_argument("--dirname", default="patch_shift_python")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--paper-magnitude", action="store_true")
    args = parser.parse_args()

    params = PCPParameters()

    logging.info(f"Using parameters: {params}")

    out_dir = prepare_output_dir(args.outdir, args.dirname, Path(__file__))

    logging.info(f"Output directory: {out_dir}")

    run_simulation(params, out_dir, seed=args.seed, paper_magnitude=args.paper_magnitude)

    logging.info(f"Simulation completed successfully")

if __name__ == "__main__":
    main()
