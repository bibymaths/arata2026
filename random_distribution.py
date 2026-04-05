"""Random high/low cell distribution PCP simulation.

Credits
-------
Mathematical model and biological study:
    Masaki Arata, Hiroshi Koyama, Toshihiko Fujimori.
    iScience 29, 114771 (2026). DOI: 10.1016/j.isci.2026.114771.

Consistency notes
-----------------
1. The Octave script divides atan2-based mean angles by cell counts in columns 7-9.
   That is inconsistent with Script 1 and with the paper's angle definition. This
   implementation defaults to the paper-consistent / Script-1-consistent behavior. Use
   --legacy-script2-angle-scaling to reproduce the Octave output exactly.
2. The Octave helper adj_conc() allocates arrays with shape (Xmax, Ymax), which is
   harmless only because Xmax == Ymax in the paper. This implementation fixes that bug.
3. The paper defines polarity magnitude as ||(Fx,Fy)|| / sum_i(FMki). The Octave
   script omits that normalization. This implementation preserves the Octave default for file-
   level reproducibility; use --paper-magnitude to switch to the paper definition.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pcp_common import (
    PCPParameters,
    compute_metrics,
    initialize_unidirectional_state,
    prepare_output_dir,
    print_logo,
    run_until_convergence,
    save_csv,
    sum2,
)

BALANCE = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=float)
FREQ_HIGH = np.array(
    [
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    ],
    dtype=float,
)

logging.basicConfig(level=logging.INFO)


def generate_random_high_mask(
    ymax: int, xmax: int, ratio_high: float, rng: np.random.Generator
) -> np.ndarray:
    high = rng.random((ymax, xmax))
    v_l = 0.0
    v_h = 1.0
    thr_prev = (v_l + v_h) / 2.0
    ratio_var = sum2((high > thr_prev).astype(float))
    ratio_ans = int(np.floor(xmax * ymax * ratio_high))
    thr_high = thr_prev
    while ratio_var != ratio_ans:
        if ratio_var < ratio_ans:
            thr_high = (v_l + thr_prev) / 2.0
            v_h = thr_prev
        else:
            thr_high = (v_h + thr_prev) / 2.0
            v_l = thr_prev
        ratio_var = sum2((high > thr_high).astype(float))
        thr_prev = thr_high
    return (high > thr_high).astype(float)


def run_simulation(
    params: PCPParameters,
    out_dir: Path,
    seed: int = 0,
    *,
    legacy_script2_angle_scaling: bool = False,
    paper_magnitude: bool = False,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nbalance = BALANCE.size
    nfreq_high = FREQ_HIGH.size
    output = np.zeros((params.nREP * nfreq_high * nbalance, 14), dtype=float)
    count = 0

    logging.info(f"Running simulation with parameters: {params}")

    total = len(FREQ_HIGH) * params.nREP * len(BALANCE) * params.noise_rep
    pbar = tqdm(total=total, desc="Simulations")

    for ratio_high in FREQ_HIGH:
        ratio_low = 1.0 - ratio_high
        for rep1 in range(1, params.nREP + 1):
            high_calc = generate_random_high_mask(params.Ymax, params.Xmax, ratio_high, rng)
            for low_conc in BALANCE:
                high = high_calc + low_conc * np.abs(1.0 - high_calc)
                for rep2 in range(1, params.noise_rep + 1):
                    t0 = 0
                    fzdmem, vanglmem, fzdint, vanglint = initialize_unidirectional_state(
                        high, params, rng
                    )

                    fname = f"{low_conc}_Lfreq{ratio_low}_noiseRep{rep2}_rep{rep1}_t{t0}.csv"
                    save_csv(out_dir / "Data" / f"HIGH_LOW{fname}", high_calc)
                    save_csv(
                        out_dir / "Data" / f"Fzdmem_LOW{fname}",
                        fzdmem.reshape(params.Ymax * params.Xmax, params.Bonds),
                    )
                    save_csv(
                        out_dir / "Data" / f"Vanglmem_LOW{fname}",
                        vanglmem.reshape(params.Ymax * params.Xmax, params.Bonds),
                    )
                    save_csv(out_dir / "Data" / f"Fzdint_LOW{fname}", fzdint)
                    save_csv(out_dir / "Data" / f"Vanglint_LOW{fname}", vanglint)

                    fzdmem, vanglmem, fzdint, vanglint, t_final = run_until_convergence(
                        fzdmem, vanglmem, fzdint, vanglint, params
                    )

                    fname = f"{low_conc}_Lfreq{ratio_low}_noiseRep{rep2}_rep{rep1}_t{t_final}.csv"
                    save_csv(
                        out_dir / "Data" / f"Fzdmem_LOW{fname}",
                        fzdmem.reshape(params.Ymax * params.Xmax, params.Bonds),
                    )
                    save_csv(
                        out_dir / "Data" / f"Vanglmem_LOW{fname}",
                        vanglmem.reshape(params.Ymax * params.Xmax, params.Bonds),
                    )
                    save_csv(out_dir / "Data" / f"Fzdint_LOW{fname}", fzdint)
                    save_csv(out_dir / "Data" / f"Vanglint_LOW{fname}", vanglint)

                    metrics = compute_metrics(
                        fzdmem,
                        high_calc,
                        params,
                        divide_angles_as_in_script2=legacy_script2_angle_scaling,
                        normalize_magnitude_as_in_paper=paper_magnitude,
                    )

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
                    output[count, 13] = low_conc
                    count += 1
                    pbar.update(1)

    pbar.close()
    save_csv(out_dir / "Summary" / "output.csv", output)
    logging.info(f"Saved output to {out_dir / 'Summary' / 'output.csv'}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path.cwd())
    parser.add_argument("--dirname", default="Random_test_python")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--legacy-script2-angle-scaling", action="store_true")
    parser.add_argument("--paper-magnitude", action="store_true")
    args = parser.parse_args()

    params = PCPParameters()
    logging.info(f"Using parameters: {params}")
    out_dir = prepare_output_dir(args.outdir, args.dirname, Path(__file__))
    logging.info(f"Output directory: {out_dir}")
    run_simulation(
        params,
        out_dir,
        seed=args.seed,
        legacy_script2_angle_scaling=args.legacy_script2_angle_scaling,
        paper_magnitude=args.paper_magnitude,
    )
    logging.info(f"Simulation completed. Results saved to {out_dir}")


if __name__ == "__main__":
    print_logo(
        name="PCP Python",
        version="0.1.0",
        tagline="Python implementation of the Arata et al. PCP simulations",
        author="Abhinav Mishra",
        email="mishraabhinav36@gmail.com",
        orcid="0009-0005-3179-7408",
        website="bibymaths.github.io",
        font="slant",
        color="bright_green",
        animate=True,
    )
    main()
