# Python implementation of the Arata et al. PCP simulations

## Credits

- Paper and mathematical model: Masaki Arata, Hiroshi Koyama, Toshihiko Fujimori.
- DOI: 10.1016/j.isci.2026.114771.
- Octave scripts: Masaki Arata, Hiroshi Koyama, Toshihiko Fujimori.
- DOI: https://doi.org/10.1016/j.isci.2026.114771
-

## Reproducibility and consistency

The implementation preserves the Octave update rules and periodic-neighborhood logic. Two points needed explicit
handling:

1. Script 'Arata_et_al_random' divides mean angles by the number of cells, but Script 1 and the paper do not. The Python
   implementation defaults to the paper-consistent behavior. Add `--legacy-script2-angle-scaling` to reproduce Script 2
   exactly.
2. The paper defines polarity magnitude as `sqrt(Fx^2 + Fy^2) / sum(FM_i)`, but both Octave scripts omit the
   denominator. The Python implementation keeps the script behavior by default and offers `--paper-magnitude` as an
   option.
3. Script 'Arata_et_al_random' allocates some temporary arrays with `(Xmax, Ymax)` instead of `(Ymax, Xmax)`. Because
   the published runs use `24 x 24`, the bug is masked there. The implementation fixes it.

## Usage

Run from this directory.

```bash
python random_distribution.py --outdir . --dirname Random_test_python
python patch_shift.py --outdir . --dirname patch_shift_python
```

Optional switches:

```bash
python random_distribution.py --legacy-script2-angle-scaling
python random_distribution.py --paper-magnitude
python patch_shift.py --paper-magnitude
```
