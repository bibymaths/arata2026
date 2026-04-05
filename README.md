# Python implementation of the Arata et al. PCP simulations

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19424946.svg)](https://doi.org/10.5281/zenodo.19424946)

Python reimplementation of the planar cell polarity (PCP) simulations described in Arata et al., preserving the original Octave logic while explicitly documenting deviations required for reproducibility.

## Credits

* **Model and paper**: Masaki Arata, Hiroshi Koyama, Toshihiko Fujimori
* **DOI**: [https://doi.org/10.1016/j.isci.2026.114771](https://doi.org/10.1016/j.isci.2026.114771)

## Reproducibility notes

The implementation follows the original Octave update rules and periodic-neighborhood logic. However, the published paper and the reference scripts are not fully consistent. The following cases are handled explicitly:

1. **Angle scaling discrepancy**
   `Arata_et_al_random` divides mean angles by the number of cells, whereas Script 1 and the paper do not.

   * Default: paper-consistent behavior
   * To reproduce Script 2 exactly:

     ```bash
     --legacy-script2-angle-scaling
     ```

2. **Polarity magnitude definition**
   The paper defines magnitude as `sqrt(Fx^2 + Fy^2) / sum(FM_i)`, but both Octave scripts omit the denominator.

   * Default: script-consistent behavior
   * To use the paper definition:

     ```bash
     --paper-magnitude
     ```

3. **Array shape bug in original script**
   `Arata_et_al_random` allocates arrays as `(Xmax, Ymax)` instead of `(Ymax, Xmax)`.
   This is masked in published runs (`24 × 24`) but incorrect in general.

   * Fixed in this implementation

## Usage

Run from the project root.

```bash
python random_distribution.py --outdir . --dirname Random_test_python
python patch_shift.py --outdir . --dirname patch_shift_python
```

### Optional flags

```bash
# Match Script 2 angle scaling
python random_distribution.py --legacy-script2-angle-scaling

# Use paper-consistent magnitude definition
python random_distribution.py --paper-magnitude
python patch_shift.py --paper-magnitude
```
---

## License

MIT