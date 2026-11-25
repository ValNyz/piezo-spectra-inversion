# Forward model and Levenberg-Marquardt inversion of piezoelectric eigenfrequency spectra

Code accompanying the ESANN 2027 paper *Learning to Invert Eigenfrequency Spectra for the Characterization of
Piezoelectric Materials*. It contains the forward model (eigenvalues $\rho\omega^2$ of a free 4mm piezoelectric
parallelepiped by the Rayleigh-Ritz method, on GPU, with exact gradients), the dataset generation and the
Levenberg-Marquardt (LM) inversion used as baseline. The networks of the paper are not included.

## Install

Python >= 3.10. Install PyTorch for your platform, then `pip install -r requirements.txt`.

## Scripts

Run from this folder; `python3 <script> -h` gives the options.

| Script | Does | Writes |
|---|---|---|
| `generate.py` | random admissible constants + their eigenvalues (forward model) | `DATA/constants.npy`, `DATA/eigenvalues.npy` |
| `lm.py` | LM inversion of test spectra from random starts or from given starts (e.g. a network) | `OUT/<sample>.json` + summary |

Library files: `constants.py` (names, geometry, sampling box), `forward.py` (Rayleigh-Ritz model, eigenvalues
and their exact gradients), `sampling.py` (box + physical filters), `dataset.py` (split, representations,
degradation of spectra).

```bash
python3 generate.py --out data --n 1331200          # dataset of the paper, about 46 h on one GPU
python3 generate.py --out data_small --n 2000       # quick try
python3 lm.py --data data --start random --out results/lm_random
python3 lm.py --data data --start random --drop_modes 10 --noise 0.01 --out results/lm_random_deg
```

LM from a network (as in the paper): export the target spectra, predict the 11 constants with your network, save
them as an array `(n_samples, 11)` in the units of `constants.npy` and the order of the spectra, then start LM from it:

```bash
python3 lm.py --data data --export spectra.npy
python3 lm.py --data data --start starts.npy --out results/lm_net
```

With `--drop_modes/--noise`, pass the same options to `--export`: the degradation is a fixed draw, so the network
and LM see the same degraded spectra (sorted eigenvalues, 0 = missing mode).

## Notes

- The forward model is invariant under $e \to s\,e$, $\varepsilon^S \to s^2\varepsilon^S$: only $c^E$ and
  $e^2/\varepsilon^S$ are identifiable from the eigenvalues. `lm.py --fix epsS33` fixes constants to their true value.
- `lm_samples.txt`: the 1000 test samples used for LM in the paper.

## Credits

Forward model: Python implementation by L. Dubois (GREMAN, VERMON), ported to PyTorch by V. Nyzam (LIFAT). Physical filters: implementation of L. Dubois.

## License

MIT, see `LICENSE`.
