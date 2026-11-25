"""Levenberg-Marquardt inversion of test spectra, from random starts or from given starts (e.g. a network).

    python3 lm.py --data data --start random --out results/lm_random            # 10 random starts
    python3 lm.py --data data --export spectra.npy                              # spectra to predict the starts from
    python3 lm.py --data data --start starts.npy --out results/lm_net           # start = network prediction
    python3 lm.py --data data --start starts.npy --fix epsS33 --out results/lm_net_eps33
    python3 lm.py --data data --drop_modes 10 --noise 0.01 --export spectra_deg.npy
    python3 lm.py --data data --start starts_deg.npy --drop_modes 10 --noise 0.01 --out results/lm_net_deg
    python3 lm.py --summary results/lm_*

Fits the 11 constants (scaled to [0, 1] in the sampling box) by least squares on the relative eigenvalue error
(scipy, "trf", exact Jacobian). The computed modes are matched to the measured ones at each iteration, in order;
with missing modes, unmatched computed modes are taken as not detected.
  --start random   10 random starts, the best one is refined
  --start FILE.npy one start per sample: array (n_samples, 11) of constants, in the units of constants.npy and
                   in the order of --samples (e.g. the prediction of a network on the spectra of --export)
  --export FILE.npy  only writes the target spectra (n_samples, 110; sorted eigenvalues, 0 = missing mode), after
                   the degradation if any, and exits. With --drop_modes/--noise, predict the starts on these
                   degraded spectra: LM fits exactly the same ones (fixed draw).
  --fix epsS33     constants fixed to their true value (as if measured separately)
  --drop_modes K --noise S   degraded spectra: half of the samples lose 1 to K random modes, half get a relative
                   frequency error up to S (fixed draw)
Samples: test-set positions read from --samples (default lm_samples.txt). If the file does not exist, --n positions
are drawn (--seed) and written to it; the defaults (1000, seed 0) give the samples of the paper.
Writes one JSON per sample in OUT (a stopped run resumes), then prints the summary. About 3 s per sample from a
network, 100 s from random starts.

This script was written with the help of an AI assistant.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares

import dataset
from constants import CONSTANT_NAMES, EIG_MAX, EIG_MIN, N_CONSTANTS
from forward import Forward

LSQ = dict(method="trf", x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10)
GAPS = 15  # computed modes the matching may skip (up to 10 missing modes + margin)
EXACT = 1e-10  # cost below which a fit is called exact
_fwd: Forward | None = None  # one forward model per worker


# ----------------------------------------------------------------------------- matching
def ordered_index(spectrum, n):
    """k-th target mode <-> k-th computed mode above EIG_MIN."""
    j0 = int(np.searchsorted(spectrum, EIG_MIN, side="right"))
    return np.minimum(j0 + np.arange(n), len(spectrum) - 1)


def aligned_index(spectrum, t, below=3, gaps=GAPS):
    """i-th target mode <-> computed mode j_i, j_0 < j_1 < ..., among the computed modes from 3 below EIG_MIN,
    minimising sum (w_j / t_i - 1)^2 (dynamic programming)."""
    j0 = int(np.searchsorted(spectrum, EIG_MIN, side="right"))
    lo = max(j0 - below, 0)
    w = spectrum[lo : min(j0 + len(t) + gaps, len(spectrum))]
    n, m = len(t), len(w)
    if m < n:
        return ordered_index(spectrum, n)
    C = (w[None, :] / t[:, None] - 1.0) ** 2
    ar = np.arange(m)
    arg = np.empty((n, m), dtype=np.int64)
    D = None
    for i in range(n):
        val = C[i].copy() if i == 0 else np.concatenate([[np.inf], D[:-1] + C[i, 1:]])
        run = np.minimum.accumulate(val)
        arg[i] = np.maximum.accumulate(np.where(val == run, ar, 0))
        D = run
    idx = np.empty(n, dtype=np.int64)
    j = arg[n - 1, m - 1]
    for i in range(n - 1, -1, -1):
        idx[i] = j
        if i:
            j = arg[i - 1, j - 1]
    return lo + idx


# ----------------------------------------------------------------------------- problem
class Problem:
    """r(u) = lambda_matched(u) / target - 1 and its exact Jacobian; some constants can be fixed (u_fix)."""

    def __init__(self, target, vmin, vmax, gaps=False, fixed=(), u_fix=None):
        self.t, self.vmin, self.scale, self.gaps = target, vmin, vmax - vmin, gaps
        self.free = np.setdiff1d(np.arange(N_CONSTANTS), fixed)
        self.u_fix = (
            np.zeros(N_CONSTANTS) if u_fix is None else np.asarray(u_fix, float)
        )
        self.n_eig, self._u = 0, None

    def full(self, uf):
        u = self.u_fix.copy()
        u[self.free] = uf
        return u

    def _eval(self, u):
        if self._u is not None and np.array_equal(u, self._u):
            return  # least_squares asks residual and Jacobian at the same point
        w, v = _fwd.eigh(self.vmin + u * self.scale)
        self.n_eig += 1
        self._idx = (
            aligned_index(w, self.t) if self.gaps else ordered_index(w, len(self.t))
        )
        self._r, self._v, self._u = w[self._idx] / self.t - 1.0, v, u.copy()

    def residual(self, uf):
        self._eval(self.full(uf))
        return self._r.copy()

    def jac(self, uf):
        u = self.full(uf)
        self._eval(u)
        d = _fwd.gradients(self.vmin + u * self.scale, self._v[:, self._idx])  # (11, n)
        return (d * self.scale[:, None] / self.t[None, :]).T[:, self.free]

    def cost(self, u):
        self._eval(np.asarray(u, float))
        return float(self._r @ self._r)


def lsq(prob, uf0, max_nfev=200):
    res = least_squares(
        prob.residual,
        np.clip(uf0, 1e-6, 1 - 1e-6),
        jac=prob.jac,
        bounds=(0, 1),
        max_nfev=max_nfev,
        **LSQ,
    )
    return res.x, res


# ----------------------------------------------------------------------------- one sample
def invert(job):
    k, target, starts, u_true, vmin, vmax, fixed, gaps, out = job
    t0 = time.time()
    prob = Problem(target, vmin, vmax, gaps, fixed, u_true)
    if (
        starts is None
    ):  # random starts (seed = sample), best one finished with the full budget
        best = None
        for s in np.random.default_rng(k).random((10, N_CONSTANTS)):
            uf, res = lsq(prob, s[prob.free], max_nfev=100)
            if best is None or res.cost < best[1].cost:
                best = (uf, res)
        uf, res = lsq(prob, best[0])
    else:
        uf, res = lsq(prob, starts[prob.free])
    u = prob.full(uf)
    r = {
        "test_index": int(k),
        "u": u.tolist(),
        "u_true": u_true.tolist(),
        "theta": (vmin + u * (vmax - vmin)).tolist(),
        "cost": prob.cost(u),
        "n_eig": prob.n_eig,
        "time_s": time.time() - t0,
        "stop": res.message,
        "fixed": [int(i) for i in fixed],
        "n_modes": int(len(target)),
    }
    r["cost_true"] = Problem(target, vmin, vmax, gaps).cost(
        u_true
    )  # > 0 with noise or missing modes
    (out / f"{k}.json").write_text(json.dumps(r, indent=1))
    return f"{k}: cost {r['cost']:.2e}, {r['n_eig']} eig, {r['time_s']:.0f}s"


def _init_worker(threads):
    global _fwd
    torch.set_num_threads(threads)
    _fwd = Forward("cpu")


# ----------------------------------------------------------------------------- data
def degrade_targets(x_clean, k_max, noise, p_aug=0.5, seed=0):
    """Degraded eigenvalues (physical units, 0 = no mode): half of the samples lose 1 to k_max random modes, half
    get a relative frequency error with a std up to noise. Fixed CPU draw (seed), the same on every machine."""
    xs = torch.from_numpy(dataset.to_sequence(x_clean))
    g, n = torch.Generator().manual_seed(seed), len(xs)
    on = lambda: torch.rand(n, generator=g) < p_aug
    k = (
        torch.randint(1, k_max + 1, (n,), generator=g) * on()
        if k_max
        else torch.zeros(n, dtype=torch.long)
    )
    sigma = noise * torch.rand(n, generator=g) * on()
    present = xs > 0
    rank = (
        torch.where(present, torch.rand(xs.shape, generator=g), torch.inf)
        .argsort(1)
        .argsort(1)
    )
    present &= rank >= k[:, None]
    eig = EIG_MIN + xs.double() * (EIG_MAX - EIG_MIN)
    eig = (
        eig
        * (
            1
            + sigma[:, None].double()
            * torch.randn(xs.shape, generator=g, dtype=torch.float64)
        )
        ** 2
    )
    xd = ((eig - EIG_MIN) / (EIG_MAX - EIG_MIN)).clamp_min(1e-6)
    xd = torch.where(present, xd, torch.inf).sort(1).values
    xd = torch.where(torch.isinf(xd), torch.zeros_like(xd), xd).numpy()
    return np.where(xd > 0, EIG_MIN + xd * (EIG_MAX - EIG_MIN), 0.0)


# ----------------------------------------------------------------------------- summary
def summary(out):
    rs = [json.loads(f.read_text()) for f in sorted(Path(out).glob("*.json"))]
    if not rs:
        return f"{out}: no results"
    err = 100 * np.abs(
        np.array([r["u"] for r in rs]) - np.array([r["u_true"] for r in rs])
    )
    cost, cost_true = (
        np.array([r["cost"] for r in rs]),
        np.array([r["cost_true"] for r in rs]),
    )
    exact, good = cost < EXACT, cost <= cost_true * (1 + 1e-6) + 1e-14
    t, eig = np.mean([r["time_s"] for r in rs]), np.mean([r["n_eig"] for r in rs])
    lines = [
        f"{out}: {len(rs)} samples, exact fits {100 * exact.mean():.1f} %, fits as good as the truth "
        f"{100 * good.mean():.1f} %, {eig:.1f} eigendecompositions and {t:.1f} s per sample",
        f"{'MAE %':>12}" + "".join(f"{c:>8}" for c in CONSTANT_NAMES + ["mean"]),
    ]
    for name, m in [("all", np.ones(len(rs), bool)), ("exact fits", exact)]:
        if m.any():
            e = err[m].mean(0)
            lines.append(
                f"{name:>12}" + "".join(f"{v:8.1f}" for v in e) + f"{e.mean():8.2f}"
            )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument(
        "--start",
        default="random",
        help="'random' (10 random starts) or a .npy of starts (n_samples, 11)",
    )
    p.add_argument(
        "--export",
        type=Path,
        default=None,
        help="write the (degraded) target spectra to this .npy and exit",
    )
    p.add_argument(
        "--fix",
        nargs="*",
        default=[],
        help="constants fixed to their true value, e.g. epsS33",
    )
    p.add_argument(
        "--samples",
        type=Path,
        default=Path("lm_samples.txt"),
        help="test-set positions (created if missing)",
    )
    p.add_argument(
        "--n",
        type=int,
        default=1000,
        help="samples drawn when --samples does not exist",
    )
    p.add_argument(
        "--drop_modes",
        type=int,
        default=0,
        help="degraded spectra: up to K missing modes",
    )
    p.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="degraded spectra: relative noise up to S",
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads", type=int, default=4, help="torch threads per worker")
    p.add_argument("--out", type=Path, default=Path("results/lm"))
    p.add_argument(
        "--summary",
        nargs="+",
        type=Path,
        default=None,
        help="only print the summary of these dirs",
    )
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.summary:
        return print("\n\n".join(summary(d) for d in a.summary))

    x_raw, y_raw = dataset.load(a.data)
    i_tr, _, i_te = dataset.split(len(x_raw))
    if not a.samples.exists():  # draw the samples once, then always reuse the same file
        a.samples.parent.mkdir(parents=True, exist_ok=True)
        draw = np.random.default_rng(a.seed).choice(len(i_te), a.n, replace=False)
        a.samples.write_text("\n".join(map(str, draw)) + "\n")
        print(f"{a.n} test samples drawn (seed {a.seed}) -> {a.samples}")
    pos = np.array(a.samples.read_text().split(), dtype=int)
    x, y = x_raw[i_te[pos]], y_raw[i_te[pos]]
    gaps = bool(a.drop_modes or a.noise)
    if gaps:
        x = degrade_targets(x, a.drop_modes, a.noise)
    if a.export:
        np.save(a.export, x)
        return print(f"{len(x)} target spectra -> {a.export}")
    scaler = dataset.LabelScaler.fit(y_raw[i_tr])  # the sampling box, as for the networks
    if a.start == "random":
        starts = [None] * len(pos)
    else:
        theta0 = np.load(a.start)
        assert theta0.shape == (len(pos), N_CONSTANTS), f"{a.start}: {theta0.shape}, expected {(len(pos), N_CONSTANTS)}"
        starts = np.clip((theta0 - scaler.vmin) / (scaler.vmax - scaler.vmin), 0.0, 1.0)
    vmin, vmax = scaler.vmin, scaler.vmax
    u_true = (y - vmin) / (vmax - vmin)
    fixed = dataset.known_index(a.fix)

    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (int(k), x[i][x[i] > 0], starts[i], u_true[i], vmin, vmax, fixed, gaps, a.out)
        for i, k in enumerate(pos)
        if not (a.out / f"{k}.json").exists()
    ]
    print(
        f"{len(jobs)} samples to invert, {a.workers} workers x {a.threads} threads",
        flush=True,
    )
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    with mp.get_context("spawn").Pool(
        a.workers, initializer=_init_worker, initargs=(a.threads,)
    ) as pool:
        for msg in pool.imap_unordered(invert, jobs):
            print(msg, flush=True)
    print(summary(a.out))


if __name__ == "__main__":
    main()
