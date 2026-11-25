"""Dataset loading, train/validation/test split, input representations and label scaling.

A dataset is a directory written by generate.py, holding constants.npy (N, 11, physical units) and
eigenvalues.npy (N, N_MODES: eigenvalues rho * omega^2 in ]EIG_MIN, EIG_MAX[, ascending, zero padded).

Input representations of an eigenvalue vector:
  seq  the sequence itself, min-max scaled on [EIG_MIN, EIG_MAX] (padding zeros stay 0)
  bin  histogram counting the modes on n_bins regular bins of [EIG_MIN, EIG_MAX]
  rgb  image of Fukuda et al.: binary histogram on side^2 bins mapped onto 3 x side x side
       (R: spiral from the centre, G: spiral from the border, flipped, B: row by row)
       rgb from loic dubois reimplementation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from constants import EIG_MAX, EIG_MIN, N_MODES

SPLIT_SEED = 4444


def load(data_dir: Path, n_max: int | None = None):
    """(eigenvalues (N, N_MODES), constants (N, 11)), float64."""
    x = np.load(Path(data_dir) / "eigenvalues.npy", mmap_mode="r")
    y = np.load(Path(data_dir) / "constants.npy", mmap_mode="r")
    n = len(x) if n_max is None else min(n_max, len(x))
    return np.asarray(x[:n, :N_MODES], dtype=np.float64), np.asarray(
        y[:n], dtype=np.float64
    )


def split(n: int, seed: int = SPLIT_SEED, fractions=(0.8, 0.1, 0.1)):
    """Reproducible random permutation -> train, validation and test indices."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).numpy()
    n_tr, n_va = int(n * fractions[0]), int(n * fractions[1])
    return perm[:n_tr], perm[n_tr : n_tr + n_va], perm[n_tr + n_va :]


@dataclass
class LabelScaler:
    """Per-constant min-max scaling fitted on the training set (= the sampling box)."""

    vmin: np.ndarray
    vmax: np.ndarray

    @classmethod
    def fit(cls, y):
        _, vmin, vmax, _, _ = normalize_labels(y, "max_min")
        return cls(vmin[0], vmax[0])

    def transform(self, y):
        return ((y - self.vmin) / (self.vmax - self.vmin)).astype(np.float32)

    def inverse(self, u):
        return self.vmin + np.asarray(u, dtype=np.float64) * (self.vmax - self.vmin)


def red(vect_freq, img_side):
    nb_pixels = img_side**2  # nombre de pixels total d'une image
    img_R = np.zeros(
        (img_side, img_side), dtype=int
    )  # création d'une matrice de zéros qui correspond à notre image

    # on calcule r qui est l'écart entre le nb de pixels et la taille d'un vecteur de fréquence
    r = nb_pixels - len(vect_freq)
    ajout = np.zeros(r, dtype=int)  # on créé un vecteur de r zéros
    # on l'ajoute au vecteur de fréquences pour avoir un vecteur de img_size entiers
    vect_img = np.append(vect_freq, ajout)

    if img_side % 2 == 0:
        i = np.int64((img_side / 2) - 1)
        n = 2
    else:
        i = np.int64((img_side - 1) / 2)
        n = 1
    while n != img_side + 2:
        m = len(vect_img)
        if n == 2 and img_side % 2 == 0:  # dans le cas d'une image "paire"
            img_R[i, i : n + i] = vect_img[0:n]
            img_R[i + 1 : n + i, n - 1 + i] = vect_img[n : 2 * n - 1]
            img_R[n - 1 + i, i : n - 1 + i] = vect_img[2 * n - 1 : 3 * n - 2]
            vect_img = vect_img[3 * n - 2 : m]
            n = n + 2
        elif n == 1:  # "impaire"
            img_R[i, i] = vect_img[0]
            vect_img = vect_img[1:m]
            n = n + 2
        else:
            img_R[i, i : n + i] = vect_img[n - 2 : 2 * n - 2]  # L1
            img_R[i + 1 : n + i, n - 1 + i] = vect_img[2 * n - 2 : 3 * n - 3]  # C1
            img_R[n - 1 + i, i : n - 1 + i] = list(
                reversed(vect_img[3 * n - 3 : 4 * n - 4])
            )  # L2
            img_R[i + 1 : n - 1 + i, i] = list(reversed(vect_img[0 : n - 2]))  # C2

            vect_img = vect_img[4 * n - 4 : m]
            n = n + 2
        i = i - 1

    return img_R


def green(vect_freq, img_side):
    nb_pixels = img_side**2  # nombre de pixels total d'une image
    img_G = np.zeros(
        (img_side, img_side), dtype=int
    )  # création d'une matrice de zéros qui correspond à notre image

    # on calcule r qui est l'écart entre le nb de pixels et la taille d'un vecteur de fréquence
    r = nb_pixels - len(vect_freq)
    ajout = np.zeros(r, dtype=int)  # on créé un vecteur de r zéros
    # on l'ajoute au vecteur de fréquences pour avoir un vecteur de img_size entiers
    vect_img = np.append(vect_freq, ajout)

    i = 0
    n = img_side
    while n != 0:
        if n == 2:  # dans le cas d'une image "paire"
            img_G[i, i : n + i] = vect_img[0:n]
            img_G[i + 1 : n + i, n - 1 + i] = vect_img[n : 2 * n - 1]
            img_G[n - 1 + i, i : n - 1 + i] = vect_img[2 * n - 1 : 3 * n - 2]
            n = n - 2
        elif n == 1:  # "impaire"
            img_G[i, i] = vect_img[0]
            n = n - 1
        else:
            img_G[i, i : n + i] = vect_img[0:n]  # L1

            img_G[i + 1 : n + i, n - 1 + i] = vect_img[n : 2 * n - 1]

            img_G[n - 1 + i, i : n - 1 + i] = list(
                reversed(vect_img[2 * n - 1 : 3 * n - 2])
            )

            img_G[i + 1 : n - 1 + i, i] = list(
                reversed(vect_img[3 * n - 2 : 4 * n - 4])
            )

            m = len(vect_img)
            vect_img = vect_img[4 * n - 4 : m]
            n = n - 2
        i = i + 1

    # L'image telle qu'elle est créée ici commence en partant du coin en haut à gauche.
    # Dans l'article, l'image est créée en partant du coin en bas à gauche.
    # Pour remédier à cela, on utilise la fonction flipud qui renverse l'image verticalement
    img_G = np.flipud(img_G)

    return img_G


def blue(vect_freq, img_side):
    nb_pixels = img_side**2  # nombre de pixels total d'une image
    img_B = np.zeros(
        (img_side, img_side), dtype=int
    )  # création d'une matrice de zéros qui correspond à notre image

    # on calcule r qui est l'écart entre le nb de pixels et la taille d'un vecteur de fréquence
    r = nb_pixels - len(vect_freq)
    ajout = np.zeros(r, dtype=int)  # on créé un vecteur de r zéros
    # on l'ajoute au vecteur de fréquences pour avoir un vecteur de img_size entiers
    vect_img = np.append(vect_freq, ajout)

    for i in range(img_side):
        # print(np.shape(vect_img[img_side*i:img_side*(i+1)]))
        img_B[i] = vect_img[img_side * i : img_side * (i + 1)]

    return img_B


def normalize_labels(y_dataset_cstes, choix):
    data_cstes_norm = np.copy(y_dataset_cstes)
    n = np.shape(data_cstes_norm)[1]
    min_cstes = np.zeros((1, n), dtype=np.float64)
    max_cstes = np.zeros((1, n), dtype=np.float64)
    moy_e31 = np.ones(np.shape(data_cstes_norm)[0], dtype=np.float64)
    std_e31 = np.ones(np.shape(data_cstes_norm)[0], dtype=np.float64)

    if choix == "max_min":
        for i in range(n):
            min = np.min(data_cstes_norm[:, i])
            max = np.max(data_cstes_norm[:, i])
            min_cstes[0, i] = min
            max_cstes[0, i] = max
            data_cstes_norm[:, i] = (data_cstes_norm[:, i] - min) / (max - min)

        moy_e31 = moy_e31 * 0
        std_e31 = std_e31 * 0

    elif choix == "moy_std_e31":
        for i in range(n):
            if i != 6:
                min = np.min(data_cstes_norm[:, i])
                max = np.max(data_cstes_norm[:, i])
                min_cstes[0, i] = min
                max_cstes[0, i] = max
                data_cstes_norm[:, i] = (data_cstes_norm[:, i] - min) / (max - min)

        std = np.std(data_cstes_norm[:, 6], dtype=np.float64)
        moy = np.mean(data_cstes_norm[:, 6], dtype=np.float64)
        moy_e31 = moy_e31 * moy
        std_e31 = std_e31 * std
        data_cstes_norm[:, 6] = (data_cstes_norm[:, 6] - moy_e31) / std_e31

    return data_cstes_norm, min_cstes, max_cstes, moy_e31, std_e31


def normalize_data(data, Low, Up):
    data_norm = np.copy(data)
    n = np.shape(data_norm)
    min_plage = Low * np.ones(n, dtype=np.float64)
    max_plage = Up * np.ones(n, dtype=np.float64)

    data_norm = (data_norm - min_plage) / (max_plage - min_plage)

    a = np.where(data == 0)
    data_norm[a] = 0

    return data_norm


def to_sequence(x, chunk=200_000):
    """normalize_data, by chunks (it allocates two arrays of the size of its input)."""
    return np.concatenate(
        [
            normalize_data(x[s : s + chunk], EIG_MIN, EIG_MAX).astype(np.float32)
            for s in range(0, len(x), chunk)
        ]
    )


def degrade(x, k=None, sigma=None, low=None):
    """Turns simulated spectra into measured-like ones (torch, on the GPU).

    x      (B, N_MODES) spectra in the seq representation: sorted eigenvalues scaled to [0, 1], 0 = padding
    k      (B,) number of modes removed from each spectrum (modes not detected in a measurement)
    low    (B,) bool, optional: True -> remove the k lowest modes instead of k random ones
    sigma  (B,) relative error on the frequencies: each frequency is multiplied by (1 + z), z ~ N(0, sigma),
           so each eigenvalue rho * omega^2 by (1 + z)^2

    Returns spectra in the same format: remaining modes sorted at the start of the row, zeros at the end.
    """
    present = x > 0
    if k is not None:
        key = torch.rand_like(x)  # removal order: random...
        if low is not None:
            key = torch.where(low[:, None], x, key)  # ...or lowest first
        rank = torch.where(present, key, torch.inf).argsort(1).argsort(1)
        present &= rank >= k[:, None]  # drop the k first in that order
    if sigma is not None:
        eig = EIG_MIN + x * (EIG_MAX - EIG_MIN)  # back to physical eigenvalues
        eig = eig * (1 + sigma[:, None] * torch.randn_like(x)) ** 2
        x = ((eig - EIG_MIN) / (EIG_MAX - EIG_MIN)).clamp_min(1e-6)  # a mode stays > 0
    x = (
        torch.where(present, x, torch.inf).sort(1).values
    )  # removed modes go to the end...
    return torch.where(
        torch.isinf(x), torch.zeros_like(x), x
    )  # ...and become padding zeros


def to_bins(x, n_bins):
    """Count histogram: each mode is counted in the bin whose centre is the nearest."""
    step = (EIG_MAX - EIG_MIN) / (n_bins - 1)
    rows, cols = np.nonzero(x)
    idx = np.rint((x[rows, cols] - EIG_MIN) / step).astype(
        np.int64
    )  # rint: index of the nearest bin centre
    counts = np.bincount(rows * n_bins + idx, minlength=len(x) * n_bins)
    return counts.reshape(len(x), n_bins).astype(np.float32)


def to_rgb(x, n_bins):
    side = int(round(n_bins**0.5))
    if side * side != n_bins:
        raise ValueError(f"rgb: n_bins must be a perfect square (got {n_bins})")
    index = np.arange(
        1, side * side + 1
    )  # pixel layout of Bin2RGB: bin number of each pixel
    maps = np.stack([red(index, side), green(index, side), blue(index, side)])
    binary = (to_bins(x, n_bins) > 0).astype(np.float32)  # = C2Bin, vectorised
    padded = np.concatenate(
        [np.zeros((len(x), 1), np.float32), binary], 1
    )  # index 0 = empty pixel
    return padded[:, maps]  # (N, 3, side, side)


def from_sequence(s, kind: str, n_bins: int = 225):
    """Builds the network input `kind` (seq, bin or rgb) from spectra in the seq representation (torch).

    s  (B, N_MODES) sorted eigenvalues scaled to [0, 1], 0 = padding

    Needed because missing modes and noise are applied to the sequence: the bin and rgb inputs are then rebuilt
    from it, batch by batch during training.
    """
    if kind == "seq":
        return s
    idx = (
        torch.round(s * (n_bins - 1)).long() + 1
    )  # bin of each mode, numbered from 1 (0 = no mode)
    idx = torch.where(
        s > 0, idx.clamp(1, n_bins), torch.zeros_like(idx)
    )  # noise may push a mode out of range
    counts = torch.zeros(len(s), n_bins + 1, device=s.device).scatter_add_(
        1, idx, torch.ones_like(s)
    )  # modes per bin
    if kind == "bin":
        return counts[:, 1:]  # histogram (drop column 0 = padding)
    if (
        kind == "rgb"
    ):  # image of Fukuda et al.: occupied bins, placed on 3 pixel layouts
        side = int(round(n_bins**0.5))
        index = np.arange(1, side * side + 1)
        maps = torch.as_tensor(
            np.stack([red(index, side), green(index, side), blue(index, side)]),
            device=s.device,
        ).long()
        binary = (counts > 0).float()
        binary[:, 0] = 0  # padding -> empty pixel
        return binary[:, maps]
    raise ValueError(f"unknown representation: {kind}")


def known_index(names):
    """Names given to --known (e.g. ["epsS33"]) -> their columns in CONSTANT_NAMES (e.g. [10])."""
    from constants import CONSTANT_NAMES

    return [
        CONSTANT_NAMES.index(n) if isinstance(n, str) else int(n) for n in names or []
    ]


def add_known(x, y_scaled, names):
    """Appends the true value of the --known constants to each input row, as if measured separately
    (e.g. eps33^S from the capacitance), which removes the invariance e -> s e, eps^S -> s^2 eps^S.

    x         (B, N) network inputs, 1D representation (seq or bin), numpy (train.py) or torch (evaluate.py, lm.py)
    y_scaled  (B, 11) true constants scaled to [0, 1]
    names     the --known constants, e.g. ["epsS33"]

    Returns (B, N + len(names)). The ResNet takes these last columns out before the convolutions and feeds them
    to the final dense layer, next to the CNN features (models.ResNet, n_extra).
    """
    idx = known_index(names)
    if not idx:
        return x
    if isinstance(x, torch.Tensor):
        return torch.cat(
            [x, torch.as_tensor(y_scaled[:, idx], dtype=x.dtype, device=x.device)], 1
        )
    return np.concatenate([x, y_scaled[:, idx]], 1).astype(np.float32)


def representation(x, kind: str, n_bins: int = 225):
    if kind == "seq":
        return to_sequence(x)
    if kind == "bin":
        return to_bins(x, n_bins)
    if kind == "rgb":
        return to_rgb(x, n_bins)
    raise ValueError(f"unknown representation: {kind}")
