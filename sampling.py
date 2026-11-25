"""Random admissible material constants: uniform in the sampling box, rejection of non-physical vectors.
From loic dubois
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from constants import (
    BOX_HIGH,
    BOX_LOW,
    EIG_MAX,
    EIG_MIN,
    EPS0,
    GEOMETRY,
    N_CONSTANTS,
    POLY_ORDER,
    PZ26,
    RHO,
)
from forward import material_tensors


class PhysicalFilters(nn.Module):
    def __init__(self):
        self.set_device = "cpu"
        self.dtype = torch.float64

        # Material and analytical model parameters
        vect_init = PZ26.copy()
        vect_init[9:] /= EPS0  # permittivities relative to eps0
        self.vect_init = torch.tensor(vect_init, device=self.set_device)
        self.n_ordre = POLY_ORDER
        self.eps0 = EPS0
        self.rho = RHO
        self.l1, self.l2, self.l3 = GEOMETRY
        self.lim_inf = EIG_MIN
        self.lim_sup = EIG_MAX

    def conv_sE_d_epsT(self, cE, e, epsS):
        sE = torch.linalg.inv(cE)
        d = torch.bmm(e, sE)
        epsT = epsS + torch.bmm(d, torch.transpose(e, 1, 2))
        return sE, d, epsT

    def conv_cD_h_betaS(self, cE, e, epsS):
        betaS = torch.linalg.inv(epsS)
        h = torch.bmm(betaS, e)
        cD = cE + torch.bmm(torch.transpose(e, 1, 2), h)
        return cD, h, betaS

    def const_2_tensor(self, cste):
        c_tens = torch.unsqueeze(cste, 1)
        c_tens = torch.unsqueeze(c_tens, 1)
        return c_tens

    def filtre_electromeca(self, cE, e, epsS, sE, d, epsT, cD):
        bsize = cE.size()[0]
        koeff = torch.zeros((bsize, 6), dtype=self.dtype, device=self.set_device)

        d31 = self.const_2_tensor(d[:, 2, 0])
        sE11 = self.const_2_tensor(sE[:, 0, 0])
        epsT33 = self.const_2_tensor(epsT[:, 2, 2])
        d33 = self.const_2_tensor(d[:, 2, 2])
        sE33 = self.const_2_tensor(sE[:, 2, 2])
        e33 = self.const_2_tensor(e[:, 2, 2])
        cD33 = self.const_2_tensor(cD[:, 2, 2])
        epsS33 = self.const_2_tensor(epsS[:, 2, 2])
        e15 = self.const_2_tensor(e[:, 0, 4])
        cD55 = self.const_2_tensor(cD[:, 4, 4])
        epsS11 = self.const_2_tensor(epsS[:, 0, 0])
        sE11 = self.const_2_tensor(sE[:, 0, 0])
        sE12 = self.const_2_tensor(sE[:, 0, 1])
        sE13 = self.const_2_tensor(sE[:, 0, 2])

        k31 = torch.div(torch.bmm(d31, d31), torch.bmm(sE11, epsT33))
        k33 = torch.div(torch.bmm(d33, d33), torch.bmm(sE33, epsT33))
        kt = torch.div(torch.bmm(e33, e33), torch.bmm(cD33, epsS33))
        k15 = torch.div(torch.bmm(e15, e15), torch.bmm(cD55, epsS11))

        sigma = -torch.div(sE12, sE11)
        dh = 2 * d31 + d33
        shE = 2 * (sE11 + sE12) + 4 * sE13 + sE33

        kp = torch.bmm(
            (2 / (1 - sigma)), torch.div(torch.bmm(d31, d31), torch.bmm(sE11, epsT33))
        )
        kh = torch.div(torch.bmm(dh, dh), torch.bmm(shE, epsT33))

        koeff[:, 0] = k31[:, 0, 0]
        koeff[:, 1] = k33[:, 0, 0]
        koeff[:, 2] = kt[:, 0, 0]
        koeff[:, 3] = k15[:, 0, 0]
        koeff[:, 4] = kp[:, 0, 0]
        koeff[:, 5] = kh[:, 0, 0]

        filtretp = torch.ones(1, bsize, device=self.set_device)

        cond1 = torch.min(koeff, 1)[0] < 0
        cond2 = torch.max(koeff, 1)[0] > 1

        filtre1 = filtretp * cond1
        filtre2 = filtretp * cond2

        filtre = filtretp * torch.logical_not(torch.logical_or(filtre1, filtre2))

        return filtre, koeff

    def filtre_thermo(self, sE, d, epsT):
        long = sE.size()[0]
        m = torch.zeros((long, 4, 4), dtype=self.dtype, device=self.set_device)

        m[:, 0, 1] = d[:, 2, 0]
        m[:, 0, 2] = d[:, 2, 0]
        m[:, 0, 3] = d[:, 2, 2]
        m[:, 1, 2] = sE[:, 0, 1]
        m[:, 1, 3] = sE[:, 0, 2]
        m[:, 2, 3] = sE[:, 0, 2]

        m = m + torch.transpose(m, 1, 2)

        m[:, 0, 0] = epsT[:, 2, 2]
        m[:, 1, 1] = sE[:, 0, 0]
        m[:, 2, 2] = sE[:, 0, 0]
        m[:, 3, 3] = sE[:, 2, 2]

        filtretp = torch.ones(1, long, device=self.set_device)
        determinant = torch.linalg.det(m)

        cond = determinant > 0
        filtre = filtretp * cond

        return filtre, determinant, m


_FILTERS = PhysicalFilters()


def physical(theta: np.ndarray) -> np.ndarray:
    """True for each constant vector (B, 11) that is physically possible: all coupling factors
    between 0 and 1, and thermodynamically stable using PhysicalFilters."""
    c, e, k = (torch.as_tensor(a) for a in material_tensors(theta))
    s_e, d, eps_t = _FILTERS.conv_sE_d_epsT(c, e, k)
    c_d, _, _ = _FILTERS.conv_cD_h_betaS(c, e, k)
    electromechanical, _ = _FILTERS.filtre_electromeca(c, e, k, s_e, d, eps_t, c_d)
    thermodynamic, _, _ = _FILTERS.filtre_thermo(s_e, d, eps_t)
    return torch.logical_and(electromechanical, thermodynamic)[0].numpy()


def sample_constants(
    n: int, rng: np.random.Generator, low=BOX_LOW, high=BOX_HIGH
) -> np.ndarray:
    """n admissible constant vectors, uniform in the box [low, high] restricted to physical() using rejection sampling."""
    out = []
    while sum(len(o) for o in out) < n:
        t = low + (high - low) * rng.random((2 * n + 16, N_CONSTANTS))
        out.append(t[physical(t)])
    return np.concatenate(out)[:n]
