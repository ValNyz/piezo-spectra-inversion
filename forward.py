"""Forward model: computes the resonance spectrum of a piezoelectric sample from its 11 material constants.

The spectrum is given by the eigenvalues rho * omega^2 of the matrix

    M = Gamma(c^E) + Omega(e) Lambda(eps^S)^-1 Omega(e)^T

built with the Rayleigh-Ritz method
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from constants import EIG_MAX, EIG_MIN, EPS0, GEOMETRY, N_MODES, POLY_ORDER, RHO

# Block (a, b) of Gamma (a, b = displacement components) = sum_k c[i_k, j_k] g_k, (i_k, j_k) listed for k = 0..8
_GAMMA = {
    (0, 0): [(0, 0), (5, 5), (4, 4), (4, 5), (4, 5), (0, 4), (0, 4), (5, 0), (0, 5)],
    (1, 1): [(5, 5), (1, 1), (3, 3), (1, 3), (1, 3), (3, 5), (3, 5), (1, 5), (1, 5)],
    (2, 2): [(4, 4), (3, 3), (2, 2), (2, 3), (2, 3), (2, 4), (2, 4), (3, 4), (3, 4)],
    (2, 1): [(4, 5), (3, 1), (2, 3), (2, 1), (3, 3), (4, 3), (2, 5), (3, 5), (4, 1)],
    (1, 2): [(4, 5), (1, 3), (2, 3), (3, 3), (1, 2), (2, 5), (3, 4), (1, 4), (3, 5)],
    (0, 2): [(0, 4), (3, 5), (2, 4), (3, 4), (2, 5), (0, 2), (4, 4), (4, 5), (0, 3)],
    (2, 0): [(0, 4), (3, 5), (2, 4), (2, 5), (3, 4), (4, 4), (0, 2), (3, 0), (4, 5)],
    (1, 0): [(5, 0), (1, 5), (3, 4), (3, 5), (1, 4), (5, 4), (3, 0), (1, 0), (5, 5)],
    (0, 1): [(0, 5), (1, 5), (3, 4), (1, 4), (3, 5), (0, 3), (4, 5), (5, 5), (0, 1)],
}
# Block a of Omega (displacement component a x potential) = sum_k e[i_k, j_k] g_k
_OMEGA = {
    0: [(0, 0), (1, 5), (2, 4), (1, 4), (2, 5), (2, 0), (0, 4), (0, 5), (1, 0)],
    1: [(0, 5), (1, 1), (2, 3), (1, 3), (2, 1), (2, 5), (0, 3), (0, 1), (1, 5)],
    2: [(0, 4), (1, 3), (2, 2), (1, 2), (2, 3), (2, 4), (0, 2), (0, 3), (1, 4)],
}
# Lambda = sum_k eps[k, k] g_k, k = 0..2


def material_tensors(theta: np.ndarray):
    """(B, 11) constants -> c^E (B, 6, 6), e (B, 3, 6), eps^S (B, 3, 3), Voigt notation, 4mm symmetry."""
    t = np.atleast_2d(theta)
    b = len(t)
    c, e, k = np.zeros((b, 6, 6)), np.zeros((b, 3, 6)), np.zeros((b, 3, 3))
    c[:, 0, 0] = c[:, 1, 1] = t[:, 0]
    c[:, 0, 1] = c[:, 1, 0] = t[:, 1]
    c[:, 0, 2] = c[:, 2, 0] = c[:, 1, 2] = c[:, 2, 1] = t[:, 2]
    c[:, 2, 2] = t[:, 3]
    c[:, 3, 3] = c[:, 4, 4] = t[:, 4]
    c[:, 5, 5] = t[:, 5]
    e[:, 2, 0] = e[:, 2, 1] = t[:, 6]
    e[:, 2, 2] = t[:, 7]
    e[:, 0, 4] = e[:, 1, 3] = t[:, 8]
    k[:, 0, 0] = k[:, 1, 1] = t[:, 9]
    k[:, 2, 2] = t[:, 10]
    return c, e, k


torch.pi = torch.tensor(math.pi)


class PiezoAnalyticalModel(nn.Module):
    def __init__(
        self,
        n_ordre,
        eps0,
        rho,
        l1,
        l2,
        l3,
        lim_inf,
        lim_sup,
        device,
        dtype=torch.float32,
    ):
        self.n = n_ordre
        self.device = device
        self.dtype = dtype

        self.rmax = torch.tensor(
            (2 * n_ordre**3 + 12 * n_ordre**2 + 22 * n_ordre) // 12, dtype=torch.int64
        )
        self.pmax = self.rmax * 3

        self.modes = self._bijection_f_V2()

        self.D = self._mat_D()  # matrice deplacement electrique
        self.E = self._mat_E()  # matrice champ electrique

        self.lambd, self.mu, self.nu, self.n = self._association()

        self.eps0 = eps0
        self.rho = rho
        self.l1 = l1 / 2
        self.l2 = l2 / 2
        self.l3 = l3 / 2
        self.lim_inf = lim_inf
        self.lim_sup = lim_sup

        if dtype == torch.float64:
            self.G_gamma = self._G_gamma().double()
            self.G_omega = self._G_omega().double()
            self.G_lambda = self._G_lambda().double()
        else:
            self.G_gamma = self._G_gamma()
            self.G_omega = self._G_omega()
            self.G_lambda = self._G_lambda()

    def to(self, device):
        self.modes.to(device)
        self.D.to(device)
        self.E.to(device)
        self.G_gamma.to(device)
        self.G_omega.to(device)
        self.G_lambda.to(device)

    def _bijection_f_V1(self, n):
        # cpu is faster for small n
        p = (n + 1) * (n + 2) // 2

        # Calcul des indices i
        i = torch.repeat_interleave(
            torch.arange(1, n + 2, device=self.device),
            torch.arange(n + 1, 0, -1, device=self.device),
        )

        # Calcul des indices j correspondants
        j = torch.cat(
            [torch.arange(1, n - k + 3, device=self.device) for k in range(1, n + 2)]
        )

        # Construction de la matrice
        mat = torch.zeros((p, 4), dtype=torch.int64, device=self.device)
        mat[:, 0] = torch.arange(1, p + 1)  # Numéro de ligne
        mat[:, 1] = i - 1  # i-1
        mat[:, 2] = j - 1  # j-1
        mat[:, 3] = n - (i - 1) - (j - 1)  # n - (i-1) - (j-1)
        return mat

    def _bijection_f_V2(self):
        # Étape 1 : calcul des tailles de blocs
        ks = torch.arange(1, self.n + 1, dtype=torch.int64, device=self.device)
        nb_lines_per_k = (ks + 1) * (ks + 2) // 2  # taille de chaque sous-bloc
        offsets = (
            2 * (ks - 1) ** 3 + 12 * (ks - 1) ** 2 + 22 * (ks - 1)
        ) // 12  # lignes de départ

        # Étape 2 : génération des blocs bijection_f_V1(k)
        blocks = [
            self._bijection_f_V1(k.item()) for k in ks
        ]  # liste de tensors [shape: (nb_lines_per_k[i], 4)]

        # Étape 3 : concaténation
        mat = torch.cat(blocks, dim=0)  # forme finale: (total_nb_lines, 4)

        # Étape 4 : incrément de la colonne 0 avec les offsets
        offsets_expanded = torch.cat(
            [offset.expand(nb) for offset, nb in zip(offsets, nb_lines_per_k)]
        )
        mat[:, 0] += offsets_expanded

        return mat

    def _mat_D(self):
        # Create index matrices for i and j
        i = torch.arange(self.n + 1, device=self.device, dtype=self.dtype).view(
            -1, 1
        )  # Shape (n+1, 1)
        j = torch.arange(self.n + 1, device=self.device, dtype=self.dtype).view(
            1, -1
        )  # Shape (1, n+1)

        # Apply condition where j >= i and (i + j) % 2 == 0
        mask = ((i + j) % 2 == 0) & (j >= i)
        D_upper = torch.sqrt((2 * i + 1) * (2 * j + 1)) * i * (i + 1) / 2 * mask

        D = D_upper + D_upper.T - torch.diag_embed(torch.diagonal(D_upper))

        return D

    def _mat_E(self):
        # Create index matrices for i and j
        i = torch.arange(0, self.n + 1, dtype=self.dtype, device=self.device).reshape(
            -1, 1
        )
        j = torch.arange(0, self.n + 1, dtype=self.dtype, device=self.device).reshape(
            1, -1
        )

        # Apply condition where j >= i and (i + j) % 2 == 1
        mask = ((i + j) % 2 == 1) & (j >= i)
        E = torch.sqrt((2 * i + 1) * (2 * j + 1)) * mask

        return E

    def _association(self):
        p = torch.arange(1, self.pmax + 1, device=self.device).unsqueeze(1)  # (Pmax, 1)

        u = torch.remainder(p, self.rmax)
        u = torch.where(u == 0, self.rmax, u)  # (Pmax, 1)
        u = u.squeeze(1)  # Shape: (Pmax,)

        n = torch.div(p, self.rmax, rounding_mode="floor")
        n = torch.where(torch.remainder(p, self.rmax) == 0, n, n + 1).squeeze(
            1
        )  # (Pmax,)

        # Create lookup table: u in [1, Rmax] → index in modes
        mode_u = self.modes[:, 0].long()
        mode_indices = torch.arange(self.modes.shape[0], device=self.device)

        lookup_table = torch.full(
            (self.rmax + 1,), -1, dtype=torch.long, device=self.device
        )
        lookup_table[mode_u] = mode_indices  # index of each u in modes

        # Get the mode index for each p via u
        matched_indices = lookup_table[u]  # Shape: (Pmax,)

        # Fetch lambda, mu, nu from modes
        lambda_arr = self.modes[matched_indices, 1]
        mu_arr = self.modes[matched_indices, 2]
        nu_arr = self.modes[matched_indices, 3]

        return lambda_arr, mu_arr, nu_arr, n

    def _G_gamma(self):
        lambda1, lambda2 = self.lambd.view(-1, 1), self.lambd.view(1, -1)
        mu1, mu2 = self.mu.view(-1, 1), self.mu.view(1, -1)
        nu1, nu2 = self.nu.view(-1, 1), self.nu.view(1, -1)

        # Precompute equality flags
        lam_eq = lambda1 == lambda2
        mu_eq = mu1 == mu2
        nu_eq = nu1 == nu2

        # Compute G terms one by one using batched indexing
        G = torch.stack(
            [
                self.D[lambda1, lambda2] * mu_eq * nu_eq / self.l1**2,
                lam_eq * self.D[mu1, mu2] * nu_eq / self.l2**2,
                lam_eq * mu_eq * self.D[nu1, nu2] / self.l3**2,
                lam_eq * self.E[mu1, mu2] * self.E[nu2, nu1] / (self.l2 * self.l3),
                lam_eq * self.E[mu2, mu1] * self.E[nu1, nu2] / (self.l2 * self.l3),
                self.E[lambda2, lambda1]
                * mu_eq
                * self.E[nu1, nu2]
                / (self.l3 * self.l1),
                self.E[lambda1, lambda2]
                * mu_eq
                * self.E[nu2, nu1]
                / (self.l3 * self.l1),
                self.E[lambda1, lambda2]
                * self.E[mu2, mu1]
                * nu_eq
                / (self.l1 * self.l2),
                self.E[lambda2, lambda1]
                * self.E[mu1, mu2]
                * nu_eq
                / (self.l1 * self.l2),
            ],
            dim=0,
        )  # Shape: (9, Pmax, Pmax)
        return G

    def gamma_matrix(self, c):
        n1, n2 = self.n.view(-1, 1), self.n.view(1, -1)

        def make_M_row(cond, values):
            vals = (
                torch.stack(values, dim=1).to(self.device).unsqueeze(-1).unsqueeze(-1)
            )  # (batch, 1, 1, 9)
            cond = cond.unsqueeze(0).float()  # (1, Pmax, Pmax)
            return (cond * vals).unsqueeze(1)  # (batch, 9, 1, Pmax, Pmax)

        rows = []
        rows.append(
            make_M_row(
                (n1 == 1) & (n2 == 1),
                [
                    c[:, 0, 0],
                    c[:, 5, 5],
                    c[:, 4, 4],
                    c[:, 4, 5],
                    c[:, 4, 5],
                    c[:, 0, 4],
                    c[:, 0, 4],
                    c[:, 5, 0],
                    c[:, 0, 5],
                ],
            )
        )
        rows.append(
            make_M_row(
                (n1 == 2) & (n2 == 2),
                [
                    c[:, 5, 5],
                    c[:, 1, 1],
                    c[:, 3, 3],
                    c[:, 1, 3],
                    c[:, 1, 3],
                    c[:, 3, 5],
                    c[:, 3, 5],
                    c[:, 1, 5],
                    c[:, 1, 5],
                ],
            )
        )
        rows.append(
            make_M_row(
                (n1 == 3) & (n2 == 3),
                [
                    c[:, 4, 4],
                    c[:, 3, 3],
                    c[:, 2, 2],
                    c[:, 2, 3],
                    c[:, 2, 3],
                    c[:, 2, 4],
                    c[:, 2, 4],
                    c[:, 3, 4],
                    c[:, 3, 4],
                ],
            )
        )
        rows.append(
            make_M_row(
                (n1 == 3) & (n2 == 2),
                [
                    c[:, 4, 5],
                    c[:, 3, 1],
                    c[:, 2, 3],
                    c[:, 2, 1],
                    c[:, 3, 3],
                    c[:, 4, 3],
                    c[:, 2, 5],
                    c[:, 3, 5],
                    c[:, 4, 1],
                ],
            )
        )
        rows.append(
            make_M_row(
                (n1 == 2) & (n2 == 3),
                [
                    c[:, 4, 5],
                    c[:, 1, 3],
                    c[:, 2, 3],
                    c[:, 3, 3],
                    c[:, 1, 2],
                    c[:, 2, 5],
                    c[:, 3, 4],
                    c[:, 1, 4],
                    c[:, 3, 5],
                ],
            )
        )
        rows.append(
            make_M_row(
                (n1 == 1) & (n2 == 3),
                [
                    c[:, 0, 4],
                    c[:, 3, 5],
                    c[:, 2, 4],
                    c[:, 3, 4],
                    c[:, 2, 5],
                    c[:, 0, 2],
                    c[:, 4, 4],
                    c[:, 4, 5],
                    c[:, 0, 3],
                ],
            )
        )
        rows.append(
            make_M_row(
                (n1 == 3) & (n2 == 1),
                [
                    c[:, 0, 4],
                    c[:, 3, 5],
                    c[:, 2, 4],
                    c[:, 2, 5],
                    c[:, 3, 4],
                    c[:, 4, 4],
                    c[:, 0, 2],
                    c[:, 3, 0],
                    c[:, 4, 5],
                ],
            )
        )
        rows.append(
            make_M_row(
                (n1 == 2) & (n2 == 1),
                [
                    c[:, 5, 0],
                    c[:, 1, 5],
                    c[:, 3, 4],
                    c[:, 3, 5],
                    c[:, 1, 4],
                    c[:, 5, 4],
                    c[:, 3, 0],
                    c[:, 1, 0],
                    c[:, 5, 5],
                ],
            )
        )
        rows.append(
            make_M_row(
                (n1 == 1) & (n2 == 2),
                [
                    c[:, 0, 5],
                    c[:, 1, 5],
                    c[:, 3, 4],
                    c[:, 1, 4],
                    c[:, 3, 5],
                    c[:, 0, 3],
                    c[:, 4, 5],
                    c[:, 5, 5],
                    c[:, 0, 1],
                ],
            )
        )

        M = torch.cat(rows, dim=1)  # (batch, 9, 9, Pmax, Pmax)

        # Tensor contraction: M[:,k,i,j] * G[k] summed over k
        gamma = torch.einsum("kij,bkij->bij", self.G_gamma, M.sum(dim=1))

        return gamma  # Shape: (batch_size, Pmax, Pmax)

    def _G_omega(self):
        lambda1, lambda2 = self.lambd.view(-1, 1), self.lambd.view(1, -1)
        mu1, mu2 = self.mu.view(-1, 1), self.mu.view(1, -1)
        nu1, nu2 = self.nu.view(-1, 1), self.nu.view(1, -1)

        # Precompute equality flags
        lam_eq = lambda1 == lambda2
        lam_eq = lam_eq[:, : self.rmax]
        mu_eq = mu1 == mu2
        mu_eq = mu_eq[:, : self.rmax]
        nu_eq = nu1 == nu2
        nu_eq = nu_eq[:, : self.rmax]

        # Compute G terms one by one using batched indexing
        G = torch.stack(
            [
                self.D[lambda1, lambda2][:, : self.rmax] * mu_eq * nu_eq / self.l1**2,
                lam_eq * self.D[mu1, mu2][:, : self.rmax] * nu_eq / self.l2**2,
                lam_eq * mu_eq * self.D[nu1, nu2][:, : self.rmax] / self.l3**2,
                lam_eq
                * self.E[mu1, mu2][:, : self.rmax]
                * self.E[nu2, nu1][:, : self.rmax]
                / (self.l2 * self.l3),
                lam_eq
                * self.E[mu2, mu1][:, : self.rmax]
                * self.E[nu1, nu2][:, : self.rmax]
                / (self.l2 * self.l3),
                self.E[lambda2, lambda1][:, : self.rmax]
                * mu_eq
                * self.E[nu1, nu2][:, : self.rmax]
                / (self.l3 * self.l1),
                self.E[lambda1, lambda2][:, : self.rmax]
                * mu_eq
                * self.E[nu2, nu1][:, : self.rmax]
                / (self.l3 * self.l1),
                self.E[lambda1, lambda2][:, : self.rmax]
                * self.E[mu2, mu1][:, : self.rmax]
                * nu_eq
                / (self.l1 * self.l2),
                self.E[lambda2, lambda1][:, : self.rmax]
                * self.E[mu1, mu2][:, : self.rmax]
                * nu_eq
                / (self.l1 * self.l2),
            ],
            dim=0,
        )  # Shape: (9, Pmax, Rmax)
        return G

    def omega_matrix(self, e):
        n1 = self.n.view(-1, 1).expand(-1, self.rmax)

        def make_M_row(cond, values):
            vals = (
                torch.stack(values, dim=1).to(self.device).unsqueeze(-1).unsqueeze(-1)
            )  # (batch, 9, 1, 1)
            cond = cond.unsqueeze(0).float()  # (1, Pmax, Rmax)
            return (cond * vals).unsqueeze(1)  # (batch, 1, 9, Pmax, Rmax)

        rows = []
        rows.append(
            make_M_row(
                n1 == 1,
                [
                    e[:, 0, 0],
                    e[:, 1, 5],
                    e[:, 2, 4],
                    e[:, 1, 4],
                    e[:, 2, 5],
                    e[:, 2, 0],
                    e[:, 0, 4],
                    e[:, 0, 5],
                    e[:, 1, 0],
                ],
            )
        )
        rows.append(
            make_M_row(
                n1 == 2,
                [
                    e[:, 0, 5],
                    e[:, 1, 1],
                    e[:, 2, 3],
                    e[:, 1, 3],
                    e[:, 2, 1],
                    e[:, 2, 5],
                    e[:, 0, 3],
                    e[:, 0, 1],
                    e[:, 1, 5],
                ],
            )
        )
        rows.append(
            make_M_row(
                n1 == 3,
                [
                    e[:, 0, 4],
                    e[:, 1, 3],
                    e[:, 2, 2],
                    e[:, 1, 2],
                    e[:, 2, 3],
                    e[:, 2, 4],
                    e[:, 0, 2],
                    e[:, 0, 3],
                    e[:, 1, 4],
                ],
            )
        )

        M = torch.cat(rows, dim=1)  # (batch, 3, 9, Pmax, Rmax)

        # Tensor contraction: M[:,k,i,j] * G[k] summed over k
        omega = torch.einsum("kij,bkij->bij", self.G_omega, M.sum(dim=1))

        return omega  # Shape: (batch_size, Pmax, Rmax)

    def _G_lambda(self):
        lambd = self.lambd[: self.rmax]
        mu = self.mu[: self.rmax]
        nu = self.nu[: self.rmax]
        lambda1, lambda2 = lambd.view(-1, 1), lambd.view(1, -1)
        mu1, mu2 = mu.view(-1, 1), mu.view(1, -1)
        nu1, nu2 = nu.view(-1, 1), nu.view(1, -1)

        # Precompute equality flags
        lam_eq = lambda1 == lambda2
        mu_eq = mu1 == mu2
        nu_eq = nu1 == nu2

        # Compute G terms one by one using batched indexing
        G = torch.stack(
            [
                self.D[lambda1, lambda2] * mu_eq * nu_eq / self.l1**2,
                lam_eq * self.D[mu1, mu2] * nu_eq / self.l2**2,
                lam_eq * mu_eq * self.D[nu1, nu2] / self.l3**2,
            ],
            dim=0,
        )  # Shape: (3, Rmax, Rmax)
        return G

    def lambda_matrix(self, k):
        # Extract scalar values (still differentiable) from k
        M0 = k[:, 0, 0].view(-1, 1, 1)  # (batch, 1, 1)
        M1 = k[:, 1, 1].view(-1, 1, 1)
        M2 = k[:, 2, 2].view(-1, 1, 1)

        # Expand to full Rmax x Rmax size
        M0 = M0.expand(-1, self.rmax, self.rmax)
        M1 = M1.expand(-1, self.rmax, self.rmax)
        M2 = M2.expand(-1, self.rmax, self.rmax)

        # Stack into M with shape (batch_size, 3, Rmax, Rmax)
        M = torch.stack([M0, M1, M2], dim=1)

        # Tensor contraction: M[k,j] * G[k] summed over k
        lambd = torch.einsum("kij,bkij->bij", self.G_lambda, M)

        return lambd  # Shape: (batch_size, Rmax, Rmax)

    def get_matrix(self, c, e, k):
        gamma = self.gamma_matrix(c)  # (B, P, P)

        omega = self.omega_matrix(e)  # (B, P, R)

        lambd = self.lambda_matrix(k)  # (B, R, R)

        # omega @ lamb_inv @ omega.T for each batch
        # torch.bmm -> batch matrix multiplication
        # (B, P, R) @ (B, R, R) -> (B, P, R)
        tmp = torch.bmm(omega, torch.linalg.inv(lambd))  # (B, P, R)

        # (B, P, R) @ (B, R, P) -> (B, P, P)
        omega_lambd_omegaT = torch.bmm(tmp, omega.transpose(-2, -1))

        # Final matrix: (B, P, P)
        matrix = gamma + omega_lambd_omegaT
        return matrix

    def _normalize_eigenvalues(self, eigenvalues):
        return torch.sqrt(eigenvalues / self.rho) / (2 * torch.pi)

    def forward(self, c, e, k):
        matrix = self.get_matrix(c, e, k)
        B, N, _ = matrix.shape

        # Step 1: Compute eigvals and eigvecs
        eigvals, eigvecs = torch.linalg.eig(
            matrix
        )  # eigvals: (B, N), eigvecs: (B, N, N)
        eigvals, eigvecs = eigvals.real, eigvecs.real
        # eigvals = self._normalize_eigenvalues(eigvals)

        # # Step 2: Mask for values within (lim_inf, lim_sup)
        mask = (eigvals > self.lim_inf) & (eigvals < self.lim_sup)  # (B, N)

        # # Step 3: Replace out-of-range values with large positive to ignore them in reverse top-k
        masked_eigvals = torch.where(
            mask, eigvals, torch.full_like(eigvals, float("inf"))
        )

        # # Step 4: Get top-k indices (safe even if some are inf, we'll pad later)
        topk_eigvals, topk_indices = torch.topk(
            masked_eigvals, k=N_MODES, dim=1, largest=False
        )

        # # Step 5: Gather top-k eigenvectors (B, 110, N)
        # eigvecs = eigvecs.transpose(1, 2)  # (B, 110, N)
        # topk_eigvecs = torch.gather(
        #     eigvecs, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, N)
        # )

        # # Step 6: Optional - zero out padded vectors (i.e., where eigval was inf)
        mask = topk_eigvals == float("inf")  # (B, 110)
        topk_eigvals = torch.where(mask, torch.zeros_like(topk_eigvals), topk_eigvals)
        # topk_eigvecs = torch.where(
        #     mask.unsqueeze(-1), torch.zeros_like(topk_eigvecs), topk_eigvecs
        # )

        return topk_eigvals, eigvecs


def elementary_integrals(order: int = POLY_ORDER, geometry=GEOMETRY) -> np.ndarray:
    ref = PiezoAnalyticalModel(
        order, EPS0, RHO, *geometry, EIG_MIN, EIG_MAX, "cpu", torch.float64
    )
    R = int(ref.rmax)
    return ref.G_gamma[:, :R, :R].numpy().copy()


def window(spectrum: np.ndarray, n: int = N_MODES) -> np.ndarray:
    """Network input convention: eigenvalues in ]EIG_MIN, EIG_MAX[, ascending, zero padded to n."""
    spectrum = np.atleast_2d(spectrum)
    out = np.zeros((len(spectrum), n))
    for i, s in enumerate(spectrum):
        s = np.sort(s[(s > EIG_MIN) & (s < EIG_MAX)])[:n]
        out[i, : len(s)] = s
    return out


class Forward:
    """theta (B, 11, physical units) -> full ascending spectrum (B, P) and network input (B, N_MODES)."""

    def __init__(
        self,
        device: str | None = None,
        order: int = POLY_ORDER,
        geometry=GEOMETRY,
        chunk: int = 16,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk = chunk  # matrices diagonalised per call (memory)
        g = elementary_integrals(order, geometry)  # (9, R, R)
        R = g.shape[1]
        self.R, self.P = R, 3 * R
        unit = np.eye(11)
        c, e, k = material_tensors(unit)
        G = np.zeros((6, 3, R, 3, R))  # Gamma = sum_m c_m G_m
        for (a, b), idx in _GAMMA.items():
            coef = np.stack([c[:6, i, j] for i, j in idx], 1)  # (6, 9)
            G[:, a, :, b, :] = np.einsum("mk,kij->mij", coef, g)
        O = np.zeros((3, 3, R, R))  # Omega = sum_m e_m O_m
        for a, idx in _OMEGA.items():
            coef = np.stack([e[6:9, i, j] for i, j in idx], 1)  # (3, 9)
            O[:, a] = np.einsum("mk,kij->mij", coef, g)
        L = np.einsum(
            "mk,kij->mij", k[9:, [0, 1, 2], [0, 1, 2]], g[:3]
        )  # Lambda = sum_m eps_m L_m, (2, R, R)
        to = dict(device=self.device, dtype=torch.float64)
        self.G = torch.as_tensor(G.reshape(6, self.P, self.P), **to)
        self.O = torch.as_tensor(O.reshape(3, self.P, R), **to)
        self.L = torch.as_tensor(L, **to)

    def _theta(self, theta):
        return torch.as_tensor(
            np.atleast_2d(theta), dtype=torch.float64, device=self.device
        )

    @torch.no_grad()
    def matrix(self, theta: np.ndarray) -> torch.Tensor:
        """Symmetric generalised stiffness M (B, P, P)."""
        y = self._theta(theta)
        gamma = torch.einsum("bk,kij->bij", y[:, :6], self.G)
        omega = torch.einsum("bk,kij->bij", y[:, 6:9], self.O)
        lambd = torch.einsum("bk,kij->bij", y[:, 9:], self.L)
        m = gamma + omega @ torch.linalg.solve(lambd, omega.transpose(-1, -2))
        return 0.5 * (m + m.transpose(-1, -2))

    @torch.no_grad()
    def spectrum(self, theta: np.ndarray) -> np.ndarray:
        """Full ascending spectrum (B, P) of a batch of constant vectors."""
        theta = np.atleast_2d(theta)
        return np.concatenate(
            [
                torch.linalg.eigvalsh(self.matrix(theta[i : i + self.chunk]))
                .cpu()
                .numpy()
                for i in range(0, len(theta), self.chunk)
            ]
        )

    def eigenvalues(self, theta: np.ndarray) -> np.ndarray:
        """Network input (B, N_MODES): windowed, zero padded."""
        return window(self.spectrum(theta))

    # ---- used by lm.py: one sample at a time, eigenvectors and exact gradients
    @torch.no_grad()
    def eigh(self, theta: np.ndarray):
        """Full spectrum (P,) and eigenvectors (P, P, on device) of a single constant vector."""
        w, v = torch.linalg.eigh(self.matrix(theta)[0])
        return w.cpu().numpy(), v

    @torch.no_grad()
    def gradients(self, theta: np.ndarray, v: torch.Tensor) -> np.ndarray:
        """d lambda_i / d theta_k (11, n) for the eigenvectors in the columns of v (P, n), first-order perturbation
        d lambda = v^T dM v, with X = Lambda^-1 Omega^T:
          dM/dc_k = G_k,   dM/de_j = O_j X + X^T O_j^T,   dM/deps_j = -X^T L_j X
        """
        y = self._theta(theta)[0]
        omega = torch.einsum("k,kij->ij", y[6:9], self.O)  # (P, R)
        lambd = torch.einsum("k,kij->ij", y[9:], self.L)  # (R, R)
        xv = torch.linalg.solve(lambd, omega.T @ v)  # X v, (R, n)
        d_c = (v.unsqueeze(0) * torch.einsum("kij,jn->kin", self.G, v)).sum(1)
        d_e = 2 * (v.unsqueeze(0) * torch.einsum("kij,jn->kin", self.O, xv)).sum(1)
        d_eps = -(xv.unsqueeze(0) * torch.einsum("kij,jn->kin", self.L, xv)).sum(1)
        return torch.cat([d_c, d_e, d_eps]).cpu().numpy()
