"""Physical constants, sample geometry and sampling box"""

import numpy as np

# The 11 independent material constants of a 4mm piezoelectric
# Units: c^E in Pa, e in C/m^2, eps^S in F/m.
CONSTANT_NAMES = [
    "cE11",
    "cE12",
    "cE13",
    "cE33",
    "cE44",
    "cE66",
    "e31",
    "e33",
    "e15",
    "epsS11",
    "epsS33",
]
N_CONSTANTS = len(CONSTANT_NAMES)

EPS0 = 8.85e-12  # F/m, vacuum permittivity
RHO = 7507.0  # kg/m^3, density
GEOMETRY = (9.17e-3, 30.40e-3, 13.49e-3)  # m, sample dimensions (l1, l2, l3)
POLY_ORDER = 16  # Legendre polynomial degree of the Rayleigh-Ritz basis

# Network input: the eigenvalues rho * omega^2 lying in ]EIG_MIN, EIG_MAX[, ascending, zero padded to N_MODES
EIG_MIN, EIG_MAX = 60e12, 13000e12
N_MODES = 110

# Pz26 (CTS Ferroperm), centre of the sampling box
PZ26 = np.array(
    [
        1.68e11,
        1.10e11,
        9.99e10,
        1.23e11,
        3.01e10,
        2.88e10,
        -2.80,
        14.7,
        9.86,
        828 * EPS0,
        700 * EPS0,
    ]
)

# Sampling box: c^E = Pz26 +- 50 %; e and eps^S: bounds of the published dataset
BOX_LOW = np.array(
    [
        8.4e10,
        5.5e10,
        4.995e10,
        6.15e10,
        1.505e10,
        1.44e10,
        -10.28,
        7.2,
        2.376324,
        77.834971 * EPS0,
        168.702849 * EPS0,
    ]
)
BOX_HIGH = np.array(
    [
        2.52e11,
        1.65e11,
        1.4985e11,
        1.845e11,
        4.515e10,
        4.32e10,
        4.68,
        22.2,
        17.34361,
        1578.157774 * EPS0,
        1231.298247 * EPS0,
    ]
)
