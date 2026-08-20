"""Science layer for the lattice QCD thermodynamics tools.

Thin wrappers around the LatticeQCD AnalysisToolbox (pip: latqcdtools),
the HotQCD collaboration's public analysis package:

- HotQCD equation of state: the published parameterization of continuum-
  extrapolated (2+1)-flavor lattice QCD results, Bazavov et al. (HotQCD),
  Phys. Rev. D 90, 094503 (2014), arXiv:1407.6387. Valid for T in
  [130, 400] MeV at mu_B = 0.
- Hadron resonance gas (HRG): non-interacting gas of all hadrons and
  resonances from the QMHRG2020 list (bundled in data/, taken from the
  AnalysisToolbox repository, MIT license). The standard low-temperature
  description of QCD matter; expected to break down near the crossover.
- QCD beta-function coefficients for the running coupling.
"""

import contextlib
import sys
from pathlib import Path

import numpy as np


def _quiet():
    """Redirect stdout to stderr while latqcdtools runs.

    Its logger prints warnings with bare print(); on the stdio MCP transport
    stdout belongs to the JSON-RPC stream, so any stray line would corrupt it.
    """
    return contextlib.redirect_stdout(sys.stderr)

HADRON_LIST = (
    Path(__file__).resolve().parents[1] / "data"
    / "QM_hadron_list_ext_strange_2020.txt"
)

# QCD crossover temperature, HotQCD: Phys. Lett. B 795 (2019) 15, arXiv:1812.08235
T_CROSSOVER_MEV = 156.5
T_CROSSOVER_ERR = 1.5

# Validity range of the 1407.6387 EoS parameterization
EOS_T_RANGE_MEV = (130.0, 400.0)

# Stefan-Boltzmann (ideal quark-gluon gas) limits for N_c=3, N_f=3
SB_P_DIV_T4 = 19.0 * np.pi**2 / 36.0  # ~ 5.209
SB_E_DIV_T4 = 3.0 * SB_P_DIV_T4       # ~ 15.63

_hrg_cache = {}


def load_hrg(hadron_list: str | None = None):
    """Build an HRG instance from the bundled QMHRG2020 hadron list (cached)."""
    path = str(hadron_list or HADRON_LIST)
    if path not in _hrg_cache:
        with _quiet():
            from latqcdtools.base.readWrite import readTable
            from latqcdtools.physics.HRG import HRG

            hadrons, M, Q, B, S, C, g = readTable(
                path, usecols=(0, 1, 2, 3, 4, 5, 6), dtype="U11,f8,i8,i8,i8,i8,i8"
            )
            w = np.array([1 if b == 0 else -1 for b in B])
            _hrg_cache[path] = (HRG(M, g, w, B, S, Q, C), len(M))
    return _hrg_cache[path]


def hotqcd_eos(T_MeV: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate the HotQCD 2014 EoS parameterization on a temperature grid.

    Returns dimensionless p/T^4, e/T^4, s/T^3, trace anomaly (e-3p)/T^4 and
    the speed of sound squared c_s^2 = dp/de (numerical derivative).
    """
    with _quiet():
        from latqcdtools.physics.HotQCDEOS import EOS

        p, e, s = EOS(T_MeV).ObsEoS()
    # cs^2 = dp/de; convert dimensionless ratios back to p, e before deriving
    p_phys = p * T_MeV**4
    e_phys = e * T_MeV**4
    cs2 = np.gradient(p_phys, T_MeV) / np.gradient(e_phys, T_MeV)
    return {"p_div_T4": p, "e_div_T4": e, "s_div_T3": s,
            "trace_anomaly": e - 3.0 * p, "cs2": cs2}


def hrg_thermodynamics(T_MeV: np.ndarray) -> tuple[dict[str, np.ndarray], int]:
    """Evaluate HRG thermodynamics on a temperature grid.

    Returns (dict of p/T^4, e/T^4, s/T^3, trace anomaly, cs^2; n_hadrons).
    """
    hrg, n_hadrons = load_hrg()
    with _quiet():
        p = hrg.P_div_T4(T_MeV)
        e = hrg.E_div_T4(T_MeV)
        s = hrg.S_div_T3(T_MeV)
    p_phys = p * T_MeV**4
    e_phys = e * T_MeV**4
    cs2 = np.gradient(p_phys, T_MeV) / np.gradient(e_phys, T_MeV)
    return ({"p_div_T4": p, "e_div_T4": e, "s_div_T3": s,
             "trace_anomaly": e - 3.0 * p, "cs2": cs2}, n_hadrons)


def hrg_charge_cumulants(
    T_MeV: np.ndarray, orders: list[int], muB_div_T: float = 0.0
) -> dict[str, np.ndarray]:
    """Baryon-number cumulants chi_n^B(T) from the HRG model."""
    hrg, _ = load_hrg()
    with _quiet():
        return {
            f"chi{n}B": hrg.gen_chi(T_MeV, B_order=n, muB_div_T=muB_div_T)
            for n in orders
        }


def alpha_s_two_loop(
    mu_MeV: np.ndarray, nf: int, lambda_MeV: float
) -> np.ndarray:
    """Two-loop running coupling alpha_s(mu) in the MS-bar scheme.

    Uses the AnalysisToolbox beta-function coefficients (b0 = 11 - 2Nf/3
    convention). Standard two-loop solution, e.g. PDG Review of Particle
    Physics, "Quantum Chromodynamics":
        alpha_s = 4 pi / (b0 L) * [1 - (b1/b0^2) ln(L)/L],  L = ln(mu^2/Lambda^2)
    """
    with _quiet():
        from latqcdtools.physics.runningCoupling import b0, b1

    mu = np.asarray(mu_MeV, dtype=float)
    if np.any(mu <= lambda_MeV):
        raise ValueError(
            f"All scales mu must exceed Lambda_QCD = {lambda_MeV} MeV: "
            "the perturbative coupling diverges at the Landau pole."
        )
    L = np.log(mu**2 / lambda_MeV**2)
    B0, B1 = b0(nf), b1(nf)
    return 4.0 * np.pi / (B0 * L) * (1.0 - (B1 / B0**2) * np.log(L) / L)
