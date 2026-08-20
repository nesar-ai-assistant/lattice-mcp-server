"""MCP tool functions for lattice QCD thermodynamics.

Plain Python functions — no MCP imports. The type hints, Field constraints,
and docstrings below become the MCP tool schema that agents see.

Data flows between tools as CSV file paths: the compute_* tools write one CSV
each, the plot_* tools read them back. Only small numbers and file paths ever
pass through the LLM context.
"""

import csv
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, Field, validate_call

from .qcd import (
    EOS_T_RANGE_MEV,
    SB_E_DIV_T4,
    SB_P_DIV_T4,
    T_CROSSOVER_ERR,
    T_CROSSOVER_MEV,
    alpha_s_two_loop,
    hotqcd_eos,
    hrg_charge_cumulants,
    hrg_thermodynamics,
)

THERMO_COLUMNS = ["T_MeV", "p_div_T4", "e_div_T4", "s_div_T3",
                  "trace_anomaly", "cs2"]


class ArtifactResult(BaseModel):
    """Uniform result contract returned by every tool."""

    status: Literal["success"]
    files: list[str]
    message: str
    metadata: dict[str, Any]


def _write_csv(path: Path, source: str, columns: list[str],
               rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write(f"# source: {source}\n")
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


def _read_csv(path_str: str, expected_columns: list[str]) -> tuple[str, dict[str, np.ndarray]]:
    """Read a CSV written by _write_csv, checking it has the right columns."""
    path = Path(path_str).expanduser().resolve()
    lines = path.read_text(encoding="utf-8").splitlines()
    source = lines[0].removeprefix("# source:").strip() if lines[0].startswith("#") else path.stem
    header = lines[1].split(",") if lines[0].startswith("#") else lines[0].split(",")
    missing = [c for c in expected_columns if c not in header]
    if missing:
        raise ValueError(
            f"{path.name} is missing columns {missing} (found {header}). "
            "Pass a CSV produced by the matching compute_* tool."
        )
    data = np.loadtxt(path, delimiter=",", skiprows=2 if lines[0].startswith("#") else 1).T
    return source, {name: col for name, col in zip(header, np.atleast_2d(data))}


def _outdir(output_dir: str) -> Path:
    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


@validate_call
def describe_qcd_thermodynamics() -> ArtifactResult:
    """Explain the physics context of the QCD thermodynamics tools.

    Use this tool first. It summarizes the two descriptions of hot strongly
    interacting matter that the other tools compute — the lattice QCD equation
    of state and the hadron resonance gas model — their validity ranges, the
    crossover temperature, and the references, so you can plan which tools to
    call and interpret their outputs.
    """
    return ArtifactResult(
        status="success",
        files=[],
        message=(
            "QCD thermodynamics at zero baryon density: lattice EoS "
            f"(valid {EOS_T_RANGE_MEV[0]:.0f}-{EOS_T_RANGE_MEV[1]:.0f} MeV) vs "
            "hadron resonance gas (valid below the crossover at "
            f"T_c = {T_CROSSOVER_MEV} +/- {T_CROSSOVER_ERR} MeV)."
        ),
        metadata={
            "equation_of_state": {
                "description": (
                    "Continuum-extrapolated (2+1)-flavor lattice QCD equation of "
                    "state at mu_B = 0, as the published HotQCD parameterization."
                ),
                "reference": "Bazavov et al. (HotQCD), PRD 90, 094503 (2014), arXiv:1407.6387",
                "valid_T_MeV": list(EOS_T_RANGE_MEV),
            },
            "hadron_resonance_gas": {
                "description": (
                    "Non-interacting gas of ~1200 hadrons and resonances "
                    "(QMHRG2020 list). Standard low-temperature description; "
                    "it must FAIL above the crossover, where quarks and gluons "
                    "are the true degrees of freedom — comparing it to the "
                    "lattice EoS locates deconfinement."
                ),
                "reference": "Bollweg et al. (HotQCD), PRD 104, 074512 (2021), arXiv:2107.10011",
                "valid_T_MeV": [30.0, 180.0],
            },
            "crossover_temperature_MeV": {
                "value": T_CROSSOVER_MEV,
                "error": T_CROSSOVER_ERR,
                "reference": "HotQCD, PLB 795 (2019) 15, arXiv:1812.08235",
            },
            "ideal_gas_limits": {
                "p_div_T4": round(SB_P_DIV_T4, 3),
                "e_div_T4": round(SB_E_DIV_T4, 3),
                "note": "Stefan-Boltzmann limit of a free quark-gluon gas (Nc=3, Nf=3).",
            },
            "early_universe": (
                "The universe passed through the QCD crossover about 10 "
                "microseconds after the Big Bang; this EoS governs its "
                "expansion during that epoch."
            ),
            "software": "LatticeQCD/AnalysisToolbox (pip: latqcdtools)",
        },
    )


@validate_call
def compute_qcd_eos(
    output_dir: Annotated[str, Field(min_length=1)],
    t_min_MeV: Annotated[float, Field(ge=130.0, le=390.0)] = 130.0,
    t_max_MeV: Annotated[float, Field(ge=140.0, le=400.0)] = 400.0,
    n_points: Annotated[int, Field(ge=10, le=500)] = 100,
) -> ArtifactResult:
    """Compute the lattice QCD equation of state (HotQCD parameterization).

    Writes a CSV with columns T_MeV, p_div_T4, e_div_T4, s_div_T3,
    trace_anomaly ((e-3p)/T^4), cs2 (speed of sound squared). Pass the
    returned file path to plot_eos or plot_speed_of_sound — never copy the
    numbers. Valid only for 130 <= T <= 400 MeV at zero baryon density.

    Args:
        output_dir: Directory where the CSV is written.
        t_min_MeV: Lowest temperature in MeV.
        t_max_MeV: Highest temperature in MeV.
        n_points: Number of temperature points.
    """
    if t_min_MeV >= t_max_MeV:
        raise ValueError("t_min_MeV must be smaller than t_max_MeV.")
    T = np.linspace(t_min_MeV, t_max_MeV, n_points)
    res = hotqcd_eos(T)
    csv_path = _outdir(output_dir) / "qcd_eos.csv"
    _write_csv(csv_path, "HotQCD lattice EoS (arXiv:1407.6387)", THERMO_COLUMNS,
               zip(T, res["p_div_T4"], res["e_div_T4"], res["s_div_T3"],
                   res["trace_anomaly"], res["cs2"]))
    peak = T[np.argmax(res["trace_anomaly"])]
    return ArtifactResult(
        status="success",
        files=[str(csv_path)],
        message=(
            f"Computed lattice QCD EoS on {n_points} points in "
            f"[{t_min_MeV:g}, {t_max_MeV:g}] MeV."
        ),
        metadata={
            "T_range_MeV": [t_min_MeV, t_max_MeV],
            "trace_anomaly_peak_T_MeV": round(float(peak), 1),
            "e_div_T4_at_Tmax": round(float(res["e_div_T4"][-1]), 3),
            "stefan_boltzmann_e_div_T4": round(SB_E_DIV_T4, 3),
            "reference": "Bazavov et al. (HotQCD), PRD 90, 094503 (2014)",
        },
    )


@validate_call
def compute_hrg_thermodynamics(
    output_dir: Annotated[str, Field(min_length=1)],
    t_min_MeV: Annotated[float, Field(ge=10.0, le=170.0)] = 30.0,
    t_max_MeV: Annotated[float, Field(ge=50.0, le=180.0)] = 180.0,
    n_points: Annotated[int, Field(ge=10, le=500)] = 100,
) -> ArtifactResult:
    """Compute hadron resonance gas thermodynamics (QMHRG2020 hadron list).

    Writes a CSV with the same columns as compute_qcd_eos (T_MeV, p_div_T4,
    e_div_T4, s_div_T3, trace_anomaly, cs2) so the two can be overlaid with
    plot_eos. The HRG is the confined-phase description: comparing it to the
    lattice EoS shows where hadrons stop being the right degrees of freedom.

    Args:
        output_dir: Directory where the CSV is written.
        t_min_MeV: Lowest temperature in MeV.
        t_max_MeV: Highest temperature in MeV (HRG is meaningless far above
            the crossover, so 180 MeV is the ceiling).
        n_points: Number of temperature points.
    """
    if t_min_MeV >= t_max_MeV:
        raise ValueError("t_min_MeV must be smaller than t_max_MeV.")
    T = np.linspace(t_min_MeV, t_max_MeV, n_points)
    res, n_hadrons = hrg_thermodynamics(T)
    csv_path = _outdir(output_dir) / "hrg_thermodynamics.csv"
    _write_csv(csv_path, f"Hadron resonance gas, QMHRG2020 ({n_hadrons} states)",
               THERMO_COLUMNS,
               zip(T, res["p_div_T4"], res["e_div_T4"], res["s_div_T3"],
                   res["trace_anomaly"], res["cs2"]))
    return ArtifactResult(
        status="success",
        files=[str(csv_path)],
        message=(
            f"Computed HRG thermodynamics ({n_hadrons} hadron states) on "
            f"{n_points} points in [{t_min_MeV:g}, {t_max_MeV:g}] MeV."
        ),
        metadata={
            "T_range_MeV": [t_min_MeV, t_max_MeV],
            "n_hadron_states": n_hadrons,
            "e_div_T4_at_Tmax": round(float(res["e_div_T4"][-1]), 3),
            "reference": "QMHRG2020 list, Bollweg et al. (HotQCD), PRD 104, 074512 (2021)",
        },
    )


@validate_call
def plot_eos(
    eos_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
    hrg_file: str | None = None,
) -> ArtifactResult:
    """Plot the QCD equation of state, optionally overlaying the HRG model.

    Use this tool after compute_qcd_eos (and, for the overlay,
    compute_hrg_thermodynamics). It draws e/T^4, 3p/T^4 and (3/4)s/T^3 vs
    temperature with the crossover region and the ideal quark-gluon gas limit
    marked. With hrg_file given, HRG curves are dashed — the temperature where
    they peel away from the lattice curves is deconfinement made visible.

    Args:
        eos_file: CSV written by compute_qcd_eos.
        output_dir: Directory where the PNG is written.
        hrg_file: Optional CSV written by compute_hrg_thermodynamics.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, eos = _read_csv(eos_file, THERMO_COLUMNS)
    fig, ax = plt.subplots(figsize=(8, 6))
    curves = [("e_div_T4", r"$\epsilon/T^4$"), ("p_div_T4", r"$3p/T^4$"),
              ("s_div_T3", r"$(3/4)\,s/T^3$")]
    scale = {"e_div_T4": 1.0, "p_div_T4": 3.0, "s_div_T3": 0.75}
    for i, (col, label) in enumerate(curves):
        ax.plot(eos["T_MeV"], scale[col] * eos[col], color=f"C{i}",
                linewidth=2, label=label + " (lattice)")
    if hrg_file is not None:
        _, hrg = _read_csv(hrg_file, THERMO_COLUMNS)
        for i, (col, label) in enumerate(curves):
            ax.plot(hrg["T_MeV"], scale[col] * hrg[col], color=f"C{i}",
                    linewidth=1.8, linestyle="--", label=label + " (HRG)")
    ax.axvspan(T_CROSSOVER_MEV - T_CROSSOVER_ERR, T_CROSSOVER_MEV + T_CROSSOVER_ERR,
               color="gray", alpha=0.35,
               label=rf"crossover $T_c = {T_CROSSOVER_MEV}\pm{T_CROSSOVER_ERR}$ MeV")
    ax.axhline(SB_E_DIV_T4, color="black", linewidth=1, linestyle=":",
               label="ideal quark-gluon gas")
    ax.set_xlabel(r"$T$ [MeV]")
    ax.set_ylabel(r"$\epsilon/T^4$, $3p/T^4$, $(3/4)\,s/T^3$")
    ax.set_title("QCD equation of state" + (" vs hadron resonance gas" if hrg_file else ""))
    ax.set_xlim(eos["T_MeV"].min() if hrg_file is None else 30.0, eos["T_MeV"].max())
    ax.legend(fontsize="small", loc="upper left")

    plot_path = _outdir(output_dir) / "qcd_eos.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ArtifactResult(
        status="success",
        files=[str(plot_path)],
        message=("Plotted the lattice QCD EoS"
                 + (" with the HRG overlay" if hrg_file else "") + "."),
        metadata={"crossover_T_MeV": [T_CROSSOVER_MEV, T_CROSSOVER_ERR],
                  "hrg_overlaid": hrg_file is not None},
    )


@validate_call
def plot_speed_of_sound(
    eos_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
    hrg_file: str | None = None,
) -> ArtifactResult:
    """Plot the speed of sound squared and the trace anomaly vs temperature.

    Use this tool after compute_qcd_eos. Two stacked panels: c_s^2(T), whose
    dip near the crossover marks the softest point of the equation of state,
    and the trace anomaly (e-3p)/T^4, whose peak measures maximal deviation
    from conformality. Optionally overlays the HRG curves for comparison.

    Args:
        eos_file: CSV written by compute_qcd_eos.
        output_dir: Directory where the PNG is written.
        hrg_file: Optional CSV written by compute_hrg_thermodynamics.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, eos = _read_csv(eos_file, THERMO_COLUMNS)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    ax1.plot(eos["T_MeV"], eos["cs2"], color="C0", linewidth=2, label="lattice EoS")
    ax2.plot(eos["T_MeV"], eos["trace_anomaly"], color="C1", linewidth=2,
             label="lattice EoS")
    if hrg_file is not None:
        _, hrg = _read_csv(hrg_file, THERMO_COLUMNS)
        ax1.plot(hrg["T_MeV"], hrg["cs2"], color="C0", linewidth=1.8,
                 linestyle="--", label="HRG")
        ax2.plot(hrg["T_MeV"], hrg["trace_anomaly"], color="C1", linewidth=1.8,
                 linestyle="--", label="HRG")
    ax1.axhline(1.0 / 3.0, color="black", linewidth=1, linestyle=":",
                label=r"conformal limit $c_s^2 = 1/3$")
    for ax in (ax1, ax2):
        ax.axvspan(T_CROSSOVER_MEV - T_CROSSOVER_ERR,
                   T_CROSSOVER_MEV + T_CROSSOVER_ERR, color="gray", alpha=0.35)
        ax.legend(fontsize="small")
    ax1.set_ylabel(r"$c_s^2$")
    ax1.set_title("Softest point and trace anomaly of the QCD medium")
    ax2.set_ylabel(r"$(\epsilon - 3p)/T^4$")
    ax2.set_xlabel(r"$T$ [MeV]")

    plot_path = _outdir(output_dir) / "qcd_speed_of_sound.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    i_min = int(np.argmin(eos["cs2"]))
    return ArtifactResult(
        status="success",
        files=[str(plot_path)],
        message="Plotted c_s^2(T) and the trace anomaly.",
        metadata={
            "cs2_minimum": {"T_MeV": round(float(eos["T_MeV"][i_min]), 1),
                            "cs2": round(float(eos["cs2"][i_min]), 4)},
            "conformal_limit": 1.0 / 3.0,
        },
    )


@validate_call
def compute_charge_cumulants(
    output_dir: Annotated[str, Field(min_length=1)],
    t_min_MeV: Annotated[float, Field(ge=100.0, le=160.0)] = 100.0,
    t_max_MeV: Annotated[float, Field(ge=120.0, le=180.0)] = 175.0,
    n_points: Annotated[int, Field(ge=10, le=200)] = 40,
    muB_div_T: Annotated[float, Field(ge=0.0, le=2.5)] = 0.0,
) -> ArtifactResult:
    """Compute baryon-number susceptibilities chi_2^B and chi_4^B in the HRG model.

    Writes a CSV with columns T_MeV, chi2B, chi4B, ratio_chi4B_chi2B. In a
    hadron gas the kurtosis ratio chi4B/chi2B stays near 1 (baryons carry
    |B| = 1); in a quark gluon plasma it drops toward (1/3)^2 x ... ~ 0.1
    (quarks carry B = 1/3). These are the fluctuation observables measured in
    heavy-ion collisions and computed on the lattice. Pass the CSV to
    plot_charge_cumulants.

    Args:
        output_dir: Directory where the CSV is written.
        t_min_MeV: Lowest temperature in MeV.
        t_max_MeV: Highest temperature in MeV.
        n_points: Number of temperature points.
        muB_div_T: Baryon chemical potential over temperature (0 = the
            early-universe / LHC regime; up to ~2.5 for RHIC energies).
    """
    if t_min_MeV >= t_max_MeV:
        raise ValueError("t_min_MeV must be smaller than t_max_MeV.")
    T = np.linspace(t_min_MeV, t_max_MeV, n_points)
    chi = hrg_charge_cumulants(T, orders=[2, 4], muB_div_T=muB_div_T)
    ratio = chi["chi4B"] / chi["chi2B"]
    csv_path = _outdir(output_dir) / "charge_cumulants.csv"
    _write_csv(csv_path, f"HRG baryon-number cumulants, muB/T={muB_div_T:g}",
               ["T_MeV", "chi2B", "chi4B", "ratio_chi4B_chi2B"],
               zip(T, chi["chi2B"], chi["chi4B"], ratio))
    return ArtifactResult(
        status="success",
        files=[str(csv_path)],
        message=(
            f"Computed chi2B and chi4B on {n_points} points in "
            f"[{t_min_MeV:g}, {t_max_MeV:g}] MeV at muB/T = {muB_div_T:g}."
        ),
        metadata={
            "muB_div_T": muB_div_T,
            "chi2B_at_Tmax": round(float(chi["chi2B"][-1]), 4),
            "hadron_gas_expectation": "chi4B/chi2B ~ 1 while baryons are the carriers",
            "reference": "HRG from QMHRG2020; observable of STAR/ALICE fluctuation programs",
        },
    )


@validate_call
def plot_charge_cumulants(
    cumulants_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
) -> ArtifactResult:
    """Plot baryon-number susceptibilities and their kurtosis ratio.

    Use this tool after compute_charge_cumulants. Two stacked panels: chi2B
    and chi4B vs temperature, and the ratio chi4B/chi2B with the hadron-gas
    expectation of 1 marked — the departure of lattice data from this HRG
    baseline near T_c is evidence that baryon number is carried by quarks.

    Args:
        cumulants_file: CSV written by compute_charge_cumulants.
        output_dir: Directory where the PNG is written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source, data = _read_csv(
        cumulants_file, ["T_MeV", "chi2B", "chi4B", "ratio_chi4B_chi2B"]
    )
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"hspace": 0.08}
    )
    ax1.plot(data["T_MeV"], data["chi2B"], color="C0", linewidth=2,
             label=r"$\chi_2^B$")
    ax1.plot(data["T_MeV"], data["chi4B"], color="C1", linewidth=2,
             linestyle="--", label=r"$\chi_4^B$")
    ax2.plot(data["T_MeV"], data["ratio_chi4B_chi2B"], color="C2", linewidth=2)
    ax2.axhline(1.0, color="black", linewidth=1, linestyle=":",
                label="hadron gas expectation")
    for ax in (ax1, ax2):
        ax.axvspan(T_CROSSOVER_MEV - T_CROSSOVER_ERR,
                   T_CROSSOVER_MEV + T_CROSSOVER_ERR, color="gray", alpha=0.35)
    ax1.set_ylabel(r"$\chi_n^B$")
    ax1.set_title("Baryon-number fluctuations in the HRG model")
    ax1.legend(fontsize="small")
    ax2.set_ylabel(r"$\chi_4^B/\chi_2^B$")
    ax2.set_xlabel(r"$T$ [MeV]")
    ax2.legend(fontsize="small")

    plot_path = _outdir(output_dir) / "charge_cumulants.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ArtifactResult(
        status="success",
        files=[str(plot_path)],
        message="Plotted chi2B, chi4B and their kurtosis ratio.",
        metadata={"source": source},
    )


@validate_call
def compute_running_coupling(
    output_dir: Annotated[str, Field(min_length=1)],
    nf: Annotated[int, Field(ge=3, le=6)] = 5,
    lambda_MeV: Annotated[float, Field(ge=100.0, le=400.0)] = 210.0,
    mu_min_MeV: Annotated[float, Field(ge=500.0)] = 1000.0,
    mu_max_MeV: Annotated[float, Field(le=1.0e7)] = 1.0e6,
    n_points: Annotated[int, Field(ge=10, le=500)] = 120,
) -> ArtifactResult:
    """Compute and plot the running QCD coupling alpha_s(mu) at two loops.

    Writes both a CSV (columns mu_MeV, alpha_s) and a log-x PNG showing
    asymptotic freedom: the coupling that confines quarks at the GeV scale
    becomes weak at high energies. Uses the AnalysisToolbox beta-function
    coefficients in the MS-bar scheme.

    Args:
        output_dir: Directory where the CSV and PNG are written.
        nf: Number of active quark flavors.
        lambda_MeV: The QCD scale Lambda_MSbar in MeV (~210 MeV for nf=5).
        mu_min_MeV: Lowest scale in MeV (must stay well above Lambda).
        mu_max_MeV: Highest scale in MeV.
        n_points: Number of log-spaced points.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if mu_min_MeV >= mu_max_MeV:
        raise ValueError("mu_min_MeV must be smaller than mu_max_MeV.")
    mu = np.logspace(np.log10(mu_min_MeV), np.log10(mu_max_MeV), n_points)
    alpha = alpha_s_two_loop(mu, nf=nf, lambda_MeV=lambda_MeV)

    outdir = _outdir(output_dir)
    csv_path = outdir / "running_coupling.csv"
    _write_csv(csv_path, f"2-loop alpha_s, nf={nf}, Lambda={lambda_MeV:g} MeV",
               ["mu_MeV", "alpha_s"], zip(mu, alpha))

    m_z = 91187.6
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.semilogx(mu / 1000.0, alpha, color="C0", linewidth=2,
                label=rf"2-loop, $n_f={nf}$, $\Lambda={lambda_MeV:g}$ MeV")
    if mu_min_MeV <= m_z <= mu_max_MeV:
        a_mz = float(alpha_s_two_loop(np.array([m_z]), nf, lambda_MeV)[0])
        ax.plot([m_z / 1000.0], [a_mz], "ko", markersize=6)
        ax.annotate(rf"$\alpha_s(M_Z) = {a_mz:.4f}$", (m_z / 1000.0, a_mz),
                    textcoords="offset points", xytext=(10, 10))
    ax.set_xlabel(r"$\mu$ [GeV]")
    ax.set_ylabel(r"$\alpha_s(\mu)$")
    ax.set_title("Asymptotic freedom: the running of the strong coupling")
    ax.legend(fontsize="small")
    plot_path = outdir / "running_coupling.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    metadata: dict[str, Any] = {
        "nf": nf,
        "lambda_MeV": lambda_MeV,
        "scheme": "MS-bar, 2-loop",
        "pdg_world_average_alpha_s_MZ": 0.1180,
    }
    if mu_min_MeV <= m_z <= mu_max_MeV:
        metadata["alpha_s_at_MZ"] = round(a_mz, 4)
    return ArtifactResult(
        status="success",
        files=[str(csv_path), str(plot_path)],
        message=(
            f"Computed 2-loop alpha_s for nf={nf} over "
            f"[{mu_min_MeV:g}, {mu_max_MeV:g}] MeV."
        ),
        metadata=metadata,
    )
