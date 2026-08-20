import asyncio

import numpy as np
import pytest

import tools
from mcp_server.server import create_server
from tools.qcd import alpha_s_two_loop, hotqcd_eos, hrg_thermodynamics


def test_server_exposes_all_tools():
    mcp = create_server()
    assert mcp.name == "Lattice MCP Server"
    listed = {t.name for t in asyncio.run(mcp.list_tools())}
    assert listed == set(tools.__all__)
    assert len(listed) == 8


def test_eos_matches_published_values():
    res = hotqcd_eos(np.array([150.0, 200.0, 400.0]))
    # arXiv:1407.6387 Fig. 5: e/T^4 ~ 4 near 150 MeV, ~13.4 at 400 MeV
    assert 3.0 < res["e_div_T4"][0] < 5.0
    assert 13.0 < res["e_div_T4"][2] < 14.0
    assert np.all(np.diff(res["e_div_T4"]) > 0)
    assert 0.0 < res["cs2"][2] < 1.0 / 3.0


def test_hrg_matches_lattice_below_crossover():
    T = np.linspace(130.0, 160.0, 7)
    hrg, n_hadrons = hrg_thermodynamics(T)
    eos = hotqcd_eos(T)
    assert n_hadrons > 1000
    # confined phase: the two descriptions agree to ~15% below T_c
    ratio = hrg["p_div_T4"] / eos["p_div_T4"]
    assert np.all(np.abs(ratio - 1.0) < 0.15)


def test_alpha_s_reproduces_world_average_scale():
    a_mz = alpha_s_two_loop(np.array([91187.6]), nf=5, lambda_MeV=210.0)[0]
    assert abs(a_mz - 0.118) < 0.005
    with pytest.raises(ValueError):
        alpha_s_two_loop(np.array([100.0]), nf=5, lambda_MeV=210.0)


def test_compute_and_plot_roundtrip(tmp_path):
    out = str(tmp_path)
    eos = tools.compute_qcd_eos(output_dir=out, n_points=20)
    hrg = tools.compute_hrg_thermodynamics(output_dir=out, n_points=20)
    plot = tools.plot_eos(eos_file=eos.files[0], output_dir=out,
                          hrg_file=hrg.files[0])
    sound = tools.plot_speed_of_sound(eos_file=eos.files[0], output_dir=out)
    assert all(r.status == "success" for r in (eos, hrg, plot, sound))
    assert (tmp_path / "qcd_eos.png").exists()
    assert (tmp_path / "qcd_speed_of_sound.png").exists()


def test_charge_cumulants_roundtrip(tmp_path):
    out = str(tmp_path)
    chi = tools.compute_charge_cumulants(output_dir=out, n_points=10)
    plot = tools.plot_charge_cumulants(cumulants_file=chi.files[0], output_dir=out)
    assert plot.status == "success"
    # hadron gas: kurtosis ratio stays near 1
    data = np.loadtxt(chi.files[0], delimiter=",", skiprows=2)
    assert np.all(np.abs(data[:, 3] - 1.0) < 0.15)


def test_plot_rejects_wrong_csv(tmp_path):
    chi = tools.compute_charge_cumulants(output_dir=str(tmp_path), n_points=10)
    with pytest.raises(ValueError, match="missing columns"):
        tools.plot_eos(eos_file=chi.files[0], output_dir=str(tmp_path))


def test_running_coupling_tool(tmp_path):
    res = tools.compute_running_coupling(output_dir=str(tmp_path))
    assert res.metadata["alpha_s_at_MZ"] == pytest.approx(0.117, abs=0.005)
    assert len(res.files) == 2
