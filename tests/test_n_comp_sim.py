import pytest
import numpy as np
from surfacetension import NCompSimulator


@pytest.fixture
def my_simulator() -> NCompSimulator:
    return NCompSimulator(-0.1, 2.5, "data", verbose=False)


def test_calc_msd_t(my_simulator: NCompSimulator):

    const_arr = np.ones((4, 2, 10))
    msd_conv = my_simulator.calc_msd_t(const_arr)
    assert np.all(msd_conv == 0)

    sin_arr = np.sin(np.linspace(0.1, 10, 10))
    sin_arr = np.vstack([sin_arr, sin_arr]).T.reshape(10, 2, 1)
    msd_nconv = my_simulator.calc_msd_t(sin_arr)
    assert msd_nconv is None


def test_get_state_init(my_simulator: NCompSimulator):

    state_init = my_simulator.get_state_init(0)
    assert state_init.shape == (2, my_simulator.N)
    assert state_init[0, 0] == my_simulator.phi_d_dil_spin[0] * 0.97
    assert state_init[0, -1] == my_simulator.phi_d_den_spin[0] * 1.03
    assert np.all(state_init[1] == my_simulator.phi_r_spin[0])

    my_simulator.is_cp_data = True
    my_simulator.profile_eq[0] = np.ones((2, my_simulator.N))
    state_init = my_simulator.get_state_init(0)
    assert np.all(state_init == my_simulator.profile_eq[0])


def test_spinodal_from_phi_r(my_simulator: NCompSimulator):
    phi_r, phi_d_dil, phi_d_den = my_simulator.spinodal_from_phi_r()
    assert phi_r.shape == phi_d_den.shape == phi_d_dil.shape
    assert np.all(phi_d_den > phi_d_dil)
    assert np.all(phi_d_dil > 0)
    assert np.all(phi_r > 0)
    assert np.diff(phi_r).min() > 0


import matplotlib.pyplot as plt

chi_ds = 2.5
for chi_dr in np.linspace(0.01, 1.5, 3):
    sim = NCompSimulator(chi_dr, chi_ds, "")
    phi_r, phi_d_dil, phi_d_den = sim.spinodal_from_phi_r()

    plt.scatter(phi_r, phi_d_dil, color="red")
    plt.scatter(phi_r, phi_d_den, color="blue")

plt.show()
