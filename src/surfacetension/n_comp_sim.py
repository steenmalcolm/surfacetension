"""Simulate the three component system with predefined chi matrix"""

import time
import os


import numpy as np
import scipy


import pde
import matplotlib.pyplot as plt
import phasesep


class NCompSimulator:
    # System parameters
    L = 400
    N = 128
    LMD = -4
    X_LIST = np.linspace(0, L, N + 1)[:-1]
    GRID = pde.CartesianGrid([(0, L)], [N], periodic=False)

    # Simulation parameters
    T_SIM = 1000000
    NUM_FRAMES = 100  # Number of frames during the simulation
    CONV_VALUE = [1e-10, 1e-10]  # MSD must be below this value for convergence

    def __init__(
        self, chi_dr: float, chi_ds: float, out_dir_path: str, verbose: bool = False
    ):
        """
        This class simulates the three component system for a set of interaction parameters.
        The resulting object are the equilibrium profiles of the system from which
        one can calculate all other thermodynamic properties such as the surface tension.

        Parameters
        ----------
        chi_dr : float
            Interaction parameter between the droplet and the regulator
        chi_ds : float
            Interaction parameter between the droplet and the solvent
        verbose : bool, optional
            Print the progress of the simulation, by default False

        """
        self.verbose = verbose

        self.chi_dr = chi_dr
        self.chi_ds = chi_ds
        chi_matrix = np.array(
            [[0, chi_dr, chi_ds], [chi_dr, 0, 0], [chi_ds, 0, 0]], dtype=float
        )

        self.f = phasesep.FloryHugginsNComponents(chis=chi_matrix, num_comp=3)

        self.pde = phasesep.CahnHilliardMultiplePDE(
            {
                "free_energy": self.f,
                "kappa": self.LMD * chi_matrix,
                "regularize_after_step": True,
                "mobility_model": "scaled_correct",
            }
        )

        # Good starting concentrations for the simulations near the spinodal
        self.phi_r_spin, self.phi_d_dil_spin, self.phi_d_den_spin = (
            self.spinodal_from_phi_r()
        )

        # File path to save the simulation data
        self.out_prof_eq = os.path.join(
            out_dir_path, f"profEq_dr{abs(chi_dr*10):.0f}_ds{chi_ds*10:.0f}.npy"
        )
        # File path for average absolute difference of the profiles
        self.out_msd = os.path.join(
            out_dir_path, f"msd_dr{abs(chi_dr*10):.0f}_ds{chi_ds*10:.0f}.npy"
        )

        if os.path.exists(self.out_prof_eq):
            self.profile_eq = np.load(self.out_prof_eq)
            self.msd_t = np.load(self.out_msd)
            self.is_cp_data = True  # Checkpoint data

        else:
            # Save the equilibrium profiles
            self.profile_eq = np.zeros((len(self.phi_r_spin), 2, self.N))
            self.msd_t = np.zeros((len(self.phi_r_spin), 2, self.NUM_FRAMES))
            self.is_cp_data = False

    def run(self):
        """Compute the binodal curve by simulating the system at different initial concentrations"""

        t_sim = self.T_SIM

        for i in range(len(self.phi_r_spin)):

            n = time.perf_counter()
            # Prevent getting stuck in a non-converging simulation
            failed_conv_count = 0

            state_init = self.get_state_init(i)

            while True:
                profile_t = self.evolve(state_init, t_sim)

                msd_t = self.calc_msd_t(profile_t)
                if msd_t is not None:
                    break

                if self.verbose:
                    print(f"\tSimulation did not converge for t_sim = {t_sim}")
                # If not converged update the initial concentrations and sim duration
                state_init = profile_t[-1]
                t_sim *= 1.5

                failed_conv_count += 1
                if failed_conv_count > 10:
                    raise RuntimeError("Simulation did not converge after 10 runs")

            self.profile_eq[i] = profile_t[-1]  # Last frame of the simulation
            self.msd_t[i] = msd_t

            # Save the intermediate results
            np.save(self.out_prof_eq, self.profile_eq)
            np.save(self.out_msd, self.msd_t)

            if self.verbose:
                print(
                    f"Finished simulation {i+1}/{len(self.phi_r_spin)} in {time.perf_counter()-n:.2f} s"
                )

    def spinodal_from_phi_r(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the spinodal curve which depends on the interaction parameter

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            phi_r, phi_d_den, phi_d_dil
        """

        # Higher resolution at low phi_r and critical point
        phi_r_c = self.find_phi_r_c()

        # Higher resolution close to the critical point
        phi_r = phi_r_c - np.exp(np.linspace(np.log(1e-3), np.log(0.02), 11))[::-1]
        phi_r = np.append(np.linspace(0.01, phi_r[0], 20)[:-1], phi_r)

        # Calculate droplet concentrations in dilute and dense phase
        q = 1 / (phi_r * (self.chi_dr - self.chi_ds) ** 2 + 2 * self.chi_ds)
        p = phi_r - 1 - 2 * self.chi_dr * phi_r * q

        phi_d_den = -p / 2 + np.sqrt(p**2 / 4 - q)
        phi_d_dil = -p / 2 - np.sqrt(p**2 / 4 - q)

        # Remove imaginary solutions
        real_idx = phi_d_dil > 0
        phi_r = phi_r[real_idx]
        phi_d_den = phi_d_den[real_idx]
        phi_d_dil = phi_d_dil[real_idx]

        return phi_r, phi_d_dil, phi_d_den

    def find_phi_r_c(self) -> float:
        """
        Find the critical concentration of the regulator
        """

        def cp(args):
            phi_r = args[0]
            q = 1 / (phi_r * (self.chi_dr - self.chi_ds) ** 2 + 2 * self.chi_ds)
            p = phi_r - 1 - 2 * self.chi_dr * phi_r * q

            return p**2 / 4 - q

        sol = scipy.optimize.fsolve(cp, 0.2)
        return sol[0]

    def evolve(self, state_init: np.ndarray, t_sim: int) -> np.ndarray:
        """
        Evolve the initial state of the simulation

        Parameters
        ----------
        state_init : np.ndarray
            Initial state of the simulation | shape (2, N)

        Returns
        -------
        np.ndarray
            Profile of the simulation at different times
        """

        field_init = pde.FieldCollection(
            [
                pde.ScalarField(self.GRID, data=state_init[0], label="Droplet"),
                pde.ScalarField(self.GRID, data=state_init[1], label="Regulator"),
            ]
        )

        storage = pde.MemoryStorage()
        self.pde.solve(
            field_init,
            t_range=t_sim,
            adaptive=True,
            tracker=storage.tracker(t_sim / self.NUM_FRAMES),
        )

        return storage.data

    def get_state_init(self, i: int) -> np.ndarray:
        """
        Get the initial state for the simulation.
        The droplet profile has a step function and the regulator profile is constant

        Parameters
        ----------
        i : int
            Index of the simulation

        Returns
        -------
        np.ndarray
            Initial state of the simulation | shape (2, N)
        """
        if self.is_cp_data:
            return self.profile_eq[i]

        # Choose point near spinodal
        phi_d_den = self.phi_d_den_spin[i]  # * 1.001
        phi_d_dil = self.phi_d_dil_spin[i]  # * 0.999
        phi_r = self.phi_r_spin[i]

        droplet_data = np.ones(self.N) * phi_d_dil
        droplet_data[self.N // 2 :] = phi_d_den  # Step function
        regulator_data = np.ones(self.N) * phi_r

        return np.array([droplet_data, regulator_data])

    def calc_msd_t(self, profile_t: np.ndarray) -> np.ndarray | None:
        """
        Check if the simulation has converged by calculating the mean square difference

        Parameters
        ----------
        profile_t : np.ndarray
            Profile of the simulation at different times

        Returns
        -------
        np.ndarray
            Mean square difference of the profiles | shape (2, T)

        """
        prof_rel_diff_t = np.mean(np.diff(profile_t, axis=0) ** 2, axis=-1) / np.mean(
            profile_t[0], axis=-1
        )  # Shape (T, 2)

        # Accept if at least one frame converged in both components
        if np.all(np.any(prof_rel_diff_t < self.CONV_VALUE, axis=0)):
            return prof_rel_diff_t.T
        return None
