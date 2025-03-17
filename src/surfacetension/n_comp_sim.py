"""Simulate the three component system with predefined chi matrix"""

import time
import os


import numpy as np
import scipy


import pde
import matplotlib.pyplot as plt
import phasesep
import flory


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
        self, chi_dr: float, chi_ds: float, out_dir: str, verbose: bool = False
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
        self.chi_matrix = np.array(
            [[0, chi_dr, chi_ds], [chi_dr, 0, 0], [chi_ds, 0, 0]], dtype=float
        )

        self.f = phasesep.FloryHugginsNComponents(chis=self.chi_matrix, num_comp=3)

        self.pde = phasesep.CahnHilliardMultiplePDE(
            {
                "free_energy": self.f,
                "kappa": self.LMD * self.chi_matrix,
                "regularize_after_step": True,
                "mobility_model": "scaled_correct",
            }
        )

        # Good starting concentrations for the simulations near the spinodal
        self.phi_d_den_spin, self.phi_d_dil_spin, self.phi_r_spin = self.calc_spinodal()

        # File path to save the simulation data
        os.makedirs(os.path.join(out_dir, "sim_results"), exist_ok=True)
        self.out_prof_eq = os.path.join(
            out_dir,
            "sim_results",
            f"profEq_dr{abs(chi_dr*10):.0f}_ds{chi_ds*10:.0f}.npy",
        )
        # File path for average absolute difference of the profiles
        self.out_msd = os.path.join(
            out_dir, "sim_results", f"msd_dr{abs(chi_dr*10):.0f}_ds{chi_ds*10:.0f}.npy"
        )
        # Cache binodal data for the given interaction parameters
        os.makedirs(os.path.join(out_dir, "binodals"), exist_ok=True)
        self.out_binod = os.path.join(
            out_dir,
            "binodals",
            f"dr{abs(self.chi_dr*10):.0f}_ds{self.chi_ds*10:.0f}.npy",
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

        binodal_phis = self.calc_binodal()

        for i, phis_init in enumerate(binodal_phis):

            n = time.perf_counter()
            # Prevent getting stuck in a non-converging simulation
            failed_conv_count = 0

            state_init = self.get_state_init(phis_init)

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

    def calc_spinodal(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        phi_r = phi_r_c - np.exp(np.linspace(np.log(1e-4), np.log(0.01), 11))[::-1]
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

        return phi_d_den, phi_d_dil, phi_r

    def calc_binodal(self) -> np.ndarray:
        """
        Calculate the binodal curve which depends on the interaction parameter

        Returns
        -------
        np.ndarray
            Binodal concentrations of the droplet and regulator in dense and dilute phase
        """
        num_comp = 3  # Set number of components

        if os.path.exists(self.out_binod):
            print("Debug: Shouldn't be here")
            return np.load(self.out_binod)

        binodal_phis = []

        # Obtain coexisting phases
        for i, phi_r in enumerate(self.phi_r_spin):
            phi_d = 0.5 * (self.phi_d_dil_spin[i] + self.phi_d_den_spin[i])
            phi_means = [
                phi_d,
                phi_r,
                1 - phi_r - phi_d,
            ]  # Set the average volume fractions
            phases = flory.find_coexisting_phases(num_comp, self.chi_matrix, phi_means)
            fracs = phases.fractions

            if len(fracs) == 2:
                binodal_phis.append(fracs[:, :2].flatten())

        # Binodals has order (phi_d_den, phi_r_den, phi_d_dil, phi_r_dil)
        binodal_phis = np.array(binodal_phis)
        np.save(self.out_binod, binodal_phis)
        return binodal_phis

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

    def get_state_init(self, phis_init: np.ndarray) -> np.ndarray:
        """
        Get the initial state for the simulation.

        Parameters
        ----------
        phis_init : np.ndarray
            Initial concentrations of the droplet and regulator in dense and dilute phase

        Returns
        -------
        np.ndarray
            Initial state of the simulation | shape (2, N)
        """

        droplet_data = np.ones(self.N) * phis_init[2]
        droplet_data[self.N // 2 :] = phis_init[0]  # Step function

        regulator_data = np.ones(self.N) * phis_init[3]
        regulator_data[self.N // 2 :] = phis_init[1]

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


if __name__ == "__main__":
    my_sim = NCompSimulator(-0.5, 2.5, "data", verbose=True)
    my_sim.run()
