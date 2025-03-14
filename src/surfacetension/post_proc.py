import re
import os
import numpy as np
import pde
from surfacetension import NCompSimulator


class PostProcessor:

    def __init__(self, raw_data_fp: str):
        self.datasets = self.fetch_data(raw_data_fp)

        # Iterate over all datasets which vary in interaction strengths
        for chis, dataset in self.datasets.items():

            # Get simulator object to calculate thermodynamic quantities
            # Chis are off by a factor of 10 for readability
            dataset["sim_obj"] = NCompSimulator(
                -chis[0] / 10, chis[1] / 10, raw_data_fp
            )
            phi_r, phi_d_dil, phi_d_den = dataset["sim_obj"].calc_spinodal()

            # Spinodal
            dataset["spinodal"] = {
                "phi_r": phi_r,
                "phi_d_dil": phi_d_dil,
                "phi_d_den": phi_d_den,
            }
            # Phase concentrations in dilute and dense phase
            dataset["phase_phis"] = self.get_phase_phis(chis)

            surf_ten, del_f, surf_exc = self.calc_surface_tension_from_dataset(chis)
            dataset["surface_tension"] = surf_ten
            dataset["del_free_energy"] = del_f
            dataset["surface_excess"] = surf_exc

    def interp_if_pos(self, phi_eq: np.ndarray, x: np.ndarray) -> tuple[float, float]:
        """
        Interpolate the position of the interface.

        Parameters
        ----------
        phi_eq : np.ndarray
            The equilibrium profile of the droplet.
        x : np.ndarray
            The x-coordinates of the grid.

        Returns
        -------
        tuple[float, float]
            The concentration and position of the interface.
        """
        phi_if = 1 / 2 * (phi_eq[0] + phi_eq[-1])
        phi_mult = (phi_eq - phi_if)[1:] * (phi_eq - phi_if)[:-1]
        idx_l = np.argmax(phi_mult < 0)
        idx_r = idx_l + 1
        x_if = x[idx_l] + (x[idx_r] - x[idx_l]) / (phi_eq[idx_r] - phi_eq[idx_l]) * (
            phi_if - phi_eq[idx_l]
        )

        return (x_if, phi_if)

    def get_phase_phis(self, chis: tuple[float, float]):
        """
        Get the concentrations in the dilute and dense phase.
        """

        prof_eq = self.datasets[chis]["prof_eq"]
        phase_phis = {
            "phi_d_dil": prof_eq[:, 0, 0],
            "phi_d_den": prof_eq[:, 0, -1],
            "phi_r_dil": prof_eq[:, 1, 0],
            "phi_r_den": prof_eq[:, 1, -1],
        }

        return phase_phis

    def calc_surface_tension_from_dataset(
        self, chis: tuple[float, float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculte the difference free energy, surface excess
        and from this the surface tension.

        Parameters
        ----------
        chis : tuple[float, float]
            The values of chi_dr and chi_ds.

        phi_r_idx : int
            The index for the list of regulator concentrations.

        Returns
        -------
        float
            The difference between the free energies.
        """

        dataset = self.datasets[chis]
        sim_obj = dataset["sim_obj"]
        prof_eq = dataset["prof_eq"]

        del_free_energies = np.zeros(prof_eq.shape[0])
        surface_excess = np.zeros(prof_eq.shape[0])
        surface_tension = np.zeros(prof_eq.shape[0])

        # Iterate over samples for increasing mean regulator concentrations
        for i, prof_eq_sample in enumerate(prof_eq):

            # In general use `NCompSimulator` class for constant system parameters
            # and class instance `sim_obj` for everything else
            _, x_if = self.interp_if_pos(prof_eq_sample[0], NCompSimulator.X_LIST)

            field_collection = pde.FieldCollection(
                [
                    pde.ScalarField(NCompSimulator.GRID, prof_eq_sample[i])
                    for i in range(2)
                ]
            )

            ###################################
            # FREE ENERGY
            ###################################
            # Soft interface free energy
            f_si = sim_obj.pde.free_energy(field_collection)

            # Hard interface free energy
            #                                   Concentrations in dilute phase
            f_hi = x_if * sim_obj.f.free_energy(prof_eq_sample[:, 0]) + (
                NCompSimulator.X_LIST[-1]
                - x_if
                #                     Concentrations in dense phase
            ) * sim_obj.f.free_energy(prof_eq_sample[:, -1])

            del_free_energies[i] = f_si - f_hi

            ###################################
            # SURFACE EXCESS
            ###################################
            sim_obj.pde.chemical_potential(field_collection)
            chem_pot = np.mean(
                sim_obj.pde.chemical_potential(field_collection).data, axis=-1
            )

            phi_si_d = prof_eq_sample[0].sum() * NCompSimulator.L / NCompSimulator.N
            phi_si_r = prof_eq_sample[1].sum() * NCompSimulator.L / NCompSimulator.N

            phi_hi_d = prof_eq_sample[0][0] * x_if + prof_eq_sample[0][-1] * (
                NCompSimulator.L - x_if
            )
            phi_hi_r = prof_eq_sample[1][0] * x_if + prof_eq_sample[1][-1] * (
                NCompSimulator.L - x_if
            )

            surface_excess[i] = (phi_si_d - phi_hi_d) * chem_pot[0] + (
                phi_si_r - phi_hi_r
            ) * chem_pot[1]

            ###################################
            # SURFACE TENSION
            ###################################
            surface_tension[i] = del_free_energies[i] - surface_excess[i]

        return surface_tension, del_free_energies, surface_excess

    def fetch_data(self, raw_data_dir: str):
        """
        Extract all unique (chi_dr, chi_ds) pairs from filenames in the given directory.
        Then load the corresponding datasets.

        Parameters
        ----------
        raw_data_dir : str
            The directory containing the raw data files.

        Returns
        -------
        dict
            A dictionary containing the datasets for each unique (chi_dr, ds) pair.
        """
        pattern = re.compile(r"^(msd|profEq)_dr(\d+)_ds(\d+)\.npy$")

        datasets = dict()

        for filename in os.listdir(raw_data_dir):
            match = pattern.match(filename)
            if match:
                if match.group(1) == "msd":
                    chi_dr = int(match.group(2))
                    chi_ds = int(match.group(3))
                    msd = np.load(os.path.join(raw_data_dir, filename))
                    prof_eq = np.load(
                        os.path.join(raw_data_dir, f"profEq_dr{chi_dr}_ds{chi_ds}.npy")
                    )
                    datasets[(chi_dr, chi_ds)] = {"prof_eq": prof_eq, "msd": msd}

        return datasets


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    post_proc = PostProcessor("data/raw_data")
    print(post_proc.datasets[(0, 0)]["surface_tension"])
    # Plot the surface tension for all datasets
    for dataset in post_proc.datasets.values():
        plt.plot(
            dataset["surface_tension"],
            label=f"chi_dr = {dataset['chi_dr']}, chi_ds = {dataset['chi_ds']}",
        )

    plt.xlabel
