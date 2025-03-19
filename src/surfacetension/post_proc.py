import re
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import pde
from surfacetension import NCompSimulator


# TODO: - Check how `phasesep.CahnHilliardMultiplePDE.free_energy` integrates numerically
#         and make sure own implementation `PostProcessor` class is correct.
#       - Quantify the error caused by the insufficient system size near the critical point.


class PostProcessor:

    COLOR_OPS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    def __init__(self, dir: str):
        self.datasets = self.fetch_data(dir)

        # Iterate over all datasets which vary in interaction strengths
        for chis, dataset in self.datasets.items():

            # Get simulator object to calculate thermodynamic quantities
            # Chis are off by a factor of 10 for readability
            dataset["sim_obj"] = NCompSimulator(-chis[0] / 10, chis[1] / 10, dir)
            dataset["phi_r_c"] = dataset["sim_obj"].find_phi_r_c()

            dataset["spinodal"] = dataset["sim_obj"].calc_spinodal()
            dataset["binodal"] = dataset["sim_obj"].calc_binodal()

            # Phase concentrations in dilute and dense phase
            dataset["phase_phis"] = self.datasets[chis]["prof_eq"][:, :, [0, -1]]

            surf_ten, del_f, surf_exc = self.calc_surface_tension_from_dataset(chis)
            dataset["surface_tension"] = surf_ten
            dataset["del_free_energy"] = del_f
            dataset["surface_excess"] = surf_exc
            dataset["power_law_fit"] = self.fit_surface_tension(chis)

    def interp_if_pos(self, x: np.ndarray, phi_eq: np.ndarray) -> tuple[float, float]:
        """
        Interpolate the position of the interface.

        Parameters
        ----------
        x : np.ndarray
            The x-coordinates of the grid.
        phi_eq : np.ndarray
            The equilibrium profile of the droplet.

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
            x_if, phi_if = self.interp_if_pos(NCompSimulator.X_LIST, prof_eq_sample[0])

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
                + NCompSimulator.X_LIST[1]
                - NCompSimulator.X_LIST[0]
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
        surface_tension = del_free_energies - surface_excess

        return surface_tension, del_free_energies, surface_excess

    def fetch_data(self, dir: str):
        """
        Extract all unique (chi_dr, chi_ds) pairs from filenames in the given directory.
        Then load the corresponding datasets.

        Parameters
        ----------
        dir : str
            The directory containing the data.

        Returns
        -------
        dict
            A dictionary containing the datasets for each unique (chi_dr, ds) pair.
        """
        dir_sim_data = os.path.join(dir, "sim_results")
        pattern = re.compile(r"^(msd|profEq)_dr(\d+)_ds(\d+)\.npy$")

        datasets = {}

        for filename in os.listdir(dir_sim_data):
            match = pattern.match(filename)
            if match:
                if match.group(1) == "msd":
                    chi_dr = int(match.group(2))
                    chi_ds = int(match.group(3))
                    msd = np.load(os.path.join(dir_sim_data, filename))
                    prof_eq = np.load(
                        os.path.join(dir_sim_data, f"profEq_dr{chi_dr}_ds{chi_ds}.npy")
                    )
                    samples_select = np.where(prof_eq[:, 0, 0] > 0.0)[0]
                    datasets[(chi_dr, chi_ds)] = {
                        "prof_eq": prof_eq[samples_select],
                        "msd": msd[samples_select],
                    }

        return datasets

    @staticmethod
    def power_func(x, a, b):
        return a * x**b

    def fit_surface_tension(self, chis: tuple[float, float]) -> tuple[float, float]:
        """
        Fit the surface tension to a power law.

        Parameters
        ----------
        chis : tuple[float, float]
            The values of chi_dr and chi_ds.

        Returns
        -------
        tuple[float, float]
            The parameters of the power law fit.
        """
        dataset = self.datasets[chis]
        surface_tension = dataset["surface_tension"]
        phi_r_dil, phi_r_c = dataset["phase_phis"][:, 1, 0], dataset["phi_r_c"]

        idx_select = np.where(surface_tension > 2e-5)[0]
        surface_tension = surface_tension[idx_select][-6:]
        phi_r_dil = phi_r_dil[idx_select][-6:]

        popt, pcov = curve_fit(
            self.power_func,
            phi_r_c - phi_r_dil,
            surface_tension,
        )
        return popt, np.diag(pcov)

    def plot_surface_tension(self):
        """
        Plot the free energy of the system.
        """

        nsamples_plot = 3
        rand_samp = np.random.choice(len(self.datasets), nsamples_plot)
        plt.figure(figsize=(8, 6))

        for i, r in enumerate(rand_samp):
            chis, dataset = list(self.datasets.items())[r]

            surface_tension = dataset["surface_tension"]
            # Near the critical point some samples don't phase separate
            # idx_select = np.where(surface_tension > 1e-6)[0]

            phi_r_dil, phi_r_c = (
                dataset["phase_phis"][:, 1, 0],
                dataset["phi_r_c"],
            )
            a, b = dataset["power_law_fit"][0]
            st_fit = self.power_func(phi_r_c - phi_r_dil, a, b)
            fit_idx = np.where(phi_r_c - phi_r_dil < 4e-2)[0]

            plt.scatter(
                phi_r_c - phi_r_dil,
                surface_tension,
                marker="o",
                color=self.COLOR_OPS[i],
                label=r"$\chi_{dr} = %.1f, \chi_{ds} = %.1f$"
                % (-chis[0] / 10, chis[1] / 10),
            )
            plt.plot(
                phi_r_c - phi_r_dil,
                st_fit,
                color=self.COLOR_OPS[i],
                linestyle="--",
            )

        plt.ylabel(r"$\gamma$")
        plt.xlabel(r"$\phi_r - \phi_{r,dil}$")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()

        # plt.savefig("st_fit.png")
        plt.show()

    def plot_profiles(self, chis: tuple[float, float]):
        """
        Plot the equilibrium profiles for a given (chi_dr, chi_ds) pair.

        Parameters
        ----------
        chis : tuple[float, float]
            The values of chi_dr and chi_ds.
        """
        dataset = self.datasets[chis]
        prof_eq = dataset["prof_eq"]
        colormap = plt.get_cmap("viridis")

        plt.figure(figsize=(8, 6))
        for i, prof in enumerate(prof_eq):
            plt.plot(
                NCompSimulator.X_LIST,
                prof[1],
                color=colormap(i / len(prof_eq)),
            )
        plt.xlabel(r"$x$")
        plt.ylabel(r"$\phi_D$")
        plt.title(
            r"$\chi_{dr} = %.1f, \chi_{ds} = %.1f$" % (-chis[0] / 10, chis[1] / 10)
        )
        # plt.savefig(f"profiles_dr{chis[0]}_ds{chis[1]}.png")
        plt.show()

    def plot_fit_params(self):

        crit_exp = {25: [], 30: []}
        crit_exp_dev = {25: [], 30: []}
        chis_dr = {25: [], 30: []}
        for i, (chis, dataset) in enumerate(self.datasets.items()):
            crit_exp[chis[1]].append(dataset["power_law_fit"][0][1])
            crit_exp_dev[chis[1]].append(np.sqrt(dataset["power_law_fit"][1][1]))
            chis_dr[chis[1]].append(chis[0] / 10 + i * 0.01)

        plt.figure(figsize=(8, 6))
        for chi_ds, crit_exps in crit_exp.items():
            plt.errorbar(
                chis_dr[chi_ds],
                crit_exps,
                yerr=crit_exp_dev[chi_ds],
                fmt="o",
                capsize=2,
                ecolor="black",
                label=r"$\chi_{ds} = %.1f$" % (chi_ds / 10),
            )
        plt.legend()
        plt.xlabel(r"$|\chi_{dr}|$")
        plt.ylabel(r"$\delta$")
        plt.title(r"$\gamma \propto (\phi_r - \phi_{r,dil})^{\delta}$")

        plt.savefig("crit_exp.png")
        # plt.show()


if __name__ == "__main__":

    post_proc = PostProcessor("data")
    # post_proc.plot_surface_tension()
    # post_proc.plot_fit_params()
    post_proc.plot_profiles((8, 30))
