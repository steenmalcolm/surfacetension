import re
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

import pde
from surfacetension import NCompSimulator


# TODO: - Check how `phasesep.CahnHilliardMultiplePDE.free_energy` integrates numerically
#         and make sure own implementation `PostProcessor` class is correct.
#       - Quantify the error caused by the insufficient system size near the critical point.


class SimulationPostProcessor:

    COLOR_OPS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    def __init__(
        self,
        chi_dr: float,
        chi_ds: float,
        box_size: int,
        prof_eq: np.ndarray,
        msd: np.ndarray,
        binodal_phis: np.ndarray,
        figures_dir: str = None,
    ):
        self.chi_dr_str, self.chi_ds_str = str(chi_dr), str(chi_ds)
        self.chi_dr, self.chi_ds = -chi_dr / 10, chi_ds / 10
        self.box_size = box_size
        self.prof_eq = prof_eq
        self.msd = msd
        self.binodal_phis = binodal_phis
        self.figures_dir = figures_dir
        self.sim_obj = NCompSimulator(-chi_dr / 10, chi_ds / 10, box_size)

        # Phase concentrations in dilute and dense phase
        phase_phis = self.prof_eq[:, :, [0, -1]].reshape(len(prof_eq), -1)
        # Sort the concentrations in the order: [phi_d_den, phi_r_den, phi_d_dil, phi_r_dil]
        self.phase_phis = phase_phis[:, [1, 3, 0, 2]]

        if figures_dir is not None:

            self.prof_fig_dir = os.path.join(
                self.figures_dir, f"box_size{self.box_size}", "profiles"
            )
            os.makedirs(self.prof_fig_dir, exist_ok=True)

            self.msd_fig_dir = os.path.join(
                self.figures_dir, f"box_size{self.box_size}", "msd"
            )
            os.makedirs(self.msd_fig_dir, exist_ok=True)

            self.phase_fig_dir = os.path.join(
                self.figures_dir, f"box_size{self.box_size}", "phase_diag"
            )
            os.makedirs(self.phase_fig_dir, exist_ok=True)

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

        del_free_energies = np.zeros(self.prof_eq.shape[0])
        surface_excess = np.zeros(self.prof_eq.shape[0])
        surface_tension = np.zeros(self.prof_eq.shape[0])

        # Iterate over samples for increasing mean regulator concentrations
        for i, prof_eq_sample in enumerate(self.prof_eq):

            # In general use `NCompSimulator` class for constant system parameters
            # and class instance `sim_obj` for everything else
            x_if, phi_if = self.interp_if_pos(self.sim_obj.x_list, prof_eq_sample[0])

            field_collection = pde.FieldCollection(
                [
                    pde.ScalarField(self.sim_obj.grid, prof_eq_sample[i])
                    for i in range(2)
                ]
            )

            ###################################
            # FREE ENERGY
            ###################################
            # Soft interface free energy
            f_si = self.sim_obj.pde.free_energy(field_collection)

            # Hard interface free energy
            #                                   Concentrations in dilute phase
            f_hi = x_if * self.sim_obj.f.free_energy(prof_eq_sample[:, 0]) + (
                self.sim_obj.box_size
                - x_if
                #                     Concentrations in dense phase
            ) * self.sim_obj.f.free_energy(prof_eq_sample[:, -1])

            del_free_energies[i] = f_si - f_hi

            ###################################
            # SURFACE EXCESS
            ###################################
            self.sim_obj.pde.chemical_potential(field_collection)
            chem_pot = np.mean(
                self.sim_obj.pde.chemical_potential(field_collection).data, axis=-1
            )

            phi_si_d = (
                prof_eq_sample[0].sum() * self.sim_obj.box_size / NCompSimulator.N
            )
            phi_si_r = (
                prof_eq_sample[1].sum() * self.sim_obj.box_size / NCompSimulator.N
            )

            phi_hi_d = prof_eq_sample[0][0] * x_if + prof_eq_sample[0][-1] * (
                self.sim_obj.box_size - x_if
            )
            phi_hi_r = prof_eq_sample[1][0] * x_if + prof_eq_sample[1][-1] * (
                self.sim_obj.box_size - x_if
            )

            surface_excess[i] = (phi_si_d - phi_hi_d) * chem_pot[0] + (
                phi_si_r - phi_hi_r
            ) * chem_pot[1]

        ###################################
        # SURFACE TENSION
        ###################################
        surface_tension = del_free_energies - surface_excess

        return surface_tension, del_free_energies, surface_excess

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
        pass
        # phi_r_dil, phi_r_c =

        # idx_select = np.where(surface_tension > 2e-5)[0]
        # surface_tension = surface_tension[idx_select][-6:]
        # phi_r_dil = phi_r_dil[idx_select][-6:]

        # popt, pcov = curve_fit(
        #     self.power_func,
        #     phi_r_c - phi_r_dil,
        #     surface_tension,
        # )
        # return popt, np.diag(pcov)

    def calc_phi_d_spin(self, phi_r: float, is_dense_phase: bool = False) -> float:
        """
        Calculate the spinodal droplet concentration as a function of the regulator concentration

        Parameters
        ----------
        phi_r : float
            Regulator concentration
        is_dense_phase : bool, optional
            Calculate the droplet concentration in the dense phase, by default False

        Returns
        -------
        float
            Droplet concentration
        """
        q = 1 / (phi_r * (self.chi_dr - self.chi_ds) ** 2 + 2 * self.chi_ds)
        p = phi_r - 1 - 2 * self.chi_dr * phi_r * q

        if is_dense_phase:
            return -p / 2 + np.sqrt(p**2 / 4 - q)
        return -p / 2 - np.sqrt(p**2 / 4 - q)

    def calc_critical_point(self, chis: tuple[float, float]) -> tuple[float, float]:
        """
        Calculate the critical point

        Parameters
        ----------
        chis : tuple[float, float]
            The values of chi_dr and chi_ds.

        Returns
        -------
        tuple[float, float]
            The critical point.
        """
        pass
        # phi_d_dil, phi_d_den =

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
                dataset["phase_phis"][:, 2],
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

    def plot_profiles(self, is_droplet: bool = True):
        """
        Plot the equilibrium profiles for a given (chi_dr, chi_ds) pair.

        Parameters
        ----------
        is_droplet : bool, optional
            Plot the droplet profiles, by default True
        """

        colormap = plt.get_cmap("viridis")

        plt.figure(figsize=(8, 6))
        for i, prof in enumerate(self.prof_eq):
            prof_show = prof[0] if is_droplet else prof[1]
            plt.plot(
                self.sim_obj.x_list,
                prof_show,
                color=colormap(i / len(self.prof_eq)),
            )
        plt.xlabel(r"$x$")
        plt.ylabel(r"$\phi_D$" if is_droplet else r"$\phi_R$")
        plt.title(r"$\chi_{dr} = %.1f, \chi_{ds} = %.1f$" % (self.chi_dr, self.chi_ds))
        plt.savefig(
            os.path.join(
                self.prof_fig_dir,
                f"{"D" if is_droplet else "R"}_dr{self.chi_dr_str}_ds{self.chi_ds_str}.png",
            )
        )
        plt.close()

    def plot_phase_diag(self):
        """
        Plot the phi_r vs phi_d phase diagram.
        """
        phi_d_bin_den, phi_r_bin_den, phi_d_bin_dil, phi_r_bin_dil = self.binodal_phis.T
        phi_r_spin = np.linspace(0, 0.5, 1000)
        phi_d_den_spin = self.calc_phi_d_spin(phi_r_spin, is_dense_phase=True)
        phi_r_spin = phi_r_spin[phi_d_den_spin > 0]
        phi_d_den_spin = phi_d_den_spin[phi_d_den_spin > 0]
        phi_d_dil_spin = self.calc_phi_d_spin(phi_r_spin)

        phi_d_den, phi_r_den, phi_d_dil, phi_r_dil = self.phase_phis.T

        plt.figure(figsize=(8, 6))
        plt.scatter(phi_r_bin_dil, phi_d_bin_dil, color="blue")
        plt.scatter(phi_r_bin_den, phi_d_bin_den, label="Binodal", color="red")
        # Tie lines
        plt.plot(
            np.array([phi_r_den, phi_r_dil]),
            np.array([phi_d_den, phi_d_dil]),
            color="black",
            alpha=0.1,
        )
        # plt.scatter(phi_r_dil, phi_d_dil, color="blue")
        # plt.scatter(phi_r_den, phi_d_den, label="Binodal", color="red")
        # # Tie lines
        # plt.plot(
        #     np.array([phi_r_den, phi_r_dil]),
        #     np.array([phi_d_den, phi_d_dil]),
        #     color="black",
        #     alpha=0.1,
        # )
        plt.plot(phi_r_spin, phi_d_dil_spin, color="blue", linestyle="--")
        plt.plot(
            phi_r_spin, phi_d_den_spin, color="red", linestyle="--", label="Spinodal"
        )
        plt.legend()
        plt.xlabel(r"$\phi_r$")
        plt.ylabel(r"$\phi_d$")
        plt.savefig(
            os.path.join(
                self.phase_fig_dir,
                f"dr{self.chi_dr_str}_ds{self.chi_ds_str}.png",
            )
        )
        plt.close()

    def plot_msd(self):
        """
        Plot the sample that converged the least according to the MSD
        """

        msd_arg_max = np.argmax(self.msd[:, 0].min(axis=-1))
        plt.figure(figsize=(8, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.msd[msd_arg_max, 0])
        plt.yscale("log")
        plt.xlabel("T")
        plt.ylabel("MSD")
        plt.axhline(1e-9, 0, len(self.msd[0]), linestyle="--", alpha=0.5)
        plt.title(
            r"$\phi_D$" + " for sample %d / %d" % (msd_arg_max + 1, len(self.msd))
        )

        msd_arg_max = np.argmax(self.msd[:, 1].min(axis=-1))
        plt.subplot(1, 2, 2)
        plt.plot(self.msd[msd_arg_max, 1])
        plt.yscale("log")
        plt.xlabel("T")
        plt.axhline(1e-9, 0, len(self.msd[0]), linestyle="--", alpha=0.5)
        plt.title(
            r"$\phi_R$" + " for sample %d / %d" % (msd_arg_max + 1, len(self.msd))
        )
        plt.savefig(
            os.path.join(
                self.msd_fig_dir,
                f"dr{self.chi_dr_str}_ds{self.chi_ds_str}.png",
            )
        )
        plt.close()

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

        # plt.savefig("crit_exp.png")


class PostProcessorCollection:
    def __init__(self, data_dir: str, figures_dir: str = None):
        """
        Initialize the post-processor collection.

        Parameters
        ----------
        data_dir : str
            The directory containing the data."""
        self.data_dir = data_dir
        self.figures_dir = figures_dir

        self.post_processors = pd.DataFrame(
            columns=["chi_dr", "chi_ds", "box_size", "post_processor"],
        )

        self.fetch_data()

    def fetch_data(self):
        """
        Extract all unique (chi_dr, chi_ds) pairs from filenames in the given directory.
        Then load the corresponding datasets.


        """
        box_pattern = re.compile(r"^box_size(\d+)$")
        sim_pattern = re.compile(r"^(msd|profEq)_dr(\d+)_ds(\d+)\.npy$")

        # Iterate over different box sizes
        for box_fp in os.listdir(self.data_dir):

            box_match = box_pattern.match(box_fp)
            if box_match:

                box_size = int(box_match.group(1))
                sim_fp = os.path.join(self.data_dir, box_fp, "sim_results")

                # Iterate over different coupling parameter
                for filename in os.listdir(
                    os.path.join(self.data_dir, box_fp, "sim_results")
                ):
                    match = sim_pattern.match(filename)
                    if match:
                        chi_dr = int(match.group(2))
                        chi_ds = int(match.group(3))
                        msd = np.load(
                            os.path.join(sim_fp, f"msd_dr{chi_dr}_ds{chi_ds}.npy")
                        )
                        prof_eq = np.load(
                            os.path.join(sim_fp, f"profEq_dr{chi_dr}_ds{chi_ds}.npy")
                        )
                        binodals = np.load(
                            os.path.join(
                                self.data_dir,
                                box_fp,
                                "binodals",
                                f"dr{chi_dr}_ds{chi_ds}.npy",
                            )
                        )
                        # Remove empty samples
                        samples_select = np.where(np.max(prof_eq[:, 0], axis=-1) > 0.0)[
                            0
                        ]
                        prof_eq = prof_eq[samples_select]
                        msd = msd[samples_select]

                        self.post_processors.loc[len(self.post_processors)] = [
                            chi_dr,
                            chi_ds,
                            box_size,
                            SimulationPostProcessor(
                                chi_dr,
                                chi_ds,
                                box_size,
                                prof_eq,
                                msd,
                                binodals,
                                self.figures_dir,
                            ),
                        ]

    def plot_profiles(self):
        for i, row in self.post_processors.iterrows():
            post_proc = row["post_processor"]
            post_proc.plot_profiles()


if __name__ == "__main__":
    post_proc_collection = PostProcessorCollection("data", "report/figures")

    # Calculate surface tension for all datasets
    for i, row in post_proc_collection.post_processors.iterrows():
        post_proc = row["post_processor"]
        post_proc.plot_profiles()
        post_proc.plot_phase_diag()
        post_proc.plot_msd()
