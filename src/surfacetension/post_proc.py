import re
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root_scalar
import pandas as pd
import logging


import pde
from surfacetension import NCompSimulator

logging.disable(logging.WARNING)


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
        self.sim_obj = NCompSimulator(self.chi_dr, self.chi_ds, box_size)

        # Phase concentrations in dilute and dense phase
        phase_phis = self.prof_eq[:, :, [0, -1]].reshape(len(prof_eq), -1)
        # Sort the concentrations in the order: [phi_d_den, phi_r_den, phi_d_dil, phi_r_dil]
        self.phase_phis = phase_phis[:, [1, 3, 0, 2]]

        self.phi_r_spin_max = SimulationPostProcessor.calc_phi_r_max(
            self.chi_dr, self.chi_ds
        )
        self.phi_r_c_list, self.is_inside = self.calc_critical_points()

        self.surface_tension, self.del_free_energies, self.surface_excess = (
            self.calc_surface_tension_from_dataset()
        )
        self.fit_params, self.fit_params_cov = self.fit_surface_tension()

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

            self.st_fig_dir = os.path.join(
                self.figures_dir, f"box_size{self.box_size}", "surface_tension"
            )
            os.makedirs(self.st_fig_dir, exist_ok=True)

    @staticmethod
    def interp_if_pos(x: np.ndarray, phi_eq: np.ndarray) -> tuple[float, float]:
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
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculte the difference free energy, surface excess
        and from this the surface tension.

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
            x_if, _ = SimulationPostProcessor.interp_if_pos(
                self.sim_obj.x_list, prof_eq_sample[0]
            )

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

    def fit_surface_tension(self) -> tuple[float, float]:
        """
        Fit the surface tension to a power law.

        Returns
        -------
        tuple[float, float]
            The parameters of the power law fit.
        """
        phi_r_dil, phi_r_c = self.phase_phis[:, 3], self.phi_r_c_list[-1]

        idx_select = np.where(phi_r_c - phi_r_dil > 0)[0]
        surface_tension = self.surface_tension[idx_select][-6:]
        phi_r_dil = phi_r_dil[idx_select][-6:]

        popt, pcov = curve_fit(self.power_func, phi_r_c - phi_r_dil, surface_tension)
        return popt, np.diag(pcov)

    @staticmethod
    def spinodal_coefficients(
        phi_r: float, chi_dr: float, chi_ds: float
    ) -> tuple[float, float]:
        """Get p and q coefficients for the spinodal equation"""

        q = 1 / (phi_r * (chi_dr - chi_ds) ** 2 + 2 * chi_ds)
        p = phi_r - 1 - 2 * chi_dr * phi_r * q

        return p, q

    @staticmethod
    def calc_phi_d_spin(
        phi_r: float, chi_dr: float, chi_ds: float, is_dense_phase: bool = False
    ) -> float:
        """
        Calculate the spinodal droplet concentration as a function of the regulator concentration

        Parameters
        ----------
        phi_r : float
            Regulator concentration
        chi_dr: float
            Regulator coupling parameter
        chi_ds: float
            Droplet coupling parameter
        is_dense_phase : bool, optional
            Calculate the droplet concentration in the dense phase, by default False

        Returns
        -------
        float
            Droplet concentration
        """
        p, q = SimulationPostProcessor.spinodal_coefficients(phi_r, chi_dr, chi_ds)

        if is_dense_phase:
            return -p / 2 + np.sqrt(p**2 / 4 - q)
        return -p / 2 - np.sqrt(p**2 / 4 - q)

    @staticmethod
    def calc_phi_r_max(chi_dr, chi_ds) -> float:
        """Calculate the maximum regulator concentration for which phase separation occurs."""

        def func(phi_r):
            p, q = SimulationPostProcessor.spinodal_coefficients(phi_r, chi_dr, chi_ds)
            return p**2 - 4 * q

        phi_r_spin_max = root_scalar(
            func, bracket=[0.15, 0.4], method="brentq", xtol=1e-8
        ).root
        return phi_r_spin_max

    def calc_deriv_phi_d_spin(self, phi_r: float, h=1e-8) -> float:
        """
        Calculate the derivative of the spinodal droplet concentration as a function of the regulator concentration
        using finite differences.
        """
        return (
            SimulationPostProcessor.calc_phi_d_spin(phi_r + h, self.chi_dr, self.chi_ds)
            - SimulationPostProcessor.calc_phi_d_spin(
                phi_r - h, self.chi_dr, self.chi_ds
            )
        ) / (2 * h)

    def calc_critical_points(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the critical point by using the fact that the derivative of the spinodal droplet concentration
        is the same as the slope of the tie line near the critical point.
        For this use the six closest points to the critical point.

        Returns
        -------
        tuple[float, float]
            phi_D and phi_R at the critical point
        """

        if self.chi_dr == 0.0:
            # If chi_dr = 0, the critical point is at the maximum regulator concentration
            # and all tie lines are inside the spinodal
            return np.ones((len(self.phase_phis),)) * self.phi_r_spin_max, np.ones(
                (len(self.phase_phis),), dtype=bool
            )

        phi_d_den, phi_r_den, phi_d_dil, phi_r_dil = self.phase_phis.T
        slopes_tie = (phi_d_den - phi_d_dil) / (phi_r_den - phi_r_dil)

        def func(phi_r, deriv):

            return self.calc_deriv_phi_d_spin(phi_r) - deriv

        phi_r_c = np.empty(len(slopes_tie))

        for i, slope in enumerate(slopes_tie[::-1]):
            ir = len(slopes_tie) - 1 - i
            if slope > 100:
                phi_r_c[ir] = phi_r_c[ir + 1]
                continue
            phi_r_c[ir] = root_scalar(
                func,
                args=(slope,),
                bracket=[0.2, self.phi_r_spin_max - 1e-6],
                method="brentq",
            ).root

        # Cross product of displacement vectors between critical point
        # and dense/dilute phase should be negative
        phi_d_c = SimulationPostProcessor.calc_phi_d_spin(
            phi_r_c, self.chi_dr, self.chi_ds
        )
        is_inside = (
            (phi_r_den - phi_r_c) * (phi_d_dil - phi_d_c)
            - (phi_d_den - phi_d_c) * (phi_r_dil - phi_r_c)
        ) > 0

        return phi_r_c, is_inside

    def plot_surface_tension(self):
        """
        Plot the surface tension of the system.
        """

        # Use critical point from tie line closest to spinodal
        phi_r_c = self.phi_r_c_list[self.is_inside][-1]
        # plt.figure(figsize=(8, 6))

        # Only plot datapoints with tie lines inside the spinodal
        phi_r_dil = self.phase_phis[self.is_inside, 3]
        surface_tension = self.surface_tension[self.is_inside]

        plt.scatter(
            phi_r_c - phi_r_dil,
            surface_tension,
            marker="o",
            facecolors="blue",
            edgecolors="black",
        )
        plt.plot(
            phi_r_c - phi_r_dil,
            self.power_func(phi_r_c - phi_r_dil, *self.fit_params),
            color="black",
            linestyle="--",
            label=r"$\propto (\phi_r - \phi_{r,dil})^{%.2f}$" % self.fit_params[1],
        )

        plt.ylabel(r"$\gamma$")
        plt.xlabel(r"$\phi_r - \phi_{r,dil}$")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()

        plt.savefig(
            os.path.join(
                self.st_fig_dir, f"dr{self.chi_dr_str}_ds{self.chi_ds_str}.png"
            )
        )
        plt.close()

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
        phi_r_spin = np.linspace(0, self.phi_r_spin_max - 1e-7, 1000)
        phi_d_den_spin = SimulationPostProcessor.calc_phi_d_spin(
            phi_r_spin, self.chi_dr, self.chi_ds, is_dense_phase=True
        )
        phi_d_dil_spin = SimulationPostProcessor.calc_phi_d_spin(
            phi_r_spin, self.chi_dr, self.chi_ds
        )

        phi_d_den, phi_r_den, phi_d_dil, phi_r_dil = self.phase_phis.T

        phi_r_c = self.phi_r_c_list
        phi_d_c = SimulationPostProcessor.calc_phi_d_spin(
            phi_r_c, self.chi_dr, self.chi_ds
        )

        plt.figure(figsize=(8, 6))
        plt.subplot(121)
        plt.title(r"$\chi_{dr} = %.1f, \chi_{ds} = %.1f$" % (self.chi_dr, self.chi_ds))

        plt.scatter(phi_r_dil, phi_d_dil, facecolors="blue", edgecolors="black")
        plt.scatter(
            phi_r_den, phi_d_den, label="Binodal", facecolors="blue", edgecolors="black"
        )
        # Tie lines
        plt.plot(
            np.array([phi_r_den, phi_r_dil]),
            np.array([phi_d_den, phi_d_dil]),
            color="black",
            alpha=0.1,
        )
        plt.plot(phi_r_spin, phi_d_dil_spin, color="orange", linestyle="--")
        plt.plot(
            phi_r_spin, phi_d_den_spin, color="orange", linestyle="--", label="Spinodal"
        )
        plt.scatter(
            phi_r_c[self.is_inside],
            phi_d_c[self.is_inside],
            color="black",
            marker="x",
            label="Critical points (inside)",
        )
        plt.scatter(
            phi_r_c[~self.is_inside],
            phi_d_c[~self.is_inside],
            color="red",
            marker="x",
            label="Critical points (outside)",
        )
        plt.legend()
        plt.xlabel(r"$\phi_R$")
        plt.ylabel(r"$\phi_D$")

        # Plot points near critical point
        plt.subplot(122)

        phi_r_den_spin = np.linspace(phi_r_den[-6], self.phi_r_spin_max - 1e-7, 1000)
        phi_d_den_spin = SimulationPostProcessor.calc_phi_d_spin(
            phi_r_den_spin, self.chi_dr, self.chi_ds, is_dense_phase=True
        )
        phi_r_dil_spin = np.linspace(phi_r_dil[-6], self.phi_r_spin_max - 1e-7, 1000)
        phi_d_dil_spin = SimulationPostProcessor.calc_phi_d_spin(
            phi_r_dil_spin, self.chi_dr, self.chi_ds
        )
        plt.plot(phi_r_dil_spin, phi_d_dil_spin, color="orange", linestyle="--")
        plt.plot(phi_r_den_spin, phi_d_den_spin, color="orange", linestyle="--")
        plt.scatter(
            phi_r_dil[-6:], phi_d_dil[-6:], facecolors="blue", edgecolors="black"
        )
        plt.scatter(
            phi_r_den[-6:], phi_d_den[-6:], facecolors="blue", edgecolors="black"
        )
        plt.plot(
            np.array(
                [phi_r_den[-6:], phi_r_dil[-6:]],
            ),
            np.array([phi_d_den[-6:], phi_d_dil[-6:]]),
            color="black",
            alpha=0.1,
        )
        plt.scatter(
            phi_r_c[self.is_inside][-6:],
            phi_d_c[self.is_inside][-6:],
            color="black",
            marker="x",
        )
        plt.scatter(
            phi_r_c[~self.is_inside][-6:],
            phi_d_c[~self.is_inside][-6:],
            color="red",
            marker="x",
        )
        plt.xlabel(r"$\phi_R$")
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.phase_fig_dir,
                f"dr{self.chi_dr_str}_ds{self.chi_ds_str}.png",
            )
        )
        # if self.chi_dr < -0.5:
        #     plt.show()
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

    def plot_fit_params(self, box_size: int):
        """
        Plot the fit parameters for all datasets.
        Plot separate curves for each unique ds value.
        """
        crit_exp = {"25": [], "30": []}
        crit_exp_dev = {"25": [], "30": []}
        chis_dr = {"25": [], "30": []}

        post_processors = self.post_processors[
            self.post_processors["box_size"] == box_size
        ]

        for i, row in post_processors.iterrows():
            post_proc = row["post_processor"]
            crit_exp[post_proc.chi_ds_str].append(post_proc.fit_params[1])
            crit_exp_dev[post_proc.chi_ds_str].append(
                np.sqrt(post_proc.fit_params_cov[1])
            )
            chis_dr[post_proc.chi_ds_str].append(abs(post_proc.chi_dr))

        plt.figure(figsize=(8, 6))
        for chi_ds, crit_exps in crit_exp.items():
            plt.errorbar(
                chis_dr[chi_ds],
                crit_exps,
                yerr=crit_exp_dev[chi_ds],
                fmt="o",
                capsize=2,
                ecolor="black",
                label=r"$\chi_{ds} = %.1f$" % (float(chi_ds) / 10),
            )
        plt.legend()
        plt.xlabel(r"$|\chi_{dr}|$")
        plt.ylabel(r"$\delta$")
        plt.title(r"$\gamma \propto (\phi_r - \phi_{r,dil})^{\delta}$")

        plt.savefig(os.path.join(self.figures_dir, "crit_exp.png"))
        plt.show()
        plt.close()


if __name__ == "__main__":
    post_proc_collection = PostProcessorCollection("data", "report/figures")

    post_proc_collection.plot_fit_params(800)
    # Calculate surface tension for all datasets
    for i, row in post_proc_collection.post_processors.iterrows():
        post_proc = row["post_processor"]
        # post_proc.plot_profiles()
        # post_proc.plot_profiles(is_droplet=False)
        # post_proc.plot_phase_diag()
        # post_proc.plot_msd()
        # post_proc.plot_surface_tension()
#
