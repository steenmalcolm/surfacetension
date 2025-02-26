import re
import os
import numpy as np
import pde
from surfacetension import NCompSimulator


class PostProcessor:
    def __init__(self, raw_data_fp: str):
        self.datasets = self.fetch_data(raw_data_fp)
        self.x_list = NCompSimulator.X_LIST.copy()

        # Get spinodal for all datasets
        for chis, dataset in self.datasets.items():
            sim_obj = NCompSimulator(chis[0], chis[1], raw_data_fp)
            phi_r, phi_d_dil, phi_d_den = sim_obj.spinodal_from_phi_r()

            dataset["sim_obj"] = sim_obj

            dataset["spinodal"] = {
                "phi_r": phi_r,
                "phi_d_dil": phi_d_dil,
                "phi_d_den": phi_d_den,
            }

            surf_ten, del_f, surf_exc = self.calc_surface_tension_from_dataset(chis)
            dataset["surface_tension"] = surf_ten
            dataset["del_free_energy"] = del_f
            dataset["surface_excess"] = surf_exc

    def interp_if_pos(self, phi_eq: np.ndarray, x: np.ndarray) -> tuple[float, float]:
        """
        Return the concentration and position of the interface

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

    def calc_surface_tension_from_dataset(
        self, chis: tuple[float, float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculte the difference between hard and soft interface free energy for dataset.

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

        # Iterate over samples for different mean regulator concentrations
        for i, prof_eq_sample in enumerate(prof_eq):

            _, x_if = self.interp_if_pos(prof_eq_sample[0], sim_obj.X_LIST)

            field_collection = pde.FieldCollection(
                [pde.ScalarField(sim_obj.GRID, prof_eq_sample[i]) for i in range(2)]
            )

            # Compute free energy of the system with gradient terms
            f_si = sim_obj.EQ.free_energy(field_collection)

            # Compute the free energy the hard interface limit
            #                                   Concentrations in dilute phase
            f_hi = x_if * sim_obj.F.free_energy(prof_eq_sample[:, 0]) + (
                sim_obj.X_LIST[-1]
                - x_if
                #                     Concentrations in dense phase
            ) * sim_obj.F.free_energy(prof_eq_sample[:, -1])

            del_free_energies[i] = f_si - f_hi

            # Calculate the surface excess
            NCompSimulator.EQ.chemical_potential(field_collection)
            chem_pot = np.mean(
                sim_obj.EQ.chemical_potential(field_collection).data, axis=-1
            )

            phi_si_d = prof_eq_sample[0].sum() * sim_obj.L / sim_obj.N
            phi_si_r = prof_eq_sample[1].sum() * sim_obj.L / sim_obj.N

            phi_hi_d = prof_eq_sample[0][0] * x_if + prof_eq_sample[0][-1] * (
                sim_obj.L - x_if
            )
            phi_hi_r = prof_eq_sample[1][0] * x_if + prof_eq_sample[1][-1] * (
                sim_obj.L - x_if
            )

            surface_excess[i] = (phi_si_d - phi_hi_d) * chem_pot[0] + (
                phi_si_r - phi_hi_r
            ) * chem_pot[1]

            # Calculate the surface tension
            surface_tension[i] = del_free_energies[i] - surface_excess[i]

        return surface_tension, del_free_energies, surface_excess

    def fetch_data(self, raw_data_dir: str):
        """
        Extract all unique (dr, ds) pairs from filenames in the given directory.

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


def compute_free_energy(fields: np.ndarray, p: int = 1) -> float:
    assert (
        fields.shape[0] == 2
    ), f"`fields` needs to have shape (2, N) not {fields.shape}"

    c_d, c_r = fields

    field_collection = pde.FieldCollection(
        [pde.ScalarField(GRID, fields[i]) for i in range(2)]
    )
    # Compute free energy of the system with surface tension

    f_sil = EQ.free_energy(field_collection)

    # Compute the free energy the hard interface limit
    x_if_d, _ = pos_if(c_d, X_LIST)
    x_if_r, _ = pos_if(c_r, X_LIST)
    x_if = p * x_if_d + (1 - p) * x_if_r
    f_hil = x_if * F.free_energy(fields[:, 0]) + (
        X_LIST[-1] + X_LIST[1] - X_LIST[0] - x_if  # Right border of grid minus x_if
    ) * F.free_energy(fields[:, -1])

    return f_sil - f_hil
