import os
import numpy as np
from surfacetension import NCompSimulator

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":

    CHI_DS = 3.0
    for chi_dr in [0.0, 0.5, 0.8, 1.2, 1.5]:
        sim = NCompSimulator(
            -chi_dr, CHI_DS, os.path.join(root_dir, "data", "raw_data"), verbose=True
        )

        # Run the simulation
        sim.run()

    # for chi_dr in np.linspace(0.0, 1.0, 5):
    #     sim = NCompSimulator(
    #         -chi_dr, CHI_DS, os.path.join(root_dir, "data", "raw_data"), verbose=True
    #     )

    #     # Run the simulation
    #     sim.run()
