import os
import numpy as np
from surfacetension import NCompSimulator

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":

    box_size = 800
    for chi_ds in [2.5, 3.0]:
        for chi_dr in [0.0, -0.2, -0.5, -0.8, -1.0]:
            print(f"Running simulation for chi_dr = {chi_dr} and chi_ds = {chi_ds}")
            sim = NCompSimulator(
                chi_dr,
                chi_ds,
                box_size,
                os.path.join(root_dir, "data", "large_box"),
                verbose=True,
            )

            # Run the simulation
            sim.run()
