import gc
import os
import random
import traceback

import numpy as np
import torch

from chroma import Chroma
from chroma.layers.structure import conditioners

# One-time only: if Chroma asks for an API key to download its weights, run
#   from chroma.utility.api import register_key; register_key("YOUR_KEY")
# once and then leave it out.


# settings --------------------------------------------------------------
OUTPUT_DIR = "/home/prometheus/Documents/protein_design/my_designs/ShapeToProtein"
NAME_PREFIX = "Shape_3_"
N_PROTEINS = 1
INVERSE_TEMPERATURE = 8

LOW, HIGH = 300, 301          # min / max residues per chain

STEPS = 500
LANGEVIN_FACTOR = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Resolve the CSV next to this script so it works from any working directory.
CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "myShape.csv")


def csv_to_point_cloud(csv_file):
    """Load an N x 3 (x, y, z) CSV with no header row as a float32 tensor."""
    # np.loadtxt (unlike pandas.read_csv's default) does not treat the first
    # row as a header, so no points are lost.
    points = np.loadtxt(csv_file, delimiter=",", dtype=np.float32, ndmin=2)
    if points.shape[1] != 3:
        raise ValueError(f"{csv_file}: expected 3 columns (x,y,z), got {points.shape[1]}")
    return torch.from_numpy(points)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    point_cloud = csv_to_point_cloud(CSV_FILE)
    print(f"Loaded {len(point_cloud)} points from {CSV_FILE}")

    chroma_instance = Chroma()      # load the model once

    for i in range(1, N_PROTEINS + 1):
        name = f"{NAME_PREFIX}{i}"
        path = os.path.join(OUTPUT_DIR, f"{name}.pdb")

        if os.path.exists(path):
            print(f"[{i}/{N_PROTEINS}] {name}: already exists, skipping")
            continue

        num_residues = random.randint(LOW, HIGH)
        chain_lengths = [num_residues]

        try:
            # Built per protein because autoscaling depends on the residue count.
            shape_conditioner = conditioners.ShapeConditioner(
                point_cloud,
                chroma_instance.backbone_network.noise_schedule,
                autoscale_num_residues=num_residues,
            ).to(DEVICE)

            protein = chroma_instance.sample(
                chain_lengths=chain_lengths,
                steps=STEPS,
                conditioner=shape_conditioner,
                langevin_factor=LANGEVIN_FACTOR,
                inverse_temperature=INVERSE_TEMPERATURE,
            )
            protein.to(path)
            print(f"[{i}/{N_PROTEINS}] {name}: saved (chain length {num_residues})")
        except Exception:
            print(f"[{i}/{N_PROTEINS}] {name}: FAILED")
            traceback.print_exc()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("Cleared cache!")

    print(f"Done. Structures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
