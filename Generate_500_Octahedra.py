"""
Generate 500 C3-symmetric proteins  and compute an ELBO score for each.


"""

import csv
import gc
import os
import traceback

import torch
from chroma import Chroma, Protein, conditioners

# settings -----------------------------
N_PROTEINS = 100
OUTPUT_DIR = "/home/wrijumitra/ProteinDesign/my_designs/Octahedral"
SCORES_FILE = os.path.join(OUTPUT_DIR, "elbo_scores.csv")   # ELBO file
NAME_PREFIX = "Octahedra_"                            
DEVICE = "cuda"

CHAIN_LENGTHS = [100]
STEPS = 500
LANGEVIN_FACTOR = 8
INVERSE_TEMPERATURE = 8
SYMMETRY_GROUP = "O"
NUM_CHAIN_NEIGHBORS = 5
# --------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_done_scores(path):
    """Return the set of protein names that already have a score saved."""
    done = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                done.add(row["name"])
    return done


def main():
    chroma_instance = Chroma(
        weights_backbone="/work/yanghj/chroma/chroma_weights/chroma_backbone_v1.0.pt",
        weights_design="/work/yanghj/chroma/chroma_weights/chroma_design_v1.0.pt",
    )

    # The conditioner is the same for every protein, so build it once.
    symmetry_conditioner = conditioners.SymmetryConditioner(
        G=SYMMETRY_GROUP, num_chain_neighbors=NUM_CHAIN_NEIGHBORS
    )

    done = load_done_scores(SCORES_FILE)
    ...
    write_header = not os.path.exists(SCORES_FILE)

    with open(SCORES_FILE, "a", newline="") as score_file:
        writer = csv.writer(score_file)
        if write_header:
            writer.writerow(["name", "elbo"])
            score_file.flush()

        for i in range(1, N_PROTEINS + 1):
            name = f"{NAME_PREFIX}{i}"
            pdb_path = os.path.join(OUTPUT_DIR, f"{name}.pdb")

            if name in done and os.path.exists(pdb_path):
                print(f"[{i}/{N_PROTEINS}] {name}: already done, skipping")
                continue

            try:
                # ---- 1) generate the symmetric protein ----
                protein = chroma_instance.sample(
                    chain_lengths=CHAIN_LENGTHS,
                    steps=STEPS,
                    conditioner=symmetry_conditioner,
                    langevin_factor=LANGEVIN_FACTOR,
                    inverse_temperature=INVERSE_TEMPERATURE,
                    sde_func="langevin",
                    potts_symmetry_order=symmetry_conditioner.potts_symmetry_order,
                )
                protein.to(pdb_path)

                # ---- 2) score it (reload from the PDB, as in your scoring script) ----
                scored_protein = Protein(pdb_path, device=DEVICE)
                elbo = float(chroma_instance.score(scored_protein)["elbo"].score)

                # ---- 3) store the score in the separate file ----
                writer.writerow([name, elbo])
                score_file.flush()  # keep results safe if the run is interrupted

                print(f"[{i}/{N_PROTEINS}] {name}: ELBO = {elbo:.4f}")

            except Exception:
                # Don't let one failed design kill the whole run
                print(f"[{i}/{N_PROTEINS}] {name}: FAILED")
                traceback.print_exc()

            finally:
                # Free GPU memory between designs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"Done. Structures in {OUTPUT_DIR}\nScores in {SCORES_FILE}")


if __name__ == "__main__":
    main()
