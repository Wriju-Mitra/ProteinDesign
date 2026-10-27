#generating octahedra
import csv
import gc
import os
import traceback

import torch
import random 




from chroma import Chroma, Protein, conditioners
from chroma.models import graph_classifier, procap, Chroma

OUTPUT_DIR = "/home/prometheus/Documents/protein_design/my_designs/OctamersfromInputProtein"
os.makedirs(OUTPUT_DIR, exist_ok=True)


Input_Protein_Path = "/home/prometheus/Documents/protein_design/my_designs/Trimer.pdb"

#Low = 60
#High = 120             #min no of residue and max number of residues





N_PROTEINS = 10
Protein_Name_Prefix = "Octamerized_"
Protein_Name=""
#CHAIN_LENGTHS = [100]

STEPS = 500
LANGEVIN_FACTOR = 8
INVERSE_TEMPERATURE = 8
SYMMETRY_GROUP = "O"
NUM_CHAIN_NEIGHBORS = 6

noiser = 0.01   # can go from 1 to 0.001. if set to 0.5 it should noise the input structure halfway and denoise it while respecting symmetry


chroma_instance = Chroma() 

#Symmetry_conditioner = conditioners.SymmetryConditioner(G = "O",num_chain_neighbors=6)

symmetry_conditioner = conditioners.SymmetryConditioner(
    G=SYMMETRY_GROUP, 
    num_chain_neighbors=NUM_CHAIN_NEIGHBORS
)

#Input_Protein = Protein(Input_Protein_Path, device="cuda")


   

for i in range(1, N_PROTEINS + 1):
    path = os.path.join(OUTPUT_DIR, f"{Protein_Name_Prefix}{i}.pdb") 
    init = p = None
    try:
        init = Protein(Input_Protein_Path, device="cuda")   # fresh copy each time
        p = chroma_instance.sample(
            protein_init=init,
            steps=STEPS,
            conditioner=symmetry_conditioner,
            langevin_factor=LANGEVIN_FACTOR,
            tspan=[noiser, 0.001],
            inverse_temperature=INVERSE_TEMPERATURE,
            sde_func="langevin",
            potts_symmetry_order=symmetry_conditioner.potts_symmetry_order,
        )
        p.to(path)
        print(f"[{i}/{N_PROTEINS}] saved {path}", flush=True)
    except Exception:
        print(f"[{i}/{N_PROTEINS}] FAILED", flush=True)
        traceback.print_exc()
    finally:
        del init, p
        gc.collect()
        torch.cuda.empty_cache()
    #finally:
                # Free GPU memory between designs
      #          gc.collect()
     #           if torch.cuda.is_available():
       #             torch.cuda.empty_cache()
#trajectories["trajectory"].to("my_protein_traj.cif")


print("Saved")
