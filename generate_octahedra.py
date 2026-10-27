#generating octahedra
import csv
import gc
import os
import traceback

import torch
import random 




from chroma import Chroma, conditioners


OUTPUT_DIR = "/home/prometheus/Documents/protein_design/my_designs/Octamers"
os.makedirs(OUTPUT_DIR, exist_ok=True)

Low = 60
High = 120             #min no of residue and max number of residues





N_PROTEINS = 10
Protein_Name_Prefix = "Octamer_nn6_"
Protein_Name=""
#CHAIN_LENGTHS = [100]

STEPS = 500
LANGEVIN_FACTOR = 8
INVERSE_TEMPERATURE = 8
SYMMETRY_GROUP = "O"
NUM_CHAIN_NEIGHBORS = 6


chroma_instance = Chroma() 

#Symmetry_conditioner = conditioners.SymmetryConditioner(G = "O",num_chain_neighbors=6)

symmetry_conditioner = conditioners.SymmetryConditioner(
    G=SYMMETRY_GROUP, 
    num_chain_neighbors=NUM_CHAIN_NEIGHBORS
)



   

for i in range(1, N_PROTEINS + 1):
    Protein_Name = f"{Protein_Name_Prefix}{i}"
    path = os.path.join(OUTPUT_DIR, f"{Protein_Name}.pdb")
    CHAIN_LENGTHS = [random.randint(Low, High)]
    my_protein = chroma_instance.sample(
        chain_lengths=CHAIN_LENGTHS,
        steps=STEPS,
        conditioner=symmetry_conditioner,
        langevin_factor=LANGEVIN_FACTOR,
        inverse_temperature=INVERSE_TEMPERATURE,
        sde_func="langevin",
        potts_symmetry_order=symmetry_conditioner.potts_symmetry_order
    )

    my_protein.to(path)
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    #finally:
                # Free GPU memory between designs
      #          gc.collect()
     #           if torch.cuda.is_available():
       #             torch.cuda.empty_cache()
#trajectories["trajectory"].to("my_protein_traj.cif")


print("Saved")
