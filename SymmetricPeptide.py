#generating symmetric protein from scratch


from chroma import Chroma, conditioners





chroma_instance = Chroma() 

Symmetry_conditioner = conditioners.SymmetryConditioner(G = "C_3",num_chain_neighbors=2)
my_protein, trajectories = chroma_instance.sample(
    chain_lengths=[80], 
    steps=500, 
    conditioner=Symmetry_conditioner,
    langevin_factor=8,
    inverse_temperature=8,  
    sde_func="langevin",
    potts_symmetry_order=Symmetry_conditioner.potts_symmetry_order,
    full_output=True)

my_protein.to("symmetric.pdb")
trajectories["trajectory"].to("my_protein_traj.cif")
print("Saved")
