import torch
import chroma
from chroma import Chroma, Protein, api

from chroma.layers.structure.conditioners import SubsequenceConditioner

chroma_instance = Chroma()


mask = torch.zeros(1, 80, device='cuda')
for i in '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24  51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80'.split(): mask[0, int(i)-1] = 1.
#redesign residues from 0-24 and 51-80 

noiser = chroma.layers.structure.diffusion.DiffusionChainCov()

my_protein = Protein.from_PDB("/home/prometheus/Documents/protein_design/my_designs/monomer.pdb", device="cuda")

X, C, S = my_protein.to_XCS()

X_noised = noiser.forward(X, C, 0.5).detach()
noised_protein = my_protein.from_XCS(X_noised, C, S)

conditioner = SubsequenceConditioner(chroma_instance.design_network,
protein)

noised_monomer = chroma_instance.sample(protein_init=noised_protein,
conditioner=conditioner,
tspan=[0.5, 0.001],
steps=250,
design_selection=mask)

noised_monomer.to('noised_monomer.pdb')
