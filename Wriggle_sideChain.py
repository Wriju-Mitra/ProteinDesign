import torch
import chroma
from chroma import Chroma, Protein, api
chroma_instance = Chroma()


mask = torch.zeros(1, 80, device='cuda')
for i in ''.split(): mask[0, int(i)-1] = 1.
#redesign residues from 0-24 and 51-80 

my_protein = Protein.from_PDB("/home/prometheus/Documents/protein_design/my_designs/monomer.pdb", device="cuda")

partially_Diffused_monomer = chroma_instance.design(my_protein, design_selection=mask)
partially_Diffused_monomer.to('sideChainWriggle_monomer.pdb')
