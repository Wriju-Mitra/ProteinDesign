import torch
import chroma
from chroma import Chroma, Protein, api
chroma_instance = Chroma()


mask = torch.zeros(1, 80, device='cuda')
for i in '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24  51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80'.split(): mask[0, int(i)-1] = 1.
#redesign residues from 0-24 and 51-80 

my_protein = Protein.from_PDB("/home/prometheus/Documents/protein_design/my_designs/monomer.pdb", device="cuda")

partially_Diffused_monomer = chroma_instance.design(my_protein, design_selection=mask)
partially_Diffused_monomer.to('partially_Diffused_monomer.pdb')
