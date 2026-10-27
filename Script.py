#generating symmetric protein from scratch


from chroma import Chroma, Protein, conditioners





chroma_instance = Chroma() 

from chroma.layers.structure.diffusion import DiffusionChainCov
protein = Protein("/home/prometheus/Documents/protein_design/my_designs/SymmetricProtein161026/symmetric_good.pdb", device="cuda")
noiser = DiffusionChainCov()
X, C, S = protein.to_XCS()
# apply forward noise to t=0.5
X_noised = noiser.forward(X, C, 0.5).detach().to("cuda")
protein.sys.update_with_XCS(X_noised, C, S)
protein.to("symmetric_Good_noised.pdb")
print("Saved")



