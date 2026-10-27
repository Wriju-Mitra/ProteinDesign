
from chroma import Chroma, Protein, conditioners





chroma_instance = Chroma() 




my_protein = Protein("/home/prometheus/Documents/protein_design/my_designs/SymmetricProtein161026/symmetric_good.pdb", device="cuda")
noised_version_of_my_protein = Protein("/home/prometheus/Documents/protein_design/my_designs/symmetric_Good_noised.pdb", device="cuda")
elbo_no_noise = chroma_instance.score(my_protein)["elbo"].score
elbo_noise = chroma_instance.score(noised_version_of_my_protein)["elbo"].score
print(elbo_no_noise, elbo_noise)

