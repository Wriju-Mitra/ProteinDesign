from chroma import Chroma

chroma = Chroma()
protein = chroma.sample(chain_lengths=[60], steps=200)
protein.to("test.cif")
print("Saved test.cif")
