import chroma
from chroma import Chroma, Protein, conditioners
chroma_instance = Chroma(weights_backbone='/work/yanghj/chroma/chroma_weights/chroma_backbone_v1.0.pt',weights_design='/work/yanghj/chroma/chroma_weights/chroma_design_v1.0.pt')
model_path = "/work/yanghj/chroma/chroma_weights/chroma_proclass_v1.0.pt"
