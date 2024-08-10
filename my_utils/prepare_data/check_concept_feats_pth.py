import torch

CLIP_CONCEPT_FEAT_PATH = './exp/asso_opt/GlomCrawl/GlomCrawl_4shot_fac/concepts_feat_ViT-L-14.pth'
CONCH_CONCEPT_FEAT_PATH = './exp/asso_opt/GlomCrawl/GlomCrawl_4shot_fac/concepts_feat_conch_ViT-B-16.pth'

clip_concepts = torch.load(CLIP_CONCEPT_FEAT_PATH)
conch_concepts = torch.load(CONCH_CONCEPT_FEAT_PATH)

print(clip_concepts.shape)
print(conch_concepts.shape)