import torch
from PIL import Image
import open_clip

# model_name = 'ViT-L-14'
# model_weights = 'laion2b_s32b_b82k'
model_name = 'EVA02-L-14'
model_weights = 'merged2b_s4b_b131k'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=model_weights, device=device)
model.eval()
tokenizer = open_clip.get_tokenizer(model_name)

image = preprocess(Image.open("/data/hqvo3/Datasets/vlm/ml-scripts/datasets/glom_refine_labo/mesangial_hypercellularity_ded5bc5d-d0f7-442b-9761-0770465831d4-1.png")).unsqueeze(0)
text = tokenizer(["endocapillary hypercellularity", "mesangial hypercellularity", "fibrous crescent", "mesangial matrix expansion", \
                  "podocyte", "visceral epithelial cell capping"])
image = image.to(device)
text = text.to(device)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image) # (1, 768)
    text_features = model.encode_text(text) # (6, 768)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs: ", text_probs) # prints: [[1., 0., 0.]]