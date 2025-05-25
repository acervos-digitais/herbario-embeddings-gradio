import gradio as gr
import json
import numpy as np
import torch

from sklearn.metrics import euclidean_distances

from transformers import AutoModel, AutoProcessor

MODEL_NAME = "google/siglip2-giant-opt-patch16-256"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, device_map="auto").to(DEVICE)

with open("./art-crops_siglip2.json", "r") as ifp:
  embeddings_data = json.load(ifp)

crop_names = np.array(list(embeddings_data.keys()))
crop_embeddings = np.array(list(embeddings_data.values()))

def get_embedding(img):
  input = processor(images=img, return_tensors="pt").to(DEVICE)
  with torch.no_grad():
    my_embedding = model.get_image_features(**input).detach().squeeze().tolist()
  return my_embedding

def find_in_art(img):
  target_embedding = get_embedding(img)
  dists = euclidean_distances(crop_embeddings, target_embedding.reshape(1,-1)).reshape(-1)
  idxs_by_dist = dists.argsort()
  all_crops_by_dist = crop_names[idxs_by_dist]

  # TODO: filter repeated ids
  seen = set()
  crops_by_dist = []

  for crop_name in all_crops_by_dist:
    qid = crop_name.split("_")[0]
    if (qid not in seen):
      crops_by_dist.append(crop_name)
      seen.add(qid)

  return crops_by_dist


with gr.Blocks() as demo:
  gr.Interface(
    fn=get_embedding,
    inputs="image",
    outputs="json",
    flagging_mode="never",
  )

  gr.Interface(
    fn=find_in_art,
    inputs="image",
    outputs="json",
    flagging_mode="never",
  )

if __name__ == "__main__":
   demo.launch()
