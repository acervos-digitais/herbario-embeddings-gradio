import gradio as gr
import json
import numpy as np
import requests
import tarfile
import torch

from io import BytesIO
from os import path, remove
from PIL import Image as PImage
from urllib import request

from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from transformers import AutoModel, AutoProcessor

IMGS_URL = "https://acervos-digitais.github.io/herbario-media/imgs/arts/900"
EMBEDS_URL = "https://media.githubusercontent.com/media/acervos-digitais/herbario-data/main/json/20250705_crops_owlv2.json.gz"

MODEL_NAME = "google/siglip2-giant-opt-patch16-256"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def download_extract_file(url, local_path="."):
  gz_file_name = url.split("/")[-1]
  gz_file_path = path.join(local_path, gz_file_name)
  file_path = gz_file_path.replace(".gz", "")

  with request.urlopen(request.Request(url), timeout=30.0) as response:
    if response.status == 200:
      with open(gz_file_path, "wb") as f:
        f.write(response.read())

  tar = tarfile.open(gz_file_path, "r:gz")
  tar.extractall(local_path, filter="data")
  tar.close()
  remove(gz_file_path)

  return file_path

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)#.to(DEVICE)

embeddings_path = download_extract_file(EMBEDS_URL)
with open(embeddings_path, "r") as ifp:
  embeddings_data = json.load(ifp)

crop_names = np.array(list(embeddings_data.keys()))
crop_embeddings = np.array(list(embeddings_data.values()))

def get_embedding(img):
  inputs = processor(images=img, return_tensors="pt")#.to(DEVICE)
  with torch.no_grad():
    img_embedding = model.get_image_features(**inputs).pooler_output.detach().squeeze().numpy()
  return (img_embedding / np.linalg.norm(img_embedding)).tolist()

def get_painting_order(img):
  target_embedding = get_embedding(img)
  dists = cosine_distances(crop_embeddings, [target_embedding]).reshape(-1)
  idxs_by_dist = dists.argsort()
  all_crops_by_dist = crop_names[idxs_by_dist]

  seen = set()
  crops_by_dist = []
  for crop_name in all_crops_by_dist:
    qid = crop_name.split("_")[0]
    if (qid not in seen):
      crops_by_dist.append(crop_name)
      seen.add(qid)

  return crops_by_dist

def display_top_painting(img):
  painting_order = get_painting_order(img)
  top_id_obj = painting_order[0]
  top_id = top_id_obj.split("_")[0]
  response = requests.get(f"{IMGS_URL}/{top_id}.jpg")
  return PImage.open(BytesIO(response.content))


with gr.Blocks() as demo:
  gr.Markdown("# Image to Closest Painting")
  gr.Interface(
    fn=display_top_painting,
    api_name="closest-painting",
    inputs="image",
    outputs="image",
    flagging_mode="never",
  )

  gr.Markdown("# Image to Painting Similarity Ranking")
  gr.Interface(
    fn=get_painting_order,
    api_name="similarity-ranking",
    inputs="image",
    outputs="json",
    flagging_mode="never",
  )

  gr.Markdown("# Image to SigLip2 Embeddings")
  gr.Interface(
    fn=get_embedding,
    api_name="embedding",
    inputs="image",
    outputs="json",
    flagging_mode="never",
  )

if __name__ == "__main__":
  demo.launch()
