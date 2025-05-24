import gradio as gr
import torch

from transformers import AutoModel, AutoProcessor

MODEL_NAME = "google/siglip2-giant-opt-patch16-256"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, device_map="auto").to(DEVICE)

def get_embeddings(img):
  input = processor(images=img, return_tensors="pt").to(DEVICE)
  with torch.no_grad():
    my_embedding = model.get_image_features(**input).detach().squeeze().tolist()
  return my_embedding

with gr.Blocks() as demo:
  gr.Interface(
    fn=get_embeddings,
    inputs="image",
    outputs="json",
    flagging_mode="never",
  )

if __name__ == "__main__":
   demo.launch()
