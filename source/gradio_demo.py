import os
import torch
import numpy as np
import gradio as gr
import torchvision

from model import UnetTMO
from PIL import Image

CKPT = "../pretrained/afifi.pth"

def read_pytorch_lightning_state_dict(ckpt):
    new_state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("model."):
            new_state_dict[k[len("model.") :]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

model = UnetTMO()
state_dict = read_pytorch_lightning_state_dict(torch.load(CKPT))
model.load_state_dict(state_dict)
model.eval()
model.cuda()

def read_image(image):
    img = np.array(image) / 255.0
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
    
    return img

def convert_torch_to_numpy(x):
    grid = torchvision.utils.make_grid(x)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
  
    return im

def image_mod(image):
    image = read_image(image).cuda()
    with torch.no_grad():
        output, _ = model(image)

    img  = convert_torch_to_numpy(output)

    return img

demo = gr.Interface(
    image_mod,
    gr.Image(type="pil"),
    "image",
    examples=[
        os.path.join(os.path.dirname(__file__), "samples/im1.png"),
        os.path.join(os.path.dirname(__file__), "samples/im2.png"),
    ],
)

if __name__ == "__main__":
    demo.launch(share=True)