# file: image_loader.py
import os
from PIL import Image
import torch
import torchvision.transforms as T

class ImageLoader:
    # loading an image from disk, resizing if needed
    
    def __init__(self, max_size=1024, device="cpu"):
        self.max_size = max_size
        self.device = device
        self.to_tensor = T.Compose([
            T.ToTensor(),
            #average Red Blue Green
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def load_image(self, path):
        #loads an image from `path` (RGB), resizes if bigger than max_size
        image = Image.open(path).convert("RGB")
        if max(image.size) > self.max_size:
            image.thumbnail((self.max_size, self.max_size), Image.LANCZOS)

        tensor = self.to_tensor(image).unsqueeze(0)
        return tensor.to(self.device)

    def tensor_to_pil(self, tensor):
        #invert to original picture
        invert = T.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std =[1/0.229,      1/0.224,      1/0.225]
        )

        image = tensor.clone().detach().cpu().squeeze(0)
        image = invert(image)
        image = torch.clamp(image, 0.0, 1.0)

        pil_image = T.ToPILImage()(image)
        return pil_image
