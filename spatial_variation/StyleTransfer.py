import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image

class StyleTransfer: 
    def __init__(self, max_dim=1024, style_weight=1e5, content_weight=1.0, num_steps=800, lr=0.003):
        self.max_dim = max_dim
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.num_steps = num_steps
        self.lr = lr

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BetterStyleTransfer] Using device: {self.device}")

        self.vgg = torchvision.models.vgg19(weights="VGG19_Weights.IMAGENET1K_V1").features.eval().to(self.device)

        self.feature_layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2', 
            '28': 'conv5_1'
        }

    def extract_features(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.feature_layers:
                features[self.feature_layers[name]] = x
        return features
    
    def gram_matrix(self, tensor):
        b, c, h, w = tensor.size()
        features = tensor.view(b, c, h*w)
        G = torch.bmm(features, features.transpose(1, 2))
        return G / (c * h * w)
        
    def preprocess(self, bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        w, h = pil_image.size
        if max(w, h) > self.max_dim:
            ratio = self.max_dim / float(max(w,h))
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
        
        final_w, final_h = pil_image.size

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        tensor = transform(pil_image).unsqueeze(0).to(self.device)
        return tensor, (final_w, final_h)
    
    def postprocess(self, tensor, out_size):
        tensor = tensor.detach().cpu().squeeze(0)

        invert_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],std=[1/0.229, 1/0.224, 1/0.225])
        unnorm = invert_normalize(tensor)
        unnorm = torch.clamp(unnorm, 0.0, 1.0)

        np_img = unnorm.numpy()
        np_img = (np_img * 255).astype(np.uint8)
        np_img = np.transpose(np_img, (1,2,0))

        pil_out = Image.fromarray(np_img, mode='RGB')
        if pil_out.size != out_size:
            pil_out = pil_out.resize(out_size, Image.LANCZOS)

        bgr = cv2.cvtColor(np.array(pil_out), cv2.COLOR_RGB2BGR)
        return bgr

    def compute_content_loss(self, target_features, content_features):
        return F.mse_loss(target_features, content_features)

    def compute_style_loss(self, target_gram, style_gram):
        return F.mse_loss(target_gram, style_gram)
    
    def run_style_transfer(self, content_bgr, style_bgr):
        content_tensor, content_size = self.preprocess(content_bgr)
        style_tensor, style_size= self.preprocess(style_bgr)

        with torch.no_grad():
            content_features = self.extract_features(content_tensor)
            style_features = self.extract_features(style_tensor)
        
        style_grams = {}
        for layer_name, feats in style_features.items():
            style_grams[layer_name] = self.gram_matrix(feats).detach()

        target = content_tensor.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([target], lr=self.lr)

        for step in range(self.num_steps):
            optimizer.zero_grad()

            target_features = self.extract_features(target)
            content_loss = self.compute_content_loss(target_features['conv4_2'], content_features['conv4_2'])

            style_loss = 0.0
            for loss in style_grams:
                target_gram = self.gram_matrix(target_features[loss])
                style_loss += self.compute_style_loss(target_gram, style_grams[loss])
            style_loss *= self.style_weight

            total_loss = self.content_weight * content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            if step % 100 == 0 or step == self.num_steps - 1:
                print(f"[Step {step:4d}/{self.num_steps}] "
                      f"Content: {content_loss.item():.2f}, "
                      f"Style: {style_loss.item():.2f}, "
                      f"Total: {total_loss.item():.2f}")
                
        out_bgr = self.postprocess(target, content_size)
        return out_bgr