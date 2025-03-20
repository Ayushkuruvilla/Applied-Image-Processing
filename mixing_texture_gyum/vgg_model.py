import torch
import torch.nn as nn
import torchvision.models as models

class VGGExtractor(nn.Module):

    def __init__(self, device="cpu"):
        super().__init__()
        # Load the pretrained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        self.layers_map = {
            'conv1_1':  0,
            'conv2_1':  5,
            'conv3_1': 10,
            'conv4_1': 19,
            'conv4_2': 21,  
            'conv5_1': 28,  
        }

        max_index = max(self.layers_map.values())
        self.vgg_sliced = vgg[:max_index+1].to(device).eval()

        for param in self.parameters():
            param.requires_grad = False

        self.device = device

    def forward(self, x):
        features = {}
        out = x
        for index, layer in enumerate(self.vgg_sliced):
            out = layer(out)
            for name, layer_index in self.layers_map.items():
                if layer_index == index:
                    features[name] = out
        return features
