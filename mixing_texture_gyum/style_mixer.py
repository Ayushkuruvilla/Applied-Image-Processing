import torch
import torch.nn.functional as F

def gram_matrix(feature_map):
    #Computes the Gram matrix for a given feature_map of shape (B,C,H,W).

    B, C, H, W = feature_map.size()
    features = feature_map.view(B, C, H*W)
    G = torch.bmm(features, features.transpose(1,2))
    return G / (C*H*W)

class StyleMixer:
    #Stores multiple style images (like Style A, Style B),precomputes their Gram matrices

    def __init__(self, vgg_extractor, style_layers=None):
        self.vgg = vgg_extractor
        #conv5_1 is extra for combining style 
        self.style_layers = style_layers if style_layers else ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1']
        self.styles_data = []

    def add_style(self, style_img_tensor, weight=1.0):
        with torch.no_grad():
            feats = self.vgg(style_img_tensor)
        
        grams = {}
        for layer in self.style_layers:
            grams[layer] = gram_matrix(feats[layer])
        
        self.styles_data.append({'grams': grams, 'weight': weight})


    def compute_style_loss(self, target_features):
        #compute the sum of MSE loss to each style's gram,weight
        total_weight = 0.0
        for data in self.styles_data:
            total_weight += data['weight']

        style_loss = 0.0

        for style in self.styles_data:
            style_grams = style['grams']
            style_weight = style['weight'] / total_weight  

            layer_loss_sum = 0.0
            for layer in self.style_layers:
                target_gram = gram_matrix(target_features[layer])
                layer_loss = F.mse_loss(target_gram, style_grams[layer])
                layer_loss_sum += layer_loss
            
            style_loss += style_weight * layer_loss_sum

        return style_loss

