import torch
import torch.nn.functional as F

class NeuralStyleTransfer:
    #minimizes total_loss = content_loss + style_loss

    def __init__(self, vgg_extractor, content_layer='conv4_2', content_weight=1.0, style_weight=1.0, num_steps=500, lr=0.02, device='cpu'):
        self.vgg = vgg_extractor
        self.content_layer = content_layer
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.num_steps = num_steps
        self.lr = lr
        self.device = device

    def run_transfer(self, content_tensor, style_mixer):
        #extract content feature and initialize target from content. 

        with torch.no_grad():
            content_features = self.vgg(content_tensor)

        #extract the features corresponding to the content later 4_2
        content_target = content_features[self.content_layer].detach()

        #tried with random noise initialize, but it wasnt merge with that
        target = content_tensor.clone().requires_grad_(True)

        optimizer = torch.optim.Adam([target], lr=self.lr)

        for step in range(self.num_steps):
            optimizer.zero_grad()
            content_features = self.vgg(target)

            current_content = content_features[self.content_layer]
            content_loss = F.mse_loss(current_content, content_target)

            style_loss = style_mixer.compute_style_loss(content_features)
            total_loss = self.content_weight * content_loss + self.style_weight * style_loss

            total_loss.backward()
            optimizer.step()

            if step % 100 == 0 or step == self.num_steps - 1:
                print(f"Step {step:4d}/{self.num_steps}, "
                      f"Content={content_loss.item():.4f}, "
                      f"Style={style_loss.item():.4f}, "
                      f"Total={total_loss.item():.4f}")

        return target.detach()
