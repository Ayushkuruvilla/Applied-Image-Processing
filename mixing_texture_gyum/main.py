import torch
import matplotlib.pyplot as plt

from image_loader import ImageLoader
from vgg_model import VGGExtractor
from style_mixer import StyleMixer
from neural_style_transfer import NeuralStyleTransfer

def main():
    content_path = r"C:\Images\house.png"     
    style1_path  = r"C:\Images\scream.png"       
    style2_path  = r"C:\Images\van_gogh.png"       
    output_path  = r"C:\Images\mixing_output.png" 
    style1_weight = 0.7    
    style2_weight = 0.3    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = ImageLoader(max_size=1024, device=device)
    content_tensor = loader.load_image(content_path) 
    style1  = loader.load_image(style1_path)
    style2  = loader.load_image(style2_path)

    vgg_model = VGGExtractor(device=device)

    mixer = StyleMixer(vgg_extractor=vgg_model, style_layers=['conv1_1','conv2_1','conv3_1','conv4_1'])
    mixer.add_style(style1, weight=style1_weight)
    mixer.add_style(style2, weight=style2_weight)

    transfer_model = NeuralStyleTransfer(vgg_extractor=vgg_model,content_layer='conv4_2',content_weight=1.0,style_weight=1e6,num_steps=2000, lr=0.002,device=device)

    final_tensor = transfer_model.run_transfer(content_tensor, mixer)

    final_pil = loader.tensor_to_pil(final_tensor)
    final_pil.save(output_path)

    plt.imshow(final_pil)
    plt.title("Style Mixed Output")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
