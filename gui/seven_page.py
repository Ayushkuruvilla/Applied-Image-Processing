import os
import random
import threading
import logging
import colorsys

import numpy as np
from PIL import Image
import pygame
from pygame.locals import *

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import (
    resnet50, ResNet50_Weights,
    densenet121,
    swin_t, Swin_T_Weights,
    vgg16, VGG16_Weights
)

from skimage import color
from sklearn.cluster import KMeans

from tkinter import filedialog, Tk, simpledialog

# Only warnings and errors will be displayed
logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

CONTENT_DIR = "input/style_mixing/content"
STYLE_DIR = "input/style_mixing/style"

MODEL_CACHE = {}

ARTIST_STYLES = {
    "1": "vangogh",
    "2": "pietmondriaan",
    "3": "picasso",
    "4": "claudemonet"
}

PREDEFINED_PALETTES = {
    "Green":   ["#00ff00", "#009900", "#66ff66", "#33cc33", "#00cc00"],
    "Mario":   ["#fed1b0", "#ee1c25", "#0065b3", "#ffffff", "#894c2f"],
    "Black":   ["#000000", "#333333", "#666666", "#999999", "#cccccc"],
    "White":   ["#ffffff", "#f0f0f0", "#e0e0e0", "#d0d0d0", "#c0c0c0"],
    "Vintage": ["#131842", "#E68369", "#ECCEAE", "#FBF6E2", "#8E9B73"],
    "Blue":    ["#000000", "#2f4550", "#586f7c", "#b8dbd9", "#f4f4f9"]
}

PALETTE_MENU = ["Green", "Mario", "Black", "White", "Vintage", "Blue"]
CHOSEN_PALETTE = "Vintage"
PALETTE_INTENSITY = 0.25

WEIGHT_CONFIGURATIONS = {
    "Swin": {"swin": 1000.0},
    "VGG":  {"layer2": 1000.0, "layer3": 1500.0},
    "default": {"layer2": 1000.0, "layer3": 1500.0}
}

MODEL_LIST = ["ResNet", "DenseNet", "Swin", "VGG"]


def safe_load_image(filepath: str):
    """
    Safely loads an image from disk and converts it to RGB.
    Returns None on failure.
    """
    try:
        return Image.open(filepath).convert("RGB")
    except Exception as e:
        logging.error(f"Error loading image {filepath}: {e}")
        return None


def hex_to_rgb_palette(hex_list):
    """
    Converts a list of hex colors (e.g. ['#ffffff', '#000000'])
    into a list of normalized RGB values in [0, 1].
    """
    rgb_palette = []
    for hex_color in hex_list:
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        rgb_palette.append([r, g, b])
    return rgb_palette


PALETTE_RGB = {name: hex_to_rgb_palette(colors) for name, colors in PREDEFINED_PALETTES.items()}


def rgb_to_hsl(rgb):
    """Convert an RGB tuple to HSL using colorsys."""
    return colorsys.rgb_to_hls(*rgb)


def hsl_to_rgb(hsl):
    """Convert an HSL tuple to RGB using colorsys."""
    return colorsys.hls_to_rgb(*hsl)


def adjust_palette_hsl(palette, saturation=1.0, hue=0.0):
    """
    Adjust the HSL values for every color in a palette.
    - saturation scales the S channel.
    - hue shifts the H channel.
    """
    adjusted = []
    for color_val in palette:
        h, l, s = rgb_to_hsl(color_val)
        s *= saturation
        h = (h + hue) % 1.0
        adjusted.append(list(hsl_to_rgb((h, l, s))))
    return adjusted


class ResNetFeatureExtractor(nn.Module):
    """
    Extracts intermediate features from a ResNet-50 model.
    """
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(DEVICE)
        resnet.eval()
        children = list(resnet.children())
        self.layer1 = nn.Sequential(*children[:5])
        self.layer2 = children[5]
        self.layer3 = children[6]
        self.layer4 = children[7]

    def forward(self, x):
        features = {}
        x = self.layer1(x)
        features['layer1'] = x
        x = self.layer2(x)
        features['layer2'] = x
        x = self.layer3(x)
        features['layer3'] = x
        x = self.layer4(x)
        features['layer4'] = x
        return features


class DenseNetFeatureExtractor(nn.Module):
    """
    Extracts intermediate features from a DenseNet-121 model.
    """
    def __init__(self):
        super().__init__()
        dnet = densenet121(weights="IMAGENET1K_V1").to(DEVICE)
        dnet.eval()
        self.features = dnet.features

    def forward(self, x):
        features = {}
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)

        x1 = self.features.denseblock1(x)
        features['layer1'] = x1
        x = self.features.transition1(x1)

        x2 = self.features.denseblock2(x)
        features['layer2'] = x2
        x = self.features.transition2(x2)

        x3 = self.features.denseblock3(x)
        features['layer3'] = x3
        x = self.features.transition3(x3)

        x4 = self.features.denseblock4(x)
        x4 = self.features.norm5(x4)
        features['layer4'] = x4
        return features


class SwinFeatureExtractor(nn.Module):
    """
    Extracts features from a Swin Transformer (tiny) model.
    """
    def __init__(self):
        super().__init__()
        swin_model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1).to(DEVICE)
        swin_model.eval()
        self.swin = swin_model

    def forward(self, x):
        features = {}
        if hasattr(self.swin, 'forward_features'):
            feats = self.swin.forward_features(x)
        else:
            feats = self.swin(x)
        features['swin'] = feats
        return features


class VGGFeatureExtractor(nn.Module):
    """
    Extracts intermediate features from a VGG16 model.
    """
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(DEVICE)
        vgg.eval()
        features = list(vgg.features.children())
        self.slice1 = nn.Sequential(*features[:4])   # Up to relu1_2
        self.slice2 = nn.Sequential(*features[4:9])  # Up to relu2_2
        self.slice3 = nn.Sequential(*features[9:16]) # Up to relu3_3
        self.slice4 = nn.Sequential(*features[16:23])# Up to relu4_3

    def forward(self, x):
        features = {}
        x = self.slice1(x)
        features['layer1'] = x
        x = self.slice2(x)
        features['layer2'] = x
        x = self.slice3(x)
        features['layer3'] = x
        x = self.slice4(x)
        features['layer4'] = x
        return features


def get_feature_extractor(model_name: str):
    """
    Retrieves a TorchScripted feature extractor for the given model name.
    Caches the model for future use.
    """
    if model_name not in MODEL_CACHE:
        if model_name == "ResNet":
            extractor = ResNetFeatureExtractor().to(DEVICE).eval()
        elif model_name == "DenseNet":
            extractor = DenseNetFeatureExtractor().to(DEVICE).eval()
        elif model_name == "Swin":
            extractor = SwinFeatureExtractor().to(DEVICE).eval()
        elif model_name == "VGG":
            extractor = VGGFeatureExtractor().to(DEVICE).eval()
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Removed the JIT compilation info log
        try:
            extractor = torch.jit.script(extractor)
        except Exception as e:
            logging.error(f"JIT compilation failed for {model_name}: {e}")

        MODEL_CACHE[model_name] = extractor
    return MODEL_CACHE[model_name]


def compute_gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    Computes the Gram matrix for the given feature tensor.
    """
    _, c, *dims = features.size()
    flattened = features.view(c, -1)
    gram = torch.matmul(flattened, flattened.t())
    return gram / flattened.numel()


def compute_style_loss(output_features: dict, style_features: dict,
                       style_layers: list, weight_config: dict) -> torch.Tensor:
    """
    Computes style loss by comparing the Gram matrices of the output and style features.
    """
    style_loss = 0
    for layer in style_layers:
        gram_out = compute_gram_matrix(output_features[layer])
        gram_style = compute_gram_matrix(style_features[layer])
        layer_loss = weight_config[layer] * torch.mean((gram_out - gram_style) ** 2)
        style_loss += layer_loss
    return style_loss


def compute_content_loss(output_features: dict, content_features: dict, content_layer: str) -> torch.Tensor:
    """
    Computes content loss as the mean squared error between the output and content features.
    """
    return torch.mean((output_features[content_layer] - content_features[content_layer]) ** 2)


def style_transfer(content_image: torch.Tensor, style_image: torch.Tensor,
                   model: str = "ResNet", iterations: int = 300,
                   content_weight: float = 0.5, style_weight: float = 1000.0,
                   progress_callback=None) -> torch.Tensor:
    """
    Performs neural style transfer using the specified model.
    Combines content and style losses over multiple iterations.
    """
    print("Starting style transfer...")
    feature_extractor = get_feature_extractor(model)

    if model == "Swin":
        content_layer = "swin"
        style_layers = ["swin"]
        weight_config = WEIGHT_CONFIGURATIONS["Swin"]
    elif model == "VGG":
        content_layer = "layer4"
        style_layers = ["layer2", "layer3"]
        weight_config = WEIGHT_CONFIGURATIONS["VGG"]
    else:
        content_layer = "layer4"
        style_layers = ["layer2", "layer3"]
        weight_config = WEIGHT_CONFIGURATIONS["default"]

    with torch.no_grad():
        content_feats = feature_extractor(content_image)
        style_feats = feature_extractor(style_image)
        content_features = {k: v.detach() for k, v in content_feats.items()}
        style_features = {k: v.detach() for k, v in style_feats.items()}

    alpha_noise = 0.6
    output = alpha_noise * content_image + (1 - alpha_noise) * torch.randn_like(content_image)
    output = output.clone().detach().requires_grad_(True).to(DEVICE)
    optimizer = optim.Adam([output], lr=0.003)

    use_amp = (DEVICE.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for i in range(iterations):
        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                output_features = feature_extractor(output)
                c_loss = compute_content_loss(output_features, content_features, content_layer)
                s_loss = compute_style_loss(output_features, style_features, style_layers, weight_config)
                total_loss = content_weight * c_loss + style_weight * s_loss
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output_features = feature_extractor(output)
            c_loss = compute_content_loss(output_features, content_features, content_layer)
            s_loss = compute_style_loss(output_features, style_features, style_layers, weight_config)
            total_loss = content_weight * c_loss + style_weight * s_loss
            total_loss.backward()
            optimizer.step()

        if progress_callback and (i % 10 == 0 or i == iterations - 1):
            progress_callback(i, iterations)
            logging.debug(f"Step {i}/{iterations}: Content Loss={c_loss.item():.4f}, Style Loss={s_loss.item():.4f}")

    print("Style transfer complete.")
    return output.detach()


def preprocess_image_adaptive(pil_image: Image.Image, result_size: int = 512, model: str = "ResNet") -> torch.Tensor:
    """
    Resizes and normalizes a PIL image for the specified model.
    """
    transform_list = [
        transforms.Resize((result_size, result_size)),
        transforms.ToTensor()
    ]
    if model == "Swin":
        transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                   std=[0.5, 0.5, 0.5]))
    else:
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225]))
    transform = transforms.Compose(transform_list)
    return transform(pil_image).unsqueeze(0).to(DEVICE)


def unnormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverses ImageNet normalization for conversion back to a PIL image.
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    t = tensor.clone().detach().cpu().squeeze(0)
    t = inv_normalize(t)
    t = torch.clamp(t, 0, 1)
    return t.unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """
    Converts a normalized Torch tensor back into a PIL Image.
    """
    tensor = tensor.squeeze(0).cpu().clone()
    return transforms.ToPILImage()(tensor)


def extract_palette(image_tensor, num_colors=5, random_state=0):
    """
    Extracts a palette of `num_colors` from an image using KMeans in LAB color space.
    """
    arr = image_tensor.clone().detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0, 1) * 255
    arr_lab = color.rgb2lab(arr)
    pixels = arr_lab.reshape(-1, 3)
    if random_state is None:
        random_state = np.random.randint(0, 10000)
    kmeans = KMeans(n_clusters=num_colors, random_state=random_state)
    kmeans.fit(pixels)
    palette_lab = kmeans.cluster_centers_
    palette_rgb = color.lab2rgb(palette_lab.reshape(1, num_colors, 3)).reshape(num_colors, 3)
    return palette_rgb


def map_colors(image_tensor, palette):
    """
    Maps each pixel of an image to the nearest color in the provided palette (using LAB color space).
    """
    arr = image_tensor.clone().detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0, 1)
    arr_lab = color.rgb2lab(arr)
    palette_arr = np.array(palette)
    palette_lab = color.rgb2lab(palette_arr)
    pixels = arr_lab.reshape(-1, 3)
    mapped = np.zeros_like(pixels)
    for i, p in enumerate(pixels):
        distances = np.linalg.norm(palette_lab - p, axis=1)
        mapped[i] = palette_lab[np.argmin(distances)]
    mapped_rgb = color.lab2rgb(mapped.reshape(arr.shape))
    new_tensor = torch.tensor(mapped_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return new_tensor.to(DEVICE)


def blend_images_with_intensity(base, over, intensity=1.0):
    """
    Blends two images (tensors) with gamma correction.
    The `intensity` parameter controls the blend ratio.
    """
    blended = (base ** 2.2 * intensity + over ** 2.2 * (1.0 - intensity)) ** (1 / 2.2)
    return blended


def apply_intensity_to_result(result_tensor, style_pil: Image.Image,
                              palette_size: int, intensity=0.3,
                              palette_name: str = CHOSEN_PALETTE) -> torch.Tensor:
    """
    Applies a palette-based recoloring to the NST result and blends it with
    the original result using the specified intensity.
    """
    denorm_tensor = unnormalize_tensor(result_tensor)
    palette = PALETTE_RGB[palette_name]
    palette = adjust_palette_hsl(palette, saturation=1.0, hue=0.0)
    mapped = map_colors(denorm_tensor, palette)
    final_tensor = blend_images_with_intensity(denorm_tensor, mapped, intensity=intensity)
    logging.debug(f"Applied intensity with value = {intensity}")
    logging.debug(f"Mean original: {torch.mean(denorm_tensor).item():.4f}, mapped: {torch.mean(mapped).item():.4f}")
    return final_tensor


def display_image(pil_img, pos, size=(512, 512), surface=None):
    """
    Scales and draws a PIL image onto a Pygame surface at the given position.
    """
    if surface is None:
        surface = pygame.display.get_surface()
    try:
        disp_img = pil_img.resize(size)
        image_surface = pygame.image.fromstring(disp_img.tobytes(), disp_img.size, disp_img.mode)
        image_rect = image_surface.get_rect(center=pos)
        surface.blit(image_surface, image_rect)
    except Exception as e:
        logging.error(f"Error displaying image: {e}")


def seven_page(screen, WIDTH, HEIGHT, main_callback=None):
    """
    Main UI page for style transfer. Users can upload images, select models,
    choose palettes, run style transfer, and apply intensity adjustments.
    """
    global CHOSEN_PALETTE, PALETTE_INTENSITY

    pygame.init()
    WHITE, BLACK = (255, 255, 255), (0, 0, 0)
    FONT_LARGE = pygame.font.Font(None, 36)
    FONT_SMALL = pygame.font.Font(None, 24)

    left_width = 250
    margin = 20
    content_preview_rect = pygame.Rect(margin, margin, left_width - 2 * margin, 150)
    upload_content = pygame.Rect(margin, content_preview_rect.bottom + 5, left_width - 2 * margin, 35)
    style_preview_rect = pygame.Rect(margin, upload_content.bottom + 10, left_width - 2 * margin, 150)
    upload_style = pygame.Rect(margin, style_preview_rect.bottom + 5, left_width - 2 * margin, 35)
    prompt_button = pygame.Rect(margin, HEIGHT - 120, 100, 40)
    processing_label_rect = pygame.Rect(margin, upload_style.bottom + 10, left_width - 2 * margin, 30)

    result_x = left_width + margin + 20
    result_y = 20
    result_width = WIDTH - result_x - 20
    result_height = HEIGHT - (20 + 60 + 50 + 20)
    result_image_rect = pygame.Rect(result_x, result_y, result_width, result_height)

    control_area_rect = pygame.Rect(result_image_rect.x, result_image_rect.bottom + 10, result_image_rect.width, 60)
    dropdown_button = pygame.Rect(control_area_rect.x + 10, control_area_rect.y + 10, 180, 30)
    dropdown_active = False
    dropdown_options_rects = []
    option_height = 25
    for i in range(len(PALETTE_MENU)):
        dropdown_options_rects.append(pygame.Rect(
            dropdown_button.x,
            dropdown_button.y - (i + 1) * option_height,
            dropdown_button.width,
            option_height
        ))
    intensity_slider_rect = pygame.Rect(dropdown_button.right + 20, control_area_rect.y + 15, 100, 20)
    apply_button = pygame.Rect(intensity_slider_rect.right + 20, control_area_rect.y + 5, 120, 30)

    back_button = pygame.Rect(10, HEIGHT - 50, 60, 40)
    current_model = "ResNet"
    model_button = pygame.Rect(80, HEIGHT - 50, 120, 40)
    run_button = pygame.Rect(WIDTH - 80, HEIGHT - 50, 60, 40)

    swatch_size = 30
    swatch_spacing = 5
    palette_preview = adjust_palette_hsl(PALETTE_RGB[CHOSEN_PALETTE], saturation=1.0, hue=0.0)

    palette_num_colors = 5
    content_image = None
    style_image = None
    result_image = None
    raw_output_tensor = None
    processing = False
    progress = 0

    def dynamic_intensity(image_tensor):
        """
        Computes a dynamic intensity value based on the image's standard deviation.
        """
        std_val = torch.std(image_tensor)
        intensity_val = 0.3 if std_val > 0.2 else 0.5
        logging.debug(f"Dynamic intensity: {intensity_val} (std={std_val:.4f})")
        return intensity_val

    def progress_callback(step, total_steps):
        nonlocal progress
        progress = int((step / total_steps) * 100)

    def run_style_transfer_thread():
        nonlocal raw_output_tensor, processing, result_image, content_image, style_image, current_model
        print("Starting style transfer process in a separate thread...")
        processing = True
        result_size = 512
        try:
            content_tensor = preprocess_image_adaptive(content_image, result_size, model=current_model)
            style_tensor = preprocess_image_adaptive(style_image, result_size, model=current_model)
        except Exception as e:
            logging.error(f"Preprocessing error: {e}")
            processing = False
            return
        output_tensor = style_transfer(
            content_tensor, style_tensor, model=current_model,
            iterations=300, content_weight=0.5, style_weight=1000.0,
            progress_callback=progress_callback
        )
        raw_output_tensor = output_tensor
        result_image = tensor_to_image(unnormalize_tensor(raw_output_tensor))
        print("Style transfer complete in background thread.")
        processing = False

    def prompt_random_selection():
        """
        Prompts the user for an artist ID and selects random content and style images.
        """
        root = Tk()
        root.withdraw()
        option = simpledialog.askstring("Prompt", "Enter prompt (1=vangogh, 2=pietmondriaan, 3=picasso, 4=claudemonet):")
        root.destroy()
        if option is None or option not in ARTIST_STYLES:
            logging.warning("Invalid prompt option")
            return
        artist = ARTIST_STYLES[option]
        try:
            content_files = [f for f in os.listdir(CONTENT_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            style_files = [f for f in os.listdir(STYLE_DIR)
                           if f.lower().endswith((".png", ".jpg", ".jpeg")) and artist in f.lower()]
            if not style_files:
                logging.warning(f"No style images found for {artist}.")
                return
            if content_files and style_files:
                chosen_content = os.path.join(CONTENT_DIR, random.choice(content_files))
                chosen_style = os.path.join(STYLE_DIR, random.choice(style_files))
                print(f"Chosen Content: {chosen_content}")
                print(f"Chosen Style: {chosen_style}")
                nonlocal content_image, style_image, result_image, raw_output_tensor, progress
                content_image = safe_load_image(chosen_content)
                style_image = safe_load_image(chosen_style)
                result_image = None
                raw_output_tensor = None
                progress = 0
                threading.Thread(target=run_style_transfer_thread, daemon=True).start()
            else:
                logging.warning("No images found in one or both directories.")
        except Exception as e:
            logging.error(f"Error selecting random files: {e}")

    def update_palette_preview():
        nonlocal palette_preview
        base_palette = PALETTE_RGB[CHOSEN_PALETTE]
        palette_preview = adjust_palette_hsl(base_palette, saturation=1.0, hue=0.0)

    def handle_style_upload():
        nonlocal style_image
        style_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if style_path:
            style_image = safe_load_image(style_path)
            if style_image is None:
                logging.error("Failed to load style image.")

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                exit()

            if not processing and event.type in [MOUSEBUTTONDOWN, MOUSEMOTION]:
                if pygame.mouse.get_pressed()[0]:
                    mouse_x, _ = pygame.mouse.get_pos()
                    if intensity_slider_rect.collidepoint(mouse_x, intensity_slider_rect.y):
                        rel_x = mouse_x - intensity_slider_rect.x
                        PALETTE_INTENSITY = round(rel_x / intensity_slider_rect.width, 2)
                        PALETTE_INTENSITY = max(0, min(PALETTE_INTENSITY, 1))

            if event.type == MOUSEBUTTONDOWN:
                if dropdown_button.collidepoint(event.pos):
                    if raw_output_tensor is None:
                        print("Palette selection only available after style transfer or prompt.")
                    else:
                        dropdown_active = not dropdown_active
                elif dropdown_active:
                    for idx, opt_rect in enumerate(dropdown_options_rects):
                        if opt_rect.collidepoint(event.pos):
                            CHOSEN_PALETTE = PALETTE_MENU[idx]
                            print(f"Palette selected: {CHOSEN_PALETTE}")
                            update_palette_preview()
                            dropdown_active = False
                elif prompt_button.collidepoint(event.pos):
                    prompt_random_selection()
                elif upload_content.collidepoint(event.pos):
                    content_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
                    if content_path:
                        content_image = safe_load_image(content_path)
                elif upload_style.collidepoint(event.pos):
                    handle_style_upload()
                elif back_button.collidepoint(event.pos):
                    running = False
                    if main_callback:
                        main_callback()
                elif model_button.collidepoint(event.pos):
                    idx = MODEL_LIST.index(current_model)
                    current_model = MODEL_LIST[(idx + 1) % len(MODEL_LIST)]
                    print(f"Model changed to {current_model}")
                elif apply_button.collidepoint(event.pos):
                    if raw_output_tensor is None:
                        print("No style transfer result available yet.")
                    else:
                        style_tensor = preprocess_image_adaptive(style_image, 512, model=current_model)
                        intensity_val = dynamic_intensity(style_tensor)
                        final_intensity = apply_intensity_to_result(
                            raw_output_tensor, style_image, palette_num_colors,
                            intensity=intensity_val, palette_name=CHOSEN_PALETTE
                        )
                        result_image = tensor_to_image(final_intensity)
                        print("Palette intensity applied to result.")
                elif run_button.collidepoint(event.pos):
                    if content_image and style_image and not processing:
                        threading.Thread(target=run_style_transfer_thread, daemon=True).start()
                    else:
                        print("Either images not loaded or processing in progress.")

        screen.fill(WHITE)

        if content_image:
            display_image(content_image, content_preview_rect.center,
                          size=(content_preview_rect.width, content_preview_rect.height), surface=screen)
        else:
            pygame.draw.rect(screen, (230, 230, 230), content_preview_rect)
            placeholder = FONT_SMALL.render("Content", True, BLACK)
            screen.blit(placeholder, placeholder.get_rect(center=content_preview_rect.center))
        pygame.draw.rect(screen, (200, 200, 200), upload_content)
        txt = FONT_SMALL.render("Upload Content", True, BLACK)
        screen.blit(txt, txt.get_rect(center=upload_content.center))

        if style_image:
            display_image(style_image, style_preview_rect.center,
                          size=(style_preview_rect.width, style_preview_rect.height), surface=screen)
        else:
            pygame.draw.rect(screen, (230, 230, 230), style_preview_rect)
            placeholder = FONT_SMALL.render("Style", True, BLACK)
            screen.blit(placeholder, placeholder.get_rect(center=style_preview_rect.center))
        pygame.draw.rect(screen, (200, 200, 200), upload_style)
        txt = FONT_SMALL.render("Upload Style", True, BLACK)
        screen.blit(txt, txt.get_rect(center=upload_style.center))

        if processing:
            proc_label = FONT_SMALL.render(f"Processing: {progress}%", True, BLACK)
            screen.blit(proc_label, proc_label.get_rect(midleft=(processing_label_rect.x,
                     processing_label_rect.y + processing_label_rect.height // 2)))

        pygame.draw.rect(screen, (200, 200, 200), prompt_button)
        prompt_txt = FONT_SMALL.render("Prompt", True, BLACK)
        screen.blit(prompt_txt, prompt_txt.get_rect(center=prompt_button.center))

        if result_image:
            display_image(result_image, result_image_rect.center,
                          size=(result_image_rect.width, result_image_rect.height), surface=screen)
        else:
            pygame.draw.rect(screen, (220, 220, 220), result_image_rect)
            placeholder = FONT_SMALL.render("Result", True, BLACK)
            screen.blit(placeholder, placeholder.get_rect(center=result_image_rect.center))

        pygame.draw.rect(screen, (240, 240, 240), control_area_rect)
        pygame.draw.rect(screen, (200, 200, 200), dropdown_button)
        dd_txt = FONT_SMALL.render(f"Palette: {CHOSEN_PALETTE}", True, BLACK)
        screen.blit(dd_txt, dd_txt.get_rect(center=dropdown_button.center))
        if dropdown_active:
            for idx, opt_rect in enumerate(dropdown_options_rects):
                pygame.draw.rect(screen, (220, 220, 220), opt_rect)
                opt_txt = FONT_SMALL.render(PALETTE_MENU[idx], True, BLACK)
                screen.blit(opt_txt, opt_txt.get_rect(center=opt_rect.center))
        pygame.draw.rect(screen, (200, 200, 200), intensity_slider_rect)
        knob_x = intensity_slider_rect.x + int(PALETTE_INTENSITY * intensity_slider_rect.width)
        knob_rect = pygame.Rect(knob_x - 5, intensity_slider_rect.y, 10, intensity_slider_rect.height)
        pygame.draw.rect(screen, (100, 100, 100), knob_rect)
        slider_txt = FONT_SMALL.render(f"Intensity: {PALETTE_INTENSITY}", True, BLACK)
        screen.blit(slider_txt, (intensity_slider_rect.x, intensity_slider_rect.y - 20))
        pygame.draw.rect(screen, (200, 200, 200), apply_button)
        apply_txt = FONT_SMALL.render("Apply", True, BLACK)
        screen.blit(apply_txt, apply_txt.get_rect(center=apply_button.center))

        common_color = (180, 180, 180)
        pygame.draw.rect(screen, common_color, back_button)
        back_txt = FONT_SMALL.render("Back", True, BLACK)
        screen.blit(back_txt, back_txt.get_rect(center=back_button.center))
        pygame.draw.rect(screen, common_color, model_button)
        model_txt = FONT_SMALL.render(f"Model: {current_model}", True, BLACK)
        screen.blit(model_txt, model_txt.get_rect(center=model_button.center))
        pygame.draw.rect(screen, (0, 200, 0), run_button)
        run_txt = FONT_SMALL.render("Run", True, BLACK)
        screen.blit(run_txt, run_txt.get_rect(center=run_button.center))

        if palette_preview is not None:
            x_start = model_button.right + 90
            y_start = model_button.centery - swatch_size // 2
            for i, color_val in enumerate(palette_preview):
                r, g, b = [int(255 * c) for c in color_val]
                swatch_rect = pygame.Rect(x_start + i * (swatch_size + swatch_spacing),
                                          y_start, swatch_size, swatch_size)
                pygame.draw.rect(screen, (r, g, b), swatch_rect)
                pygame.draw.rect(screen, BLACK, swatch_rect, 1)

        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


def main():
    """
    Main entry point for the Pygame application.
    Displays a start screen that launches the style transfer UI.
    """
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Neural Style Transfer")
    running = True

    while running:
        screen.fill((50, 50, 50))
        start_button = pygame.Rect(WIDTH // 2 - 150, HEIGHT // 2 - 25, 300, 50)
        pygame.draw.rect(screen, (70, 130, 180), start_button)
        font = pygame.font.Font(None, 28)
        txt = font.render("Start Style Transfer", True, (255, 255, 255))
        screen.blit(txt, txt.get_rect(center=start_button.center))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                exit()
            elif event.type == MOUSEBUTTONDOWN:
                if start_button.collidepoint(event.pos):
                    seven_page(screen, WIDTH, HEIGHT, main)
        pygame.event.pump()
    pygame.quit()


if __name__ == "__main__":
    main()