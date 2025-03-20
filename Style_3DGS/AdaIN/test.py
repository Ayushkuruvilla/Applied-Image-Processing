from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import cv2
from torchvision.utils import save_image

from . import net
from .function import adaptive_instance_normalization, coral


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def get_style_embeddings(
    style_img,
    vgg_str="Style_3DGS/AdaIN/models/vgg_normalised.pth",
    style_size=512,
    crop=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vgg = net.vgg
    vgg.eval()
    vgg.load_state_dict(torch.load(vgg_str))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    vgg.to(device)

    style_tf = test_transform(style_size, crop)
    style = style_tf(style_img)
    style = style.to(device).unsqueeze(0)

    # might have alpha channel
    if style.shape[1] == 4:
        style = style[:, :3, :, :]
    style_f = vgg(style)
    return style_f


def style_transfer(
    vgg, decoder, content, style, depth_map, alpha=1.0, offset=0.15, prominence=20
):
    assert 0.0 <= alpha <= 1.0
    assert 0.0 <= offset <= 1.0
    content_f = vgg(content)

    # might have alpha channel
    if style.shape[1] == 4:
        style = style[:, :3, :, :]

    style_f = vgg(style)

    # compute P
    _, _, Hc, Wc = content_f.shape
    P = compute_stylization_strength_map(depth_map, (Hc, Wc), offset, prominence)

    AdaIN_feat = adaptive_instance_normalization(content_f, style_f)
    feat = AdaIN_feat * (1 - P) + content_f * P
    return decoder(feat)


def style_transfer_simple(vgg, decoder, content, style, alpha=0.5):
    assert 0.0 <= alpha <= 1.0
    content_f = vgg(content)
    style_f = vgg(style)

    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


def midas_depth_map_est(img):
    # model_type = "DPT_Large"
    # model_type = "DPT_Hybrid"
    model_type = "MiDaS_small"

    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    midas.to(device)
    midas.eval()

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    if isinstance(img, Image.Image):  # Check if it's a PIL image
        img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 3:  # Already RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction


def compute_stylization_strength_map(
    depth_map, encoder_size, offset=0.15, prominence=20
):
    # Step 1 and 2 skipped because Midas gives directly proximity map
    # D_max = depth_map.max()
    # proximity_map = D_max - depth_map

    # rescale
    Hc, Wc = encoder_size
    proximity_map_resized = F.interpolate(
        depth_map.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
        size=(Hc, Wc),
        mode="bicubic",
        align_corners=False,
    )

    # normalize P with min-max normalization
    min_val = proximity_map_resized.min()
    max_val = proximity_map_resized.max()
    if max_val > min_val:
        P = (proximity_map_resized - min_val) / (max_val - min_val)
    else:
        # the entire map is constant
        return torch.zeros_like(proximity_map_resized)

    P = P - P.mean()

    # apply prominence and clamp
    P = 1.0 / (1.0 + torch.exp(-prominence * P))
    P = torch.clamp(P, max=1.0 - offset)

    return P


def adain_inference(
    content_img,
    style_img,
    vgg_str="Style_3DGS/AdaIN/models/vgg_normalised.pth",
    decoder_str="Style_3DGS/AdaIN/models/decoder.pth",
    depth_offset=0.5,
    depth_prominence=20,
    content_size=512,
    style_size=512,
    alpha=0.5,
    crop=False,
    save_ext=".jpg",
    output="output",
    file_name="test",
    preserve_color=False,
    content_mask=None,
    use_depth=False,
):
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(output)
    output_dir.mkdir(exist_ok=True, parents=True)

    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(decoder_str))
    vgg.load_state_dict(torch.load(vgg_str))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    content_tf = test_transform(content_size, crop)
    style_tf = test_transform(style_size, crop)

    # process one content and one style
    if type(content_img) == str:
        content_img = Image.open(content_img)
    if type(style_img) == str:
        style_img = Image.open(str(style_img))

    content = content_tf(content_img)
    style = style_tf(style_img)
    if preserve_color:
        style = coral(style, content)
    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)

    with torch.no_grad():
        if use_depth:
            depth_map_est = midas_depth_map_est(content_img)
            output_img = style_transfer(
                vgg,
                decoder,
                content,
                style,
                depth_map_est,
                alpha,
                depth_offset,
                depth_prominence,
            )
        else:
            output_img = style_transfer_simple(vgg, decoder, content, style, alpha)

        if content_mask is not None:
            mask_tensor = torch.from_numpy(content_mask).float().to(device)
            mask_tensor = mask_tensor.unsqueeze(0)

            # Use nearest interpolation for a binary mask
            mask_tensor = F.interpolate(
                mask_tensor, size=content.shape[-2:], mode="nearest"
            )
            output_img = F.interpolate(
                output_img,
                size=content.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )  # this one might few pixels off in some edge cases
            stylized_tensor = content * (1.0 - mask_tensor) + output_img * mask_tensor
        else:
            stylized_tensor = output_img

    if stylized_tensor.shape[1] == 4:  # If it has 4 channels (RGBA)
        stylized_tensor = stylized_tensor[:, :3, :, :]

    output_path = output_dir / f"{file_name}{save_ext}"
    save_image(stylized_tensor, str(output_path))
    print(f"Image saved to {output_path}")

    return output_path
