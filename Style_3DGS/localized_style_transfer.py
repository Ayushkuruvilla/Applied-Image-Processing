import torch
import matplotlib.pyplot as plt
from torchvision import models
from torchvision.transforms import functional as F
from Style_3DGS.AdaIN import adain_inference
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA


# Matrices for Reinhard's transform
RGB_TO_LMS = np.array(
    [[0.3811, 0.5783, 0.0402], [0.1967, 0.7244, 0.0782], [0.0241, 0.1288, 0.8444]]
)
LMS_TO_LAB = np.array(
    [[1 / np.sqrt(3), 0, 0], [0, 1 / np.sqrt(6), 0], [0, 0, 1 / np.sqrt(2)]]
) @ np.array([[1, 1, 1], [1, 1, -2], [1, -1, 0]])
LAB_TO_LMS = np.linalg.inv(LMS_TO_LAB)
LMS_TO_RGB = np.linalg.inv(RGB_TO_LMS)


def rgb_to_lab_image(image_uint8):
    """
    Convert an 8-bit RGB image [H,W,3] to Reinhard's lab space [H,W,3].
    """
    # scale to [0..1]
    image_float = image_uint8.astype(np.float32) / 255.0

    H, W, C = image_float.shape
    flat = image_float.reshape(-1, 3)

    # RGB -> LMS
    LMS = np.dot(flat, RGB_TO_LMS.T)
    LMS = np.maximum(LMS, 1e-6)  # avoid log(0)
    log_LMS = np.log10(LMS)

    # LMS -> lab
    lab_flat = np.dot(log_LMS, LMS_TO_LAB.T)
    lab = lab_flat.reshape(H, W, 3)
    return lab


def lab_to_rgb_image(lab):
    """
    Convert an lab image [H,W,3] back to 8-bit RGB [H,W,3].
    """
    H, W, C = lab.shape
    flat = lab.reshape(-1, 3)

    # lab -> log_LMS
    log_LMS = np.dot(flat, LAB_TO_LMS.T)
    LMS = np.power(10, log_LMS)

    # LMS -> RGB
    rgb_float = np.dot(LMS, LMS_TO_RGB.T)
    rgb_float = np.clip(rgb_float, 0, 1)

    # scale to [0..255]
    rgb_uint8 = (rgb_float * 255).astype(np.uint8)
    rgb_uint8 = rgb_uint8.reshape(H, W, 3)
    return rgb_uint8


# define color-space transforms for pixels im (N,3)


def rgb_to_lab_pixels(pixels_uint8):
    """
    Convert an Nx3 array of 8-bit RGB pixels to Nx3 lab floats.
    """
    # [0..1]
    float_pixels = pixels_uint8.astype(np.float32) / 255.0
    LMS = np.dot(float_pixels, RGB_TO_LMS.T)
    LMS = np.maximum(LMS, 1e-6)
    log_LMS = np.log10(LMS)
    lab = np.dot(log_LMS, LMS_TO_LAB.T)
    return lab


def lab_to_rgb_pixels(lab_pixels):
    """
    Convert an Nx3 array of lab floats back to Nx3 8-bit RGB.
    """
    log_LMS = np.dot(lab_pixels, LAB_TO_LMS.T)
    LMS = np.power(10, log_LMS)
    rgb_float = np.dot(LMS, LMS_TO_RGB.T)
    rgb_float = np.clip(rgb_float, 0, 1)
    rgb_uint8 = (rgb_float * 255).astype(np.uint8)
    return rgb_uint8


def apply_pca(lab_data):
    pca = PCA(n_components=1)
    # only inerested in most predominant colour
    projection = pca.fit_transform(lab_data)
    return projection, pca


def match_cdf(target_proj, source_proj):
    """
    Match the CDF of target_proj to source_proj along a 1D dimension.
    Returns the matched target_proj.
    """
    t_sorted = np.sort(target_proj, axis=0).flatten()
    s_sorted = np.sort(source_proj, axis=0).flatten()

    # Ensure they are of the same length
    if len(t_sorted) != len(s_sorted):
        # Interpolate the smaller array to match the length of the larger
        if len(t_sorted) > len(s_sorted):
            s_sorted = np.interp(
                np.linspace(0, 1, len(t_sorted)),
                np.linspace(0, 1, len(s_sorted)),
                s_sorted,
            )
        else:
            t_sorted = np.interp(
                np.linspace(0, 1, len(s_sorted)),
                np.linspace(0, 1, len(t_sorted)),
                t_sorted,
            )

    # find the matches
    matched = np.interp(target_proj.flatten(), t_sorted, s_sorted)
    return matched.reshape(-1, 1)


def color_transfer_foreground(foreground_img, background_img):
    """
    Apply color transfer to `foreground_img` harmonizes with `background_img`
    Returns an adjusted foreground in [H,W,3], uint8.
    """
    # get non-black -> do not contribute to PCA
    fg_mask = foreground_img.sum(axis=-1) > 0
    bg_mask = background_img.sum(axis=-1) > 0

    fg_pixels = foreground_img[fg_mask]
    bg_pixels = background_img[bg_mask]

    # Edge cases
    if fg_pixels.size == 0:
        print("Warning: No foreground pixels found.")
        return foreground_img.copy()

    if bg_pixels.size == 0:
        print("Warning: No background pixels found for color transfer.")
        return foreground_img.copy()

    # convert those pixels to lab space
    fg_lab = rgb_to_lab_pixels(fg_pixels)
    bg_lab = rgb_to_lab_pixels(bg_pixels)

    # PCA on both sets
    fg_projection, fg_pca = apply_pca(fg_lab)
    bg_projection, bg_pca = apply_pca(bg_lab)

    # match the CDF along that principal component
    matched_projection = match_cdf(fg_projection, bg_projection)

    # reconstruct matched pixels back in lab -> RGB
    adjusted_lab = fg_pca.inverse_transform(matched_projection)
    adjusted_rgb = lab_to_rgb_pixels(adjusted_lab)

    # place these adjusted pixels back into a full image
    adjusted_foreground = foreground_img.copy()
    adjusted_foreground[fg_mask] = adjusted_rgb

    return adjusted_foreground


def extract_foreground_deeplab(content_img, threshold=0.5):
    # Load pre-trained DeepLabV3 model for smenatic sgementation
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()

    content_tensor = F.to_tensor(content_img).unsqueeze(0)

    with torch.no_grad():
        output = model(content_tensor)["out"][0]

    probs = torch.softmax(output, dim=0)

    # background probabilities (class 0)
    background_probs = probs[0]
    background_mask = (background_probs > threshold).float()
    background_mask = background_mask.unsqueeze(0).numpy().astype(np.uint8)

    return background_mask


def run_localized_style_transfer(
    content_img_path,
    style_img_path,
    output_path="../output",
    file_name="test",
    use_depth=False,
    depth_offset=0.5,
    depth_prominence=20,
):
    content_img = Image.open(content_img_path).convert("RGB")
    content_np = np.array(content_img)

    # get background mask
    background_mask = extract_foreground_deeplab(content_img)

    # style transfer just the background
    stylized_image_path = adain_inference(
        content_img=content_img_path,
        style_img=style_img_path,
        content_mask=background_mask,
        output=output_path,
        file_name=file_name,
        use_depth=use_depth,
        depth_offset=depth_offset,
        depth_prominence=depth_prominence,
        alpha=1,
    )
    background_mask = background_mask[0]
    stylized_image = Image.open(stylized_image_path).convert("RGB")
    stylized_np = np.array(stylized_image)

    # resize maybe (edge cases)
    if (stylized_np.shape[0] != background_mask.shape[0]) or (
        stylized_np.shape[1] != background_mask.shape[1]
    ):
        stylized_np = np.array(
            Image.fromarray(stylized_np).resize(
                (background_mask.shape[1], background_mask.shape[0]), Image.NEAREST
            )
        )

    foreground_mask = 1 - background_mask
    foreground_np = content_np * foreground_mask[..., None]
    background_np = stylized_np * background_mask[..., None]

    # apply color transfer
    adjusted_foreground = color_transfer_foreground(foreground_np, background_np)
    combined_image = adjusted_foreground * foreground_mask[..., None] + background_np

    # save
    combined_image_pil = Image.fromarray(combined_image.astype(np.uint8))
    save_path = f"{output_path}/localized_style_transfer_result.jpg"
    combined_image_pil.save(save_path)

    return save_path
