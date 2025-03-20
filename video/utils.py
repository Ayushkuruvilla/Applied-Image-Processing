import os
import cv2
import numpy as np
import PIL.Image
import tensorflow as tf
import tensorflow_hub as hub
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from Style_3DGS import adain_inference

# Load the Fast Neural Style Transfer model
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Convert a TensorFlow tensor to an image
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def video_to_frames(video_path, output_dir):
    """Extract frames from a video and save them in a sequential format."""
    os.makedirs(output_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    frame_counter = 0

    while vidcap.isOpened():
        success, image = vidcap.read()
        if not success:
            break
        
        # Save frame with sequential numbering: frame_0001.jpg, frame_0002.jpg, ...
        frame_path = os.path.join(output_dir, f'frame_{frame_counter:04d}.jpg')
        cv2.imwrite(frame_path, image)
        frame_counter += 1

    vidcap.release()
    cv2.destroyAllWindows()
    print(f"Extracted {frame_counter} frames from video.")
    
# Load and preprocess an image for style transfer
def load_img(path_to_img, target_shape=None, max_dim=512):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    if target_shape:
        # Resize directly to target shape
        img = tf.image.resize(img, target_shape)
    else:
        # Preserve aspect ratio while resizing longest side to max_dim
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = tf.reduce_max(shape)
        scale = max_dim / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)

    img = img[tf.newaxis, :]  # Add batch dimension
    return img

def imshow(image, title=None):
    """Display an image using Matplotlib."""
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)  # Remove batch dimension
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()
    
# Estimate optical flow between two frames
def estimate_optical_flow(frame1, frame2, method='farneback'):
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    print("Estimating optical flow...")
    if method == 'farneback':
        flow = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, None, 0.5, 5, 15, 3, 7, 1.5, 0)
    elif method == 'dualtvl1':
        print("Using Dual TV-L1 Optical Flow")
        flow_calculator = cv2.optflow.DualTVL1OpticalFlow_create()
        flow = flow_calculator.calc(frame1_gray, frame2_gray, None)
    print("Done estimating...")
    return torch.from_numpy(flow.transpose(2, 0, 1))

# Warp an image using optical flow
def warp_image(image, flow):
    """Warp image using optical flow."""
    print("Warping image using optical flow...")
    h, w = flow.shape[1:]
    flow = flow.numpy().transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
    
    # Generate mesh grid for pixel locations
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Compute new pixel locations using flow
    map_x = (x + flow[..., 0]).astype(np.float32)
    map_y = (y + flow[..., 1]).astype(np.float32)
    
    # Warp image using remap function
    warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    print("Done warping...")
    return warped

# Apply style transfer with optical flow-based temporal consistency
def apply_style_transfer(content_dir, style_image_path, output_dir,flow_method ='farneback', alpha=0.7,target_resolution=None,cancel_flag=None):
    os.makedirs(output_dir, exist_ok=True)
    style_image = load_img(style_image_path, target_shape=target_resolution)

    frames_list = sorted([f for f in os.listdir(content_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    prev_stylized = None
    prev_frame = None

    print("Processing frames with optical flow-based consistency...")
    for i, frame in enumerate(tqdm(frames_list)):
        if cancel_flag and cancel_flag.is_set():
            print("Stopping style transfer...")
            return 
        frame_path = os.path.join(content_dir, frame)
        content_image = load_img(frame_path, target_shape=target_resolution)
        current_frame = cv2.imread(frame_path)

        # Resize current frame to match target_resolution
        current_frame = cv2.resize(current_frame, target_resolution)

        # Apply style transfer to the current frame
        current_stylized = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
        current_stylized = np.array(current_stylized[0])

        if prev_stylized is not None and prev_frame is not None:
            # Compute optical flow (forward and backward)
            flow_forward = estimate_optical_flow(prev_frame, current_frame,method = flow_method)

            # Warp the previous stylized frame
            warped_prev = warp_image(prev_stylized, flow_forward)

            # Blend warped previous frame with current stylized frame
            current_stylized = (alpha * current_stylized + (1 - alpha) * warped_prev)
            #current_stylized = np.clip(current_stylized, 0, 255).astype(np.uint8)

        # Convert to image and save
        stylized_img = tensor_to_image(tf.convert_to_tensor(current_stylized))
        #print(f"Displaying Final Stylized Image for Frame {i} AFTER blending 2")
        #imshow(np.array(stylized_img), title=f"Final Stylized Image (Frame {i}) AFTER blending 2")
        output_path = os.path.join(output_dir, frame)
        stylized_img.save(output_path)

        # Store for next iteration
        prev_stylized = current_stylized
        prev_frame = current_frame
        
        print(f"Stylized and saved: {output_path}")

def apply_style_transfer_multi(content_dir, style_dir, output_dir, flow_method ='farneback', alpha=0.7,target_resolution=None,cancel_flag=None):
    os.makedirs(output_dir, exist_ok=True)
    
    # Retrieve sorted lists of content and style images
    content_frames = sorted([f for f in os.listdir(content_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    style_images = sorted(os.listdir(style_dir))
    
    num_content_imgs = len(content_frames)
    num_style_imgs = len(style_images)
    
    if num_style_imgs == 0:
        raise ValueError("No style images found in the style directory.")
    
    # Ensure at least one style image is applied
    frames_per_style = max(1, num_content_imgs // num_style_imgs)

    prev_stylized = None
    prev_frame = None
    style_idx = 0  # Start with the first style image

    print("Processing frames with multiple style images...")

    for i, frame in enumerate(tqdm(content_frames)):
        if cancel_flag and cancel_flag.is_set():
            print("Stopping style transfer...")
            return 
        frame_path = os.path.join(content_dir, frame)
        content_image = load_img(frame_path)
        current_frame = cv2.imread(frame_path)

        current_frame = cv2.resize(current_frame, target_resolution)
        
        # Switch to the next style image at intervals
        if i > 0 and i % frames_per_style == 0:
            style_idx = min(style_idx + 1, num_style_imgs - 1)

        style_image_path = os.path.join(style_dir, style_images[style_idx])
        style_image = load_img(style_image_path)

        # Apply style transfer
        current_stylized = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
        current_stylized = np.array(current_stylized[0])

        if prev_stylized is not None and prev_frame is not None:
            # Compute optical flow (forward and backward)
            flow_forward = estimate_optical_flow(prev_frame, current_frame, flow_method)

            # Warp the previous stylized frame
            warped_prev = warp_image(prev_stylized, flow_forward)
            current_stylized = (alpha * current_stylized + (1 - alpha) * warped_prev)

        # Convert to image and save
        stylized_img = tensor_to_image(tf.convert_to_tensor(current_stylized))
        output_path = os.path.join(output_dir, frame)
        stylized_img.save(output_path)

        prev_stylized = current_stylized
        prev_frame = current_frame

        print(f"Stylized and saved: {output_path}")
        
def normalize_image(image):
    """Ensure image is float32 in range [0,1] before blending."""
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image

def blend_images(stylized, warped, alpha):
    """Perform adaptive blending with correct normalization."""
    stylized = normalize_image(stylized)
    warped = normalize_image(warped)

    blended = alpha * stylized + (1 - alpha) * warped
    return (np.clip(blended * 255, 0, 255)).astype(np.uint8) 

def tensor_to_image_ada(tensor):
    """Convert a TensorFlow tensor to an image without altering values."""
    tensor = np.array(tensor)  # Convert tensor to NumPy
    if np.ndim(tensor) > 3:
        tensor = tensor[0]  # Remove batch dimension if present

    return Image.fromarray(tensor.astype(np.uint8))

# Apply style transfer with optical flow-based temporal consistency
def apply_style_transfer_ada(content_dir, style_image_path, output_dir, flow_method ='farneback', alpha=0.7,target_resolution=None,cancel_flag=None, offset = 0.30, prominence = 20):
    ada_temp_dir = "input/videos/ada_outputs/"
    clear_frames(ada_temp_dir)
    os.makedirs(output_dir, exist_ok=True)
    frames_list = sorted([f for f in os.listdir(content_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    prev_stylized = None
    prev_frame = None

    print("Processing frames with optical flow-based consistency...")
    for i, frame in enumerate(tqdm(frames_list)):
        if cancel_flag and cancel_flag.is_set():
            print("Stopping style transfer...")
            return 

        frame_path = os.path.join(content_dir, frame)
        #content_image = load_img(frame_path, target_shape=target_resolution)
        current_frame = cv2.imread(frame_path)

        # Resize current frame to match target_resolution
        current_frame = cv2.resize(current_frame, target_resolution)

        stylized_img_path = adain_inference(
            frame_path,
            style_image_path,
            content_size = 256,
            output=ada_temp_dir,
            file_name = frame.rsplit(".", 1)[0],
            depth_offset=offset,
            depth_prominence=prominence,
            use_depth=True
        )

        current_stylized = np.array(Image.open(stylized_img_path))
        current_stylized = cv2.resize(current_stylized, target_resolution, interpolation=cv2.INTER_AREA)
        #imshow(np.array(current_stylized), title=f"Final Stylized Image (Frame {i}) AFTER resize")
        #print(f"🔍 Shape of `current_stylized`: {current_stylized.shape}") 
        if prev_stylized is not None and prev_frame is not None:
            # Compute optical flow (forward and backward)
            flow_forward = estimate_optical_flow(prev_frame, current_frame, method=flow_method)

            # Warp the previous stylized frame
            warped_prev = warp_image(prev_stylized, flow_forward)
            #imshow(np.array(warped_prev), title=f"Final Stylized Image (Frame {i}) AFTER warping")
            
            #print(f"🔍 Shape of `warped_prev`: {warped_prev.shape}") 
            # Blend warped previous frame with current stylized frame
            current_stylized = blend_images(current_stylized, warped_prev, alpha)

            #imshow(np.array(current_stylized), title=f"Final Stylized Image (Frame {i}) AFTER Blending")
            #current_stylized = np.clip(current_stylized, 0, 255).astype(np.uint8)

        # Convert to image and save
        stylized_img = tensor_to_image_ada(current_stylized)
        #print(f"Displaying Final Stylized Image for Frame {i} AFTER blending 2")
        #imshow(np.array(stylized_img), title=f"Final Stylized Image (Frame {i}) AFTER blending 2")
        output_path = os.path.join(output_dir, frame)
        stylized_img.save(output_path)

        # Store for next iteration
        prev_stylized = current_stylized
        prev_frame = current_frame
        
        print(f"Stylized and saved: {output_path}")

def apply_style_transfer_multi_ada(content_dir, style_dir, output_dir, flow_method ='farneback', alpha=0.7,target_resolution=None,cancel_flag=None,offset = 0.30, prominence = 20):
    os.makedirs(output_dir, exist_ok=True)
    ada_temp_dir = "input/videos/ada_outputs/"
    clear_frames(ada_temp_dir)
    # Retrieve sorted lists of content and style images
    content_frames = sorted([f for f in os.listdir(content_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    style_images = sorted(os.listdir(style_dir))
    
    num_content_imgs = len(content_frames)
    num_style_imgs = len(style_images)
    
    if num_style_imgs == 0:
        raise ValueError("No style images found in the style directory.")
    
    # Ensure at least one style image is applied
    frames_per_style = max(1, num_content_imgs // num_style_imgs)

    prev_stylized = None
    prev_frame = None
    style_idx = 0  # Start with the first style image

    print("Processing frames with multiple style images...")

    for i, frame in enumerate(tqdm(content_frames)):
        if cancel_flag and cancel_flag.is_set():
            print("Stopping style transfer...")
            return 
        frame_path = os.path.join(content_dir, frame)
        current_frame = cv2.imread(frame_path)
        current_frame = cv2.resize(current_frame, target_resolution)
        # Switch to the next style image at intervals
        if i > 0 and i % frames_per_style == 0:
            style_idx = min(style_idx + 1, num_style_imgs - 1)

        style_image_path = os.path.join(style_dir, style_images[style_idx])
        

        stylized_img_path = adain_inference(
            frame_path,
            style_image_path,
            content_size = 256,
            output=ada_temp_dir,
            file_name = frame.rsplit(".", 1)[0],
            depth_offset=offset,
            depth_prominence=prominence,
            use_depth=True
        )

        current_stylized = np.array(Image.open(stylized_img_path))
        current_stylized = cv2.resize(current_stylized, target_resolution, interpolation=cv2.INTER_AREA)

        if prev_stylized is not None and prev_frame is not None:
            # Compute optical flow (forward and backward)
            flow_forward = estimate_optical_flow(prev_frame, current_frame, flow_method)

            # Warp the previous stylized frame
            warped_prev = warp_image(prev_stylized, flow_forward)
            current_stylized = blend_images(current_stylized, warped_prev, alpha)

        # Convert to image and save
        stylized_img = tensor_to_image_ada(current_stylized)
        output_path = os.path.join(output_dir, frame)
        stylized_img.save(output_path)

        prev_stylized = current_stylized
        prev_frame = current_frame

        print(f"Stylized and saved: {output_path}")
        
# Convert processed frames into a video
def frames_to_video(image_folder, output_video, fps=20):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
    if not images:
        print("No images found for video creation.")
        return

    first_frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        frame_path = os.path.join(image_folder, image)
        frame = cv2.imread(frame_path)
        video.write(frame)

    video.release()
    print(f"Stylized video saved to {output_video}")
 
#Removes all files from the specified directory except for .gitkeep.    
def clear_frames(directory: str):
    print(f"Clearing directory '{directory}' of old frames...")
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and entry.name != ".gitkeep":  # Skip .gitkeep
                os.remove(entry.path)
                
# Run the below from a created py file in main dir to test out style transfer manually
def run_style_transfer():
    content_dir = "input/videos/content_frames/"
    styled_dir = "input/videos/styled_frames/"
    output_video = "video/outputs/stylized_video_manual.mp4"
    styles = "input/videos/styles/"
    selected_video = "input/videos/sample.mp4"
    selected_style_image = "input/videos/styles/style_3.mp4"
    offset = 0.30
    prominence = 20
    optical_flow_method = 'dualtvl1'

    clear_frames(content_dir)
    clear_frames(styled_dir)
    video_to_frames(selected_video, content_dir)
    apply_style_transfer_multi_ada(content_dir, styles, styled_dir, target_resolution=(256, 256), flow_method=optical_flow_method , offset = offset, prominence = prominence)
    #apply_style_transfer_multi(content_dir, styles, styled_dir, target_resolution=(512, 288),flow_method=optical_flow_method)
    #apply_style_transfer_ada(content_dir, selected_style_image, styled_dir, target_resolution=(256, 256), flow_method=optical_flow_method, offset= offset, prominence=prominence)
    #apply_style_transfer(content_dir, selected_style_image, styled_dir, target_resolution=(256, 256), flow_method=optical_flow_method)
    frames_to_video(styled_dir, output_video, fps=20)
    
