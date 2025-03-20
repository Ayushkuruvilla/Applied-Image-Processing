import os
import threading
import random
from tkinter import Tk, filedialog

import cv2
import numpy as np
import pygame
from PIL import Image
from pygame.locals import QUIT

from Style_3DGS import adain_inference
from utils import (
    handle_slider_event,
    draw_group_box,
    draw_button_box,
    draw_radio_button,
    draw_sliders,
    display_image_with_style,
)

CONTENT_DIR = "input/content/"
STYLE_DIR = "input/style/"

# Initialize Pygame
pygame.init()
pipeline_running = False

# Screen Dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Image/Video Loader")

# Colors
DARK_GREY = (59, 58, 57)
BLACK = (0, 0, 0)
LIGHT_GREY = (200, 200, 200, 128)
FONT = pygame.font.Font(None, 24)

# Radio button states
radio_depth_aware_state = False
radio_semantic_segmentation_state = False

# Group box definitions
upload_group = pygame.Rect(5, 20, 160, 150)
pipeline_group = pygame.Rect(5, 180, 160, 200)
slider_background_rect = pygame.Rect(180, HEIGHT - 120, 440, 100)

# Run button
run_button = pygame.Rect(WIDTH - 160, HEIGHT - 60, 140, 40)
running_rect = pygame.Rect(WIDTH / 2 - 40, 30, 80, 40)

# Button definitions inside groups
upload_button = pygame.Rect(15, 50, 140, 40)
style_button = pygame.Rect(15, 100, 140, 40)

# Pipeline buttons
radio_depth_aware = pygame.Rect(15, 210, 20, 20)
radio_semantic_segmentation = pygame.Rect(15, 240, 20, 20)

# Sliders for parameters
control_sliders = {
    "depth_proximity_offset": {
        "rect": pygame.Rect(200, HEIGHT - 100, 200, 20),
        "value": 0.15,
        "min": 0,
        "max": 1,
    },
    "depth_map_prominence": {
        "rect": pygame.Rect(200, HEIGHT - 50, 200, 20),
        "value": 20,
        "min": 0,
        "max": 100,
    },
}


def load_file_dialog():
    """Open a file dialog to select a file."""
    Tk().withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename()
    return file_path


def get_random_file(directory, valid_extensions=(".png", ".jpg", ".jpeg")):
    try:
        files = [
            f for f in os.listdir(directory) if f.lower().endswith(valid_extensions)
        ]
        if not files:
            print(f"No valid files found in directory: {directory}")
            return None
        return os.path.join(directory, random.choice(files))
    except Exception as e:
        print(f"Error accessing directory {directory}: {e}")
        return None


def play_video(video_path):
    """Play a video on the Pygame screen."""
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(np.rot90(frame))
        frame_surface = pygame.transform.scale(frame_surface, (WIDTH, HEIGHT))
        screen.blit(frame_surface, (0, 0))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == QUIT:
                cap.release()
                pygame.quit()
                return

    cap.release()


def run_pipeline(content_path, style_path):
    # updates global running tasks and output image
    def pipeline_task(content, path):
        global pipeline_running, resulting_image
        pipeline_running = True  # Set running state
        try:
            if radio_depth_aware_state:
                output_path = adain_inference(
                    content,
                    path,
                    depth_offset=control_sliders["depth_minimal_offset"]["value"],
                    depth_prominence=control_sliders["depth_map_prominence"]["value"],
                )
                resulting_image = Image.open(output_path).convert("RGB")

            elif radio_semantic_segmentation_state:
                print("Semantic segmentation logic goes here.")
        except Exception as e:
            print(f"Error running pipeline: {e}")
        finally:
            pipeline_running = False  # Reset running state

    # Run the pipeline task in a new thread
    thread = threading.Thread(target=pipeline_task, args=(content_path, style_path))
    thread.start()


# MAIN DRAWING LOOP. Order of drawing is important, otherwise components might occlude others unintentionally.
def main():
    global pipeline_running, resulting_image
    global radio_depth_aware_state, radio_semantic_segmentation_state

    # start with random input for faster debugging
    running = True
    content_path = get_random_file(CONTENT_DIR)
    content_image = Image.open(content_path).convert("RGB")
    style_path = get_random_file(STYLE_DIR)
    style_image = Image.open(style_path).convert("RGB")
    resulting_image = None

    while running:
        screen.fill(DARK_GREY)

        if content_image or style_image:
            if resulting_image:
                display_image_with_style(resulting_image, style_image)
            else:
                display_image_with_style(content_image, style_image)

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if upload_button.collidepoint(event.pos):
                    file_path = load_file_dialog()
                    if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                        resulting_image = None
                        content_image = Image.open(file_path).convert("RGB")
                        content_path = file_path
                    elif file_path.lower().endswith((".hdr")):
                        resulting_image = None
                        hdr_image = cv2.imread(
                            file_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR
                        )
                        hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)
                        content_image = Image.fromarray(
                            (hdr_image * 255).astype(np.uint8)
                        )
                        content_path = file_path
                    elif file_path.lower().endswith((".mp4", ".avi", ".mov")):
                        resulting_image = None
                        content_path = file_path
                        play_video(file_path)
                elif style_button.collidepoint(event.pos):
                    resulting_image = None
                    file_path = load_file_dialog()
                    if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                        style_image = Image.open(file_path).convert("RGB")
                        style_path = file_path
                elif radio_depth_aware.collidepoint(event.pos):
                    radio_depth_aware_state = not radio_depth_aware_state
                elif radio_semantic_segmentation.collidepoint(event.pos):
                    radio_semantic_segmentation_state = (
                        not radio_semantic_segmentation_state
                    )
                elif run_button.collidepoint(event.pos):
                    # block multiple calls because it will spawn threads
                    if not pipeline_running:
                        run_pipeline(content_path, style_path)

            elif event.type in [
                pygame.MOUSEBUTTONDOWN,
                pygame.MOUSEMOTION,
                pygame.MOUSEBUTTONUP,
            ]:
                handle_slider_event(event, control_sliders)

        if pipeline_running:
            draw_group_box(running_rect, "Running")

        # group boxes
        draw_group_box(upload_group, "Upload")
        draw_group_box(pipeline_group, "Style Transfer")

        # Draw buttons inside groups
        draw_button_box(upload_button, "Content", pipeline_group)
        draw_button_box(style_button, "Style", pipeline_group)

        # Draw radio buttons
        draw_radio_button(
            radio_depth_aware, "Depth-aware", radio_depth_aware_state, pipeline_group
        )
        draw_radio_button(
            radio_semantic_segmentation,
            "Semantic Segmentation",
            radio_semantic_segmentation_state,
            pipeline_group,
        )

        # Draw sliders
        if radio_depth_aware_state:
            draw_sliders(slider_background_rect, control_sliders)

        # Draw run button
        draw_button_box(run_button, "Run")

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
