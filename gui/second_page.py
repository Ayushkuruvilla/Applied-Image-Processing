import pygame
import sys
import json
import threading
from pygame.locals import *
from PIL import Image, ImageColor
import numpy as np
from typing import Tuple
import tkinter as tk
from tkinter import filedialog
from utils.draw_helpers import loading_animation,draw_text,display_style_image,display_image,open_file_dialog,get_random_file,handle_slider_event,draw_radio_button,draw_sliders
import cv2
import pixel_art.utils as pix
import os
import networkx as nx
import tensorflow as tf
import tensorflow_hub as hub
from video.utils import tensor_to_image
from Style_3DGS import adain_inference

from scipy.spatial import KDTree
RGBColor = Tuple[int, int, int]

pygame.init()

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
WHITE, BLACK, LIGHT_GREY, DARK_GREY, GRAY = (255, 255, 255), (0, 0, 0), (200, 200, 200), (50, 50, 50),(200, 200, 200)
FONT = pygame.font.SysFont("Arial", 24)    
PALETTES_JSON_PATH = "assets/lospec-palettes-c16-n1024.json"
with open(PALETTES_JSON_PATH) as json_file:
    PALETTES = json.load(json_file)

RESAMPLING_MODES = {mode: getattr(Image.Resampling, mode) for mode in Image.Resampling.__members__}
RESAMPLING_MODE_KEYS = list(RESAMPLING_MODES.keys())


def second_page(screen, main_callback):
    """Second page with buttons for Pixelize and Depixelize Pipeline"""
    buttons = {
        "Pixelize Image": pygame.Rect(300, 200, 220, 50),
        "Depixelize Pipeline": pygame.Rect(300, 300, 220, 50),
        "Back": pygame.Rect(10, SCREEN_HEIGHT - 100, 100, 50)  # Back to main_gui.py
    }

    running = True
    while running:
        screen.fill(WHITE)
        draw_text("Pixel Art Pipeline", FONT, BLACK, screen, SCREEN_WIDTH // 2, 100)

        for key, rect in buttons.items():
            pygame.draw.rect(screen, BLACK, rect, 3)
            draw_text(key, FONT, BLACK, screen, rect.centerx, rect.centery)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if buttons["Pixelize Image"].collidepoint(event.pos):
                    Pixelize(screen, lambda: second_page(screen, main_callback))
                elif buttons["Depixelize Pipeline"].collidepoint(event.pos):
                    Depixelize_pipeline(screen, lambda: second_page(screen, main_callback))
                elif buttons["Back"].collidepoint(event.pos):
                    main_callback()  # Go back to main_gui.py
                    
def Pixelize(screen, main_callback=None):
    class MainWindow:
        def __init__(self, title: str):
            pygame.display.set_caption(title)
            self.screen = screen
            self.resize = False
            self.running = True
            self.image = Image.open(get_random_file("input/content/")).convert("RGB")
            self.converted_image = None
            self.palette_index = -1
            self.resampling_index = 0
            self.palette = None
            self.brightness = 0
            self.contrast = 0
            self.grayscale = False
            self.resampling_mode = RESAMPLING_MODES[RESAMPLING_MODE_KEYS[self.resampling_index]]
            self.recolor_methods = ["RGB","kd-tree", "LAB", "Floyd-Steinberg",]
            self.selected_recolor_method = 0 
            self.downsampling_factor = 1
            self._setup_gui_elements()

        def _setup_gui_elements(self):
            """Sets up menu elements and their positions."""
            self.menu_elements = {
                "Open Image": pygame.Rect(10, 20, 220, 40),
                "Save Image": pygame.Rect(10, 70, 220, 40),
                "Convert Image": pygame.Rect(10, 120, 220, 40),
                "Palette": pygame.Rect(10, 180, 220, 40),
                "Brightness": pygame.Rect(10, 250, 220, 10),
                "Contrast": pygame.Rect(10, 290, 220, 10),
                "Grayscale": pygame.Rect(10, 330, 220, 40),
                "Resampling": pygame.Rect(10, 380, 220, 40),
                "Factor": pygame.Rect(10, 450, 220, 10),
                "Recolor Method": pygame.Rect(10, 490, 220, 40),
                "Back": pygame.Rect(10, 550, 100, 50),
            }
            self.slider_positions = {"Brightness": 0, "Contrast": 0, "Factor": 1}
        def _handle_events(self):
            """Handles UI interactions."""
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.resize = not self.resize
                    self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    self._draw()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.menu_elements["Palette"].collidepoint(event.pos):
                        self.palette_index = (self.palette_index + 1) % len(PALETTES)
                        self.palette = [ImageColor.getrgb(color) for color in list(PALETTES.values())[self.palette_index]['colors']]
                    elif self.menu_elements["Resampling"].collidepoint(event.pos):
                        self.resampling_index = (self.resampling_index + 1) % len(RESAMPLING_MODE_KEYS)
                        self.resampling_mode = RESAMPLING_MODES[RESAMPLING_MODE_KEYS[self.resampling_index]]
                        print(f"Resampling mode changed to: {RESAMPLING_MODE_KEYS[self.resampling_index]}")
                    elif self.menu_elements["Grayscale"].collidepoint(event.pos):
                        self.grayscale = not self.grayscale
                        print(f"Grayscale: {self.grayscale}")
                    elif self.menu_elements["Recolor Method"].collidepoint(event.pos):
                        self.selected_recolor_method = (self.selected_recolor_method + 1) % len(self.recolor_methods)
                        print(f"Recolor method changed to: {self.recolor_methods[self.selected_recolor_method]}")
                    elif self.menu_elements["Open Image"].collidepoint(event.pos):
                        self.open_image()
                    elif self.menu_elements["Save Image"].collidepoint(event.pos):
                        self.save_image()
                    elif self.menu_elements["Convert Image"].collidepoint(event.pos):
                        self.convert_image()
                    elif self.menu_elements["Back"].collidepoint(event.pos):
                        running = False
                        if main_callback:
                            main_callback()
                elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
                    self._update_sliders(event.pos)
                        
        def _update_sliders(self, position):
            for slider, rect in self.menu_elements.items():
                if slider in ["Brightness", "Contrast", "Factor"] and rect.collidepoint(position):
                    if slider == "Factor":
                        self.slider_positions[slider] = round(min(max((position[0] - rect.x) / rect.width * 98 + 1, 1), 99))
                        print(int(self.slider_positions["Factor"]))
                    else:
                        self.slider_positions[slider] = min(max((position[0] - rect.x) / rect.width * 2 - 1, -1), 1)
                        
        def open_image(self):
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
            if file_path:
                self.image = Image.open(file_path).convert("RGB")
                self.original_image = self.image.copy()  # Store original image for processing
                self.converted_image = None  # Reset converted image

        def save_image(self):
            if self.converted_image:
                self.converted_image.save("pixel_art/outputs/output_image.png")

        def convert_image(self):
            if self.image:
                self.converted_image = self._convert_image(
                    self.image,
                    int(self.slider_positions["Factor"]),
                    self.resampling_mode,
                    self.grayscale,
                    self.slider_positions["Brightness"],
                    self.slider_positions["Contrast"],
                    self.palette,
                )
                self._draw()

        def _convert_image(self, image, downsampling_factor, resampling_mode, grayscale, brightness_adjustment, contrast_adjustment, colors):
            if downsampling_factor > 1:
                image = self._downsample_image(image, downsampling_factor, resampling_mode)
            if grayscale:
                image = image.convert("L").convert("RGB")
            if brightness_adjustment != 0 or contrast_adjustment != 0:
                image = self._adjust_brightness_and_contrast(image, brightness_adjustment, contrast_adjustment)
            if colors:
                if self.recolor_methods[self.selected_recolor_method] == "LAB":
                    image = self._recolor_image_LAB(image, colors)
                elif self.recolor_methods[self.selected_recolor_method] == "Floyd-Steinberg":
                    image = self._recolor_image_floyd(image, colors)
                elif self.recolor_methods[self.selected_recolor_method] == "kd-tree":
                    image = self._recolor_image_kd(image, colors)
                else:
                    image = self._recolor_image(image, colors)
            return image

        def _downsample_image(self, image, factor, resampling_mode):
            new_width = image.width // factor
            new_height = image.height // factor
            return image.resize((new_width, new_height), resample=resampling_mode)

        def _adjust_brightness_and_contrast(self, image, brightness_adjustment, contrast_adjustment):
            image_array = np.asarray(image) / 255
            if brightness_adjustment != 0:
                image_array += brightness_adjustment
            if contrast_adjustment != 0:
                contrast_factor = np.tan((0.5 + contrast_adjustment) * np.pi / 4)
                image_array = (image_array - 0.5) * contrast_factor + 0.5
            return Image.fromarray((image_array.clip(0, 1) * 255).astype(np.uint8))

        def _recolor_image(self, image, colors):
            image_array = np.asarray(image)
            if image_array.shape[-1] == 4:
                image_array = image_array[:, :, :3]
            colors_array = np.asarray(colors, dtype=np.uint8)
            rgb_difference_vectors = image_array[:, :, np.newaxis, :] - colors_array
            rgb_distances = np.linalg.norm(rgb_difference_vectors, axis=-1)
            closest_color_indices = rgb_distances.argmin(axis=-1)
            new_image_array = colors_array[closest_color_indices].astype(np.uint8)
            return Image.fromarray(new_image_array)


        def _recolor_image_kd(self, image, colors):
            image_array = np.asarray(image)
        
            alpha_channel = None
            if image_array.shape[-1] == 4:
                alpha_channel = image_array[:, :, 3]
                image_array = image_array[:, :, :3]
        
            colors_array = np.asarray(colors, dtype=np.uint8)
        
            tree = KDTree(colors_array)
            
            pixels = image_array.reshape(-1, 3)
            _, indices = tree.query(pixels)
        
            recolored_pixels = colors_array[indices].reshape(image_array.shape)
        
            if alpha_channel is not None:
                recolored_pixels = np.dstack((recolored_pixels, alpha_channel))
        
            return Image.fromarray(recolored_pixels.astype(np.uint8))
        
        def _recolor_image_LAB(self,image, colors):
            image_array = np.asarray(image, dtype=np.uint8)
            
            alpha_channel = None
            if image_array.shape[-1] == 4:
                alpha_channel = image_array[:, :, 3]
                image_array = image_array[:, :, :3]

            image_lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)

            colors_lab = np.array([cv2.cvtColor(np.array([[c]], dtype=np.uint8), cv2.COLOR_RGB2LAB)[0][0] for c in colors])

            lab_difference_vectors = image_lab[:, :, np.newaxis, :] - colors_lab
            lab_distances = np.linalg.norm(lab_difference_vectors, axis=-1)

            closest_color_indices = lab_distances.argmin(axis=-1)

            new_image_array = np.array(colors, dtype=np.uint8)[closest_color_indices]

            if alpha_channel is not None:
                new_image_array = np.dstack((new_image_array, alpha_channel))

            return Image.fromarray(new_image_array.astype(np.uint8))
        
        def _recolor_image_floyd(self, image, colors):
            print("running")
            image_array = np.asarray(image, dtype=np.float32)  # Convert to float for error diffusion
            if image_array.shape[-1] == 4:
                image_array = image_array[:, :, :3]  # Remove alpha channel

            colors_array = np.asarray(colors, dtype=np.uint8)  # Convert palette to array

            height, width, _ = image_array.shape
            for y in range(height):
                for x in range(width):
                    # Find the nearest palette color
                    original_color = image_array[y, x]
                    distances = np.linalg.norm(colors_array - original_color, axis=1)
                    closest_index = np.argmin(distances)
                    new_color = colors_array[closest_index]
                    image_array[y, x] = new_color

                    error = original_color - new_color

                    # Floyd-Steinberg error diffusion (distributes error to neighboring pixels)
                    if x < width - 1:
                        image_array[y, x + 1] += error * (7 / 16)
                    if y < height - 1 and x > 0:
                        image_array[y + 1, x - 1] += error * (3 / 16)
                    if y < height - 1:
                        image_array[y + 1, x] += error * (5 / 16)
                    if y < height - 1 and x < width - 1:
                        image_array[y + 1, x + 1] += error * (1 / 16)

            image_array = np.clip(image_array, 0, 255).astype(np.uint8)

            return Image.fromarray(image_array)
                
        def run(self):
            while self.running:
                self.screen.fill(WHITE)
                self._handle_events()
                self._draw()
                pygame.display.flip()

        def _draw(self): 
            """Draws UI elements including buttons, sliders, and images."""
            self.screen.fill(WHITE)
            draw_text("Pixelization", FONT, BLACK, self.screen, SCREEN_WIDTH // 2, 10)

            image_display_width = int(self.screen.get_width() * 0.67)  # 67% of screen width
            image_display_height = int(self.screen.get_height() * 0.85)
            display_x = self.screen.get_width() - image_display_width - 20  # Right-side positioning
            display_y = (self.screen.get_height() - image_display_height) // 2  # Centered vertically

            # UI Text
            palette_name = "None" if self.palette is None else list(PALETTES.keys())[self.palette_index]
            recolor_method_text = f"Recolor: {self.recolor_methods[self.selected_recolor_method]}" 
            text_elements = {
                "Palette": f"Palette: {palette_name}",
                "Brightness": f"Brightness: {self.slider_positions['Brightness']:.2f}",
                "Contrast": f"Contrast: {self.slider_positions['Contrast']:.2f}",
                "Grayscale": f"Grayscale: {'ON' if self.grayscale else 'OFF'}",
                "Resampling": f"Resampling: {RESAMPLING_MODE_KEYS[self.resampling_index]}",
                "Factor": f"Factor: {self.slider_positions['Factor']:.0f}",
                "Recolor Method": recolor_method_text,
                "Open Image": "Press to Open Image",
                "Save Image": "Press to Save Image",
                "Convert Image": "Press to Convert Image",
                "Back": "Back",
            }

            for key, rect in self.menu_elements.items():
                if key not in ["Brightness", "Contrast", "Factor"]:  
                    pygame.draw.rect(self.screen, DARK_GREY, rect, border_radius=10)
                    pygame.draw.rect(self.screen, WHITE, rect, width=2, border_radius=10)
                    text_surface = FONT.render(text_elements[key], True, WHITE)
                    text_rect = text_surface.get_rect(center=rect.center)
                    self.screen.blit(text_surface, text_rect)

            slider_y_offset = 25  # Move text slightly above the slider
            for key in ["Brightness", "Contrast", "Factor"]:
                slider_rect = self.menu_elements[key]

                if key == "Factor":
                    text_surface = FONT.render(f"{key}: {self.slider_positions[key]:.0f}", True, BLACK)
                    min_value, max_value = 1, 99 
                else:
                    text_surface = FONT.render(f"{key}: {self.slider_positions[key]:.2f}", True, BLACK)
                    min_value, max_value = -1, 1
                self.screen.blit(text_surface, (slider_rect.x, slider_rect.y - slider_y_offset))

                pygame.draw.line(
                    self.screen,
                    GRAY,
                    (slider_rect.x, slider_rect.y + slider_rect.height // 2),
                    (slider_rect.x + slider_rect.width, slider_rect.y + slider_rect.height // 2),
                    2,
                )

                if key in ["Brightness", "Contrast"]:
                    normalized_value = (self.slider_positions[key] + 1) / 2  
                else:
                    normalized_value = (self.slider_positions[key] - min_value) / (max_value - min_value)
                    
                handle_x = slider_rect.x + int(normalized_value * slider_rect.width)

                handle_rect = pygame.Rect(handle_x - 5, slider_rect.y + slider_rect.height // 2 - 5, 10, 10)
                pygame.draw.rect(self.screen, BLACK, handle_rect)
        
            image_to_display = self.converted_image if self.converted_image else self.image
            if image_to_display:
                image_surface = pygame.image.fromstring(image_to_display.tobytes(), image_to_display.size, image_to_display.mode)
                
                # Scale image to fit bounding box while keeping aspect ratio
                image_surface = pygame.transform.scale(image_surface, (image_display_width, image_display_height))
                
                image_rect = image_surface.get_rect(topleft=(display_x, display_y))
                self.screen.blit(image_surface, image_rect)
    
    app = MainWindow("Pixel Art Converter")
    app.run()
    return

def Depixelize_pipeline(screen, main_callback=None):
    """Depixelize pipeline page with a Back button"""
    class MainWindow:
        def __init__(self):
            pygame.display.set_caption("Depixelize Pipeline")
            self.screen = screen
            self.running = True
            self.selected_image = get_random_file("input/pixel_art/")
            self.selected_style_image = get_random_file("input/videos/styles/")
            self.processed_image = None
            self.end_image = None
            self.graph = None
            self.image_display_width = int(self.screen.get_width() * 0.50)  # 67% of screen width
            self.image_display_height = int(self.screen.get_height() * 0.85)  # 85% of screen height
            self.display_x = SCREEN_WIDTH - self.image_display_width - 20  # Right-side positioning
            self.display_y = (SCREEN_HEIGHT - self.image_display_height) // 2  # Centered vertically
            self.menu_elements = {
                "Back": pygame.Rect(10, SCREEN_HEIGHT - 70, 100, 50),
                "Select Image": pygame.Rect(10, 20, 200, 30),
                "Run Depixelization": pygame.Rect(10, 50, 200, 30),
                "Select Style Image": pygame.Rect(10, 80, 200, 30),
                "Run Style Transfer": pygame.Rect(10, 110, 200, 30)
            }
            self.radio_depth_aware_state = False
            self.radio_depth_aware = pygame.Rect(15, 150, 20, 20) 
            self.slider_background_rect = pygame.Rect(180, SCREEN_HEIGHT - 120, 440, 100) 
            self.control_sliders = {
                "depth_proximity_offset": {
                    "rect": pygame.Rect(200, SCREEN_HEIGHT - 100, 200, 20),
                    "value": 0.15,
                    "min": 0,
                    "max": 1,
                },
                "depth_map_prominence": {
                    "rect": pygame.Rect(200, SCREEN_HEIGHT - 50, 200, 20),
                    "value": 20,
                    "min": 0,
                    "max": 100,
                },
            }
            
        def _handle_events(self):
            """Handles user input events."""
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.menu_elements["Back"].collidepoint(event.pos):
                        return False
                    
                    elif self.menu_elements["Select Image"].collidepoint(event.pos):
                        self.selected_image = open_file_dialog()
                        if self.selected_image:
                            self.original_image = self.selected_image
                            self.processed_image = None
                            self.graph = None
                    
                    # Run Depixelization button -> Run vectorization
                    elif self.menu_elements["Run Depixelization"].collidepoint(event.pos):
                        if self.selected_image:
                            self._run_vectorization()
                            
                    # Select Style Image button -> Open file dialog
                    elif self.menu_elements["Select Style Image"].collidepoint(event.pos):
                        self.selected_style_image = open_file_dialog()
                        #if self.selected_style_image:
                        #    self.processed_image = None
                        #    self.graph = None
                    elif self.radio_depth_aware.collidepoint(event.pos): 
                        self.radio_depth_aware_state = not self.radio_depth_aware_state
                        print(self.radio_depth_aware_state)                        
                    # Run Style Transfer button -> style_transfer
                    elif self.menu_elements["Run Style Transfer"].collidepoint(event.pos):
                        if self.selected_style_image and self.selected_image:
                            self._style_transfer()
                elif event.type == pygame.MOUSEMOTION and event.buttons[0]:  
                    if self.radio_depth_aware_state:
                        handle_slider_event(event, self.control_sliders) 
            return True  # Keep running
        
        def _load_img(self,path_to_img):
            max_dim = 512
            img = tf.io.read_file(path_to_img)
            img = tf.image.decode_image(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)

            shape = tf.cast(tf.shape(img)[:-1], tf.float32)
            long_dim = max(shape)
            scale = max_dim / long_dim

            new_shape = tf.cast(shape * scale, tf.int32)

            img = tf.image.resize(img, new_shape)
            img = img[tf.newaxis, :]
            return img
        
        def _style_transfer(self):
            """Runs the style transfer process on the selected image."""
            if self.radio_depth_aware_state:
                stop_flag = threading.Event()
                loading_thread = threading.Thread(target=loading_animation, args=(self.screen, stop_flag, "Style Transfer in Progress"))
                loading_thread.start()
                self.graph = None
                self.processed_image = None
                stylized_img_path = adain_inference(self.selected_image,self.selected_style_image,output="pixel_art/outputs/",file_name = "styled",depth_offset=self.control_sliders["depth_proximity_offset"]["value"],depth_prominence=self.control_sliders["depth_map_prominence"]["value"],use_depth=True)
                self.end_image = Image.open(stylized_img_path)
                stop_flag.set()  # Signal loading thread to stop
                loading_thread.join() 
                self._draw()
            else:
                stop_flag = threading.Event()
                loading_thread = threading.Thread(target=loading_animation, args=(self.screen, stop_flag, "Style Transfer in Progress"))
                loading_thread.start()
                selected_image = self._load_img(self.selected_image)
                self.graph = None
                self.processed_image = None
                style_image = self._load_img(self.selected_style_image)
                hub_module = hub.load('https://www.kaggle.com/models/google/arbitrary-image-stylization-v1/TensorFlow1/256/2')
                stylized_image = hub_module(tf.constant(selected_image), tf.constant(style_image))[0]
                stylized_frame = tensor_to_image(stylized_image)
                self.end_image = stylized_frame
                stop_flag.set()  # Signal loading thread to stop
                loading_thread.join() 
                self._draw()

                     
        def _run_vectorization(self):
            """Runs the vectorization process on the selected image."""
            output_png_path = "pixel_art/outputs/vectorized_output"
            print(f"Processing: {self.selected_image} -> {output_png_path}")
            image = Image.open(self.selected_image)
            img_converted = image.convert('YCbCr')

            # Step 1: Create Similarity Graph
            similarity_graph = pix.create_similarity_graph(image, img_converted)

            # Step 2: Process Diagonal Edges
            similarity_graph = pix.process_diagonal_edges(similarity_graph, img_converted.width, img_converted.height)

            # Step 3: Generate Voronoi Cells
            similarity_graph = pix.create_voronoi_cells(similarity_graph, img_converted)

            # Step 4: Remove Low-Valency Voronoi Points
            valency = pix.calculate_valencies(similarity_graph, img_converted)
            similarity_graph = pix.remove_valency_2_voronoi_points(similarity_graph, valency,img_converted)

            # Step 5: Apply Voronoi Smoothing using Chaikin's Algorithm
            similarity_graph = pix.smooth_voronoi_graph(
                similarity_graph,
                num_iterations=4,
                num_different_colors_threshold=3,
                diagonal_length_threshold=0.8,
                width=img_converted.width,
                height=img_converted.height
            )
            #graph, img = vectorize_image(self.selected_image, output_png_path)
            self.processed_image = img_converted
            self.graph = similarity_graph
            pix.render_as_png(similarity_graph, img_converted.width, img_converted.height, 10, output_png_path)
            self.selected_image = output_png_path + ".png"
            self._draw()
                        
        def _draw_voronoi_cells(self):
            """Draws the Voronoi cells from the processed graph inside the fixed bounding box."""
            if not self.graph:
                return

            img_width, img_height = self.processed_image.size  # Use processed image dimensions

            # Scaling factors to fit Voronoi cells inside bounding box
            scale_x = self.image_display_width / img_width
            scale_y = self.image_display_height / img_height

            for node in self.graph.nodes():
                voronoi_cell_vertices = self.graph.nodes[node]["voronoi_vertices"]

                # Scale and reposition vertices inside the bounding box
                vertices = [
                    [
                        self.display_x + v[0] * scale_x,  # Scale x and adjust for bounding box
                        self.display_y + v[1] * scale_y   # Scale y and adjust for bounding box
                    ]
                    for v in voronoi_cell_vertices
                ]

                color = self.graph.nodes[node]["rgb_color"]  # Assuming 'rgb_value' exists in graph

                pygame.draw.polygon(self.screen, color, vertices)
                
        def _draw(self):
            """Draws the UI elements on the screen."""
            self.screen.fill(WHITE)
            draw_text("Depixelize Pipeline", FONT, BLACK, self.screen, SCREEN_WIDTH // 2, 10)

            # Draw buttons
            for key, rect in self.menu_elements.items():
                pygame.draw.rect(self.screen, DARK_GREY, rect, border_radius=10)  # Darker box
                pygame.draw.rect(self.screen, WHITE, rect, width=2, border_radius=10)  # White border
                draw_text(key, FONT, WHITE, self.screen, rect.centerx, rect.centery)  # White text

            # Display selected image path
            #if self.selected_image is not None:
            #    draw_text(f"Selected: {os.path.basename(self.selected_image)}", FONT, BLACK, self.screen, SCREEN_WIDTH // 2, 300)
            
                        # Display processed image
                        
            if self.selected_style_image is not None:
                display_style_image(self.screen, self.selected_style_image, SCREEN_HEIGHT, BLACK)
                
            if isinstance(self.end_image,Image.Image):
                display_image(self.screen, self.end_image, self.display_x, self.display_y, self.image_display_width, self.image_display_height)
            elif self.graph:
                self._draw_voronoi_cells()
            
            elif self.selected_image:
                display_image(self.screen, self.selected_image, self.display_x, self.display_y, self.image_display_width, self.image_display_height)
                
            draw_radio_button(self.radio_depth_aware,"Depth-aware",self.radio_depth_aware_state,pygame.Rect(10, 140, 200, 50))

            if self.radio_depth_aware_state:
                draw_sliders(self.slider_background_rect, self.control_sliders)



        def run(self):
            """Runs the main loop."""
            while self.running:
                self._draw()
                pygame.display.flip()
                self.running = self._handle_events()

    MainWindow().run()
    main_callback()  # Return to second_page