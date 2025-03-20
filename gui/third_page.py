import pygame
import sys
import json
import threading
import os
import cv2
import numpy as np
import PIL.Image
import tensorflow as tf
import tensorflow_hub as hub
import torch
from tqdm import tqdm
from utils.draw_helpers import loading_animation, draw_text, display_style_image, open_file_dialog, open_video_file, play_video, handle_slider_event, draw_radio_button, draw_sliders
from video.utils import video_to_frames, frames_to_video, apply_style_transfer, apply_style_transfer_multi,apply_style_transfer_ada, apply_style_transfer_multi_ada, clear_frames

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
WHITE, BLACK, LIGHT_GREY, DARK_GREY, GRAY = (255, 255, 255), (0, 0, 0), (200, 200, 200), (50, 50, 50),(200, 200, 200)
FONT = pygame.font.SysFont("Arial", 24)

def draw_text(text, font, color, surface, x, y):
    text_obj = font.render(text, True, color)
    text_rect = text_obj.get_rect(center=(x, y))
    surface.blit(text_obj, text_rect)
    
def third_page(screen, main_callback=None):
    """Video Style Transfer page of the GUI."""
    class MainWindow:
        def __init__(self):
            pygame.display.set_caption("Video Style Transfer")
            self.screen = screen
            self.running = True
            self.selected_video = None
            self.selected_style_image = None
            self.end_video = None
            #self.end_video = "video/outputs/stylized_video_256.mp4"
            self.multi_style = False
            self.radio_depth_aware_state = False
            self.optical_flow_method = "farneback" 
            self.image_display_width = int(self.screen.get_width() * 0.50)  # 67% of screen width
            self.image_display_height = int(self.screen.get_height() * 0.85)  # 85% of screen height
            self.display_x = SCREEN_WIDTH - self.image_display_width - 20  # Right-side positioning
            self.display_y = (SCREEN_HEIGHT - self.image_display_height) // 2  # Centered vertically
            self.menu_elements = {
                "Back": pygame.Rect(10, SCREEN_HEIGHT - 70, 100, 50),
                "Select Video": pygame.Rect(10, 20, 200, 30),
                "Select Style Image": pygame.Rect(10, 80, 200, 30),
                "Run Style Transfer": pygame.Rect(10, 110, 200, 30),
                "Run Multi-Image Style Transfer": pygame.Rect(10, 140, 300, 30),
                "Optical Flow: farneback": pygame.Rect(10, 220, 270, 30)
            }
            self.play_video_button = pygame.Rect(10, 50, 200, 30)
            self.play_created_video_button = pygame.Rect(10, 170, 200, 30) 
            self.radio_depth_aware = pygame.Rect(15, 200, 20, 20) 
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
                    
                    elif self.menu_elements["Select Video"].collidepoint(event.pos):
                        self.selected_video = open_video_file()
                        
                    elif self.selected_video and self.play_video_button.collidepoint(event.pos):
                        play_video(self.screen, self.selected_video)                    

                    elif self.menu_elements["Select Style Image"].collidepoint(event.pos):
                        self.selected_style_image = open_file_dialog()
                        
                    elif self.menu_elements["Run Style Transfer"].collidepoint(event.pos):
                        if self.selected_style_image and self.selected_video:
                            self.multi_style = False
                            self._style_transfer()
                    elif self.menu_elements["Run Multi-Image Style Transfer"].collidepoint(event.pos):
                        if self.selected_video:
                            self.multi_style = True
                            self._style_transfer()
                            
                    elif self.end_video and self.play_created_video_button.collidepoint(event.pos):
                        play_video(self.screen, self.end_video)
                    elif self.radio_depth_aware.collidepoint(event.pos): 
                        self.radio_depth_aware_state = not self.radio_depth_aware_state
                        print(self.radio_depth_aware_state)
                    elif self.menu_elements[f"Optical Flow: {self.optical_flow_method}"].collidepoint(event.pos):
                        self.optical_flow_method = "dualtvl1" if self.optical_flow_method == "farneback" else "farneback"
                        self.menu_elements[f"Optical Flow: {self.optical_flow_method}"] = self.menu_elements.pop(f"Optical Flow: {'dualtvl1' if self.optical_flow_method == 'farneback' else 'farneback'}")
                elif event.type == pygame.MOUSEMOTION and event.buttons[0]:  
                    if self.radio_depth_aware_state:
                        handle_slider_event(event, self.control_sliders) 
                        
            return True # Continue running
        
        def _style_transfer(self):
            """Runs the style transfer pipeline in a separate thread to keep Pygame responsive."""
            stop_flag = threading.Event()
            cancel_flag = threading.Event()

            loading_thread = threading.Thread(target=loading_animation, args=(self.screen, stop_flag, "Style Transfer in Progress\nThis may take up to 15 minutes... \nProcessing logs are in the cmd prompt. \nPress ESC to cancel."))
            loading_thread.start()

            def process_style_transfer():
                """Runs the actual style transfer process, with support for cancellation."""
                content_dir = "input/videos/content_frames/"
                styled_dir = "input/videos/styled_frames/"
                styles = "input/videos/styles/"
                #styles = "input/style/"

                try:
                    if cancel_flag.is_set():
                        print("Process canceled before frame extraction.")
                        return
                    clear_frames(content_dir)
                    clear_frames(styled_dir)
                    video_to_frames(self.selected_video, content_dir)

                    if cancel_flag.is_set():
                        print("Process canceled before style transfer.")
                        return 

                    if self.radio_depth_aware_state and self.multi_style:
                        output_video = "video/outputs/stylized_video_multi_depth.mp4"
                        apply_style_transfer_multi_ada(content_dir, styles, styled_dir, target_resolution=(256, 256), cancel_flag=cancel_flag, flow_method=self.optical_flow_method , offset= self.control_sliders["depth_proximity_offset"]["value"], prominence=self.control_sliders["depth_map_prominence"]["value"])
                    elif self.multi_style:
                        output_video = "video/outputs/stylized_video_multi.mp4"
                        apply_style_transfer_multi(content_dir, styles, styled_dir, target_resolution=(512, 288), cancel_flag=cancel_flag, flow_method=self.optical_flow_method)
                    elif self.radio_depth_aware_state:
                        output_video = "video/outputs/stylized_video_depth.mp4"
                        apply_style_transfer_ada(content_dir, self.selected_style_image, styled_dir, target_resolution=(256, 256), cancel_flag=cancel_flag, flow_method=self.optical_flow_method, offset= self.control_sliders["depth_proximity_offset"]["value"], prominence=self.control_sliders["depth_map_prominence"]["value"])
                    else:
                        output_video = "video/outputs/stylized_video_tensorflow.mp4"
                        apply_style_transfer(content_dir, self.selected_style_image, styled_dir, target_resolution=(256, 256), cancel_flag=cancel_flag, flow_method=self.optical_flow_method)

                    if cancel_flag.is_set():
                        print("Process canceled before video conversion.")
                        return

                    frames_to_video(styled_dir, output_video, fps=20)

                    if cancel_flag.is_set():
                        print("Process canceled before finalizing video.")
                        return

                    self.end_video = output_video  
                    print("Style transfer completed successfully!")

                except Exception as e:
                    print(f"Error during style transfer: {e}")

                finally:
                    stop_flag.set() 

            
            processing_thread = threading.Thread(target=process_style_transfer)
            processing_thread.start()

            
            while processing_thread.is_alive():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        cancel_flag.set() 
                        stop_flag.set()
                        return
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        print("Cancelling process...")
                        cancel_flag.set()  
                        stop_flag.set()
                        return

                pygame.time.delay(100)  

            processing_thread.join() 
            loading_thread.join()  
        
        def _draw(self):
            """Draws the UI elements on the screen."""
            self.screen.fill(WHITE)
            draw_text("Video Style Transfer", FONT, BLACK, self.screen, SCREEN_WIDTH // 2, 10)

            # Draw buttons
            for key, rect in self.menu_elements.items():
                pygame.draw.rect(self.screen, DARK_GREY, rect, border_radius=10)
                pygame.draw.rect(self.screen, WHITE, rect, width=2, border_radius=10) 
                draw_text(key, FONT, WHITE, self.screen, rect.centerx, rect.centery)
                
            if self.selected_style_image is not None:
                display_style_image(self.screen, self.selected_style_image, SCREEN_HEIGHT, BLACK, y_offset=150)
                
            if self.selected_video:
                pygame.draw.rect(self.screen, DARK_GREY, self.play_video_button, border_radius=10) 
                pygame.draw.rect(self.screen, WHITE, self.play_video_button, width=2, border_radius=10)  
                draw_text("Play Video", FONT, WHITE, self.screen, self.play_video_button.centerx, self.play_video_button.centery)
            
            if self.end_video:
                pygame.draw.rect(self.screen, (0, 180, 0), self.play_created_video_button, border_radius=10)  
                pygame.draw.rect(self.screen, WHITE, self.play_created_video_button, width=2, border_radius=10)
                draw_text("Play Created Video", FONT, WHITE, self.screen, self.play_created_video_button.centerx, self.play_created_video_button.centery)

            draw_radio_button(
                self.radio_depth_aware,
                "Depth-aware",
                self.radio_depth_aware_state,
                pygame.Rect(10, 140, 200, 50),
            )

            if self.radio_depth_aware_state:
                draw_sliders(self.slider_background_rect, self.control_sliders)

        def run(self):
            """Runs the main loop."""
            while self.running:
                self._draw()
                pygame.display.flip()
                self.running = self._handle_events()

    MainWindow().run()
    main_callback()  # Go back to main menu