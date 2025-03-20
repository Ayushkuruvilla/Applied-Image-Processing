import pygame
import os
import pygame
from tkinter import Tk, filedialog
from PIL import Image
from pygame.locals import QUIT
import threading

from mixing_texture_gyum.image_loader import ImageLoader
from mixing_texture_gyum.vgg_model import VGGExtractor
from mixing_texture_gyum.style_mixer import StyleMixer
from mixing_texture_gyum.neural_style_transfer import NeuralStyleTransfer

import io
import torch

def six_page(screen, WIDTH, HEIGHT, callback=None):
    DARK_GREY = (59, 58, 57)
    WHITE = (255, 255, 255)
    LIGHT_GREY = (200, 200, 200)

    content_button = pygame.Rect(20, 50, 140, 40)
    style1_button  = pygame.Rect(20, 100, 140, 40)
    style2_button  = pygame.Rect(20, 150, 140, 40)
    run_button     = pygame.Rect(WIDTH - 180, HEIGHT - 60, 160, 40)
    back_button    = pygame.Rect(20, HEIGHT - 60, 100, 40)

    # content_path = r"C:\\work\\dsait4120-24-group13\\input\style_mixing_inputs\\house.png"
    # style1_path  = r"C:\\work\\dsait4120-24-group13\\input\style_mixing_inputs\\van_gogh.png"
    # style2_path  = r"C:\\work\\dsait4120-24-group13\\input\style_mixing_inputs\\scream.png"

    content_path = None
    style1_path = None
    style2_path = None
    content_surface = None
    result_surface = None
    pipeline_running = False

    style1_weight = 0.3
    style2_weight = 0.7

    def load_file():
        Tk().withdraw()
        return filedialog.askopenfilename()

    def run_pipeline(content, style1, style2):

        nonlocal pipeline_running, result_surface
        pipeline_running = True
        result_surface = None

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            loader = ImageLoader(max_size=1024, device=device)
            content_tensor = loader.load_image(content)
            style1_tensor = loader.load_image(style1)
            style2_tensor = loader.load_image(style2)

            vgg_model = VGGExtractor(device=device)
            mixer = StyleMixer(
                vgg_extractor=vgg_model,
                style_layers=['conv1_1','conv2_1','conv3_1','conv4_1']
            )
            mixer.add_style(style1_tensor, weight = style1_weight)
            mixer.add_style(style2_tensor, weight = style2_weight)

            nst = NeuralStyleTransfer(
                vgg_extractor=vgg_model,
                content_layer='conv4_2',
                content_weight=1.0,
                style_weight=1e6,  
                num_steps=2000,
                lr=0.002,
                device=device
            )

            final = nst.run_transfer(content_tensor, mixer)
            final_PIL = loader.tensor_to_pil(final)

            result_surface = pil_to_surface(final_PIL)
        except Exception as e:
            print("Error occur")
        finally:
            pipeline_running = False

    def pil_to_surface(pil_img):
        mode = pil_img.mode
        size = pil_img.size
        data = pil_img.tobytes()
        if mode == "RGBA":
            surface = pygame.image.fromstring(data, size, mode)
        else:
            pil_img = pil_img.convert("RGBA")
            data = pil_img.tobytes()
            surface = pygame.image.fromstring(data, pil_img.size, "RGBA")
        return surface

    def content_surface_fill(path):
        try:
            img = Image.open(path).convert("RGB")
            return pil_to_surface(img)
        except Exception as e:
            print("No Content file", e)
            return None

    active = True
    content_surface = content_surface_fill(content_path)
    while active:
        screen.fill(DARK_GREY)

        pygame.draw.rect(screen, LIGHT_GREY, content_button)
        pygame.draw.rect(screen, LIGHT_GREY, style1_button)
        pygame.draw.rect(screen, LIGHT_GREY, style2_button)
        pygame.draw.rect(screen, LIGHT_GREY, run_button)
        pygame.draw.rect(screen, LIGHT_GREY, back_button)

        draw_text(screen, "Content",  content_button.center)
        draw_text(screen, "Style #1", style1_button.center)
        draw_text(screen, "Style #2", style2_button.center)
        if pipeline_running:
            draw_text(screen, "Running...", run_button.center)
        else:
            draw_text(screen, "Run", run_button.center)
        draw_text(screen, "Back", back_button.center)

        if result_surface:
            rectangle = result_surface.get_rect(center=(WIDTH//2, HEIGHT//2))
            screen.blit(result_surface, rectangle)
        # else: 
        #     if content_surface:
        #         content_rect = content_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
        #         screen.blit(content_surface, content_rect)

        for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if content_button.collidepoint(event.pos):
                        file_path = load_file()
                        if file_path:
                            content_path = file_path
                            print("Content selected:", content_path)

                    elif style1_button.collidepoint(event.pos):
                        file_path = load_file()
                        if file_path:
                            style1_path = file_path
                            print("Style #1 selected:", style1_path)

                    elif style2_button.collidepoint(event.pos):
                        file_path = load_file()
                        if file_path:
                            style2_path = file_path
                            print("Style #2 selected:", style2_path)

                    elif run_button.collidepoint(event.pos):
                        if not pipeline_running:
                            if content_path and style1_path and style2_path:
                                threading.Thread(
                                    target=run_pipeline,
                                    args=(content_path, style1_path, style2_path),
                                    daemon=True
                                ).start()
                            else:
                                print("Please select content, style1, and style2 first.")

                    elif back_button.collidepoint(event.pos):
                        active = False
                        if callback:
                            callback()

        pygame.display.flip()


def draw_text(screen, text, center_pos, color=(0,0,0), font_size=20):
    font = pygame.font.SysFont(None, font_size)
    txt_surface = font.render(text, True, color)
    txt_rect = txt_surface.get_rect(center=center_pos)
    screen.blit(txt_surface, txt_rect)
