import pygame
import threading
from tkinter import Tk, filedialog
from PIL import Image
from pygame.locals import QUIT
import cv2
import numpy as np
import torch
import sys

from spatial_variation.StyleTransfer import StyleTransfer

def eight_page(screen, WIDTH, HEIGHT, main_callback = None):

    DARK_GREY  = (59, 58, 57)
    LIGHT_GREY = (200, 200, 200)
    WHITE      = (255, 255, 255)

    content_button = pygame.Rect(20, 50, 140, 40)
    mask_button    = pygame.Rect(20, 100, 140, 40)
    style1_button  = pygame.Rect(20, 150, 140, 40)
    style2_button  = pygame.Rect(20, 200, 140, 40)
    style3_button  = pygame.Rect(20, 250, 140, 40)

    run_button     = pygame.Rect(WIDTH - 180, HEIGHT - 60, 160, 40)
    back_button    = pygame.Rect(20, HEIGHT - 60, 100, 40)

    content_path = None
    mask_path    = None
    style1_path  = None
    style2_path  = None
    style3_path  = None

    content_surface = None
    result_surface = None 

    pipeline_running = False

    def load_file():
        Tk().withdraw()
        return filedialog.askopenfilename()
    
    def pil_to_surface(pil_img):
        mode = pil_img.mode
        size = pil_img.size
        data = pil_img.tobytes()
        if mode == "RGBA":
            return pygame.image.fromstring(data, size, mode)
        else:
            pil_img = pil_img.convert("RGBA")
            data = pil_img.tobytes()
            return pygame.image.fromstring(data, pil_img.size, "RGBA")
        
    def load_content_surface(path, max_size=(400,400)):
        try:
            pil_img = Image.open(path).convert("RGB")
            w,h = pil_img.size
            if w>max_size[0] or h>max_size[1]:
                ratio = min(max_size[0]/w, max_size[1]/h)
                new_w = int(w*ratio)
                new_h = int(h*ratio)
                pil_img = pil_img.resize((new_w,new_h), Image.LANCZOS)
            return pil_to_surface(pil_img)
        except Exception as e:
            print("Failed to load content image:", e)
            return None
        
    def spatial_pipeline():
        nonlocal pipeline_running, result_surface
        pipeline_running = True
        result_surface = None

        try: 
            original = cv2.imread(content_path)
            mask_bgr = cv2.imread(mask_path)
            if original is None or mask_bgr is None:
                raise FileNotFoundError("Missing content/mask")
            
            style1 = cv2.imread(style1_path)
            style2 = cv2.imread(style2_path)
            style3 = cv2.imread(style3_path)
            if style1 is None or style2 is None or style3 is None:
                raise FileNotFoundError("no style")

            h,w = original.shape[:2]

            style = StyleTransfer(max_dim=1024, style_weight=1e5, content_weight=1.0, num_steps=800, lr=0.003)

            print("Applying style1 to full image ...")
            styled1 = style.run_style_transfer(original, style1)
            styled1 = cv2.resize(styled1, (w,h), interpolation=cv2.INTER_CUBIC)

            print("Applying style2 to full image ...")
            styled2 = style.run_style_transfer(original, style2)
            styled2 = cv2.resize(styled2, (w,h), interpolation=cv2.INTER_CUBIC)

            print("Applying style3 to full image ...")
            styled3 = style.run_style_transfer(original, style3)
            styled3 = cv2.resize(styled3, (w,h), interpolation=cv2.INTER_CUBIC)

            final = np.zeros_like(original)

            def color_mask(img_bgr, color_bgr, feather = 5):
                binmask = cv2.inRange(img_bgr, color_bgr, color_bgr)
                binmask = binmask.astype(np.float32)/255.0
                if feather>0:
                    binmask = cv2.GaussianBlur(binmask, (2*feather+1,2*feather+1),0)
                return binmask
            
            def blend(source1, source2, const):
                blender = cv2.merge([const, const, const])
                output = blender*source1 + (1 - blender) * source2
                return output.astype(np.uint8)
            
            print("Combining final ...")
            red = color_mask(mask_bgr, (0,0,255), 5)
            final = blend(styled1, final, red)
            green = color_mask(mask_bgr, (0,255,0), 5)
            final = blend(styled2, final, green)
            blue = color_mask(mask_bgr, (255,0,0), 5)
            final = blend(styled3, final, blue)

            final_rgb = cv2.cvtColor(final,cv2.COLOR_BGR2RGB)
            pil_final = Image.fromarray(final_rgb)
            w2, h2 = (400, 300)
            pil_final.thumbnail((w2,h2))
            result_surface = pil_to_surface(pil_final)

        except Exception as e:
            print("Error occur in pipeline", e)
        finally:
            pipeline_running = False

    active = True
    while active:
        screen.fill(DARK_GREY)

        pygame.draw.rect(screen, LIGHT_GREY, content_button)
        pygame.draw.rect(screen, LIGHT_GREY, mask_button)
        pygame.draw.rect(screen, LIGHT_GREY, style1_button)
        pygame.draw.rect(screen, LIGHT_GREY, style2_button)
        pygame.draw.rect(screen, LIGHT_GREY, style3_button)

        pygame.draw.rect(screen, LIGHT_GREY, run_button)
        pygame.draw.rect(screen, LIGHT_GREY, back_button)

        draw_text(screen, "Content", content_button.center)
        draw_text(screen, "Mask",    mask_button.center)
        draw_text(screen, "Style #1",style1_button.center)
        draw_text(screen, "Style #2",style2_button.center)
        draw_text(screen, "Style #3",style3_button.center)

        if pipeline_running:
            draw_text(screen, "Running...", run_button.center)
        else:
            draw_text(screen, "Run", run_button.center)

        draw_text(screen, "Back", back_button.center)

        if result_surface:
            rectangle = result_surface.get_rect(center=(WIDTH//2, HEIGHT//2))
            screen.blit(result_surface, rectangle)
        else:
            if content_surface:
                content_rect = content_surface.get_rect(center=(WIDTH//2, HEIGHT//2))
                screen.blit(content_surface, content_rect)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if content_button.collidepoint(event.pos):
                    path = load_file()
                    if path:
                        content_path = path
                        content_surface = load_content_surface(path)
                        print("Content selected:", content_path)

                elif mask_button.collidepoint(event.pos):
                    path = load_file()
                    if path:
                        mask_path = path
                        print("Mask selected:", mask_path)

                elif style1_button.collidepoint(event.pos):
                    path = load_file()
                    if path:
                        style1_path = path
                        print("Style #1 selected:", style1_path)

                elif style2_button.collidepoint(event.pos):
                    path = load_file()
                    if path:
                        style2_path = path
                        print("Style #2 selected:", style2_path)

                elif style3_button.collidepoint(event.pos):
                    path = load_file()
                    if path:
                        style3_path = path
                        print("Style #3 selected:", style3_path)

                elif run_button.collidepoint(event.pos):
                    if not pipeline_running:
                        if all([content_path, mask_path, style1_path, style2_path, style3_path]):
                            threading.Thread(target=spatial_pipeline, daemon=True).start()
                        else:
                            print("Please select content, mask, style1, style2, style3 first.")
                elif back_button.collidepoint(event.pos):
                    active = False
                    if main_callback:
                        main_callback()

        pygame.display.flip()

def draw_text(screen, txt, center_pos, color=(0,0,0), font_size=20):
    import pygame
    font = pygame.font.SysFont(None, font_size)
    surface = font.render(txt, True, color)
    rect = surface.get_rect(center=center_pos)
    screen.blit(surface, rect)