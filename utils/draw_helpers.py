import pygame
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import cv2
import os
import random

# Screen Dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Image/Video Loader")

# Colors
DARK_GREY = (59, 58, 57)
BLACK = (0, 0, 0)
LIGHT_GREY = (200, 200, 200, 128)
FONT = pygame.font.Font(None, 24)


def display_image_with_style(content_image, style_image):
    """Display the content image with the style image in the top right corner, maintaining aspect ratio."""
    if content_image:
        content_width, content_height = content_image.size
        scale_factor = min(WIDTH / content_width, HEIGHT / content_height)
        new_width = int(content_width * scale_factor)
        new_height = int(content_height * scale_factor)
        content_surface = pygame.image.fromstring(
            content_image.tobytes(), content_image.size, content_image.mode
        )
        content_surface = pygame.transform.scale(
            content_surface, (new_width, new_height)
        )
        screen.blit(
            content_surface, ((WIDTH - new_width) // 2, (HEIGHT - new_height) // 2)
        )

    if style_image:
        style_width, style_height = style_image.size
        scale_factor = min(200 / style_width, 200 / style_height)
        new_width = int(style_width * scale_factor)
        new_height = int(style_height * scale_factor)
        style_surface = pygame.image.fromstring(
            style_image.tobytes(), style_image.size, style_image.mode
        )
        style_surface = pygame.transform.scale(style_surface, (new_width, new_height))
        style_border_rect = pygame.Rect(WIDTH - 205, 15, new_width, new_height)
        pygame.draw.rect(
            screen, BLACK, style_border_rect.inflate(10, 10), 5
        )  # Black border around actual dimensions
        screen.blit(style_surface, (WIDTH - 205, 15))


def blit_text(surface, text, rect, max_rect):
    words = [
        word.split(" ") for word in text.splitlines()
    ]  # 2D array where each row is a list of words.
    space = FONT.size(" ")[0]  # The width of a space.
    max_width, max_height = max_rect.width, max_rect.height
    x, y = rect.topleft
    for line in words:
        for word in line:
            word_surface = FONT.render(word, True, BLACK)
            word_width, word_height = word_surface.get_size()
            if x + word_width >= max_width:
                x = rect.topleft[0]  # Reset the x.
                y += word_height  # Start on new row.
            surface.blit(word_surface, (x, y))
            x += word_width + space
        x = rect.topleft[0]  # Reset the x.
        y += word_height


def draw_button_box(rect, text, max_rect=None):
    """Draw a rounded box with text wrapped inside using blit_text."""
    pygame.draw.rect(screen, LIGHT_GREY, rect, border_radius=10)
    pygame.draw.rect(screen, BLACK, rect, width=2, border_radius=10)  # Border
    padding = 10  # Padding inside the button
    text_rect = pygame.Rect(
        rect.left + padding,
        rect.top + padding,
        rect.width - 2 * padding,
        rect.height - 2 * padding,
    )
    if max_rect:
        blit_text(screen, text, text_rect, max_rect)
    else:
        text_surface = FONT.render(text, True, BLACK)
        text_rect = text_surface.get_rect(center=rect.center)
        screen.blit(text_surface, text_rect)


def draw_group_box(rect, title):
    """Draw a group box with a title."""
    pygame.draw.rect(screen, LIGHT_GREY, rect, border_radius=10)
    pygame.draw.rect(screen, BLACK, rect, width=2, border_radius=10)
    text_surface = FONT.render(title, True, BLACK)
    text_rect = text_surface.get_rect(midtop=(rect.centerx, rect.top + 5))
    screen.blit(text_surface, text_rect)


def draw_radio_button(rect, text, state, max_rect):
    """Draw a radio button with a label and its current state."""
    pygame.draw.circle(screen, BLACK, rect.center, rect.width // 2, 2)  # Circle border
    if state:
        pygame.draw.circle(screen, BLACK, rect.center, rect.width // 4)  # Filled circle

    text_surface = FONT.render(text, True, BLACK)
    text_rect = text_surface.get_rect(midleft=(rect.right + 10, rect.centery))
    blit_text(screen, text, text_rect, max_rect)


def draw_sliders(slider_rect, sliders):
    """Draw sliders and their values."""
    pygame.draw.rect(screen, LIGHT_GREY, slider_rect, border_radius=10)
    pygame.draw.rect(screen, BLACK, slider_rect, width=2, border_radius=10)  # Border

    slider_y = slider_rect.top + 20
    for slider_key, slider in sliders.items():
        pygame.draw.line(
            screen,
            BLACK,
            (slider["rect"].left, slider_y),
            (slider["rect"].right, slider_y),
            2,
        )
        normalized_value = (slider["value"] - slider["min"]) / (
            slider["max"] - slider["min"]
        )
        handle_x = slider["rect"].left + int(normalized_value * slider["rect"].width)
        handle_rect = pygame.Rect(handle_x - 5, slider_y - 5, 10, 10)
        pygame.draw.rect(screen, BLACK, handle_rect)

        # Display slider label and current value
        text_surface = FONT.render(f"{slider_key}: {slider['value']:.2f}", True, BLACK)
        screen.blit(text_surface, (slider["rect"].left, slider_y + 15))
        slider_y += 50


def handle_slider_event(event, sliders):
    """Handle slider interaction events."""
    if event.type == pygame.MOUSEBUTTONDOWN or (
        event.type == pygame.MOUSEMOTION and event.buttons[0]
    ):
        for slider_key, slider in sliders.items():
            if slider["rect"].collidepoint(event.pos):
                rel_x = event.pos[0] - slider["rect"].left
                normalized_value = max(0, min(1, rel_x / slider["rect"].width))

                sliders[slider_key]["value"] = slider["min"] + normalized_value * (
                    slider["max"] - slider["min"]
                )


def draw_text(text, font, color, surface, x, y):
    """Renders text on a given surface at a specified position."""
    text_obj = font.render(text, True, color)
    text_rect = text_obj.get_rect(center=(x, y))
    surface.blit(text_obj, text_rect)


def loading_animation(screen, stop_flag, message="Processing..."):
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 40)

    message_lines = message.split("\n")

    while not stop_flag.is_set():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                stop_flag.set()
                return

        screen.fill((255, 255, 255))

        # Cycle dots from "." to "..." every 500ms
        dots = "." * ((pygame.time.get_ticks() // 500) % 4)

        # Render multi-line message
        text_y = screen.get_height() // 2 - 60
        for line in message_lines:
            text_surface = font.render(line, True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=(screen.get_width() // 2, text_y))
            screen.blit(text_surface, text_rect)
            text_y += 40

        # Render processing animation
        dots_surface = font.render(f"Processing{dots}", True, (0, 0, 0))
        dots_rect = dots_surface.get_rect(
            center=(screen.get_width() // 2, screen.get_height() // 2 + 100)
        )
        screen.blit(dots_surface, dots_rect)

        pygame.display.flip()
        clock.tick(30)  # Limit frame rate


def display_style_image(
    screen, style_image_path, screen_height, black_color, y_offset=75
):
    style_image = Image.open(style_image_path).convert("RGB")
    style_width, style_height = style_image.size
    scale_factor = min(200 / style_width, 200 / style_height)
    new_width = int(style_width * scale_factor)
    new_height = int(style_height * scale_factor)

    # Convert style image to Pygame surface
    style_surface = pygame.image.fromstring(
        style_image.tobytes(), style_image.size, style_image.mode
    )
    style_surface = pygame.transform.scale(style_surface, (new_width, new_height))

    style_x = 15
    style_y = screen_height - new_height - y_offset

    # Draw border around the image
    style_border_rect = pygame.Rect(style_x, style_y, new_width, new_height)
    pygame.draw.rect(screen, black_color, style_border_rect.inflate(10, 10), 5)

    screen.blit(style_surface, (style_x, style_y))


def display_image(
    screen, image_path, display_x, display_y, image_display_width, image_display_height
):
    """Displays an image inside a fixed box while maintaining aspect ratio."""
    # Load image (if given a path, otherwise assume it's a PIL image)
    image = Image.open(image_path) if isinstance(image_path, str) else image_path

    # Resize to fit the display area
    image = image.resize((image_display_width, image_display_height))

    image_surface = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
    img_x = display_x + (image_display_width - image.size[0]) // 2
    img_y = display_y + (image_display_height - image.size[1]) // 2
    screen.blit(image_surface, (img_x, img_y))


def get_random_file(directory, valid_extensions=(".png", ".jpg", ".jpeg")):
    files = [f for f in os.listdir(directory) if f.lower().endswith(valid_extensions)]
    return os.path.join(directory, random.choice(files)) if files else None


def open_file_dialog():
    """Opens a file dialog for selecting an image and returns the file path."""
    root = tk.Tk()
    root.withdraw()  # Hide the root Tkinter window
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]
    )
    return file_path if file_path else None


def open_video_file():
    """Opens a file dialog for selecting an MP4 video file and returns the file path."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("MP4 Video files", "*.mp4")])
    return file_path if file_path else None


def play_video(screen, video_path):
    """Plays an MP4 video in the background of the Pygame window, has ESC key exit."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    clock = pygame.time.Clock()
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    screen_width, screen_height = screen.get_size()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                print("ESC pressed. Stopping video playback.")
                running = False

        ret, frame = cap.read()
        if not ret:
            print("Video playback finished.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Rotate 90 degrees cause pygame is acting weird
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        frame_surface = pygame.surfarray.make_surface(frame)

        frame_surface = pygame.transform.scale(
            frame_surface, (screen_width, screen_height)
        )

        screen.blit(frame_surface, (0, 0))
        pygame.display.update()

        clock.tick(fps)

    cap.release()
