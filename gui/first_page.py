import os
import threading
from tkinter import Tk, filedialog
import pygame
from PIL import Image
from pygame.locals import QUIT
from Style_3DGS import (
    adain_inference,
    run_3dgs_training,
    run_3dgs_rendering,
)
from utils.draw_helpers import (
    get_random_file,
    handle_slider_event,
    draw_button_box,
    draw_radio_button,
    draw_sliders,
    display_image_with_style,
)


def first_page(screen, WIDTH, HEIGHT, main_callback=None):
    CONTENT_DIR = "input/content/"
    STYLE_DIR = "input/style/"

    # Colors
    DARK_GREY = (59, 58, 57)
    BLACK = (0, 0, 0)
    WHITEISH = (240, 240, 240)
    FONT = pygame.font.Font(None, 24)

    radio_depth_aware_state = False
    radio_3d_state = False
    pipeline_running = False

    upload_button = pygame.Rect(15, 50, 140, 40)
    style_button = pygame.Rect(15, 100, 140, 40)
    run_button = pygame.Rect(WIDTH - 160, HEIGHT - 60, 140, 40)
    back_button = pygame.Rect(50, HEIGHT - 100, 100, 50)

    radio_depth_aware = pygame.Rect(15, 150, 20, 20)
    radio_3d = pygame.Rect(15, 180, 20, 20)

    slider_background_rect = pygame.Rect(180, HEIGHT - 120, 440, 100)
    info_box_rect = pygame.Rect(WIDTH // 2 - 230, 10, 300, 50)  # For messages at top

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
        Tk().withdraw()
        return filedialog.askopenfilename()

    def load_folder_dialog():
        Tk().withdraw()
        return filedialog.askdirectory()

    def get_first_image(directory, valid_extensions=(".png", ".jpg", ".jpeg")):
        files = sorted(
            [f for f in os.listdir(directory) if f.lower().endswith(valid_extensions)]
        )
        return os.path.join(directory, files[0]) if files else None

    def load_gif_frames(gif_path):
        gif_image = Image.open(gif_path)
        frames = []
        try:
            while True:
                frame = gif_image.copy()
                frames.append(
                    pygame.image.fromstring(frame.tobytes(), frame.size, frame.mode)
                )
                gif_image.seek(len(frames))  # Go to the next frame
        except EOFError:
            pass  # End of GIF
        return frames

    def run_pipeline(content_path, style_path):
        def pipeline_task(content, path):
            nonlocal pipeline_running, resulting_image, gif_frames
            resulting_image = None
            pipeline_running = True
            try:
                if radio_3d_state:
                    content_dir_name = os.path.basename(os.path.dirname(content))
                    style_name = os.path.splitext(os.path.basename(style_path))[0]
                    output_dir = os.path.join(
                        "output", f"{content_dir_name}_{style_name}"
                    )

                    run_3dgs_training(
                        source_path=content,
                        style_image=path,
                        output_folder=output_dir,
                        use_depth=radio_depth_aware_state,
                        depth_offset=control_sliders["depth_proximity_offset"]["value"]
                        if radio_depth_aware_state
                        else 0.5,
                        depth_prominence=control_sliders["depth_map_prominence"][
                            "value"
                        ]
                        if radio_depth_aware_state
                        else 20,
                    )

                    gif_path = run_3dgs_rendering(
                        style_image=path, model_path=output_dir
                    )
                    gif_frames = load_gif_frames(gif_path)
                elif radio_depth_aware_state:
                    output_path = adain_inference(
                        content,
                        path,
                        use_depth=True,
                        output="output",
                        depth_offset=control_sliders["depth_proximity_offset"]["value"],
                        depth_prominence=control_sliders["depth_map_prominence"][
                            "value"
                        ],
                    )
                    resulting_image = Image.open(output_path).convert("RGB")
            except Exception as e:
                print(f"Error running pipeline: {e}")
            finally:
                pipeline_running = False

        thread = threading.Thread(target=pipeline_task, args=(content_path, style_path))
        thread.start()

    running = True
    content_path = get_random_file(CONTENT_DIR)
    content_dir_path = None  # for 3DGS
    content_image = Image.open(content_path).convert("RGB") if content_path else None
    style_path = get_random_file(STYLE_DIR)
    style_image = Image.open(style_path).convert("RGB") if style_path else None
    resulting_image, gif_frames = None, None
    gif_index = 0
    gif_timer = 0

    while running:
        screen.fill(DARK_GREY)

        if content_image and style_image:
            if gif_frames:
                gif_timer += 1
                if gif_timer >= 60:
                    gif_index = (gif_index + 1) % len(gif_frames)
                    gif_timer = 0
                frame = gif_frames[gif_index]
                frame_rect = frame.get_rect(center=(WIDTH // 2, HEIGHT // 2))
                screen.blit(frame, frame_rect)
            else:
                display_image_with_style(
                    content_image if not resulting_image else resulting_image,
                    style_image,
                )

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if upload_button.collidepoint(event.pos):
                    if not radio_3d_state:
                        file_path = load_file_dialog()
                        if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                            content_image = Image.open(file_path).convert("RGB")
                            content_path = file_path
                            content_dir_path = None
                    else:
                        folder_path = load_folder_dialog()
                        first_image_path = get_first_image(folder_path)
                        if first_image_path:
                            content_image = Image.open(first_image_path).convert("RGB")
                            content_path = first_image_path
                            content_dir_path = folder_path

                elif style_button.collidepoint(event.pos):
                    file_path = load_file_dialog()
                    if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                        style_image = Image.open(file_path).convert("RGB")
                        style_path = file_path
                elif radio_depth_aware.collidepoint(event.pos):
                    radio_depth_aware_state = not radio_depth_aware_state
                elif radio_3d.collidepoint(event.pos):
                    radio_3d_state = not radio_3d_state
                elif run_button.collidepoint(event.pos):
                    if not pipeline_running:
                        run_pipeline(
                            content_dir_path if content_dir_path else content_path,
                            style_path,
                        )
                elif back_button.collidepoint(event.pos):
                    running = False
                    if main_callback:
                        main_callback()
            elif event.type in [
                pygame.MOUSEBUTTONDOWN,
                pygame.MOUSEMOTION,
                pygame.MOUSEBUTTONUP,
            ]:
                if radio_depth_aware_state:
                    handle_slider_event(event, control_sliders)

        # background left
        pygame.draw.rect(
            screen, WHITEISH, pygame.Rect(15, 145, 140, 80), border_radius=15
        )

        if radio_3d_state:
            pygame.draw.rect(screen, WHITEISH, info_box_rect, border_radius=10)
            text = FONT.render("Choose a folder of images to upload", True, BLACK)
            screen.blit(text, (info_box_rect.x + 10, info_box_rect.y + 15))

        draw_button_box(upload_button, "Content")
        draw_button_box(style_button, "Style")
        if pipeline_running:
            draw_button_box(run_button, "Running...")
        else:
            draw_button_box(run_button, "Run")
        draw_button_box(back_button, "Back")
        draw_radio_button(
            radio_depth_aware,
            "Depth-aware",
            radio_depth_aware_state,
            pygame.Rect(10, 140, 200, 50),
        )
        draw_radio_button(
            radio_3d, "3DGS", radio_3d_state, pygame.Rect(10, 140, 200, 50)
        )

        if radio_depth_aware_state:
            draw_sliders(slider_background_rect, control_sliders)

        pygame.display.flip()

    return
