import pygame
import sys

pygame.init()
pygame.font.init()

from gui.first_page import first_page  # noqa: E402
from gui.second_page import second_page  # noqa: E402, F401

from gui.third_page import third_page  # noqa: E402, F401
from gui.fourth_page import fourth_page  # noqa: E402, F401
from gui.fifth_page import fifth_page  # noqa: E402, F401
from gui.seven_page import seven_page
from gui.six_page import six_page 
from gui.eight_page import eight_page

# Screen settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Style Transfer")


background = pygame.image.load("assets/background.png")
background = pygame.transform.scale(background, (WIDTH, HEIGHT))


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
TRANSLUCENT_WHITE = (255, 255, 255, 180)

font = pygame.font.Font(None, 50)
button_font = pygame.font.Font(None, 30)


def draw_text(text, font, color, surface, x, y, bg_color=None):
    text_obj = font.render(text, True, color, bg_color)
    text_rect = text_obj.get_rect(center=(x, y))
    surface.blit(text_obj, text_rect)


def button_action(action):
    if action is not None:
        action()


# Change these to point to the correct pages when developed
buttons = [
    {
        "text": "3DGS Pipeline",
        "pos": (200, 250),
        "size": (220, 50),
        "action": lambda: first_page(screen, WIDTH, HEIGHT, main),
    },
    {
        "text": "Pixel Art Pipeline",
        "pos": (200, 320),
        "size": (220, 50),
        "action": lambda: second_page(screen, main),
    },
    {
        "text": "Sem. Segm. Pipeline",
        "pos": (200, 390),
        "size": (220, 50),
        "action": lambda: fourth_page(screen, WIDTH, HEIGHT, main),
    },
    {
        "text": "Depth-Guided",
        "pos": (450, 390),
        "size": (220, 50),
        "action": lambda: fifth_page(screen, WIDTH, HEIGHT, main),
    },
    {
        "text": "Video Pipeline",
        "pos": (450, 250),
        "size": (220, 50),
        "action": lambda: third_page(screen, main),
    },
    {
        "text": "Style Mixer", 
        "pos": (200, 460),
        "size": (220, 50), 
        "action": lambda: seven_page(screen, WIDTH, HEIGHT, main),
    },
    {
        "text": "Style Mixing",  
        "pos": (450, 320),
        "size": (220, 50), 
        "action": lambda: six_page(screen, WIDTH, HEIGHT, main),  
    },
    {
        "text": "Spatial variation",
        "pos": (450, 460),
        "size": (220, 50),
        "action": lambda: eight_page(screen, WIDTH, HEIGHT, main),
    }
]


def main():
    running = True
    while running:
        screen.blit(background, (0, 0))
        draw_text(
            "Style Transfer", font, BLACK, screen, WIDTH // 2, 50, TRANSLUCENT_WHITE
        )

        for button in buttons:
            rect = pygame.Rect(button["pos"], button["size"])
            pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, BLACK, rect, 2)
            draw_text(
                button["text"],
                button_font,
                BLACK,
                screen,
                rect.centerx,
                rect.centery,
                TRANSLUCENT_WHITE,
            )

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                for button in buttons:
                    rect = pygame.Rect(button["pos"], button["size"])
                    if rect.collidepoint(x, y):
                        button_action(button["action"])

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
