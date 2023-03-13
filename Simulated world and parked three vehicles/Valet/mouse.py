
import pygame, sys
from pygame.locals import *

def mouse_trajectory(obstacle, width, height):
    pygame.init()
    trajectory = []
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    mouse_position = (0, 0)
    drawing = False
    screen = pygame.display.set_mode((width, height))
    screen.fill(WHITE)
    pygame.display.set_caption("Valet")
    
    last_pos = None
    running = True
    count = 0
    while running:
        
        for obstacles in obstacle:
            for i in range(len(obstacles)-1):
                # print(obstacles[i])
                pygame.draw.line(screen, BLACK, obstacles[i], obstacles[i+1], 8)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEMOTION:
                if (drawing):
                    mouse_position = pygame.mouse.get_pos()
                    if last_pos is not None:
                        pygame.draw.line(screen, BLACK, last_pos, mouse_position, 1)
                        trajectory.append(mouse_position)
                    last_pos = mouse_position
            elif event.type == MOUSEBUTTONUP:
                mouse_position = (0, 0)
                drawing = False
                if count!=0:
                    running = False
            elif event.type == MOUSEBUTTONDOWN:
                drawing = True
                count = 1

        pygame.display.update()
    
    return trajectory
