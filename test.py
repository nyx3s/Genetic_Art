import os
import numpy as np
from numpy.random import choice, random , normal
from colour import Color
import pygame

class Population:
    def __init__(self, path):
        """ Load in the reference image and create a surface we can draw on. """
        pygame.init()
        pygame.display.set_mode((500,500))
        self.ref = pygame.surfarray.pixels3d(pygame.image.load(path))
        w, h, d = self.ref.shape
        self.screen = pygame.Surface((w, h))
        #self.screen.fill((255, 255, 255))
        
        self.best_organism = None


p = Population('download.jpeg')
while True:
    pygame.display.flip()

