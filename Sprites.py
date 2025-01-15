import pygame
import math

GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GREY = (125, 125, 125)

class Forest(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.tree_size = (20, 40)
        self.image = pygame.Surface(self.tree_size)
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class Flag(pygame.sprite.Sprite):
    def __init__(self, x=660, y=40):
        pygame.sprite.Sprite.__init__(self)
        self.flag_size = (40, 60)
        self.image = pygame.Surface(self.flag_size)
        self.image.fill(YELLOW)
        self.start_x = x
        self.start_y = y
        self.rect = self.image.get_rect(center=(self.start_x, self.start_y))
        self.rect.x = self.start_x
        self.rect.y = self.start_y

class Road(pygame.sprite.Sprite):
    def __init__(self, x=0, y=410):
        pygame.sprite.Sprite.__init__(self)
        self.road_size = (800, 40)
        self.image = pygame.Surface(self.road_size)
        self.image.fill(GREY)
        self.start_x = x
        self.start_y = y
        self.rect = self.image.get_rect(center=(self.start_x, self.start_y))
        self.rect.x = self.start_x
        self.rect.y = self.start_y

class Car(pygame.sprite.Sprite):
    def __init__(self, x=10, y=420):
        pygame.sprite.Sprite.__init__(self)
        self.car_width = 40
        self.car_height = 20
        self.car_original = pygame.image.load('E:\VS_project\prac\car.png').convert_alpha()
        self.car_original = pygame.transform.rotate(self.car_original, 180)
        self.car_original = pygame.transform.scale(self.car_original, (self.car_width, self.car_height))
        self.car_new = self.car_original
        self.start_x = x
        self.start_y = y
        self.rect = self.car_new.get_rect(center=(self.start_x, self.start_y))
        self.rect.x = self.start_x
        self.rect.y = self.start_y
        self.speed = 0
        self.angle = 0
        

    def update(self, action):
        if action == 0:
            pass
        if action == 1:
            self.rotate_car(3)
        if action == 2:
            self.rotate_car(-3)
        if action == 3:
            self.go_forward()
            self.speed = 0
        if action == 4:
            self.go_back()
            self.speed = 0
    
    def rotate_car(self, rotation_speed):
        self.angle += rotation_speed
        self.car_new = pygame.transform.rotate(self.car_original, self.angle)
        self.rect = self.car_new.get_rect(center=self.rect.center)
    
    def go_forward(self):
        self.rect.x += self.speed * math.cos(math.radians(-self.angle))
        self.rect.y += self.speed * math.sin(math.radians(-self.angle))

    def go_back(self):
        self.rect.x -= (self.speed-1) * math.cos(math.radians(self.angle))
        self.rect.y += (self.speed-1) * math.sin(math.radians(self.angle))
    
    def restart(self):
        self.angle = 0
        self.car_new = pygame.transform.rotate(self.car_original, self.angle)
        self.rect = self.car_new.get_rect(center=self.rect.center)
        self.rect.x = self.start_x
        self.rect.y = self.start_y