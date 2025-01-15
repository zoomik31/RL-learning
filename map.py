import pygame
from Sprites import *
from model import DQL
from collections import deque
import random
import torch
import numpy as np
import time
from pymongo import MongoClient

def get_collection(map_name):
    return db[map_name]

def train(model, states, actions, rewards, next_states, WIN):
    if (model.memory_len == 0):
            return
    memory_len = model.memory_len
    states, actions, rewards, next_states, WIN = model.samplebatch()

    NeuroNowAnswer = model.forward(states)
    NeuroNextAnswer = model.forward(next_states)

    predicted_now_value = NeuroNowAnswer[range(memory_len), actions]
    predicted_future_value = torch.max(NeuroNextAnswer, dim=1)[0]
    predict_target = rewards + 0.8 * predicted_future_value * WIN

    loss = model.criterion(predict_target, predicted_now_value)
    model.loses.append(loss.cpu().item())

    model.rewards_per_epoch.append(torch.sum(rewards.cpu()).item())
    model.optimizer.zero_grad()

    loss.backward()
    
    if (model.inp.weight.grad.norm() < 0.0001):
        model.inp.weight.grad.data += torch.FloatTensor([0.001]).cpu()
    model.optimizer.step()

def draw_map1(screen, clock, WIDTH, HEIGHT, WIN, EPS):
    positions = []
    model = DQL(num_layers=423)
    trees = pygame.sprite.Group()
    flag = Flag(x=660, y=40)
    car = Car(x=10, y=420)
    road = Road(x=0, y=410)
    dist = math.sqrt((car.rect.x - flag.rect.x)**2 + (car.rect.y - flag.rect.y)**2)

    positions.append(flag.rect.center[0])
    positions.append(flag.rect.center[1])

    for y in range(10, HEIGHT, 50):
            for x in range(10, WIDTH, 30):
                if ((y < 400 or y > 440)) and (580 > x):
                    tree = Forest(x, y)
                    positions.append(tree.rect.center[0])
                    positions.append(tree.rect.center[1])
                    trees.add(tree)
                else: continue
    
    collection = get_collection('map1')

    for i in range(EPOCHS):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        prev_dist = dist
        dist = math.sqrt((car.rect.x - flag.rect.x)**2 + (car.rect.y - flag.rect.y)**2)

        positions.append(car.rect.center[0])
        positions.append(car.rect.center[1])
        positions.append(car.angle)
        
        states = np.array(positions)
        state = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
        del positions[len(positions)-3:len(positions)+1]
        
        if (random.random() < EPS):
            action = random.choice(range(5))
        else:
            answer = model.forward((torch.FloatTensor(state)))
            #print(answer)
            action = torch.argmax(answer).item()
        
        screen.fill((180,255,153))
        
        car.update(action)

        positions.append(car.rect.center[0])
        positions.append(car.rect.center[1])
        positions.append(car.angle)
        
        states = np.array(positions)
        next_state = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
        del positions[len(positions)-3:len(positions)+1]
        
        if pygame.sprite.collide_rect(car, flag):
            reward = 1000
            WIN = True
            car.restart()
        if pygame.sprite.spritecollideany(car, trees):
            reward += -200
            car.restart()
        elif pygame.sprite.collide_rect(car, road):
            car.speed = 50
            if dist < prev_dist:
                reward = 50
            else:
                reward = 5
            car.speed = 3
        if car.rect.right > 800:
            car.rect.right = 800
            reward = -200
            car.restart()
        if car.rect.left < 0:
            car.rect.left = 0
            reward = -200
            car.restart()
        
        model.remember(state, action, reward, next_state, WIN)
        
        data_entry = {
            'state': state.tolist(), 
            'action': action,
            'reward': reward,
            'next_state': next_state.tolist(),
            'win': WIN,
        }
        collection.insert_one(data_entry)
        
        trees.draw(screen)
        screen.blit(flag.image, flag.rect)
        screen.blit(road.image, road.rect)
        screen.blit(car.car_new, car.rect)
        
        #print(reward)
        pygame.display.flip() 

        clock.tick(30)  
        train(model, state, action, reward, next_state, WIN)

def draw_map2(screen, clock, WIDTH, HEIGHT, WIN, EPS):
    positions = []
    model = DQL(num_layers=599)
    trees = pygame.sprite.Group()
    flag = Flag(x=710, y=250)
    car = Car(x=10, y=270)
    road = Road(x=0, y=260)
    dist = math.sqrt((car.rect.x - flag.rect.x)**2 + (car.rect.y - flag.rect.y)**2)

    positions.append(flag.rect.center[0])
    positions.append(flag.rect.center[1])

    for y in range(10, HEIGHT, 50):
            for x in range(10, WIDTH, 30):
                if ((y < 240 or y > 300)):
                    tree = Forest(x, y)
                    positions.append(tree.rect.center[0])
                    positions.append(tree.rect.center[1])
                    trees.add(tree)
                else: continue
    
    collection = get_collection('map2')

    for i in range(EPOCHS):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        prev_dist = dist
        dist = math.sqrt((car.rect.x - flag.rect.x)**2 + (car.rect.y - flag.rect.y)**2)

        positions.append(car.rect.center[0])
        positions.append(car.rect.center[1])
        positions.append(car.angle)
        
        states = np.array(positions)
        state = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
        del positions[len(positions)-3:len(positions)+1]
        
        if (random.random() < EPS):
            action = random.choice(range(5))
        else:
            answer = model.forward((torch.FloatTensor(state)))
            #print(answer)
            action = torch.argmax(answer).item()

        screen.fill((180,255,153))

        car.update(action)

        positions.append(car.rect.center[0])
        positions.append(car.rect.center[1])
        positions.append(car.angle)
        
        states = np.array(positions)
        next_state = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
        del positions[len(positions)-3:len(positions)+1]
        
        if pygame.sprite.collide_rect(car, flag):
            reward = 1000
            WIN = True
            car.restart()
        if pygame.sprite.spritecollideany(car, trees):
            reward = -200
            car.restart()
        elif car.rect.right > 800:
            car.rect.right = 800
            reward = -200
            car.restart()
        elif car.rect.left < 0:
            car.rect.left = 0
            reward = -200
            car.restart()
        if pygame.sprite.collide_rect(car, road):
            car.speed = 3
        if dist < prev_dist:
            reward = 50
        else:
            reward = 5
        
        model.remember(state, action, reward, next_state, WIN)
        
        data_entry = {
            'state': state.tolist(),  
            'action': action,
            'reward': reward,
            'next_state': next_state.tolist(),
            'win': WIN,
        }
        
        collection.insert_one(data_entry)
        
        trees.draw(screen)
        screen.blit(road.image, road.rect)
        screen.blit(flag.image, flag.rect)
        screen.blit(car.car_new, car.rect)
        #print(reward)
        
        pygame.display.flip() 
        
        clock.tick(30)  

def draw_map3(screen, clock, WIDTH, HEIGHT, WIN, EPS):
    positions = []
    model = DQL()
    trees = pygame.sprite.Group()
    flag = Flag(x=660, y=40)
    car = Car(x=10, y=500)
    road = Road(x=0, y=260)
    dist = math.sqrt((car.rect.x - flag.rect.x)**2 + (car.rect.y - flag.rect.y)**2)

    positions.append(flag.rect.center[0])
    positions.append(flag.rect.center[1])

    for y in range(10, HEIGHT, 50):
            for x in range(10, WIDTH, 30):
                if ((y < 240 ) and (580 > x)):
                    tree = Forest(x, y)
                    positions.append(tree.rect.center[0])
                    positions.append(tree.rect.center[1])
                    trees.add(tree)
                else: continue
    
    collection = get_collection('map3')
    
    for i in range(EPOCHS):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        prev_dist = dist
        dist = math.sqrt((car.rect.x - flag.rect.x)**2 + (car.rect.y - flag.rect.y)**2)

        positions.append(car.rect.center[0])
        positions.append(car.rect.center[1])
        positions.append(car.angle)
        
        states = np.array(positions)
        state = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
        del positions[len(positions)-3:len(positions)+1]
        
        if (random.random() < EPS):
            action = random.choice(range(5))
        else:
            answer = model.forward((torch.FloatTensor(state)))
            #print(answer)
            action = torch.argmax(answer).item()
        
        screen.fill((180,255,153))
        
        car.update(action)

        positions.append(car.rect.center[0])
        positions.append(car.rect.center[1])
        positions.append(car.angle)
        
        states = np.array(positions)
        next_state = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
        del positions[len(positions)-3:len(positions)+1]
        
        if pygame.sprite.collide_rect(car, flag):
            reward = 1000
            WIN = True
            car.restart()
        if pygame.sprite.spritecollideany(car, trees):
            reward = -200
            car.restart()
        elif car.rect.right > 800:
            car.rect.right = 800
            reward = -200
            car.restart()
        elif car.rect.left < 0:
            car.rect.left = 0
            reward = -200
            car.restart()
        if pygame.sprite.collide_rect(car, road):
            car.speed = 3
        if dist < prev_dist:
            reward = 50
        else:
            reward = 5
        
        model.remember(state, action, reward, next_state, WIN)
        
        data_entry = {
            'state': state.tolist(), 
            'action': action,
            'reward': reward,
            'next_state': next_state.tolist(),
            'win': WIN,
        }
        
        collection.insert_one(data_entry)
        
        trees.draw(screen)
        screen.blit(road.image, road.rect)
        screen.blit(flag.image, flag.rect)
        screen.blit(car.car_new, car.rect)
        #print(reward)
        
        pygame.display.flip() 
         
        clock.tick(30)

if __name__ == "__main__":

    # инициализируем библиотеку Pygame
    pygame.init()
    # Подклбчение к бд
    client = MongoClient("mongodb+srv://sofakonstantin45:wyVNHzgtClvsE3Ao@mycluster.6mk6i.mongodb.net/")
    db = client.MyClass  # Замените на имя вашей базы данных
        
    # определяем размеры окна и скорость игры
    WIDTH, HEIGHT = 800, 600
    FPS = 30
    EPOCHS = 1000
    EPS = 0.3
    WIN = False
    # создаем окно
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # задаем название окна
    pygame.display.set_caption("game")
    clock = pygame.time.Clock()

    font = pygame.font.Font(None, 24)
    # Cjplfybt ryjgrb
    button_map1 = pygame.Surface((150, 50))
    button_map1.fill((255, 255, 255))
    button_map2 = pygame.Surface((150, 50))
    button_map2.fill((255, 255, 255))
    button_map3 = pygame.Surface((150, 50))
    button_map3.fill((255, 255, 255))

    # Отображение текста на кнопке
    text_map1 = font.render("1 map", True, (0, 0, 0))
    text_map2 = font.render("2 map", True, (0, 0, 0))
    text_map3 = font.render("3 map", True, (0, 0, 0))
    text_rect1 = text_map1.get_rect(
        center=(button_map1.get_width() /2, 
                button_map1.get_height()/2))
    text_rect2 = text_map2.get_rect(
        center=(button_map2.get_width() /2, 
                button_map2.get_height()/2))
    text_rect3 = text_map3.get_rect(
        center=(button_map3.get_width() /2, 
                button_map3.get_height()/2))

    # Создайте объект pygame.Rect, который представляет границы кнопки
    button_rect1 = button_map1.get_rect(center=(400, 250))
    button_rect2 = button_map2.get_rect(center=(310, 320))
    button_rect3 = button_map3.get_rect(center=(490, 319))  # Отрегулируйте положение

    # Цикл игры
    while True:
        clock.tick(60)
        screen.fill((155, 255, 155))

        for event in pygame.event.get():
        # Проверка события выхода
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Проверка нажатия на кнопки с созданием карт
                if button_rect1.collidepoint(event.pos):
                    draw_map1(screen, clock, WIDTH, HEIGHT, WIN, EPS)
                if button_rect2.collidepoint(event.pos):
                    draw_map2(screen, clock, WIDTH, HEIGHT, WIN, EPS)
                if button_rect3.collidepoint(event.pos):
                    draw_map3(screen, clock, WIDTH, HEIGHT, WIN, EPS)
        #Отрисовка текста
        button_map1.blit(text_map1, text_rect1)
        button_map2.blit(text_map2, text_rect2)
        button_map3.blit(text_map3, text_rect3)
        #Отрисовка кнопок
        screen.blit(button_map1, (button_rect1.x, button_rect1.y))
        screen.blit(button_map2, (button_rect2.x, button_rect2.y))
        screen.blit(button_map3, (button_rect3.x, button_rect3.y))

        # Обновить состояние
        pygame.display.update()