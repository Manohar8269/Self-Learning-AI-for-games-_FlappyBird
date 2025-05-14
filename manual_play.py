import pygame
import csv
import os
from flappy_bird import Bird, Pipe, Base

pygame.font.init()

WIN_WIDTH = 500
WIN_HEIGHT = 800
STAT_FONT = pygame.font.SysFont("comicsans", 50)

WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird Manual Play")


def log_human_score(score):
    filename = 'human_scores.csv'
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['score'])
        writer.writerow([score])


def draw_window(win, bird, pipes, base, score):
    win.fill((135, 206, 235)) 

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)
    bird.draw(win)

    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    pygame.display.update()


def manual_play():
    bird = Bird(230, 350)
    base = Base(730)
    pipes = [Pipe(600)]
    clock = pygame.time.Clock()
    score = 0

    run = True
    while run:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                log_human_score(score)
                run = False
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    bird.jump()

        bird.move()
        add_pipe = False
        rem = []

        for pipe in pipes:
            pipe.move()

            if pipe.collide(bird):
                log_human_score(score)
                run = False

            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

        if add_pipe:
            score += 1
            pipes.append(Pipe(600))

        for r in rem:
            pipes.remove(r)

        if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
            log_human_score(score)
            run = False

        base.move()
        draw_window(WIN, bird, pipes, base, score)


manual_play()
