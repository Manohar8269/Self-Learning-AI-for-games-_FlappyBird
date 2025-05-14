import pygame
import neat
import os
import csv
from flappy_bird import Bird, Pipe, Base

pygame.font.init()

WIN_WIDTH = 800
WIN_HEIGHT = 800
STAT_FONT = pygame.font.SysFont("comicsans", 20)

WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird AI")

gen = 0

def log_ai_score(score):
    filename = 'ai_scores.csv'
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['score'])
        writer.writerow([score])


def draw_window(win, birds, pipes, base, score, gen):
    win.fill((135, 206, 235)) 
    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)
    for bird in birds:
        bird.draw(win)

    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    text = STAT_FONT.render("Gen: " + str(gen), 1, (255, 255, 255))
    win.blit(text, (10, 10))

    text = STAT_FONT.render("Alive: " + str(len(birds)), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 90, 35))

    if len(birds) > 0:
        output = round(birds[0].last_output, 3) if hasattr(birds[0], "last_output") else 0
        text = STAT_FONT.render(f"Output: {output}", 1, (255, 255, 255))
        win.blit(text, (WIN_WIDTH - 120, 60))

    pygame.display.update()


def main(genomes, config):
    global WIN, gen
    gen += 1

    nets = []
    ge = []
    birds = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        g.fitness = 0
        ge.append(g)

    print(f"[GEN {gen}] Starting with {len(birds)} birds.")

    base = Base(730)
    pipes = [Pipe(600)]
    score = 0

    clock = pygame.time.Clock()
    run = True

    dataset = open("flappy_dataset.csv", mode="w", newline='')
    writer = csv.writer(dataset)
    writer.writerow(["bird_y", "dist_top", "dist_bottom", "velocity", "dist_to_pipe", "action"])

    try:
        while run and len(birds) > 0:
            clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    pygame.quit()
                    quit()

            pipe_ind = 0
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1

            for x, bird in enumerate(birds):
                bird.move()
                ge[x].fitness += 0.1

                bird_y = bird.y
                dist_top = abs(bird.y - pipes[pipe_ind].height)
                dist_bottom = abs(bird.y - pipes[pipe_ind].bottom)
                velocity = bird.vel
                dist_to_pipe = pipes[pipe_ind].x - bird.x

                output = nets[x].activate((bird_y, dist_top, dist_bottom))
                action = 1 if output[0] > 0.5 else 0

                bird.last_output = output[0]

                if action:
                    bird.jump()

                writer.writerow([bird_y, dist_top, dist_bottom, velocity, dist_to_pipe, action])

            base.move()

            rem = []
            add_pipe = False

            for pipe in pipes:
                pipe.move()
                if len(birds) == 0:
                    break

                for x in range(len(birds)-1, -1, -1):
                    bird = birds[x]
                    if pipe.collide(bird):
                        ge[x].fitness -= 1
                        birds.pop(x)
                        nets.pop(x)
                        ge.pop(x)

                if not pipe.passed and len(birds) > 0 and pipe.x < birds[0].x:
                    pipe.passed = True
                    add_pipe = True

                if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                    rem.append(pipe)

            if add_pipe:
                score += 1
                for g in ge:
                    g.fitness += 5
                pipes.append(Pipe(600))

                if score == 1 and len(birds) > 1:
                    birds = [birds[0]]
                    nets = [nets[0]]
                    ge = [ge[0]]
                    print("[INFO] First pipe passed – keeping only one bird alive.")

            for r in rem:
                pipes.remove(r)

            for x in range(len(birds)-1, -1, -1):
                bird = birds[x]
                if bird.y + bird.img.get_height() >= 760 or bird.y < 0:
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

            draw_window(WIN, birds, pipes, base, score, gen)

    finally:
        dataset.close()
        log_ai_score(score)
        print(f"[GEN {gen}] Final Score Logged: {score}")


def run(config_file):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())

    winner = p.run(main, 50)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
