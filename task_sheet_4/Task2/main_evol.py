from evol_classes import create_walls, Vector2, run_multiple_ea_and_plot_fitness
import matplotlib.pyplot as plt

if __name__ == "__main__":
    WIDTH, HEIGHT = 800, 600
    walls = create_walls(WIDTH, HEIGHT)
    start_pos = Vector2(WIDTH * 0.1, HEIGHT * 0.1)
    start_heading = 0.0

    # parameters you can tune:
    N_RUNS = 5
    GENERATIONS = 50
    POP_SIZE = 50
    EVAL_TIME = 2000  # frames per evaluation

    run_multiple_ea_and_plot_fitness(
        n_runs=N_RUNS,
        width=WIDTH,
        height=HEIGHT,
        walls=walls,
        generations=GENERATIONS,
        pop_size=POP_SIZE,
        eval_time=EVAL_TIME,
        start_pos=start_pos,
        start_heading=start_heading,
        out_path="./fitness_runs.png"
    )
