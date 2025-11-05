from evol_classes import create_walls, Vector2, run_multiple_and_plot
import matplotlib.pyplot as plt


# Main execution
if __name__ == "__main__":
    WIDTH, HEIGHT = 800, 600
    walls = create_walls(WIDTH, HEIGHT)
    start_pos = Vector2(WIDTH * 0.1, HEIGHT * 0.1)
    start_heading = 0.0

    # 5–10 independent runs on one plot (no duplicate final runs)
    run_multiple_and_plot(
        n_runs=6,
        width=WIDTH, height=HEIGHT,
        walls=walls,
        generations=50,
        eval_time=3000,
        start_pos=start_pos,
        start_heading=start_heading,
        out_path="./multi_run_trajectories.png"
    )
