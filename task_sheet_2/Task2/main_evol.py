from evol_classes import create_walls,Vector2,hill_climber,visualize_evolved_behavior

if __name__ == "__main__":
    WIDTH, HEIGHT = 800, 600
    walls = create_walls(WIDTH, HEIGHT)
    start_pos = Vector2(WIDTH * 0.1, HEIGHT * 0.1)
    start_heading = 0.0
    
    # Run hill climber
    best_genome, best_fitness, fitness_history = hill_climber(
        WIDTH, HEIGHT, walls, 
        generations=50,
        eval_time=1000,
        start_pos=start_pos,
        start_heading=start_heading
    )
    
    # Visualize best behavior
    visualize_evolved_behavior(best_genome, WIDTH, HEIGHT, walls, 
                              eval_time=2000, start_pos=start_pos, 
                              start_heading=start_heading)