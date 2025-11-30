import pygame
import math
import random
from typing import List, Optional, Tuple, Dict, Set
from pygame import Vector2
import numpy as np
import matplotlib.pyplot as plt
import os

class Wall:
    """Represents a wall segment for collision detection"""
    def __init__(self, starting_pt: Vector2, ending_pt: Vector2):
        self.starting_pt = starting_pt.copy()
        self.ending_pt = ending_pt.copy()
        self.direction = self.ending_pt - self.starting_pt
    
    def draw(self, screen: pygame.Surface):
        """Draw the wall with antialiasing"""
        pygame.draw.line(screen, (255, 255, 255), self.starting_pt, self.ending_pt, 2)


class Ray:
    """Ray for distance sensing with line segment intersection"""
    def __init__(self, local_offset: Vector2, relative_angle: float, max_length: float):
        self.local_offset = local_offset.copy()
        self.relative_angle = relative_angle
        self.max_length = max_length
        
        self.pos = Vector2(0, 0)
        self.dir = Vector2(0, 0)
        self.hit_point: Optional[Vector2] = None
        self.current_length = max_length
        
        self.origin_surf = pygame.Surface((8, 8), pygame.SRCALPHA)
        pygame.draw.circle(self.origin_surf, (255, 0, 0), (4, 4), 4)
        
        self.hit_surf = pygame.Surface((12, 12), pygame.SRCALPHA)
        pygame.draw.circle(self.hit_surf, (255, 50, 50), (6, 6), 6)
    
    def update(self, robot_pos: Vector2, robot_heading: float, walls: List[Wall]):
        """Update ray position and check for wall intersections"""
        offset_rotated = self.local_offset.rotate(-math.degrees(robot_heading))
        self.pos = robot_pos + offset_rotated
        
        total_angle = robot_heading + self.relative_angle
        self.dir = Vector2()
        self.dir.from_polar((self.max_length, -math.degrees(total_angle)))
        
        closest_point = None
        closest_distance = float('inf')
        
        for wall in walls:
            hit = self.intersect_segment(wall.starting_pt, wall.ending_pt)
            if hit and hit['dist'] < closest_distance:
                closest_distance = hit['dist']
                closest_point = hit['point']
        
        if closest_point:
            self.hit_point = closest_point.copy()
            self.current_length = closest_distance
        else:
            self.hit_point = None
            self.current_length = self.max_length
    
    def intersect_segment(self, seg_a: Vector2, seg_b: Vector2) -> Optional[Dict]:
        """Ray-segment intersection using parametric line equations"""
        p = self.pos
        r = self.dir
        q = seg_a
        s = seg_b - seg_a
        
        rxs = r.x * s.y - r.y * s.x
        
        if abs(rxs) < 1e-8:
            return None
        
        q_p = q - p
        t = (q_p.x * s.y - q_p.y * s.x) / rxs
        u = (q_p.x * r.y - q_p.y * r.x) / rxs
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection = p + r * t
            distance = p.distance_to(intersection)
            return {'point': intersection, 'dist': distance}
        
        return None
    
    def draw(self, screen: pygame.Surface):
        """Visualize the ray and hit point"""
        end = self.hit_point if self.hit_point else (self.pos + self.dir)
        pygame.draw.line(screen, (255, 255, 255, 200), self.pos, end, 2)
        screen.blit(self.origin_surf, self.origin_surf.get_rect(center=self.pos))
        
        if self.hit_point:
            screen.blit(self.hit_surf, self.hit_surf.get_rect(center=self.hit_point))

def ann_forward_from_genome(genome: np.ndarray,
                            inputs: Tuple[float, float, float]) -> np.ndarray:
    """
    3-2-2 ANN forward pass using genome as flat parameter vector.

    Genome layout:
    [ 0-5 ] -> W1 (3x2)
    [ 6-7 ] -> b1 (2,)
    [ 8-11] -> W2 (2x2)
    [12-13] -> b2 (2,)
    """
    x = np.array(inputs, dtype=float).reshape(1, 3)

    # Decode genome
    W1 = genome[0:6].reshape(3, 2)
    b1 = genome[6:8].reshape(1, 2)
    W2 = genome[8:12].reshape(2, 2)
    b2 = genome[12:14].reshape(1, 2)

    # Hidden layer
    z1 = x @ W1 + b1
    h1 = np.tanh(z1)  # simple activation

    # Output layer
    z2 = h1 @ W2 + b2
    y = np.tanh(z2)   # outputs in [-1, 1]

    return y.flatten()  # shape (2,)



class Robot(pygame.sprite.Sprite):
    """Robot with evolved linear reactive behavior and FSM obstacle avoidance"""
    
    # FSM States
    STATE_GENOME_CONTROL = "Genome Control"
    STATE_FORWARD = "Forward"
    STATE_ROTATING = "Rotating"
    STATE_STEERING_LEFT = "Steering Left"
    STATE_STEERING_RIGHT = "Steering Right"
    
    def __init__(self, pos: Vector2, walls: List[Wall], world_width: int, genome: np.ndarray):
        super().__init__()
        self.pos = pos.copy()
        self.walls = walls
        self.heading = 0.0
        self.genome = genome  # [m0, c0, m1, c1, m2, c2]
        
        # Robot dimensions
        self.length = 40
        self.width = 20
        
        # Wheel velocities
        self.v_left = 0.0
        self.v_right = 0.0
        self.max_speed = 3.0
        self.rotation_speed = 0.03
        self.speed = 2.0
        
        # FSM state
        self.state = self.STATE_GENOME_CONTROL
        self.rotation_dir = 0
        
        # Create base image
        self.base_image = pygame.Surface((self.length, self.width), pygame.SRCALPHA)
        pygame.draw.rect(self.base_image, (0, 150, 255), self.base_image.get_rect())
        pygame.draw.rect(self.base_image, (255, 255, 255), self.base_image.get_rect(), 2)
        
        self.image = self.base_image.copy()
        self.rect = self.image.get_rect(center=self.pos)
        
        # Create three forward-facing rays
        max_ray_length = 0.15 * world_width
        half_length = self.length / 2
        half_width = self.width / 2
        
        self.rays = [
            Ray(Vector2(half_length, -half_width), math.radians(25), max_ray_length),   # left
            Ray(Vector2(half_length, 0), 0, max_ray_length),                            # center
            Ray(Vector2(half_length, half_width), math.radians(-25), max_ray_length),  # right
        ]
        
        # Safe distance threshold for FSM activation
        self.safe_distance = 0.6 * self.rays[1].max_length
        
        # Grid tracking for fitness
        self.visited_cells: Set[Tuple[int, int]] = set()
        self.cell_size = 5  # Grid cell size in pixels
        
        # Trajectory tracking
        self.trajectory: List[Vector2] = []
        self.trajectory_interval = 5  # Record position every N frames
        self.frame_count = 0
    
    def get_sensor_readings(self) -> Tuple[float, float, float]:
        """Get normalized sensor readings (0 = max distance, 1 = no distance)"""
        left = 1.0 - (self.rays[0].current_length / self.rays[0].max_length)
        mid = 1.0 - (self.rays[1].current_length / self.rays[1].max_length)
        right = 1.0 - (self.rays[2].current_length / self.rays[2].max_length)
        return left, mid, right
    
    def compute_wheel_velocities(self, s_left: float, s_mid: float, s_right: float):
        """Compute wheel velocities using a 3-2-2 ANN encoded in self.genome."""
        # ANN forward: from sensors → two outputs in [-1, 1]
        ann_out = ann_forward_from_genome(self.genome, (s_left, s_mid, s_right))
        # ann_out[0] = left neuron output, ann_out[1] = right neuron output

        # Scale ANN outputs to wheel speeds
        self.v_left = float(ann_out[0]) * self.max_speed
        self.v_right = float(ann_out[1]) * self.max_speed

        #clamp, in case tanh*max_speed overshoots a bit due to mutation
        self.v_left = np.clip(self.v_left, -self.max_speed, self.max_speed)
        self.v_right = np.clip(self.v_right, -self.max_speed, self.max_speed)

    
    def update_fsm(self, dt: float):
        """Obstacle-avoidance FSM that runs only when near walls. 
        Outside obstacles we switch to Genome Control."""
        # Measurements
        left_len  = self.rays[0].current_length
        mid_len   = self.rays[1].current_length
        right_len = self.rays[2].current_length
        max_len   = self.rays[1].max_length

        # Are we near any obstacle?
        near_obstacle = (
            mid_len  < self.safe_distance or
            left_len < self.safe_distance * 0.7 or
            right_len < self.safe_distance * 0.7
        )

        # If path is clear, leave the FSM (if we were in it) and return to genome
        if not near_obstacle:
            self.state = self.STATE_GENOME_CONTROL
            return

        # If we were roaming by genome and now something is near, enter the FSM
        if self.state == self.STATE_GENOME_CONTROL:
            # Choose initial FSM state
            if mid_len < self.safe_distance:
                self.state = self.STATE_ROTATING
                self.rotation_dir = 1 if left_len > right_len else -1
            elif left_len < self.safe_distance * 0.7:
                self.state = self.STATE_STEERING_RIGHT
            elif right_len < self.safe_distance * 0.7:
                self.state = self.STATE_STEERING_LEFT
            else:
                self.state = self.STATE_FORWARD

        # --- FSM proper (your provided logic) ---
        if self.state == self.STATE_FORWARD:
            if mid_len < self.safe_distance:
                self.state = self.STATE_ROTATING
                self.rotation_dir = 1 if left_len > right_len else -1
            elif left_len < self.safe_distance * 0.7:
                self.state = self.STATE_STEERING_RIGHT
            elif right_len < self.safe_distance * 0.7:
                self.state = self.STATE_STEERING_LEFT

        elif self.state == self.STATE_ROTATING:
            self.heading += self.rotation_speed * self.rotation_dir * dt * 60
            if mid_len > 0.7 * max_len and min(left_len, right_len) > self.safe_distance * 0.5:
                self.state = self.STATE_FORWARD

        elif self.state == self.STATE_STEERING_LEFT:
            self.heading += self.rotation_speed * dt * 60
            # Return to forward when clear OR if middle gets too close
            if right_len > self.safe_distance or mid_len < self.safe_distance:
                # If mid is too close, we’ll rotate on next loop; otherwise go forward
                if mid_len < self.safe_distance:
                    self.state = self.STATE_ROTATING
                    self.rotation_dir = 1 if left_len > right_len else -1
                else:
                    self.state = self.STATE_FORWARD

        elif self.state == self.STATE_STEERING_RIGHT:
            self.heading -= self.rotation_speed * dt * 60
            if left_len > self.safe_distance or mid_len < self.safe_distance:
                if mid_len < self.safe_distance:
                    self.state = self.STATE_ROTATING
                    self.rotation_dir = 1 if left_len > right_len else -1
                else:
                    self.state = self.STATE_FORWARD

        # --- FSM movement (only when inside the FSM) ---
        half_robot_length = self.length / 2
        can_move = self.state in [self.STATE_FORWARD, self.STATE_STEERING_LEFT, self.STATE_STEERING_RIGHT]
        if can_move and mid_len > half_robot_length * 1.5:
            # self.speed is in pixels/frame at 60 FPS
            self.move_forward_if_clear(self.speed)

    
    # def update_motion_fsm(self, dt: float):
    #     """Update motion using FSM (for obstacle avoidance)"""
    #     # Get ray measurements
    #     mid_len = self.rays[1].current_length
    #     half_robot_length = self.length / 2
        
    #     # Movement during FSM states
    #     can_move = self.state in [self.STATE_STEERING_LEFT, self.STATE_STEERING_RIGHT]
        
    #     if can_move and mid_len > half_robot_length * 1.5:
    #         # Move forward with fixed speed during FSM
    #         velocity = Vector2()
    #         velocity.from_polar((2.0 * dt * 60, -math.degrees(self.heading)))
    #         new_pos = self.pos + velocity
            
    #         if not self.check_collision(new_pos):
    #             self.pos = new_pos
    #             self.rect.center = self.pos
                
    #             # Record visited cell
    #             cell_x = int(self.pos.x // self.cell_size)
    #             cell_y = int(self.pos.y // self.cell_size)
    #             self.visited_cells.add((cell_x, cell_y))
    
    def check_collision(self, new_pos: Vector2) -> bool:
        """Return True if the robot at new_pos would collide with any wall."""
        collision_radius = math.sqrt((self.length/2)**2 + (self.width/2)**2) * 0.8
        for wall in self.walls:
            dist = self.point_to_segment_distance(new_pos, wall.starting_pt, wall.ending_pt)
            if dist < collision_radius:
                return True
        return False

    def move_forward_if_clear(self, pixels_per_frame: float) -> None:
        """Move forward by pixels_per_frame if it won't collide; also mark visited cells."""
        velocity = Vector2()
        velocity.from_polar((pixels_per_frame, -math.degrees(self.heading)))
        new_pos = self.pos + velocity
        if not self.check_collision(new_pos):
            self.pos = new_pos
            self.rect.center = self.pos
            cell_x = int(self.pos.x // self.cell_size)
            cell_y = int(self.pos.y // self.cell_size)
            self.visited_cells.add((cell_x, cell_y))

    
    
    def point_to_segment_distance(self, point: Vector2, seg_a: Vector2, seg_b: Vector2) -> float:
        """Calculate minimum distance from point to line segment"""
        # Vector from seg_a to seg_b
        segment = seg_b - seg_a
        segment_length_sq = segment.length_squared()
        
        if segment_length_sq == 0:
            return point.distance_to(seg_a)
        
        # Project point onto line, clamped to segment
        t = max(0, min(1, (point - seg_a).dot(segment) / segment_length_sq))
        projection = seg_a + segment * t
        
        return point.distance_to(projection)
    
    def update(self, dt: float):
        # Update sensors
        for ray in self.rays:
            ray.update(self.pos, self.heading, self.walls)

        # Run avoidance FSM (may switch state in/out of Genome Control and may move robot)
        self.update_fsm(dt)

        # If not inside FSM (i.e., state == GENOME_CONTROL), use genome-controlled differential drive
        if self.state == self.STATE_GENOME_CONTROL:
            s_left, s_mid, s_right = self.get_sensor_readings()
            self.compute_wheel_velocities(s_left, s_mid, s_right)

            v_avg = (self.v_left + self.v_right) / 2.0
            omega = (self.v_right - self.v_left) / self.width

            new_heading = self.heading + omega * dt * 60
            velocity = Vector2()
            velocity.from_polar((v_avg * dt * 60, -math.degrees(new_heading)))
            new_pos = self.pos + velocity

            if not self.check_collision(new_pos):
                self.heading = new_heading
                self.pos = new_pos
                self.rect.center = self.pos
                cell_x = int(self.pos.x // self.cell_size)
                cell_y = int(self.pos.y // self.cell_size)
                self.visited_cells.add((cell_x, cell_y))

        # Trajectory
        self.frame_count += 1
        if self.frame_count % self.trajectory_interval == 0:
            self.trajectory.append(self.pos.copy())

        # Visual rotation
        self.image = pygame.transform.rotate(self.base_image, math.degrees(self.heading))
        self.rect = self.image.get_rect(center=self.pos)

    
    def get_fitness(self) -> int:
        """Return number of unique cells visited"""
        return len(self.visited_cells)
    
    def draw_rays(self, screen: pygame.Surface):
        """Draw all rays"""
        for ray in self.rays:
            ray.draw(screen)
    
    def draw_trajectory(self, screen: pygame.Surface):
        """Draw the robot's trajectory"""
        if len(self.trajectory) > 1:
            pygame.draw.lines(screen, (255, 255, 0), False, self.trajectory, 2)
    
    def draw_visited_grid(self, screen: pygame.Surface):
        """Draw visited grid cells"""
        for cell_x, cell_y in self.visited_cells:
            rect = pygame.Rect(
                cell_x * self.cell_size,
                cell_y * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            # Draw visited cells with semi-transparent green
            surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            pygame.draw.rect(surf, (0, 255, 100, 80), surf.get_rect())
            screen.blit(surf, rect)


def create_walls(width: int, height: int) -> List[Wall]:
    """Create the maze walls"""
    walls = [
        # Boundary walls
        Wall(Vector2(0, 0), Vector2(width, 0)),
        Wall(Vector2(width, 0), Vector2(width, height)),
        Wall(Vector2(width, height), Vector2(0, height)),
        Wall(Vector2(0, height), Vector2(0, 0)),
        
        # Interior obstacles
        Wall(Vector2(0, 0.6 * height), Vector2(0.2 * width, 0.6 * height)),
        Wall(Vector2(0.56 * width, 0.5 * height), Vector2(0.56 * width, height)),
        Wall(Vector2(0.78 * width, 0.4 * height), Vector2(width, 0.4 * height)),
    ]
    return walls


def random_genome() -> np.ndarray:
    """
    Generate a random genome representing a 3-2-2 ANN:
    - W1: (3x2) = 6
    - b1: (2)   = 2
    - W2: (2x2) = 4
    - b2: (2)   = 2
    Total: 14 parameters
    """
    # random weights and biases in range [-1.0, 1.0]
    genome = np.random.uniform(-1.0, 1.0, size=14)
    return genome


def mutate_genome(genome: np.ndarray, mutation_rate: float = 0.3) -> np.ndarray:
    """Mutate genome by adding Gaussian noise to each parameter with some probability."""
    new_genome = genome.copy()
    for i in range(len(new_genome)):
        if random.random() < mutation_rate:
            noise = random.gauss(0, 0.3)
            new_genome[i] += noise
    return new_genome

def evaluate_genome(genome: np.ndarray, width: int, height: int, 
                    walls: List[Wall], eval_time: int = 1000,
                    start_pos: Vector2 = None, start_heading: float = 0.0,
                    visualize: bool = False) -> Tuple[int, Robot]:
    """Evaluate a genome by running robot simulation"""
    if start_pos is None:
        start_pos = Vector2(width * 0.1, height * 0.1)
    
    pygame.init()
    clock = pygame.time.Clock()
    
    if visualize:
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Robot Evaluation")
        
        # Create background grid for unvisited cells
        grid_bg = pygame.Surface((width, height))
        grid_bg.fill((20, 20, 20))
        cell_size = 10
        # Draw grid lines
        for x in range(0, width, cell_size):
            pygame.draw.line(grid_bg, (40, 40, 40), (x, 0), (x, height), 1)
        for y in range(0, height, cell_size):
            pygame.draw.line(grid_bg, (40, 40, 40), (0, y), (width, y), 1)
    else:
        screen = None
        grid_bg = None
    
    robot = Robot(start_pos, walls, width, genome)
    robot.heading = start_heading
    
    # Font for displaying stats
    if visualize:
        font = pygame.font.Font(None, 36)
    
    for frame in range(eval_time):
        if visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return robot.get_fitness(), robot
            
            # Draw background grid
            screen.blit(grid_bg, (0, 0))
            
            # Draw visited cells (green overlay)
            robot.draw_visited_grid(screen)
            
            # Draw walls
            for wall in walls:
                wall.draw(screen)
            
            # Draw robot sensors and body
            robot.draw_rays(screen)
            screen.blit(robot.image, robot.rect)
            
            # Draw trajectory
            robot.draw_trajectory(screen)
            
            # Display fitness counter
            fitness_text = font.render(f'Cells: {robot.get_fitness()}', True, (255, 255, 255))
            screen.blit(fitness_text, (10, 10))
            
            # Display current state
            state_text = font.render(f'State: {robot.state}', True, (255, 255, 255))
            screen.blit(state_text, (10, 50))
            
            # Display generation progress
            progress_text = font.render(f'Frame: {frame}/{eval_time}', True, (200, 200, 200))
            screen.blit(progress_text, (10, 90))
            
            pygame.display.flip()
            clock.tick(60)
        else:
            # Headless evaluation
            pass
        
        robot.update(1/60)
    
    fitness = robot.get_fitness()
    
    if visualize:
        pygame.quit()
    
    return fitness, robot

def evolutionary_run(width: int, height: int, walls: List[Wall],
                     generations: int = 50,
                     pop_size: int = 50,
                     eval_time: int = 1000,
                     start_pos: Vector2 = None,
                     start_heading: float = 0.0):
    """
    Population-based EA:
    - population of genomes
    - evaluate all
    - keep elites
    - create new generation by mutating elites
    Returns:
        best_genome, best_fitness, best_history, avg_history
    """
    if start_pos is None:
        start_pos = Vector2(width * 0.1, height * 0.1)

    # 1) initial population
    population = [random_genome() for _ in range(pop_size)]

    best_history = []
    avg_history = []
    best_genome = None
    best_fitness = -1

    ELITE_SIZE = max(2, pop_size // 10)  # top 10% as elites

    for gen in range(generations):
        fitnesses = []

        # 2) evaluate population
        for genome in population:
            fit, _ = evaluate_genome(
                genome, width, height, walls,
                eval_time=eval_time,
                start_pos=start_pos,
                start_heading=start_heading,
                visualize=False
            )
            fitnesses.append(fit)

        fitnesses = np.array(fitnesses)
        gen_best = fitnesses.max()
        gen_avg = fitnesses.mean()

        best_history.append(gen_best)
        avg_history.append(gen_avg)

        # track global best
        if gen_best > best_fitness:
            best_fitness = gen_best
            best_genome = population[fitnesses.argmax()].copy()

        print(f"[Gen {gen}] best = {gen_best:.1f}, avg = {gen_avg:.1f}")

        # 3) selection: pick elites
        elite_indices = fitnesses.argsort()[-ELITE_SIZE:]
        elites = [population[i] for i in elite_indices]

        # 4) create next generation: elites + mutated children
        new_population = []
        new_population.extend(elites)  # elitism

        while len(new_population) < pop_size:
            parent = random.choice(elites)
            child = mutate_genome(parent)
            new_population.append(child)

        population = new_population

    print(f"Evolution finished. Best fitness = {best_fitness}")
    return best_genome, best_fitness, best_history, avg_history


def run_multiple_ea_and_plot_fitness(n_runs: int,
                                     width: int,
                                     height: int,
                                     walls: List[Wall],
                                     generations: int,
                                     pop_size: int,
                                     eval_time: int,
                                     start_pos: Vector2,
                                     start_heading: float,
                                     out_path: str = "./fitness_runs.png"):
    """
    Run evolutionary_run n_runs times independently.
    For each run, log best and average fitness per generation and plot them.
    """
    all_best_hist = []
    all_avg_hist = []

    for r in range(n_runs):
        print(f"\n=== EA run {r+1}/{n_runs} ===")
        _, best_fit, best_hist, avg_hist = evolutionary_run(
            width, height, walls,
            generations=generations,
            pop_size=pop_size,
            eval_time=eval_time,
            start_pos=start_pos,
            start_heading=start_heading
        )
        all_best_hist.append(best_hist)
        all_avg_hist.append(avg_hist)
        print(f"Run {r+1}: final best fitness = {best_fit}")

    # --- Plot ---
    gens = np.arange(generations)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i in range(n_runs):
        ax.plot(gens, all_best_hist[i], alpha=0.6, label=f"Run {i+1} best")
        ax.plot(gens, all_avg_hist[i], alpha=0.4, linestyle="--", label=f"Run {i+1} avg")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (visited cells)")
    ax.set_title("Best and average fitness per generation (multiple runs)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved fitness plot to: {out_path}")

    