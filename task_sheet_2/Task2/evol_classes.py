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
        self.cell_size = 10  # Grid cell size in pixels
        
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
        """Compute wheel velocities using linear genome mapping"""
        # Extract genome parameters: [m0, c0, m1, c1, m2, c2]
        m0, c0, m1, c1, m2, c2 = self.genome
        
        # vl = m0 * sl + c0
        # vr = m1 * sr + c1 + m2 * sm + c2
        self.v_left = m0 * s_left + c0
        self.v_right = m1 * s_right + c1 + m2 * s_mid + c2
        
        # Clamp velocities to prevent excessive backward motion
        self.v_left = np.clip(self.v_left, -self.max_speed * 0.5, self.max_speed)
        self.v_right = np.clip(self.v_right, -self.max_speed * 0.5, self.max_speed)
    
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
    """Generate a random genome with appropriate parameter ranges"""
    # [m0, c0, m1, c1, m2, c2]
    # Initialize with bias toward forward motion
    # When sensors read 0 (free space), we want positive forward velocity
    # Intercepts (c0, c1) should be positive for forward movement
    # Slopes should be negative so obstacles (high sensor values) reduce speed
    genome = np.array([
        random.uniform(-5, -0.5),   # m0 - negative slope for left sensor
        random.uniform(0.5, 2.5),   # c0 - positive base speed for left wheel
        random.uniform(-5, -0.5),   # m1 - negative slope for right sensor
        random.uniform(0.5, 2.5),   # c1 - positive base speed for right wheel
        random.uniform(-5, -0.5),   # m2 - negative slope for middle sensor
        random.uniform(-1, 1),      # c2 - additional adjustment for right wheel
    ])
    return genome


def mutate_genome(genome: np.ndarray, mutation_rate: float = 0.3) -> np.ndarray:
    """Mutate genome by adding Gaussian noise with smarter constraints"""
    new_genome = genome.copy()
    for i in range(len(new_genome)):
        if random.random() < mutation_rate:
            # Add Gaussian noise with smaller variance for more gradual changes
            noise = random.gauss(0, 0.3)
            new_genome[i] += noise
            
            # Apply constraints based on parameter type
            if i == 0 or i == 2 or i == 4:  # Slopes (m0, m1, m2)
                # Keep slopes mostly negative to reduce speed when obstacles detected
                new_genome[i] = np.clip(new_genome[i], -8, 2)
            elif i == 1 or i == 3:  # Base intercepts (c0, c1)
                # Keep base speeds positive for forward motion
                new_genome[i] = np.clip(new_genome[i], 0, 4)
            else:  # c2 - additional adjustment
                new_genome[i] = np.clip(new_genome[i], -2, 2)
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


def hill_climber(width: int, height: int, walls: List[Wall], 
                 generations: int = 100, eval_time: int = 1000,
                 start_pos: Vector2 = None, start_heading: float = 0.0,
                 visualize_final: bool = True):
    """Hill climber evolution algorithm; optionally visualize the final best genome once."""
    if start_pos is None:
        start_pos = Vector2(width * 0.1, height * 0.1)

    # Initialize with random genome
    current_genome = random_genome()
    current_fitness, _ = evaluate_genome(current_genome, width, height, walls, 
                                         eval_time, start_pos, start_heading, visualize=False)
    print(f"Generation 0: Fitness = {current_fitness}")
    print(f"Initial genome: {current_genome}")

    fitness_history = [current_fitness]
    best_genome = current_genome.copy()
    best_fitness = current_fitness

    for gen in range(1, generations + 1):
        candidate_genome = mutate_genome(current_genome)
        candidate_fitness, _ = evaluate_genome(candidate_genome, width, height, 
                                               walls, eval_time, start_pos, start_heading, visualize=False)

        if candidate_fitness >= current_fitness:
            current_genome = candidate_genome
            current_fitness = candidate_fitness
            print(f"Generation {gen}: Fitness = {current_fitness} (ACCEPTED)")
        else:
            print(f"Generation {gen}: Fitness = {candidate_fitness} (rejected, keeping {current_fitness})")

        fitness_history.append(current_fitness)
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_genome = current_genome.copy()

    print(f"\nEvolution complete!")
    print(f"Best fitness: {best_fitness}")
    print(f"Best genome: {best_genome}")

    # One (and only) visualization run if requested
    best_robot = None
    if visualize_final:
        print("\nVisualizing best genome...")
        vis_fitness, best_robot = evaluate_genome(best_genome, width, height, walls,
                                                  eval_time, start_pos, start_heading, visualize=True)
        print(f"Final fitness (visualized): {vis_fitness}")

    return best_genome, best_fitness, fitness_history, best_robot



def visualize_evolved_behavior(genome: np.ndarray, width: int, height: int,
                               walls: List[Wall], eval_time: int = 2000,
                               start_pos: Vector2 = None, start_heading: float = 0.0):
    """Visualize the behavior of an evolved genome"""
    print("\nVisualizing evolved behavior...")
    fitness, robot = evaluate_genome(genome, width, height, walls, eval_time,
                                     start_pos, start_heading, visualize=True)
    print(f"Final fitness: {fitness}")
    save_trajectory_png(robot, walls, WIDTH, HEIGHT, "./run_trajectory.png")
    return robot

def save_trajectory_png(robot: "Robot",
                        walls: List[Wall],
                        width: int,
                        height: int,
                        out_path: str = "trajectory.png") -> None:
    """
    Save a 2D plot of the robot's trajectory and walls using matplotlib.
    """
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # invert Y to match pygame screen coords
    ax.set_aspect('equal', adjustable='box')

    # Draw walls
    for w in walls:
        ax.plot([w.starting_pt.x, w.ending_pt.x],
                [w.starting_pt.y, w.ending_pt.y], linewidth=2)

    # Draw trajectory
    if len(robot.trajectory) > 1:
        xs = [p.x for p in robot.trajectory]
        ys = [p.y for p in robot.trajectory]
        ax.plot(xs, ys, linewidth=2)

    # Draw final robot position
    ax.scatter([robot.pos.x], [robot.pos.y], s=40)

    ax.set_title("Robot Trajectory")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_multiple_and_plot(n_runs: int,
                          width: int, height: int,
                          walls,
                          generations: int,
                          eval_time: int,
                          start_pos: Vector2,
                          start_heading: float,
                          out_path: str = "./multi_run_trajectories.png") -> None:
    """
    Runs hill_climber n_runs times independently (headless), evaluates the best genome once
    per run to collect its trajectory, and plots all trajectories on one matplotlib figure.
    """
    robots = []

    for i in range(n_runs):
        # Evolve headless (no extra visual run)
        best_genome, best_fitness, _, _ = hill_climber(
            width, height, walls,
            generations=generations,
            eval_time=eval_time,
            start_pos=start_pos,
            start_heading=start_heading,
            visualize_final=False
        )
        # Evaluate the best genome once to get its trajectory (headless)
        _, robot = evaluate_genome(
            best_genome, width, height, walls,
            eval_time, start_pos, start_heading, visualize=False
        )
        robots.append(robot)
        print(f"Run {i+1}: best fitness = {best_fitness}")

    # --- Plot all trajectories together ---
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # invert Y to match pygame coordinates
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Multi-run Trajectories")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")

    # Draw walls
    for w in walls:
        ax.plot([w.starting_pt.x, w.ending_pt.x],
                [w.starting_pt.y, w.ending_pt.y],
                linewidth=2)

    # Draw trajectories, each with a different color/label
    for i, robot in enumerate(robots):
        if len(robot.trajectory) > 1:
            xs = [p.x for p in robot.trajectory]
            ys = [p.y for p in robot.trajectory]
            ax.plot(xs, ys, linewidth=2, label=f"Run {i+1}")
            ax.scatter([robot.pos.x], [robot.pos.y], s=20)  # final position marker

    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved combined plot to: {out_path}")

    