import pygame
import math
import random
from typing import List, Optional, Tuple, Dict
from pygame import Vector2



class Wall:
    """Represents a wall segment for collision detection"""
    def __init__(self, starting_pt: Vector2, ending_pt: Vector2):
        self.starting_pt = starting_pt.copy()
        self.ending_pt = ending_pt.copy()
        # Pre-calculate for performance
        self.direction = self.ending_pt - self.starting_pt
    
    def draw(self, screen: pygame.Surface):
        """Draw the wall with antialiasing"""
        pygame.draw.line(
            screen,
            (255, 255, 255),
            self.starting_pt,
            self.ending_pt,
            2
        )


class Ray:
    """Ray for distance sensing with line segment intersection"""
    def __init__(self, local_offset: Vector2, relative_angle: float, max_length: float):
        self.local_offset = local_offset.copy()
        self.relative_angle = relative_angle
        self.max_length = max_length
        
        # Current state
        self.pos = Vector2(0, 0)
        self.dir = Vector2(0, 0)
        self.hit_point: Optional[Vector2] = None
        self.current_length = max_length
        
        # Pre-create surfaces for drawing
        self.origin_surf = pygame.Surface((8, 8), pygame.SRCALPHA)
        pygame.draw.circle(self.origin_surf, (255, 0, 0), (4, 4), 4)
        
        self.hit_surf = pygame.Surface((12, 12), pygame.SRCALPHA)
        pygame.draw.circle(self.hit_surf, (255, 50, 50), (6, 6), 6)
    
    def update(self, robot_pos: Vector2, robot_heading: float, walls: List[Wall]):
        """Update ray position and check for wall intersections"""
        # Calculate ray origin in world space using pygame's rotation
        # Rotate the local offset by the robot's heading
        offset_rotated = self.local_offset.rotate(-math.degrees(robot_heading))
        self.pos = robot_pos + offset_rotated
        
        # Calculate ray direction (heading + relative angle)
        total_angle = robot_heading + self.relative_angle
        self.dir = Vector2()
        self.dir.from_polar((self.max_length, -math.degrees(total_angle)))
        
        # Find closest intersection with walls
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
        """
        Ray-segment intersection using parametric line equations.
        Returns dict with 'point' and 'dist' if intersection exists.
        """
        p = self.pos
        r = self.dir
        q = seg_a
        s = seg_b - seg_a
        
        # Calculate cross product r × s
        rxs = r.x * s.y - r.y * s.x
        
        # Parallel or collinear
        if abs(rxs) < 1e-8:
            return None
        
        q_p = q - p
        
        # Calculate t and u parameters
        t = (q_p.x * s.y - q_p.y * s.x) / rxs
        u = (q_p.x * r.y - q_p.y * r.x) / rxs
        
        # Check if intersection is within both segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection = p + r * t
            distance = p.distance_to(intersection)
            return {'point': intersection, 'dist': distance}
        
        return None
    
    def draw(self, screen: pygame.Surface):
        """Visualize the ray and hit point"""
        # Calculate ray end point
        end = self.hit_point if self.hit_point else (self.pos + self.dir)
        
        # Draw ray line with alpha
        pygame.draw.line(screen, (255, 255, 255, 200), self.pos, end, 2)
        
        # Draw ray origin
        screen.blit(self.origin_surf, self.origin_surf.get_rect(center=self.pos))
        
        # Draw hit point if it exists
        if self.hit_point:
            screen.blit(self.hit_surf, self.hit_surf.get_rect(center=self.hit_point))


class Robot(pygame.sprite.Sprite):
    """Autonomous robot with FSM-based navigation using ray sensors"""
    
    # FSM States
    STATE_FORWARD = "Forward"
    STATE_ROTATING = "Rotating"
    STATE_STEERING_LEFT = "Steering Left"
    STATE_STEERING_RIGHT = "Steering Right"
    
    def __init__(self, pos: Vector2, walls: List[Wall], world_width: int):
        super().__init__()
        self.pos = pos.copy()
        self.walls = walls
        self.heading = 0.0
        self.speed = 2.0
        self.rotation_speed = 0.05
        
        # FSM state
        self.state = self.STATE_FORWARD
        self.rotation_dir = 0
        
        # Robot dimensions
        self.length = 40
        self.width = 20
        
        # Create base image (rectangle)
        self.base_image = pygame.Surface((self.length, self.width), pygame.SRCALPHA)
        pygame.draw.rect(self.base_image, (0, 150, 255), self.base_image.get_rect())
        pygame.draw.rect(self.base_image, (255, 255, 255), self.base_image.get_rect(), 2)
        
        self.image = self.base_image.copy()
        self.rect = self.image.get_rect(center=self.pos)
        
        # Create three forward-facing rays
        # FIXED: Swapped the signs of the angles
        # Positive angle rotates clockwise (right), negative rotates counterclockwise (left)
        max_ray_length = 0.15 * world_width
        half_length = self.length / 2
        half_width = self.width / 2
        
        self.rays = [
            Ray(Vector2(half_length, -half_width), math.radians(25), max_ray_length),   # left sensor (positive angle)
            Ray(Vector2(half_length, 0), 0, max_ray_length),                            # center sensor
            Ray(Vector2(half_length, half_width), math.radians(-25), max_ray_length),  # right sensor (negative angle)
        ]
        
        self.safe_distance = 0.6 * self.rays[1].max_length
    
    def update_fsm(self, dt: float):
        """Finite State Machine logic"""
        # Get ray measurements
        left_len = self.rays[0].current_length
        mid_len = self.rays[1].current_length
        right_len = self.rays[2].current_length
        max_len = self.rays[1].max_length
        
        # State transitions
        if self.state == self.STATE_FORWARD:
            if mid_len < self.safe_distance:
                # Obstacle ahead, start rotating
                self.state = self.STATE_ROTATING
                # Choose to rotate towards the side with more space
                if left_len > right_len:
                    self.rotation_dir = 1  # Rotate left (counterclockwise, positive in our system)
                else:
                    self.rotation_dir = -1  # Rotate right (clockwise, negative)
            elif left_len < self.safe_distance * 0.7:
                # Wall getting close on left, steer right
                self.state = self.STATE_STEERING_RIGHT
            elif right_len < self.safe_distance * 0.7:
                # Wall getting close on right, steer left
                self.state = self.STATE_STEERING_LEFT
        
        elif self.state == self.STATE_ROTATING:
            self.heading += self.rotation_speed * self.rotation_dir * dt * 60
            # Exit rotation when path is clear
            if mid_len > 0.7 * max_len and min(left_len, right_len) > self.safe_distance * 0.5:
                self.state = self.STATE_FORWARD
        
        elif self.state == self.STATE_STEERING_LEFT:
            self.heading += self.rotation_speed * dt * 60  # Positive = turn left
            # Return to forward when clear or if middle sensor gets too close
            if right_len > self.safe_distance or mid_len < self.safe_distance:
                self.state = self.STATE_FORWARD
        
        elif self.state == self.STATE_STEERING_RIGHT:
            self.heading -= self.rotation_speed * dt * 60  # Negative = turn right
            # Return to forward when clear or if middle sensor gets too close
            if left_len > self.safe_distance or mid_len < self.safe_distance:
                self.state = self.STATE_FORWARD
        
        # Movement
        half_robot_length = self.length / 2
        can_move = self.state in [self.STATE_FORWARD, self.STATE_STEERING_LEFT, 
                                   self.STATE_STEERING_RIGHT]
        
        if can_move and mid_len > half_robot_length * 1.5:
            velocity = Vector2()
            velocity.from_polar((self.speed * dt * 60, -math.degrees(self.heading)))
            self.pos += velocity
            self.rect.center = self.pos
    
    def update(self, dt: float):
        """Main update loop"""
        # Update all rays
        for ray in self.rays:
            ray.update(self.pos, self.heading, self.walls)
        
        # Run FSM logic
        self.update_fsm(dt)
        
        # Update visual rotation (pygame.transform.rotate uses degrees)
        self.image = pygame.transform.rotate(self.base_image, math.degrees(self.heading))
        self.rect = self.image.get_rect(center=self.pos)
    
    def draw_rays(self, screen: pygame.Surface):
        """Draw all rays"""
        for ray in self.rays:
            ray.draw(screen)


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