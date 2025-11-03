import pygame
from classes import Robot, create_walls
from pygame import Vector2


def main():
    # Constants
    WIDTH, HEIGHT = 1000, 1000
    FPS = 60
    
    # Setup display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Ray-Casting Robot Navigator")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)
    
    # Create environment
    walls = create_walls(WIDTH, HEIGHT)
    
    # Create robot
    robot = Robot(Vector2(300, 800), walls, WIDTH)
    
    # Main game loop
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # Delta time in seconds
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Reset robot
                    robot.pos = Vector2(300, 800)
                    robot.heading = 0
                    robot.state = Robot.STATE_FORWARD
                    robot.rect.center = robot.pos
        
        # Clear screen
        screen.fill((30, 30, 30))
        
        # Draw walls
        for wall in walls:
            wall.draw(screen)
        
        # Update robot
        robot.update(dt)
        
        # Draw robot (sprite system handles this efficiently)
        screen.blit(robot.image, robot.rect)
        
        # Draw rays on top
        robot.draw_rays(screen)
        
        # Draw state
        text = font.render(f"State: {robot.state}", True, (255, 255, 255))
        screen.blit(text, (10, 10))
        
        # Update display
        pygame.display.flip()
    
    pygame.quit()


if __name__ == "__main__":
    pygame.init()  # Initialize pygame BEFORE calling main()
    main()