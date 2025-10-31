from hill_climbing import hill_climber


def main():
    goal_state = "charles darwin was always seasick"
    climber = hill_climber(goal_state)
    print("Initial state: ", climber.current_state)
    climber.run_iteration()
    print("Updated state: ", climber.current_state)

if __name__ == "__main__":
    main()