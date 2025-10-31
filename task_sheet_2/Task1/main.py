from hill_climbing import hill_climber


def main():
    goal_state = "charles darwin was always seasick"
    climber = hill_climber(goal_state)
    print("Initial state: ", climber.current_state)
    climber.run_iteration()
    print("Updated state: ", climber.current_state)


    #For Visualization 
    # climber = hill_climber("charles darwin was always seasick")
    # climber.print_landscape_analysis(num_samples=5000)
    # results = climber.analyze_fitness_landscape(num_samples=10000)
    # print(results['statistics'])

if __name__ == "__main__":
    main()