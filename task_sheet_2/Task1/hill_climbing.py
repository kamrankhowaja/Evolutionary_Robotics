from random import choice, randint


class hill_climber():

    def __init__(self, goal_state):

        self.goal_state = goal_state
        self.current_state = list(choice([chr(i) for i in range(97, 123)] + [chr(32)]) for _ in range(len(goal_state)))
        self.current_fitness = 0
        self.random_chars = [*[chr(i) for i in range(97, 123)], chr(32)] # a-z     

    def update_state(self):

        random_chr_i_to_replace = randint(0, len(self.current_state) - 1)
        random_chr_chosen = choice(self.random_chars)
        return random_chr_i_to_replace, random_chr_chosen

    def run_iteration(self):

        while self.current_fitness < len(self.goal_state):
            self.update_fitness()
    
    def update_fitness(self):
        
        test_current_state = self.current_state.copy()

        random_chr_i_to_replace, random_chr_chosen = self.update_state()
        test_current_state[random_chr_i_to_replace] = random_chr_chosen

        test_fitness = sum(1 for i in range(len(test_current_state)) 
                        if test_current_state[i] == self.goal_state[i])

        print("Testing state: ", "".join(test_current_state))
        print("Test fitness: ", test_fitness, " | Current fitness: ", self.current_fitness)

        if test_fitness >= self.current_fitness:
            self.current_state = test_current_state.copy()
            self.current_fitness = test_fitness
            print("State accepted! New fitness:", self.current_fitness)
        else:
            print("State rejected")

    

    # def analyze_fitness_landscape(self, num_samples=10000):
    #     """
    #     Analyzes the fitness landscape by sampling random states.
        
    #     Args:
    #         num_samples: Number of random states to sample
            
    #     Returns:
    #         dict: Contains fitness distribution, statistics, and theoretical values
    #     """
    #     import math
    #     from collections import Counter
        
    #     # Helper function to calculate fitness
    #     def calc_fitness(state, goal):
    #         return sum(1 for i in range(len(state)) if state[i] == goal[i])
        
    #     # Helper function for factorial
    #     def factorial(n):
    #         if n <= 1:
    #             return 1
    #         result = 1
    #         for i in range(2, n + 1):
    #             result *= i
    #         return result
        
    #     # Sample random states
    #     fitness_counts = Counter()
    #     n = len(self.goal_state)
        
    #     for _ in range(num_samples):
    #         random_state = [choice(self.random_chars) for _ in range(n)]
    #         fitness = calc_fitness(random_state, self.goal_state)
    #         fitness_counts[fitness] += 1
        
    #     # Calculate theoretical distribution
    #     theoretical = {}
    #     total_states = 27 ** n
        
    #     for k in range(n + 1):
    #         # Binomial coefficient: C(n, k)
    #         combinations = factorial(n) / (factorial(k) * factorial(n - k))
    #         # States with k correct chars: C(n,k) * 26^(n-k)
    #         # (k positions are correct, n-k positions can be any of 26 wrong chars)
    #         num_states = combinations * (26 ** (n - k))
    #         theoretical[k] = {
    #             'count': num_states,
    #             'probability': num_states / total_states
    #         }
        
    #     # Calculate statistics
    #     sampled_fitness_values = []
    #     for fitness, count in fitness_counts.items():
    #         sampled_fitness_values.extend([fitness] * count)
        
    #     avg_fitness = sum(sampled_fitness_values) / len(sampled_fitness_values)
    #     theoretical_avg = n / 27  # Expected value for random string
        
    #     # Prepare results
    #     results = {
    #         'fitness_distribution': dict(fitness_counts),
    #         'theoretical_distribution': theoretical,
    #         'statistics': {
    #             'n': n,
    #             'total_possible_states': total_states,
    #             'samples_taken': num_samples,
    #             'average_fitness_sampled': avg_fitness,
    #             'average_fitness_theoretical': theoretical_avg,
    #             'fitness_range': (min(fitness_counts.keys()), max(fitness_counts.keys()))
    #         }
    #     }
        
    #     return results

    # def print_landscape_analysis(self, num_samples=10000):
    #     """
    #     Prints a formatted analysis of the fitness landscape.
    #     """
    #     results = self.analyze_fitness_landscape(num_samples)
    #     stats = results['statistics']
        
    #     print("\n" + "="*60)
    #     print("FITNESS LANDSCAPE ANALYSIS")
    #     print("="*60)
    #     print(f"Goal string: '{self.goal_state}'")
    #     print(f"String length: {stats['n']}")
    #     print(f"Total possible states: {stats['total_possible_states']:.2e}")
    #     print(f"Samples analyzed: {stats['samples_taken']}")
    #     print(f"\nAverage fitness (sampled): {stats['average_fitness_sampled']:.2f}")
    #     print(f"Average fitness (theoretical): {stats['average_fitness_theoretical']:.2f}")
    #     print(f"Fitness range observed: {stats['fitness_range'][0]} to {stats['fitness_range'][1]}")
        
    #     print("\n" + "-"*60)
    #     print("FITNESS DISTRIBUTION (Sampled vs Theoretical)")
    #     print("-"*60)
    #     print(f"{'Fitness':<10} {'Sampled':<15} {'Theoretical':<20} {'Prob':<10}")
    #     print("-"*60)
        
    #     fitness_dist = results['fitness_distribution']
    #     theoretical = results['theoretical_distribution']
        
    #     for fitness in sorted(set(list(fitness_dist.keys()) + list(theoretical.keys()))):
    #         sampled_count = fitness_dist.get(fitness, 0)
    #         theo = theoretical.get(fitness, {'count': 0, 'probability': 0})
    #         theo_count = theo['count']
    #         theo_prob = theo['probability']
            
    #         print(f"{fitness:<10} {sampled_count:<15} {theo_count:<20.2e} {theo_prob:<10.6f}")
        
    #     print("="*60)
    #     print("\nWHY THIS IS A 'GOOD-NATURED' PROBLEM:")
    #     print("- No local optima (single peak at fitness = n)")
    #     print("- Monotonic landscape (each correct char independently adds +1)")
    #     print("- No epistasis (character positions don't interact)")
    #     print("- Guaranteed progress with hill climbing")
    #     print("="*60 + "\n")