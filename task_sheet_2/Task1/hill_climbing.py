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

    