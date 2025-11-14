import pandas as pd
from math import exp
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt

def load_file(file_path):
    # 4 columns: example number, class, x, y
    col_names = ['example_id', 'label', 'x', 'y']
    data = pd.read_csv(
        file_path,
        sep=r"\s+",          # one or more spaces / tabs
        header=None,         # no header row in file
        names=col_names
    )
    print(f"File '{file_path}' loaded successfully.")
    return data

def evaluate_ann_on_data(ann, weights, data):
    correct = 0

    for _, row in data.iterrows():
        x_val = row["x"]
        y_val = row["y"]
        true_label = row["label"]

        pred_label = ann.predict(x_val, y_val, weights)

        if pred_label == true_label:
            correct += 1

    total = len(data)
    accuracy = correct / total
    return accuracy
    # print(f"Correct: {correct} out of {total} (accuracy = {accuracy:.3f})")


class ANN():
    def __init__(self, inputs = 2, outputs = 1, bias = 1.0):
        self.inputs = inputs
        self.output = outputs
        self.bias = bias
        self.activation_value = 0.0

    def weighted_sum(self, weights, x, y):
        total = self.bias * weights[0] + weights[1] * x + weights[2] * y
        return total

    def activation_function(self, x):
        denomonator = 1 + exp(-2*x)
        self.activation_value = (2 / denomonator) - 1
        return self.activation_value
    
    def predict(self, x, y, weights):
        net = self.weighted_sum(weights, x, y)
        a = self.activation_function(net)
        # print(a)
        if a < 0:
            # print("Class 0")
            return 0
        else:
            # print("Class 1")
            return 1


#### TASK1 CODE using FOR TASK 2 RECAP ####
@dataclass
class Individual:
    weights: np.ndarray    # shape (3,)
    fitness: float = 0.0

    def calculate_fitness(self, ann, data):
        self.fitness = evaluate_ann_on_data(ann, self.weights, data)

class EvolutionaryAlgorithmANN:
    
    def __init__(self, 
                 ann,
                 data,
                 population_size: int = 50,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.7,
                 weight_bounds: Tuple[float, float] = (-5.0, 5.0),
                 elitism_count: int = 1,
                 per_gene_mutation: bool = True):
        self.ann = ann
        self.data = data
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.weight_bounds = weight_bounds
        self.elitism_count = elitism_count
        self.per_gene_mutation = per_gene_mutation
        self.population: List[Individual] = []
        self.best_individual: Individual | None = None
        self.history = {'best_fitness': [], 'avg_fitness': [], 'best_weights': []}

    
    def initialize_population(self):
        self.population = []
        low, high = self.weight_bounds
        for _ in range(self.population_size):
            weights = np.random.uniform(low, high, size=3)
            ind = Individual(weights=weights)
            ind.calculate_fitness(self.ann, self.data)
            self.population.append(ind)
        self.update_best()
    
    def update_best(self):
        current_best = max(self.population, key=lambda ind: ind.fitness)
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            # Deep copy
            self.best_individual = Individual(weights=current_best.weights.copy(),
                                              fitness=current_best.fitness)

    def tournament_selection(self, k: int = 3) -> Individual:
        """Tournament selection: pick k random, return the best of them."""
        # randomly pick k indices
        indices = np.random.randint(0, len(self.population), size=k)
        best = None
        for idx in indices:
            cand = self.population[idx]
            if best is None or cand.fitness > best.fitness:
                best = cand
        return best

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        if np.random.random() < self.crossover_rate:
            alpha = np.random.random()
            child1_weights = alpha * parent1.weights + (1 - alpha) * parent2.weights
            child2_weights = (1 - alpha) * parent1.weights + alpha * parent2.weights
            child1 = Individual(weights=child1_weights)
            child2 = Individual(weights=child2_weights)
        else:
            child1 = Individual(weights=parent1.weights.copy())
            child2 = Individual(weights=parent2.weights.copy())
        return child1, child2

    def mutate(self, individual: Individual):
        """Mutate individual's weights with Gaussian noise."""
        low, high = self.weight_bounds
        sigma = (high - low) * 0.1
        
        if self.per_gene_mutation:
            # Mutate each weight independently
            for i in range(len(individual.weights)):
                if np.random.random() < self.mutation_rate:
                    individual.weights[i] += np.random.normal(0, sigma)
        else:
            # Mutate all weights together
            if np.random.random() < self.mutation_rate:
                individual.weights += np.random.normal(0, sigma, size=3)
        
        individual.weights = np.clip(individual.weights, low, high)

    def evolve_generation(self):
        """Evolve one generation."""
        # Sort by fitness descending
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)

        # Elitism: preserve best individuals
        new_population: List[Individual] = []
        for i in range(self.elitism_count):
            elite = self.population[i]
            new_population.append(Individual(
                weights=elite.weights.copy(),
                fitness=elite.fitness
            ))
        
        # Generate offspring
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection(k=3)
            parent2 = self.tournament_selection(k=3)

            child1, child2 = self.crossover(parent1, parent2)
            self.mutate(child1)
            self.mutate(child2)

            child1.calculate_fitness(self.ann, self.data)
            child2.calculate_fitness(self.ann, self.data)

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        self.population = new_population
        self.update_best()

    def run(self, generations: int, verbose: bool = True):
        """Run the evolutionary algorithm."""
        self.initialize_population()

        # Log generation 0
        best_fitness = self.best_individual.fitness
        avg_fitness = np.mean([ind.fitness for ind in self.population])
        self.history['best_fitness'].append(best_fitness)
        self.history['avg_fitness'].append(avg_fitness)
        self.history['best_weights'].append(self.best_individual.weights.copy())

        if verbose:
            print(f"Generation 0: Best = {best_fitness:.4f}, Avg = {avg_fitness:.4f}")

        # Evolve
        for gen in range(1, generations + 1):
            self.evolve_generation()

            best_fitness = self.best_individual.fitness
            avg_fitness = np.mean([ind.fitness for ind in self.population])
            self.history['best_fitness'].append(best_fitness)
            self.history['avg_fitness'].append(avg_fitness)
            self.history['best_weights'].append(self.best_individual.weights.copy())

            if verbose and (gen % 10 == 0 or gen == generations):
                print(f"Generation {gen}: Best = {best_fitness:.4f}, Avg = {avg_fitness:.4f}")

        return self.best_individual


    def plot_convergence(self, save_path="fitness_convergence.png"):
        """Plot fitness convergence over generations."""
        plt.figure(figsize=(10, 6))
        generations = range(len(self.history['best_fitness']))
        
        plt.plot(generations, self.history['best_fitness'], 
                label='Best Fitness', linewidth=2, color='blue')
        plt.plot(generations, self.history['avg_fitness'], 
                label='Average Fitness', linewidth=2, color='orange', alpha=0.7)
        
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Fitness (Accuracy)', fontsize=12)
        plt.title('EA Convergence: Best vs Average Fitness', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved: {save_path}")
        plt.show()


def plot_data_and_decision_boundary(data, 
                                    weights,
                                    ann=None,
                                    save_path="decision_boundary.png",
                                    title="Data and ANN Decision Boundary"):
    """
    Plots dataset and the decision boundary of the ANN.
    Saves the figure to save_path.
    """
    w0, w1, w2 = weights

    plt.figure(figsize=(6, 6))

    class0 = data[data['label'] == 0]
    class1 = data[data['label'] == 1]

    plt.scatter(class0['x'], class0['y'], c='blue', label='Class 0', 
               alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    plt.scatter(class1['x'], class1['y'], c='red', label='Class 1', 
               alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    # Plot decision boundary
    x_min, x_max = data['x'].min() - 0.5, data['x'].max() + 0.5
    y_min, y_max = data['y'].min() - 0.5, data['y'].max() + 0.5
    
    if abs(w2) > 1e-8:
        # Decision boundary: w0 + w1*x + w2*y = 0 → y = -w0/w2 - (w1/w2)*x
        xs = np.linspace(x_min, x_max, 200)
        ys = -(w0 / w2) - (w1 / w2) * xs
        plt.plot(xs, ys, 'k-', linewidth=2.5, label='Decision Boundary')
    else:
        print("Warning: w2 ≈ 0. Boundary is nearly vertical.")
        if abs(w1) > 1e-8:
            x_boundary = -w0 / w1
            plt.axvline(x=x_boundary, color='k', linewidth=2.5, 
                       label='Decision Boundary')
    if ann is not None:
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        Z = np.array([[ann.predict(x, y, weights) for x, y in zip(x_row, y_row)] 
                      for x_row, y_row in zip(xx, yy)])
        plt.contourf(xx, yy, Z, alpha=0.2, levels=[-0.5, 0.5, 1.5], 
                    colors=['blue', 'red'])

    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Decision boundary plot saved: {save_path}")
    plt.show()