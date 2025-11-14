import pandas as pd
from math import exp
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt

def load_file(file_path):
    """Load classification data from file."""
    col_names = ['example_id', 'label', 'x', 'y']
    data = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=col_names
    )
    print(f"File '{file_path}' loaded successfully. Shape: {data.shape}")
    return data

def evaluate_ann_on_data(ann, weights, data):
    """Evaluate ANN accuracy on dataset."""
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

class ANN():
    """Simple Artificial Neural Network with no hidden layer."""
    
    def __init__(self, inputs=2, outputs=1, bias=1.0):
        self.inputs = inputs
        self.output = outputs
        self.bias = bias
        self.activation_value = 0.0
        self.num_weights = 3  # bias + 2 inputs

    def weighted_sum(self, weights, x, y):
        """Calculate weighted sum: bias*w0 + x*w1 + y*w2"""
        total = self.bias * weights[0] + weights[1] * x + weights[2] * y
        return total

    def activation_function(self, x):
        """Hyperbolic tangent-like activation: φ(x) = 2/(1+exp(-2x)) - 1"""
        denominator = 1 + exp(-2*x)
        self.activation_value = (2 / denominator) - 1
        return self.activation_value
    
    def predict(self, x, y, weights):
        """Make binary classification prediction."""
        net = self.weighted_sum(weights, x, y)
        a = self.activation_function(net)
        return 0 if a < 0 else 1


class ANNWithHiddenLayer():
    """Artificial Neural Network with one hidden layer."""
    
    def __init__(self, inputs=2, hidden=3, outputs=1, bias=1.0):
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs
        self.bias = bias
        
        # Calculate number of weights needed
        # Hidden layer: (inputs + 1 bias) * hidden_neurons
        # Output layer: (hidden + 1 bias) * output_neurons
        self.weights_to_hidden = (inputs + 1) * hidden
        self.weights_to_output = (hidden + 1) * outputs
        self.num_weights = self.weights_to_hidden + self.weights_to_output
        
    def activation_function(self, x):
        """Hyperbolic tangent-like activation: φ(x) = 2/(1+exp(-2x)) - 1"""
        try:
            denominator = 1 + exp(-2*x)
            return (2 / denominator) - 1
        except OverflowError:
            # Handle extreme values
            return -1.0 if x < 0 else 1.0
    
    def predict(self, x, y, weights):
        """Make binary classification prediction with hidden layer."""
        # Parse weights
        # First part: weights to hidden layer
        # Second part: weights to output layer
        
        w_hidden_end = self.weights_to_hidden
        w_hidden = weights[:w_hidden_end]
        w_output = weights[w_hidden_end:]
        
        # Reshape hidden weights: each row is weights for one hidden neuron
        # [bias_weight, x_weight, y_weight] for each hidden neuron
        w_hidden = w_hidden.reshape(self.hidden, self.inputs + 1)
        
        # Calculate hidden layer activations
        hidden_activations = []
        for h in range(self.hidden):
            # net_h = bias*w[h,0] + x*w[h,1] + y*w[h,2]
            net = self.bias * w_hidden[h, 0] + x * w_hidden[h, 1] + y * w_hidden[h, 2]
            activation = self.activation_function(net)
            hidden_activations.append(activation)
        
        # Calculate output layer
        # net_output = bias*w_output[0] + sum(hidden[i] * w_output[i+1])
        net_output = self.bias * w_output[0]
        for i, h_act in enumerate(hidden_activations):
            net_output += h_act * w_output[i + 1]
        
        output_activation = self.activation_function(net_output)
        
        return 0 if output_activation < 0 else 1

@dataclass
class Individual:
    """Individual in the evolutionary algorithm."""
    weights: np.ndarray
    fitness: float = 0.0

    def calculate_fitness(self, ann, data):
        """Calculate fitness as classification accuracy."""
        self.fitness = evaluate_ann_on_data(ann, self.weights, data)

class EvolutionaryAlgorithmANN:
    """Evolutionary Algorithm for training ANN weights."""
    
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
        """Create initial random population."""
        self.population = []
        low, high = self.weight_bounds
        num_weights = self.ann.num_weights
        for _ in range(self.population_size):
            weights = np.random.uniform(low, high, size=num_weights)
            ind = Individual(weights=weights)
            ind.calculate_fitness(self.ann, self.data)
            self.population.append(ind)
        self.update_best()
    
    def update_best(self):
        """Update best individual found so far."""
        current_best = max(self.population, key=lambda ind: ind.fitness)
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = Individual(
                weights=current_best.weights.copy(),
                fitness=current_best.fitness
            )

    def tournament_selection(self, k: int = 3) -> Individual:
        """Tournament selection: pick k random individuals, return best."""
        indices = np.random.randint(0, len(self.population), size=k)
        best = None
        for idx in indices:
            cand = self.population[idx]
            if best is None or cand.fitness > best.fitness:
                best = cand
        return best

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Blend crossover (BLX-α style)."""
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
                individual.weights += np.random.normal(0, sigma, size=len(individual.weights))
        
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

def plot_data_and_decision_boundary(data, weights, ann=None, 
                                   save_path="decision_boundary.png",
                                   title="Data and ANN Decision Boundary"):
    """Plot dataset and decision boundary."""
    plt.figure(figsize=(8, 8))

    # Plot data points
    class0 = data[data['label'] == 0]
    class1 = data[data['label'] == 1]

    plt.scatter(class0['x'], class0['y'], c='blue', label='Class 0', 
               alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    plt.scatter(class1['x'], class1['y'], c='red', label='Class 1', 
               alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    # Plot decision boundary based on ANN type
    x_min, x_max = data['x'].min() - 0.5, data['x'].max() + 0.5
    y_min, y_max = data['y'].min() - 0.5, data['y'].max() + 0.5
    
    # For simple ANN (no hidden layer), plot analytical boundary
    if ann is not None and isinstance(ann, ANN):
        w0, w1, w2 = weights
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
    
    # For any ANN (including hidden layer), show classification regions
    if ann is not None:
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                             np.linspace(y_min, y_max, 150))
        Z = np.array([[ann.predict(x, y, weights) for x, y in zip(x_row, y_row)] 
                      for x_row, y_row in zip(xx, yy)])
        plt.contourf(xx, yy, Z, alpha=0.2, levels=[-0.5, 0.5, 1.5], 
                    colors=['blue', 'red'])
        
        # Add contour line for decision boundary
        plt.contour(xx, yy, Z, levels=[0.5], colors='black', 
                   linewidths=2.5, linestyles='solid')

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

def run_experiment(data_path, generations=100, population_size=100, 
                  crossover_rate=0.7, mutation_rate=0.3, use_hidden_layer=False,
                  hidden_neurons=3):
    """Run complete experiment: load data, evolve ANN, plot results."""
    
    print(f"\n{'='*60}")
    print(f"Running experiment on: {data_path}")
    print(f"Hidden layer: {use_hidden_layer}" + (f" ({hidden_neurons} neurons)" if use_hidden_layer else ""))
    print(f"{'='*60}")
    
    # Load data
    data = load_file(data_path)
    print(f"Class distribution:\n{data['label'].value_counts()}")
    
    # Create ANN
    if use_hidden_layer:
        ann = ANNWithHiddenLayer(inputs=2, hidden=hidden_neurons, outputs=1)
        print(f"ANN architecture: 2 inputs → {hidden_neurons} hidden → 1 output")
        print(f"Total weights: {ann.num_weights}")
    else:
        ann = ANN()
        print(f"ANN architecture: 2 inputs → 1 output (no hidden layer)")
        print(f"Total weights: {ann.num_weights}")
    
    # Create EA
    ea = EvolutionaryAlgorithmANN(
        ann=ann,
        data=data,
        population_size=population_size,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        elitism_count=2,  # Keep top 2
        per_gene_mutation=True
    )
    
    # Evolve
    print(f"\nEvolving for {generations} generations...")
    best = ea.run(generations=generations, verbose=True)
    
    # Results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Best weights: {best.weights}")
    print(f"Best accuracy: {best.fitness:.4f} ({best.fitness*100:.2f}%)")
    
    if not use_hidden_layer and len(best.weights) == 3:
        w0, w1, w2 = best.weights
        if abs(w2) > 1e-8:
            print(f"Decision boundary equation: y = {-w0/w2:.4f} - {w1/w2:.4f}*x")
    
    # Plot convergence
    ea.plot_convergence(save_path=f"{data_path}_convergence.png")
    
    # Plot decision boundary
    plot_data_and_decision_boundary(
        data, 
        best.weights, 
        ann=ann,
        save_path=f"{data_path}_boundary.png",
        title=f"Decision Boundary (Accuracy: {best.fitness:.2%})"
    )
    
    return ea, best

# Example usage
if __name__ == "__main__":
    # Run on data (linearly separable - no hidden layer needed)
    print("\n" + "="*70)
    print("EXPERIMENT 1: DATA (linearly separable)")
    print("="*70)
    ea1, best1 = run_experiment('task_sheet_3/data', generations=100, use_hidden_layer=False)
    
    # Run on data2 (NOT linearly separable - needs hidden layer)
    print("\n" + "="*70)
    print("EXPERIMENT 2: DATA2 (non-linearly separable)")
    print("="*70)
    ea2, best2 = run_experiment('task_sheet_3/data2', generations=200, 
                                population_size=150,
                                use_hidden_layer=True, 
                                hidden_neurons=4,
                                mutation_rate=0.25,
                                crossover_rate=0.8)