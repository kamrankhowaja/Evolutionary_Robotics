import numpy as np
import matplotlib.pyplot as plt
from ANN_XOR import ANN_XOR

POP_SIZE = 1000
GENERATIONS = 2000
MUTATION_STD = 0.1
ELITE_SIZE = 10

def create_child_from_parent(parent, mutation_std=MUTATION_STD):
    """
    Make a new ANN_XOR whose weights are a slightly mutated copy
    of the parent weights.
    """
    child = ANN_XOR()

    child.W1 = parent.W1 + np.random.normal(0, mutation_std, parent.W1.shape)
    child.b1 = parent.b1 + np.random.normal(0, mutation_std, parent.b1.shape)
    child.W2 = parent.W2 + np.random.normal(0, mutation_std, parent.W2.shape)
    child.b2 = parent.b2 + np.random.normal(0, mutation_std, parent.b2.shape)

    return child

def run_evolution():
    #random population
    population = [ANN_XOR() for _ in range(POP_SIZE)]

    best_ann = None
    best_fitness = -1.0

    for gen in range(GENERATIONS):
        # evaluate fitness of all individuals
        fitnesses = np.array([ind.fitness_function() for ind in population])

        # track the best one
        gen_best_idx = np.argmax(fitnesses)
        gen_best_fit = fitnesses[gen_best_idx]

        # update overall best
        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_ann = population[gen_best_idx]

        print(f"Generation {gen} - best fitness: {gen_best_fit:.4f}")

        # Check if its fitness is good enough to stop
        if best_fitness > 0.90:
            print("Target fitness reached!")
            break

        # selection: pick ELITE_SIZE best as parents
        elite_indices = fitnesses.argsort()[-ELITE_SIZE:]
        elites = [population[i] for i in elite_indices]
        new_population = []
        new_population.extend(elites)

        # fill the rest with children of random elites
        while len(new_population) < POP_SIZE:
            parent = np.random.choice(elites)
            child = create_child_from_parent(parent)
            new_population.append(child)

        population = new_population

    return best_ann, best_fitness

def plot_output_surface(ann: ANN_XOR, resolution=50):
    """
    Plot ANN output over the continuous input space [0,1]x[0,1].
    """
    xs = np.linspace(0, 1, resolution)
    ys = np.linspace(0, 1, resolution)

    # generate grid of all (a,b) pairs
    grid = np.array([[x, y] for x in xs for y in ys])
    outputs = ann.forward(grid)  # shape (resolution^2, 1)
    Z = outputs.reshape(resolution, resolution)

    X, Y = np.meshgrid(xs, ys)

    # 3D surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis")
    ax.set_xlabel("Input a")
    ax.set_ylabel("Input b")
    ax.set_zlabel("ANN(a,b)")
    ax.set_title("XOR ANN output over [0,1]x[0,1]")
    plt.savefig("xor_ann.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)

    best_ann, best_fit = run_evolution()
    print(f"Best fitness found: {best_fit:.4f}")

    plot_output_surface(best_ann)