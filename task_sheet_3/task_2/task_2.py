import task2_functions as task2

#Loading the data
# Task 2a 

# data = task2.load_file('task_sheet_3/data')
# data2 = task2.load_file('task_sheet_3/data2')
# print(data.head())

#Creating an ANN instance and making a prediction
#Task 2b

# ann = task2.ANN()
# weights = [0.1, -0.5, 0.3]
# task2.evaluate_ann_on_data(ann, weights, data)
# task2.evaluate_ann_on_data(ann, weights, data2)

#Task 2c
data = task2.load_file('task_sheet_3/data')
data2 = task2.load_file('task_sheet_3/data2')
ann = task2.ANN()
ea = task2.EvolutionaryAlgorithmANN(ann=ann, 
                              data=data,
                              population_size=100,
                              mutation_rate=0.3,
                              crossover_rate=0.7,
                              elitism_count=2,
                              per_gene_mutation=True)

best = ea.run(generations=100, verbose=True)
print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"Best weights: w0={best.weights[0]:.4f}, w1={best.weights[1]:.4f}, w2={best.weights[2]:.4f}")
print(f"Best accuracy: {best.fitness:.4f} ({best.fitness*100:.2f}%)")
print(f"Decision boundary equation: y = {-best.weights[0]/best.weights[2]:.4f} - {best.weights[1]/best.weights[2]:.4f}*x")
    

ea.plot_convergence()

task2.plot_data_and_decision_boundary(
        data, 
        best.weights,
        ann=ann,
        save_path="task2_decision_boundary.png",
        title=f"Decision Boundary (Accuracy: {best.fitness:.2%})"
        )
