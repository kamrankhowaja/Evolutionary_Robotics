import numpy as np

class ANN_XOR:
    def __init__(self, input_size=2, hidden_size=2, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def activation_func(self, x):
        denominator = (1 + np.exp(-2.0*x))
        return (2.0 / denominator) - 1.0
    
    def fitness_function(self):
        X_inputs = np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1]
        ], dtype=float)
        
        Targets = np.array([[0.0], [1.0], [1.0], [0.0]])
        output = self.forward(X_inputs)
        errors = np.abs(Targets - output)
        scores = 1.0 - errors

        fitness = np.mean(scores)
        return fitness
    
    def forward(self, X):
        #Hidden layer
        z1 = X @ self.W1 + self.b1
        h1 = self.activation_func(z1)

        # Output layer
        z2 = h1 @ self.W2 + self.b2
        y = self.activation_func(z2)
        return y
