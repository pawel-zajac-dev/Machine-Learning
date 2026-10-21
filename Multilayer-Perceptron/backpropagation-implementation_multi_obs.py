import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        layer_sizes: list of ints, e.g., [2, 3, 3, 1] for 2D input, two hidden 
        layers with 3 neurons each, and 1 output neuron.
        """
        self.num_layers = len(layer_sizes) - 1  # Number of weight transformations
        self.W = []
        self.b = []
        
        # Cache for intermediate values needed during backpropagation
        self.A = []
        self.Z = []
        
        # Initialize weights and biases for each layer connection
        for i in range(self.num_layers):
            # He initialization (optimal for ReLU / Leaky ReLU activations)
            scale = np.sqrt(2.0 / layer_sizes[i])
            W_layer = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            b_layer = np.zeros((1, layer_sizes[i+1]))
            
            self.W.append(W_layer)
            self.b.append(b_layer)

    @staticmethod
    def sigmoid(z):
        # np.clip prevents numerical overflow in np.exp for extreme values
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def leaky_relu(z, alpha=0.01):
        """
        Leaky ReLU is used instead of standard ReLU to prevent the 'Dying ReLU' problem.
        In small networks (e.g., 3 neurons per layer), standard ReLU outputs 0 for all 
        negative inputs, causing gradients to become completely zero and freezing weight updates.
        Leaky ReLU provides a small non-zero slope (alpha) for negative values, 
        ensuring gradients can always flow backwards during training.
        """
        return np.where(z > 0, z, alpha * z)

    @staticmethod
    def leaky_relu_derivative(z, alpha=0.01):
        """
        Derivative of Leaky ReLU: returns 1.0 for z > 0, and alpha (0.01) for z <= 0.
        This guarantees a non-zero gradient even when pre-activation values (Z) are negative.
        """
        return np.where(z > 0, 1.0, alpha)

    def forward(self, X):
        """
        X: input matrix of shape (num_samples, input_dim)
        Returns: Y_hat (network output predictions)
        """
        # A[0] stores the raw input data
        self.A = [X]    # A = [X, A1, A2, A3], where A3 = Y_pred
        self.Z = []     # Z = [Z1, Z2, Z3]
        
        for i in range(self.num_layers):
            # 1. Linear transformation: Z = A_prev @ W + b
            z = self.A[i] @ self.W[i] + self.b[i]
            self.Z.append(z)
            
            # 2. Non-linear activation
            if i == self.num_layers - 1:
                # Output layer — Sigmoid for binary classification
                a = self.sigmoid(z)
            else:
                # Hidden layers — Leaky ReLU to avoid dead neurons
                a = self.leaky_relu(z)
                
            self.A.append(a)
            
        # The last element of list A is the output prediction Y_hat
        return self.A[-1]
    
    def backward(self, y_true):
        """
        Computes gradients (dL_dW, dL_db) for each layer of the network.
        y_true: ground truth labels of shape (num_samples, output_dim)
        """
        m = y_true.shape[0]
        
        # Initialize gradient lists before the loop
        self.dL_dW = []
        self.dL_db = []
        
        # Iterate backwards from the last layer to the first
        for i in reversed(range(self.num_layers)):
            if i == self.num_layers - 1:
                # 1. Output layer (BCE + Sigmoid)
                dL_dA = (self.A[i+1] - y_true) / (self.A[i+1] * (1.0 - self.A[i+1]) + 1e-8)
                dA_dZ = self.A[i+1] * (1.0 - self.A[i+1])
                delta = dL_dA * dA_dZ
            else:
                # 2. Hidden layers (Leaky ReLU)
                dZ_dA_e = self.W[i+1].T
                dA_dZ = self.leaky_relu_derivative(self.Z[i])
                delta = (delta @ dZ_dA_e) * dA_dZ
                
            # 3. Compute weight and bias gradients for the current layer
            dL_dW = (self.A[i].T @ delta) / m
            dL_db = np.sum(delta, axis=0, keepdims=True) / m
            
            self.dL_dW.append(dL_dW)
            self.dL_db.append(dL_db)
            
        # Reverse the lists so that index 0 corresponds to layer 0 (first layer)
        self.dL_dW.reverse()
        self.dL_db.reverse()
        
        return self.dL_dW, self.dL_db

    def update_params(self, learning_rate):
        """
        Updates weights and biases using Gradient Descent.
        """
        for i in range(self.num_layers):
            self.W[i] -= learning_rate * self.dL_dW[i]
            self.b[i] -= learning_rate * self.dL_db[i]

    def train(self, X, y, epochs, learning_rate):
        """
        Training loop combining forward pass, loss computation, backward pass, and parameter updates.
        """
        for epoch in range(epochs):
            # 1. Forward pass
            y_pred = self.forward(X)

            # 2. Compute loss (Binary Cross-Entropy)
            loss = -np.mean(y * np.log(y_pred + 1e-8) + (1.0 - y) * np.log(1.0 - y_pred + 1e-8))

            # 3. Backward pass
            self.backward(y)

            # 4. Update parameters
            self.update_params(learning_rate)

            # Print progress every 200 epochs
            if epoch % 200 == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")


if __name__ == "__main__":
    # Set random seed for reproducible weight initialization
    np.random.seed(42)

    # Dataset for the XOR problem (2 inputs)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    # Network architecture: [2, 3, 3, 1]
    nn = NeuralNetwork([2, 3, 3, 1])
    
    # Training the network
    nn.train(X, y, epochs=3000, learning_rate=0.2)

    # Results after training
    print("\nGround truth labels (y):")
    print(y)
    
    print("\nRaw output probabilities (Y_hat):")
    print(np.round(nn.forward(X), 4))

    print("\nFinal classification (rounded to 0 or 1):")
    print((nn.forward(X) > 0.5).astype(int))