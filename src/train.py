import sys, os
# Ensures the script can find the 'ann' and 'utils' folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np

# Core ANN components
from ann.neural_layer import Dense
from ann.activations import ReLU, Sigmoid, Tanh
from ann.objective_functions import CrossEntropy, MSE
from ann.neural_network import NeuralNetwork
from ann.optimizers import SGD, Momentum, NAG, RMSProp, Adam, Nadam

# Utility functions
from utils.data_loader import load_dataset
from utils.serialization import save_model, save_config

# Mapping strings from CLI to actual Classes
act_map = {"relu": ReLU, "sigmoid": Sigmoid, "tanh": Tanh}
opt_map = {
    "sgd": SGD, 
    "momentum": Momentum, 
    "nag": NAG, 
    "rmsprop": RMSProp, 
    "adam": Adam, 
    "nadam": Nadam
}
loss_map = {"cross_entropy": CrossEntropy, "mse": MSE}

def build_network(hidden_sizes, activation_str, init_type):
    """
    Dynamically builds the MLP architecture based on CLI inputs.
    """
    layers = []
    input_dim = 784 # Standard for MNIST (28x28)
    
    # Hidden Layers
    prev_dim = input_dim
    for h in hidden_sizes:
        layers.append(Dense(prev_dim, h, init_type))
        layers.append(act_map[activation_str]())
        prev_dim = h

    # Output Layer (10 classes for digits/clothing)
    layers.append(Dense(prev_dim, 10, init_type))
    
    return NeuralNetwork(layers)

def parse_arguments():
    """
    Defines the Command Line Interface as per Assignment requirements.
    """
    parser = argparse.ArgumentParser(description="Train MLP for Image Classification")

    # Dataset and Training Hyperparameters
    parser.add_argument("-d", "--dataset", default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-o", "--optimizer", default="adam", choices=list(opt_map.keys()))
    
    # Architecture Hyperparameters
    parser.add_argument("-nhl", "--num_layers", type=int, default=1, help="Number of hidden layers")
    # Using nargs="+" allows: -sz 128 64 32
    parser.add_argument("-sz", "--hidden_size", nargs="+", type=int, default=[128], 
                        help="Number of neurons in each hidden layer")
    
    # Activation and Loss
    parser.add_argument("-a", "--activation", default="relu", choices=list(act_map.keys()))
    parser.add_argument("-l", "--loss", default="cross_entropy", choices=list(loss_map.keys()))
    
    # Initialization and Regularization
    parser.add_argument("-w_i", "--weight_init", default="xavier", choices=["random", "xavier"])
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)

    return parser.parse_args()

def main():
    args = parse_arguments()

    # 1. Load Data
    print(f"Loading {args.dataset} dataset...")
    X_train, y_train, X_val, y_val = load_dataset(args.dataset)

    # 2. Build Model
    # Note: Use args.hidden_size directly as it is now a list thanks to nargs="+"
    model = build_network(args.hidden_size, args.activation, args.weight_init)
    
    # 3. Initialize Loss and Optimizer
    loss_fn = loss_map[args.loss]()
    optimizer = opt_map[args.optimizer]()

    print(f"Starting training: {args.optimizer} optimizer, {args.activation} activation...")

    # 4. Training Loop
    num_samples = X_train.shape[0]
    for epoch in range(args.epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(num_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_loss = 0
        num_batches = 0

        for i in range(0, num_samples, args.batch_size):
            x_batch = X_shuffled[i : i + args.batch_size]
            y_batch = y_shuffled[i : i + args.batch_size]

            # Forward pass
            logits = model.forward(x_batch)
            loss = loss_fn.forward(logits, y_batch)
            
            # Backward pass
            grad = loss_fn.backward()
            model.backward(grad)

            # Update weights (Optimizer)
            optimizer.step(model.layers, args.learning_rate, args.weight_decay)

            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {avg_loss:.4f}")

    # 5. Final Save
    # Create src directory if it doesn't exist for Gradescope compatibility
    os.makedirs("src", exist_ok=True)
    
    print("Saving model and configuration...")
    save_model(model.layers, "src/best_model.npy")
    save_config(vars(args), "src/best_config.json")
    
    print("Training complete.")

if __name__ == "__main__":
    main()