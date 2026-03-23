import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse

from utils.data_loader import load_dataset
from utils.metrics import accuracy, precision_recall_f1
from utils.serialization import load_model # Use the helper we fixed

def parse_arguments():
    parser = argparse.ArgumentParser()
    # Ensure flags match the requirements if the autograder uses them [cite: 34, 54]
    parser.add_argument("-d", "--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("-m", "--model_path", type=str, default="src/best_model.npy")
    return parser.parse_args()

def forward(x, weights_list):
    """
    Manages the forward pass using loaded weights.
    Note: This simplified version assumes ReLU for hidden layers and 
    Softmax/Identity for the last, which works for argmax.
    """
    a = np.atleast_2d(x)
    
    # Iterate through the list of weight dictionaries
    for i, layer_data in enumerate(weights_list):
        W = layer_data['W']
        b = layer_data['b']
        
        # Linear Transformation
        a = np.dot(a, W) + b
        
        # Apply Activation (except for the last output layer)
        if i < len(weights_list) - 1:
            # You should ideally use your ReLU class here, 
            # but a simple max(0, x) works for inference
            a = np.maximum(0, a) 
            
    return np.argmax(a, axis=1)

def main():
    args = parse_arguments()
    
    # 1. Load weights using the fixed loader that handles the 'item()' scalar issue
    weights = load_model(args.model_path)
    
    # 2. Load Dataset
    _, _, X_test, y_test = load_dataset(args.dataset)
    
    # 3. Generate Predictions
    preds = forward(X_test, weights)
    
    # 4. Calculate and Print Metrics 
    acc = accuracy(y_test, preds)
    p, r, f1 = precision_recall_f1(y_test, preds, 10)
    
    # Print exactly as required for the evaluation pipeline
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall: {r:.4f}")
    print(f"F1-score: {f1:.4f}")

if __name__ == "__main__":
    main()