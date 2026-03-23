import numpy as np
import json
import os

def save_model(layers, path="src/best_model.npy"):
    # Ensure directory exists [cite: 11, 55]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Extract only layers that have weights (Dense layers) [cite: 52, 53]
    weights_to_save = []
    for l in layers:
        if hasattr(l, 'W') and hasattr(l, 'b'):
            weights_to_save.append({
                'W': l.W,
                'b': l.b
            })
    
    # Save as a single object 
    # We wrap it in a list to ensure it remains iterable for the autograder
    np.save(path, np.array(weights_to_save, dtype=object), allow_pickle=True)

def load_model(path="src/best_model.npy"):
    # allow_pickle=True is required for object arrays 
    data = np.load(path, allow_pickle=True)
    # If NumPy wrapped the list in a 0-d array, extract it
    if data.ndim == 0:
        return data.item()
    return data

def save_config(cfg, path="src/best_config.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        # Save hyperparameters for verification [cite: 55, 63]
        json.dump(cfg, f, indent=4)