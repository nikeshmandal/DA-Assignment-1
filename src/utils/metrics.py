import numpy as np

def accuracy(y_true, y_pred):
    """Calculates the ratio of correctly predicted samples to total samples."""
    return np.mean(y_true == y_pred)

def precision_recall_f1(y_true, y_pred, num_classes=10):
    """
    Calculates Macro-Averaged Precision, Recall, and F1-score.
    
    """
    precisions = []
    recalls = []
    f1_scores = []

    for i in range(num_classes):
        # True Positives: Predicted i and actually i
        tp = np.sum((y_pred == i) & (y_true == i))
        # False Positives: Predicted i but actually not i
        fp = np.sum((y_pred == i) & (y_true != i))
        # False Negatives: Predicted not i but actually i
        fn = np.sum((y_pred != i) & (y_true == i))

        # Handle division by zero cases
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0

        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)

    # Return macro-averaged results
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1_scores)

    return macro_precision, macro_recall, macro_f1