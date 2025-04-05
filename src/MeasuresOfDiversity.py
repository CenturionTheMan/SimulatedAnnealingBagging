import numpy as np
from itertools import combinations

def compute_contingency_table(y1, y2):
    """
    Compute the 2x2 contingency table for two classifiers.
    y1, y2: Binary vectors (1=correct, 0=incorrect).
    Returns: N11, N10, N01, N00
    """
    N11 = np.sum((y1 == 1) & (y2 == 1))
    N10 = np.sum((y1 == 1) & (y2 == 0))
    N01 = np.sum((y1 == 0) & (y2 == 1))
    N00 = np.sum((y1 == 0) & (y2 == 0))
    return N11, N10, N01, N00

def q_statistic(y1, y2):
    """
    Compute Q-statistic for two classifiers.
    """
    N11, N10, N01, N00 = compute_contingency_table(y1, y2)
    numerator = N11 * N00 - N01 * N10
    denominator = N11 * N00 + N01 * N10
    return numerator / denominator if denominator != 0 else 0.0

def average_q_statistic(ensemble_outputs):
    """
    Compute average Q-statistic for an ensemble of classifiers.
    ensemble_outputs: List of binary vectors (one per classifier).
    """
    L = len(ensemble_outputs)
    if L < 2:
        return 0.0  # No pairs to compare
    
    q_values = []
    for (i, j) in combinations(range(L), 2):
        q = q_statistic(ensemble_outputs[i], ensemble_outputs[j])
        q_values.append(q)
    
    return np.mean(q_values)

# # Example Usage
# if __name__ == "__main__":
#     # Example: 3 classifiers, 5 samples
#     # Each row is a classifier's output (1=correct, 0=incorrect)
#     y1 = np.array([1, 1, 0, 1, 0])  # Classifier 1
#     y2 = np.array([1, 0, 0, 1, 1])  # Classifier 2
#     y3 = np.array([0, 1, 0, 1, 0])  # Classifier 3
#     ensemble = [y1, y2, y3]

#     # Q-statistic for Classifier 1 vs. Classifier 2
#     q_12 = q_statistic(y1, y2)
#     print(f"Q (Classifier 1 vs. Classifier 2): {q_12:.3f}")

#     # Average Q-statistic for the ensemble
#     q_avg = average_q_statistic(ensemble)
#     print(f"Average Q-statistic for ensemble: {q_avg:.3f}")