from copy import deepcopy
from itertools import combinations
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import random
from typing import Tuple
from dataclasses import dataclass
from typing import List, Dict, Any

    
@dataclass
class Bag:
    X_bin: np.ndarray[np.bool]
    y_bin: np.ndarray[np.bool]
    features: List[int]
    
    def get_mapped_data(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        X_mapped = X[self.X_bin][:, self.features]
        y_mapped = y[self.X_bin]
        return X_mapped, y_mapped
    
    def count_samples(self) -> int:
        return np.sum(self.X_bin)   
    
    def size_ratio(self) -> float:
        return self.count_samples() / len(self.X_bin)
    
    def copy(self) -> "Bag":
        return Bag(
            X_bin=deepcopy(self.X_bin),
            y_bin=deepcopy(self.y_bin),
            features=deepcopy(self.features)
        )

@dataclass
class BaggingModel:
    model: DecisionTreeClassifier
    bag: Bag
    
    def copy(self) -> "BaggingModel":
        return BaggingModel(
            model=DecisionTreeClassifier(**self.model.get_params()),
            bag=self.bag.copy()
        )


def create_bag(X) -> Bag:
    indices = np.random.choice(range(len(X)), size=int(len(X)/2), replace=False)

    X_bin = np.zeros(len(X), dtype=bool)
    X_bin[indices] = True
    y_bin = np.zeros(len(X), dtype=bool)
    y_bin[indices] = True
    
    data_features_amount = X.shape[1]
    features = np.random.choice(
            range(data_features_amount),
            size=int(np.sqrt(data_features_amount)), 
            replace=False
        )
    bag = Bag(X_bin, y_bin, features)
    return bag

def create_bags(X, bags_amount: int) -> List[Bag]:
    bags = [create_bag(X) for _ in range(bags_amount)]
    return bags

def create_model(X, y, bag: Bag) -> BaggingModel:
    X_mapped, y_mapped = bag.get_mapped_data(X, y)
    model = DecisionTreeClassifier()
    model.fit(X_mapped, y_mapped)
    return BaggingModel(model, bag)

def create_models(X, y, bags: List[Bag]) -> List[BaggingModel]:
    models = [create_model(X, y, bag) for bag in bags]
    return models

def predict(X, models: List[BaggingModel]) -> np.ndarray:
    predictions = [ model.model.predict(X[:,model.bag.features]) for model in models ]
    predictions = np.array(predictions)
    final_predictions = [np.bincount(pred).argmax() for pred in predictions.T]
    return np.array(final_predictions)

def evaluate(X, y, models: List[BaggingModel]) -> float:
    predictions = predict(X, models)
    accuracy = accuracy_score(y, predictions)
    return accuracy

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy



def q_statistic_for_ensemble(X: np.ndarray, y: np.ndarray, models: List['BaggingModel']) -> float:
    predictions = np.array([
        model.model.predict(X[:, model.bag.features]) == y
        for model in models
    ])  # (n_models, n_samples)

    n_models = len(models)
    q_values = []

    for i in range(n_models):
        for j in range(i + 1, n_models):
            p1 = predictions[i]
            p2 = predictions[j]

            n11 = np.sum(p1 & p2)
            n10 = np.sum(p1 & ~p2)
            n01 = np.sum(~p1 & p2)
            n00 = np.sum(~p1 & ~p2)

            denominator = n11 * n00 + n10 * n01
            if denominator == 0:
                continue 
            q = (n11 * n00 - n10 * n01) / denominator
            q_values.append(q)

    return float(np.mean(q_values)) if q_values else 0.0


def q_statistic_for_ensemble_fast(X: np.ndarray, y: np.ndarray, models: List['BaggingModel']) -> float:
    # Step 1: Generate all predictions as a binary matrix (correct = 1, incorrect = 0)
    n_models = len(models)
    n_samples = len(y)
    correct_matrix = np.empty((n_models, n_samples), dtype=bool)

    for idx, model in enumerate(models):
        preds = model.model.predict(X[:, model.bag.features])
        correct_matrix[idx] = preds == y

    # Step 2: Compute pairwise Q-statistics
    q_stats = []
    for i, j in combinations(range(n_models), 2):
        c_i = correct_matrix[i]
        c_j = correct_matrix[j]

        n11 = np.count_nonzero(c_i & c_j)
        n00 = np.count_nonzero(~c_i & ~c_j)
        n10 = np.count_nonzero(c_i & ~c_j)
        n01 = np.count_nonzero(~c_i & c_j)

        denom = n11 * n00 + n10 * n01
        if denom == 0:
            continue

        q = (n11 * n00 - n10 * n01) / denom
        q_stats.append(q)

    return float(np.mean(q_stats)) if q_stats else 0.0