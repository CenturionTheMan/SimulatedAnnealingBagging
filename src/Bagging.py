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
    X: np.ndarray[np.bool]
    y: np.ndarray[np.bool]
    features: List[int]
    features_max_amount: int
    
    def get_mapped_data(self) -> Tuple[np.ndarray, np.ndarray]:
        X_mapped = self.X if self.features_max_amount == len(self.features) else self.X[:, self.features]
        y_mapped = self.y
        return X_mapped, y_mapped
    
    def map_X_by_features(self, X_out):
        return X_out if len(self.features) == self.features_max_amount else X_out[:, self.features]
            
    def count_samples(self) -> int:
        return len(self.X)   
    
    # def size_ratio(self) -> float:
    #     return self.count_samples() / len(self.X_bin)
    
    def copy(self) -> "Bag":
        return Bag(
            X=self.X.copy(),
            y=self.y.copy(),
            features=self.features.copy(),
            features_max_amount=self.features_max_amount
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


def create_bag(X, y, replace:bool, cut_features:bool) -> Bag:
    if replace:
        indices = np.random.choice(range(len(X)), size=len(X), replace=True)
    else:
        indices = np.random.choice(range(len(X)), size=int(len(X)*0.5), replace=False)

    tmp_X = X[indices].copy()
    tmp_y = y[indices].copy()

    if cut_features:
        data_features_amount = X.shape[1]
        features = np.random.choice(
                range(data_features_amount),
                size=int(np.sqrt(data_features_amount)), 
                replace=False
            )
    else:
        features = list(range(X.shape[1]))
    bag = Bag(tmp_X, tmp_y, features, features_max_amount=X.shape[1])
    return bag

def create_bags(X, y, bags_amount: int, replace:bool=True, cut_features:bool=False) -> List[Bag]:
    bags = [create_bag(X, y, replace=replace, cut_features=cut_features) for _ in range(bags_amount)]
    return bags

def create_model(bag: Bag) -> BaggingModel:
    X_mapped, y_mapped = bag.get_mapped_data()
    model = DecisionTreeClassifier()
    model.fit(X_mapped, y_mapped)
    return BaggingModel(model, bag)

def create_models(bags: List[Bag]) -> List[BaggingModel]:
    models = [create_model(bag) for bag in bags]
    return models

def predict(X, models: List[BaggingModel]) -> np.ndarray:
    predictions = [ model.model.predict(model.bag.map_X_by_features(X)) for model in models ]
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
    # Step 1: Generate all predictions as a binary matrix (correct = 1, incorrect = 0)
    n_models = len(models)
    n_samples = len(y)
    correct_matrix = np.empty((n_models, n_samples), dtype=bool)

    for idx, model in enumerate(models):
        preds = model.model.predict(model.bag.map_X_by_features(X))
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