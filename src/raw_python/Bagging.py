from copy import deepcopy
from functools import wraps
from itertools import combinations
import time
import pandas as pd
from sklearn.calibration import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import numpy as np
import random
from typing import Tuple
from dataclasses import dataclass
from typing import List, Dict, Any
from collections import Counter

rng = np.random.RandomState(42)

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper
    
@dataclass
class Bag:
    X: np.ndarray[np.bool_]
    y: np.ndarray[np.bool_]
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

@dataclass
class Ensemble:
    models: List[BaggingModel]
    output_ratios: np.ndarray

def create_bag(X, y) -> Bag:
    indices = rng.choice(len(X), size=len(X), replace=True)
    tmp_X = X[indices].copy()
    tmp_y = y[indices].copy()
    features = list(range(X.shape[1]))
    bag = Bag(tmp_X, tmp_y, features, features_max_amount=X.shape[1])
    return bag

def create_bags(X, y, bags_amount: int) -> List[Bag]:
    bags = [create_bag(X, y) for _ in range(bags_amount)]
    return bags

def create_model(bag: Bag) -> BaggingModel:
    X_mapped, y_mapped = bag.get_mapped_data()
    model = DecisionTreeClassifier()
    model.fit(X_mapped, y_mapped)
    return BaggingModel(model, bag)

def create_models(bags: List[Bag], n_jobs: int = -1) -> List[BaggingModel]:
    return Parallel(n_jobs=n_jobs)(
        delayed(create_model)(bag) for bag in bags
    )
    
def create_ensemble(bags: List[Bag], n_jobs: int = -1) -> Ensemble:
    models = Parallel(n_jobs=n_jobs)(
        delayed(create_model)(bag) for bag in bags
    )
    output_ratios = np.ones_like(models, dtype=float) / len(models)
    return Ensemble(models=models, output_ratios=output_ratios)


def majority_vote(labels: np.ndarray) -> int:
    return Counter(labels).most_common(1)[0][0]

def weighted_majority_vote(pred: np.ndarray, weights: np.ndarray, n_classes: int) -> int:
    bincount = np.zeros(n_classes)
    for label, weight in zip(pred, weights):
        bincount[label] += weight
    return bincount.argmax()

def predict_ensemble(X: np.ndarray, ensemble: Ensemble) -> np.ndarray:
    predictions = [model.model.predict(model.bag.map_X_by_features(X)) for model in ensemble.models]
    predictions = np.array(predictions)  # shape: (n_models, n_samples)
    
    n_classes = max(model.model.n_classes_ for model in ensemble.models)
    final_predictions = [
        weighted_majority_vote(pred, ensemble.output_ratios, n_classes) for pred in predictions.T
    ]
    return np.array(final_predictions)

def predict(X: np.ndarray, models: List[BaggingModel]) -> np.ndarray:
    predictions = [model.model.predict(model.bag.map_X_by_features(X)) for model in models]
    predictions = np.array(predictions)
    final_predictions = [
        majority_vote(pred) for pred in predictions.T
    ]
    return np.array(final_predictions)

def evaluate_stats(X, y, models: List[BaggingModel], average='weighted', return_arr:bool=False)-> pd.DataFrame:
    y_pred = predict(X, models)
    
    result = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average=average),
        'recall': recall_score(y, y_pred, average=average),
        'f1': f1_score(y, y_pred, average=average),
        'cm': confusion_matrix(y, y_pred),
    }
    
    if return_arr:
        return list(result.values())
    else:
        return result
    

def evaluate(X, y, models: List[BaggingModel]) -> float:
    predictions = predict(X, models)
    accuracy = accuracy_score(y, predictions)
    return accuracy

def evaluate_ensemble(X, y, ensemble: Ensemble) -> float:
    predictions = predict_ensemble(X, ensemble)
    accuracy = accuracy_score(y, predictions)
    return accuracy

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def compute_disagreement(X: np.ndarray, models: List[BaggingModel]) -> float:
    n_models = len(models)
    all_preds = []

    # get predictions per model
    for model in models:
        preds = model.model.predict(model.bag.map_X_by_features(X))
        all_preds.append(preds)
    
    all_preds = np.array(all_preds)  # (n_models, n_samples)

    # calc disagreements
    disagreements = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            disagreement_rate = np.mean(all_preds[i] != all_preds[j])
            disagreements.append(disagreement_rate)

    return np.mean(disagreements)
    

    

def q_statistic_for_ensemble(X: np.ndarray, y: np.ndarray, models: List['BaggingModel']) -> float:
    n_models = len(models)
    n_samples = len(y)
    correct_matrix = np.empty((n_models, n_samples), dtype=bool)

    for idx, model in enumerate(models):
        preds = model.model.predict(model.bag.map_X_by_features(X))
        correct_matrix[idx] = preds == y

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