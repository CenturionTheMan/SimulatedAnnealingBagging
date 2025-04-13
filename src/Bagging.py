from copy import deepcopy
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



def q_statistic_for_ensemble(X: np.ndarray, y: np.ndarray, models: List[BaggingModel]) -> float:
    predictions = [model.model.predict(X[:,model.bag.features]) for model in models ]
    predictions = [p == y for p in predictions]
    predictions = np.array(predictions)
    print(predictions)    