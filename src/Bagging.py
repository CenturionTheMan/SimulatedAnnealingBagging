from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import random
from typing import Tuple
from dataclasses import dataclass
from typing import List, Dict, Any

from MeasuresOfDiversity import average_q_statistic

@dataclass
class BaggingModel:
    model: DecisionTreeClassifier
    features: List[int]
    
@dataclass
class Bag:
    X: np.ndarray
    y: np.ndarray
    features: List[int]


def create_bag(X, y, with_replacement: bool) -> Bag:
    """Create a bootstrap sample."""
    if with_replacement:
        indices = np.random.choice(range(len(X)), size=len(X), replace=True)
    else:
        indices = np.random.choice(range(len(X)), size=int(len(X)/2), replace=False)
    X_sample = X[indices]
    y_sample = y[indices]
    
    data_features_amount = X.shape[1]
    random_features = np.random.choice(
            range(data_features_amount),
            size=int(np.sqrt(data_features_amount)), 
            replace=False
        )

    X_sample = X_sample[:, random_features]
    return Bag(X_sample, y_sample, random_features)


def create_bags(X, y, n_bags: int, with_replacement:bool) -> List[Bag]:
    return [create_bag(X, y, with_replacement=with_replacement) for _ in range(n_bags)]

        
def predict(X: np.ndarray, models: List[BaggingModel]) -> np.ndarray:
        """Predict the class of the given data."""
        
        if models is None or len(models) == 0:
            raise ValueError("The model has not been fitted yet.")
        
        predictions = []
        for model in models:
            X_sample = X[model.features]
            X_sample = X_sample.reshape(1, -1)
            prediction = model.model.predict(X_sample)
            predictions.append(prediction)
        
        predictions = np.array(predictions)
        final_predictions = [np.bincount(pred).argmax() for pred in predictions.T]
        
        return final_predictions
    
def get_accuracy(X: np.ndarray, y: np.ndarray, models: List[BaggingModel]) -> float:
    accuracy, _ = get_accuracy_and_predictions(X, y, models)
    return accuracy

def get_accuracy_and_predictions(X: np.ndarray, y: np.ndarray, models: List[BaggingModel]) -> Tuple[float, np.ndarray]:
    predictions = [ model.model.predict(X[:,model.features]) for model in models ]
    predictions = np.array(predictions)
    final_predictions = [np.bincount(pred).argmax() for pred in predictions.T]
    accuracy = np.mean([1 if pred == real else 0 for pred, real in zip(final_predictions, y)])
    return accuracy, predictions    
    
def create_models(bags: list[Bag], n_trees: int, seed:int = None) -> List[BaggingModel]:
        """Create the model."""
        models = []
        
        if len(bags) != n_trees:
            raise ValueError("The number of bags must be equal to the number of estimators.")
        
        for i in range(n_trees):
            X_sample, y_sample, random_features = bags[i].X, bags[i].y, bags[i].features
            model = DecisionTreeClassifier(random_state=seed) if seed is not None else DecisionTreeClassifier()
            model.fit(X_sample, y_sample)
            models.append(BaggingModel(model, random_features))
            
        return models