from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import random
from typing import Tuple
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class BaggingModel:
    model: DecisionTreeClassifier
    features: List[int]

class BaggingDT:
    """docstring for BaggingDT."""
    def __init__(self, X: np.ndarray, y: np.ndarray, n_estimators: int = 10, seed = None) -> None:
        self.X = X
        self.y = y
        self.n_estimators = n_estimators
        self.models = []
        self.data_size = len(X)
        self.data_features_amount = X.shape[1]
        self.seed = seed
        self.data_range = range(self.data_size)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def __create_bag(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create a bootstrap sample."""
        indices = np.random.choice(self.data_range, size=int(self.data_size), replace=True)
        
        X_sample = self.X[indices]
        y_sample = self.y[indices]
        
        random_features = np.random.choice(
                range(self.data_features_amount),
                size=int(np.sqrt(self.data_features_amount)), 
                replace=False
            )
        X_sample = X_sample[:, random_features]
        
        return X_sample, y_sample, random_features

    def __create_models(self) -> None:
        """Create the model."""
        self.models = []
        for _ in range(self.n_estimators):
            X_sample, y_sample, random_features = self.__create_bag()
            model = DecisionTreeClassifier(random_state=self.seed) if self.seed is not None else DecisionTreeClassifier()
            model.fit(X_sample, y_sample)
            self.models.append(BaggingModel(model, random_features))

    def fit(self) -> None:
        """Fit the model."""
        self.__create_models()
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class of the given data."""
        predictions = []
        for model in self.models:
            X_sample = X[model.features]
            X_sample = X_sample.reshape(1, -1)
            prediction = model.model.predict(X_sample)
            predictions.append(prediction)
        
        # Combine the predictions from all models
        predictions = np.array(predictions)
        final_predictions = [np.bincount(pred).argmax() for pred in predictions.T]
        
        return np.array(final_predictions)
    
    def estimate_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Estimate the accuracy of the model."""
        predictions = [self.predict(x) for x in X]
        accuracy = accuracy_score(y, predictions)
        return accuracy