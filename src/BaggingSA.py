from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import random
from typing import Tuple
from dataclasses import dataclass
from typing import List, Dict, Any
from Bagging import Bag, create_bags, create_models, get_accuracy
from sklearn.model_selection import train_test_split



class BaggingSA:
    """docstring for BaggingSA."""
    def __init__(self, 
                 X: np.ndarray, y: np.ndarray, bags_with_replacement: bool,
                 T0: float, alpha:float, max_iterations: int, n_trees: int
                 ):
        self.bags_with_replacement = bags_with_replacement
        self.T0 = T0
        self.alpha = alpha
        self.n_trees = n_trees
        self.X, self.y = X, y
        self.max_iterations = max_iterations
        
        
    def run_simulated_annealing(self) -> List[DecisionTreeClassifier]:
        T = self.T0
        iteration = 0
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3)
        
        bags = create_bags(X=X_train, y=y_train, n_bags=self.n_trees, with_replacement=self.bags_with_replacement)
        models = create_models(bags=bags, n_trees=self.n_trees)
        best_models = models.copy()
        
        #X_test, y_test = get_data_subset(X=self.X, y=self.y)
        accuracy = get_accuracy(X=X_test, y=y_test, models=models)
        best_accuracy = accuracy
        
        while T > 0.0001 and iteration < self.max_iterations and accuracy < 1.0:
            bags = [get_neighbor_bag(X_train, y_train, bag) for bag in bags]
            new_models = create_models(bags=bags, n_trees=self.n_trees)
            
            #X_test, y_test = get_data_subset(X=self.X, y=self.y)
            new_accuracy = get_accuracy(X=X_test, y=y_test, models=new_models)
            
            print(f"Iteration: {iteration}, Temperature: {T:.4f}, Accuracy: {accuracy:.2f}, New Accuracy: {new_accuracy:.2f}")
            
            if best_accuracy < new_accuracy:
                best_accuracy = new_accuracy
                best_models = new_models.copy()
                
            if accuracy < new_accuracy:
                models = new_models.copy()
                accuracy = new_accuracy
            else:
                threshold = random.uniform(0, 1)
                prob = np.exp((new_accuracy - accuracy) / T)
                
                if prob > threshold:
                    models = new_models.copy()
                    accuracy = new_accuracy

            T *= self.alpha
            iteration += 1
        
        return best_models
        
            
def get_data_subset(X: np.ndarray, y: np.ndarray, sub_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Get a subset of the data."""
    if sub_size >= 1:
        return X, y
    n_samples = int(len(X) * sub_size)
    indices = np.random.choice(range(len(X)), size=n_samples, replace=False)
    return X[indices], y[indices]

def get_neighbor_bag(X, y, bag: Bag) -> Bag:
    #TODO: check if amout of swap is correct
        swap_amount = int(len(bag.X) / 1000.0)
        if swap_amount == 0:
            swap_amount = 1
        new_bag = Bag(X=bag.X.copy(), y=bag.y.copy(), features=bag.features.copy())
        
        for _ in range(swap_amount):
            swap_index = np.random.randint(0, len(X))
            tmpX = X[swap_index]
            tmpy = y[swap_index]
            new_bag.X[np.random.randint(0, len(new_bag.X))] = tmpX[bag.features]
            new_bag.y[np.random.randint(0, len(new_bag.y))] = tmpy
            
        return new_bag
        
        

        
        
            