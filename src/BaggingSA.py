from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import random
from typing import Tuple
from dataclasses import dataclass
from typing import List, Dict, Any
from BaggingDT import Bag, create_bags, create_models, get_accuracy



class BaggingSA:
    """docstring for BaggingSA."""
    def __init__(self, 
                 X: np.ndarray, y: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray, 
                 T0: float, alpha:float, max_iterations: int, n_trees: int
                 ):
        self.T0 = T0
        self.alpha = alpha
        self.n_trees = n_trees
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.max_iterations = max_iterations
        
    
        
    def run_simulated_annealing(self) -> List[DecisionTreeClassifier]:
        T = self.T0
        iteration = 0
        bags = create_bags(X=self.X, y=self.y, n_bags=self.n_trees)
        models = create_models(bags=bags, n_trees=self.n_trees)
        best_models = models.copy()
        accuracy = get_accuracy(X=self.X_test, y=self.y_test, models=models)
        best_accuracy = accuracy
        
        while T > 0.001 and iteration < self.max_iterations:
            bags = [get_neighbor_bag(self.X, self.y, bag) for bag in bags]
            new_models = create_models(bags=bags, n_trees=self.n_trees)
            new_accuracy = get_accuracy(X=self.X_test, y=self.y_test, models=new_models)
            
            print(f"Iteration: {iteration}, Temperature: {T}, Accuracy: {accuracy}, New Accuracy: {new_accuracy}")
            
            if new_accuracy > best_accuracy:
                best_accuracy = new_accuracy
                best_models = new_models.copy()
                
            if accuracy < new_accuracy:
                models = new_models.copy()
                accuracy = new_accuracy
            else:
                p = np.exp((accuracy - new_accuracy) / T)
                if random.random() < p:
                    models = new_models.copy()
                    accuracy = new_accuracy
                
            T *= self.alpha
            iteration += 1
        
        return best_models
        
            

def get_neighbor_bag(X, y, bag: Bag) -> Bag:
        swap_amount = int(len(bag.X) / 100)
        if swap_amount == 0:
            swap_amount = 1
        
        new_bag = Bag(X=bag.X.copy(), y=bag.y.copy(), features=bag.features.copy())
        
        for _ in range(swap_amount):
            # Swap a random sample with a random sample from the original data
            swap_index = np.random.randint(0, len(X))
            tmpX = X[swap_index]
            tmpy = y[swap_index]
            new_bag.X[np.random.randint(0, len(new_bag.X))] = tmpX[bag.features]
            new_bag.y[np.random.randint(0, len(new_bag.y))] = tmpy
            
        return new_bag
        
        

        
        
            