from itertools import combinations
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import random
from typing import Tuple
from dataclasses import dataclass
from typing import List, Dict, Any
from Bagging import Bag, BaggingModel, create_bags, create_models, get_accuracy_and_predictions, predict
from sklearn.model_selection import train_test_split

from MeasuresOfDiversity import average_q_statistic



class BaggingSA:
    """docstring for BaggingSA."""
    def __init__(self, 
                 X: np.ndarray, y: np.ndarray, bags_with_replacement: bool,
                 T0: float, alpha:float, max_iterations: int, n_trees: int,
                 fitness_acc: float = 0.5, fitness_div: float = 0.25, fitness_qstat: float = 0.25,
                 ):
        self.bags_with_replacement = bags_with_replacement
        self.T0 = T0
        self.alpha = alpha
        self.n_trees = n_trees
        self.X, self.y = X, y
        self.max_iterations = max_iterations
        self.fitness_acc = fitness_acc
        self.fitness_div = fitness_div
        self.fitness_qstat = fitness_qstat
        
        
    # def calculate_fitness(self, models: List[BaggingModel], X: np.ndarray, y: np.ndarray) -> float:
    #     """
    #     Calculate the fitness of the models based on their accuracy.
    #     """
    #     accuracy, predictions = get_accuracy_and_predictions(models=models, X=X, y=y)
        
    #     q_statistic = average_q_statistic(predictions)
    #     q_statistic_norm = (q_statistic+1) /2
    #     return accuracy * (1-q_statistic_norm)       
    
    def pairwise_disagreement(self, predictions):
        """
        Diversity measure: average disagreement between all pairs of classifiers
        """
        disagreement = 0
        pairs = list(combinations(predictions, 2))
        for pred1, pred2 in pairs:
            disagreement += np.mean(pred1 != pred2)
        return disagreement / len(pairs)
    
    def calculate_fitness(self, models: List[BaggingModel], X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the fitness of the models based on their accuracy.
        """
        accuracy, predictions = get_accuracy_and_predictions(models=models, X=X, y=y)
        div = self.pairwise_disagreement(predictions)
        q_statistic = 1 - (average_q_statistic(predictions) + 1) / 2
        res = self.fitness_acc * accuracy + self.fitness_div * div + self.fitness_qstat * q_statistic
        return res
        
    
    
    
    def run_simulated_annealing(self) -> Tuple[List[BaggingModel], List[BaggingModel], float]:
        T = self.T0
        iteration = 0
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.1)
        
        bags = create_bags(X=X_train, y=y_train, n_bags=self.n_trees, with_replacement=self.bags_with_replacement)
        models = create_models(bags=bags, n_trees=self.n_trees)
        init_models = models.copy()
        best_models = models.copy()
        
        fitness = self.calculate_fitness(X=X_test, y=y_test, models=models)
        best_fitness = fitness
        
        while iteration < self.max_iterations:
            bags = [get_neighbor_bag(X_train, y_train, bag) for bag in bags]
            new_models = create_models(bags=bags, n_trees=self.n_trees)
            new_fitness = self.calculate_fitness(X=X_test, y=y_test, models=new_models)
            
            print(f"Iteration: {iteration}, Temperature: {T:.5f}, Best fitness: {best_fitness:.3f}, Fitness: {fitness:.3f}, New fitness: {new_fitness:.3f}")
            
            if best_fitness < new_fitness:
                best_fitness = new_fitness
                best_models = new_models.copy()
                
            if fitness < new_fitness:
                models = new_models.copy()
                fitness = new_fitness
            else:
                threshold = random.uniform(0, 1)
                prob = np.exp((new_fitness - fitness) / T)
                
                if prob > threshold:
                    models = new_models.copy()
                    fitness = new_fitness

            T *= self.alpha
            iteration += 1
        
        return best_models, init_models, best_fitness
        
            
def get_neighbor_bag(X, y, bag: Bag) -> Bag:
        x_length = len(bag.X)
        swap_amount = int(x_length * 0.001)
        if swap_amount == 0:
            swap_amount = 1
            
        new_bag = Bag(
            X=bag.X.copy(),
            y=bag.y.copy(),
            features=bag.features.copy()
        )
        
        for _ in range(swap_amount):
            is_add = random.choice([True, False])
            
            if is_add:
                index_to_add = np.random.randint(0, len(X))
                tmpX = X[index_to_add][new_bag.features].reshape(1, -1)
                tmpy = y[index_to_add]
                new_bag.X = np.append(new_bag.X, tmpX, axis=0)
                new_bag.y = np.append(new_bag.y, tmpy)
            else:
                index_to_remove = np.random.randint(0, len(new_bag.X))
                new_bag.X = np.delete(new_bag.X, index_to_remove, axis=0)
                new_bag.y = np.delete(new_bag.y, index_to_remove)
                
        return new_bag 
        
        

        
        
            