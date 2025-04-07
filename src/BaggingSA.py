from typing import List
from Bagging import Bag, BaggingModel, create_models, create_bags, evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import random



class BaggingSA:
    def __init__(self, 
                 X: np.ndarray, y: np.ndarray,
                 T0: float, alpha:float, max_iterations: int, n_trees: int,
                 ):
        self.T0 = T0
        self.alpha = alpha
        self.n_trees = n_trees
        self.X, self.y = X, y
        self.max_iterations = max_iterations
    
    
    def calculate_fitness(self, models: List[BaggingModel], X: np.ndarray, y: np.ndarray) -> float:
        accuracy = evaluate(X=X, y=y, models=models)
        return accuracy
    
    def mutate(self, population: List[Bag]) -> List[BaggingModel]:
        for single in population:
            selected = np.random.choice(range(len(single.X_bin)), size=int(len(single.X_bin) * 0.01), replace=False)
            single.X_bin[selected] = np.random.randint(0, 2, size=len(selected))
        return population
    
    def run(self, X_for_test = None, y_for_test = None) -> List[BaggingModel]:
        T = self.T0
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, shuffle=True)
        
        bags = create_bags(X_train, self.n_trees)
        models = create_models(X_train, y_train, bags)
        fitness = self.calculate_fitness(models, X_test, y_test)
        
        best_models = models.copy()
        best_fitness = fitness
        
        iteration = 0
        
        while T > 1e-10 and iteration < self.max_iterations:
            new_bags = self.mutate(bags)
            new_models = create_models(X_train, y_train, new_bags)
            new_fitness = self.calculate_fitness(models, X_test, y_test)
            
            if X_for_test is not None and y_for_test is not None:
                accuracy = evaluate(X_for_test, y_for_test, new_models)
            else:
                accuracy = None
            
            print(f"Iteration: {iteration}, Temperature: {T:.4f}, Best: {best_fitness:.4f}, Fitness: {fitness:.4f}, New Fitness: {new_fitness:.4f}, Accuracy: {accuracy:.4f}")
            
            if new_fitness > best_fitness:
                best_models = new_models.copy()
                best_fitness = new_fitness
            
            if new_fitness > fitness:
                fitness = new_fitness
                models = new_models.copy()
            else:
                delta = fitness - best_fitness
                prob = np.exp(delta / T)
                if random.random() < prob:
                    models = new_models.copy()
                    fitness = new_fitness
                    
                    
                    
            T *= self.alpha
            iteration += 1
        
        return best_models
        