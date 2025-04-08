from typing import List
from Bagging import Bag, BaggingModel, create_models, create_bags, evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import random



class BaggingSAv2:
    def __init__(self, 
                 X: np.ndarray, y: np.ndarray,
                 T0: float, alpha:float, max_iterations: int, n_trees: int,
                 ):
        self.T0 = T0
        self.alpha = alpha
        self.n_trees = n_trees
        self.max_iterations = max_iterations
        self.test_split_amount = 10
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        self.X_train = X_train
        self.y_train = y_train
        self.sub_groups_X_test = np.array_split(X_test, self.test_split_amount)
        self.sub_groups_y_test = np.array_split(y_test, self.test_split_amount)
        self.features = X.shape[1]
        
    
    
    def calculate_fitness(self, models: List[BaggingModel]) -> float:
        acc_sum = 0
        for i in range(self.test_split_amount):
            sub_X = self.sub_groups_X_test[i]
            sub_y = self.sub_groups_y_test[i]
            
            accuracy = evaluate(X=sub_X, y=sub_y, models=models)
            acc_sum += accuracy
        accuracy = acc_sum / self.test_split_amount
        return accuracy
    
    def mutate(self, population: List[Bag]) -> List[BaggingModel]:
        for single in population:
            is_feature_mutation = random.random() < 0.1
            if is_feature_mutation:
                mutation_point = random.randint(0, len(single.features) - 1)
                new_feature = None
                while new_feature is None or new_feature in single.features:
                    new_feature = random.randint(0, self.features - 1)
                single.features[mutation_point] = new_feature
            else:            
                mutation_point = random.randint(0, len(single.X_bin) - 1)
                single.X_bin[mutation_point] = not single.X_bin[mutation_point]
        return population
    
    def run(self, X_for_test = None, y_for_test = None, monitor_fun = None) -> List[BaggingModel]:
        T = self.T0
        
        bags = create_bags(self.X_train, self.n_trees)
        models = create_models(self.X_train, self.y_train, bags)
        fitness = self.calculate_fitness(models)
        
        best_models = models.copy()
        best_fitness = fitness
        
        iteration = 0
        
        while T > 1e-10 and iteration < self.max_iterations:
            new_bags = self.mutate(bags)
            new_models = create_models(self.X_train, self.y_train, new_bags)
            new_fitness = self.calculate_fitness(models)
            
            if X_for_test is not None and y_for_test is not None:
                accuracy = evaluate(X_for_test, y_for_test, new_models)
            else:
                accuracy = None
            
            if fitness == new_fitness:
                print("Fitness is the same, skipping iteration")
            
            if monitor_fun is not None:
                monitor_fun(iteration, T, best_fitness, fitness, new_fitness, accuracy)
            
            if new_fitness > best_fitness:
                best_models = new_models.copy()
                best_fitness = new_fitness
            
            if new_fitness > fitness:
                fitness = new_fitness
                models = new_models.copy()
            else:
                delta = new_fitness - fitness
                prob = np.exp(delta / T)
                if random.random() < prob:
                    models = new_models.copy()
                    fitness = new_fitness
                    
            T *= self.alpha
            iteration += 1
        
        return best_models
        