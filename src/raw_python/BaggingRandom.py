from copy import deepcopy
import math
from typing import List, Literal, Tuple
from raw_python.Bagging import Bag, BaggingModel, compute_disagreement, create_models, create_bags, evaluate, predict, q_statistic_for_ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import random
from functools import wraps
import time
from concurrent.futures import ThreadPoolExecutor

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

class BaggingRandom:
    def __init__(self, 
                 X: np.ndarray, y: np.ndarray,
                 n_trees: int,
                 test_split_amount: int,
                 pop_size: int
                 ):
        self.n_trees = n_trees
        self.test_split_amount = test_split_amount
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
        self.X_train = X_train
        self.y_train = y_train
        self.rows_validate = [(X, y) for X, y in zip(X_test, y_test)]
        self.classes = np.unique(y)
        self.pop_size = pop_size

    def get_validate_sets(self):
        random.shuffle(self.rows_validate)
        X_test, y_test = zip(*self.rows_validate)
        
        # add noise to labels
        y_test = np.array(y_test)
        y_test = np.where(np.random.rand(len(y_test)) < 0.1, np.random.choice(self.classes, len(y_test)), y_test)
        
        if self.test_split_amount <= 1:
            return np.array([X_test]), np.array([y_test])
        
        sub_groups_X_test = np.array_split(np.array(X_test), self.test_split_amount)
        sub_groups_y_test = np.array_split(np.array(y_test), self.test_split_amount)
        return sub_groups_X_test, sub_groups_y_test
        
    def calculate_fitness(self, models: List[BaggingModel]) -> float:
        sub_groups_X_test, sub_groups_y_test = self.get_validate_sets()
        accuracies = [
            evaluate(X=sub_groups_X_test[i], y=sub_groups_y_test[i], models=models)
            for i in range(self.test_split_amount)
        ]
        accuracy = np.mean(accuracies)
        
        return accuracy
    
    

    def get_initial_population(self) -> Tuple[List[Bag], List[BaggingModel], float]:
        tmp_bags = create_bags(self.X_train, self.y_train, self.n_trees, replace=True, cut_features=False)
        tmp_models = create_models(tmp_bags)     
        tmp_fit = self.calculate_fitness(tmp_models)
        return tmp_bags, tmp_models, tmp_fit   

        
    
    def run(self) -> List[BaggingModel]:
        pop_bags = [create_bags(self.X_train, self.y_train, self.n_trees, replace=True, cut_features=False) for _ in range(self.pop_size)]
        pop_models = [create_models(bags) for bags in pop_bags]
        pop_fit = [self.calculate_fitness(pop) for pop in pop_models]
        best_pop = pop_models[np.argmax(pop_fit)]
        # best_fit = max(pops_fit)
        return best_pop