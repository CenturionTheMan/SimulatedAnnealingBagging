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

class BaggingSA:
    def __init__(self, 
                 X: np.ndarray, y: np.ndarray,
                 T0: float, alpha: float, cooling_method: Literal['linear', 'geometric', 'logarithmic'], max_iterations: int, n_trees: int,
                 feature_mutation_chance: float, test_split_amount: int,
                 theta: float, beta: float, gamma: float,
                 ):
        self.T0 = T0
        self.n_trees = n_trees
        self.max_iterations = max_iterations
        self.feature_mutation_chance = feature_mutation_chance
        self.test_split_amount = test_split_amount
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        self.X_train = X_train
        self.y_train = y_train
        self.rows_validate = [(X, y) for X, y in zip(X_test, y_test)]
        self.features_amount = X.shape[1]
        self.alpha = alpha
        self.cooling_method = cooling_method
        self.classes = np.unique(y)
        self.theta = theta
        self.beta = beta
        self.gamma = gamma

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
        if self.theta > 0:        
            sub_groups_X_test, sub_groups_y_test = self.get_validate_sets()
            accuracies = [
                evaluate(X=sub_groups_X_test[i], y=sub_groups_y_test[i], models=models)
                for i in range(self.test_split_amount)
            ]
            accuracy = np.mean(accuracies)
        else:
            accuracy = 0
        
        if self.beta > 0:
            X_test, _ = zip(*self.rows_validate)
            disagreement = compute_disagreement(X=np.array(X_test), models=models)
        else:
            disagreement = 0

        if self.gamma > 0:
            complexities = np.array([
                0.3 * min(len(m.bag.features), self.features_amount) / max(len(m.bag.features), self.features_amount) +
                0.7 * min(m.bag.count_samples(), len(self.X_train)) / max(m.bag.count_samples(), len(self.X_train))
                for m in models
            ])
            complexity = 1 - complexities.mean()
        else:
            complexity = 0
        
        accuracy = accuracy * self.theta
        disagreement = disagreement* self.beta
        complexity = complexity* self.gamma
        
        fitness = accuracy + disagreement - complexity
        
        #print(f"   Acc: {accuracy:.4f}, Dis: {disagreement:.4f}, Com: {complexity:.4f} => Fit: {fitness:.4f}")
        
        return fitness
    
    
    def get_neighbors(self, population: List[Bag]) -> List[Bag]:
        def should_add_feature(features_len: int) -> bool:
            if self.features_amount <= features_len:
                return False
            elif int(np.sqrt(self.features_amount)) >= features_len:
                return True
            return random.random() < 0.5

        def should_add_sample(current_size: int) -> bool:
            if len(self.X_train) <= current_size:
                return False
            elif int(len(self.X_train) / 2) >= current_size:
                return False
            return random.random() < 0.5

        bags = [single.copy() for single in population]

        for single in bags:
            if random.random() < self.feature_mutation_chance:
                # Feature mutation
                is_add = should_add_feature(len(single.features))
                
                if is_add:
                    available = list(set(range(self.features_amount)) - set(single.features))
                    if available:  # Safety check
                        to_add = random.choice(available)
                        single.features = np.append(single.features, to_add)
                elif len(single.features) > 1:
                    to_remove = random.choice(single.features)
                    single.features = np.setdiff1d(single.features, [to_remove])
            else:
                # Sample mutation
                current_size = single.count_samples()
                is_add = should_add_sample(current_size)

                if is_add:
                    to_add = random.randint(0, len(self.X_train) - 1)
                    add_X = self.X_train[to_add].reshape(1, -1)
                    add_y = self.y_train[to_add].reshape(1,)
                    single.X = np.concatenate([single.X, add_X], axis=0)
                    single.y = np.concatenate([single.y, add_y], axis=0)
                elif len(single.X) > 1:
                    to_remove = random.randint(0, len(single.X) - 1)
                    single.X = np.delete(single.X, to_remove, axis=0)
                    single.y = np.delete(single.y, to_remove, axis=0)

        return bags

    def get_initial_population(self) -> Tuple[List[Bag], List[BaggingModel], float]:
        tmp_bags = create_bags(self.X_train, self.y_train, self.n_trees, replace=True, cut_features=False)
        tmp_models = create_models(tmp_bags)     
        tmp_fit = self.calculate_fitness(tmp_models)
        return tmp_bags, tmp_models, tmp_fit   

    def calculate_probability(self, newFitness: float, oldFitness: float, temperature: float) -> float:
        diff = newFitness - oldFitness
        if diff / temperature > 709:
            prob = 1.0 
        else:
            prob = np.exp(diff / temperature)
        return prob

    def calculate_temperature(self, method: str, T: float, iteration: int) -> float:
        if method == 'linear':
            return T - self.alpha
        elif method == 'geometric':
            return T * self.alpha
        elif method == 'logarithmic':
            return self.alpha / math.log(iteration + 1)
        else:
            raise ValueError("Invalid temperature calculation method.")
        
    
    def run(self, X_for_test = None, y_for_test = None, monitor_fun = None, get_fitness = False) -> List[BaggingModel]:
        T = self.T0
        
        bags, models, fitness = self.get_initial_population()
        
        best_models = models.copy()
        best_fitness = fitness
        
        iteration = 1
        
        while iteration <= self.max_iterations:
            new_bags = self.get_neighbors(bags)
            models = create_models(new_bags)
            new_fitness = self.calculate_fitness(models)
            
            accuracy = None
            if X_for_test is not None and y_for_test is not None:
                accuracy = evaluate(X_for_test, y_for_test, models)
            
            if monitor_fun is not None:
                monitor_fun(iteration, T, best_fitness, fitness, new_fitness, accuracy)
    
            if best_fitness < new_fitness:
                best_models = models.copy()
                best_fitness = new_fitness
    
            if fitness < new_fitness:
                fitness = new_fitness
                bags = new_bags
            else:
                threshold = random.random()
                prob = self.calculate_probability(new_fitness, fitness, T)
                if prob > threshold:
                    fitness = new_fitness
                    bags = new_bags
                    
            T = self.calculate_temperature(self.cooling_method, T, iteration)
            if T <= 1e-10:
                T = 1e-10
            
            iteration += 1
        
        if get_fitness:
            return best_models, best_fitness
        else:
            return best_models
        