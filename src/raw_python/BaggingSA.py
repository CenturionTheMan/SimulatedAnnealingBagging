from copy import deepcopy
import math
from typing import List, Literal, Tuple
from raw_python.Bagging import Bag, BaggingModel, compute_disagreement, create_models, create_bags, evaluate, evaluate_f1, predict, q_statistic_for_ensemble
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
                 feature_mutation_chance: float, validation_split_amount: int,
                 beta: float, gamma: float, delta: float
                 ):
        self.T0 = T0
        self.n_trees = n_trees
        self.max_iterations = max_iterations
        self.feature_mutation_chance = feature_mutation_chance
        self.validation_split_amount = validation_split_amount
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
        self.X_train = X_train
        self.y_train = y_train
        self.rows_validate = [(X, y) for X, y in zip(X_test, y_test)]
        self.features_amount = X.shape[1]
        self.alpha = alpha
        self.cooling_method = cooling_method
        self.classes = np.unique(y)
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def get_validate_sets(self):
        random.shuffle(self.rows_validate)
        X_test, y_test = zip(*self.rows_validate)
        
        # add noise to labels
        y_test = np.array(y_test)
        y_test_noise = np.where(np.random.rand(len(y_test)) < self.delta, np.random.choice(self.classes, len(y_test)), y_test)
        
        if self.validation_split_amount <= 1:
            return np.array([X_test]), np.array([y_test_noise])
        
        sub_groups_X_test = np.array_split(np.array(X_test), self.validation_split_amount)
        sub_groups_y_test = np.array_split(np.array(y_test_noise), self.validation_split_amount)
        return sub_groups_X_test, sub_groups_y_test
        
    def calculate_fitness(self, models: List[BaggingModel]) -> float:
        if self.beta > 0:        
            sub_groups_X_test, sub_groups_y_test = self.get_validate_sets()
            evals = [
                evaluate_f1(X=sub_groups_X_test[i], y=sub_groups_y_test[i], models=models)
                for i in range(self.validation_split_amount)
            ]
            eval_score = np.mean(evals)
        else:
            eval_score = 0
        
        
        
        if self.gamma > 0:
            X_test, _ = zip(*self.rows_validate)
            disagreement = compute_disagreement(X=np.array(X_test), models=models)
        else:
            disagreement = 0


        
        eval_score = eval_score * self.beta
        disagreement = disagreement* self.gamma
        
        fitness = eval_score + disagreement
        
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
        pop_size = 20
        
        tmp_bags = [create_bags(self.X_train, self.y_train, pop_size) for _ in range(pop_size)]
        tmp_models = [create_models(bag) for bag in tmp_bags]
        tmp_fits = [self.calculate_fitness(models) for models in tmp_models]
        
        best_fit = max(tmp_fits)
        best_bags = tmp_bags[np.argmax(tmp_fits)]
        best_models = tmp_models[np.argmax(tmp_fits)]
        
        return best_bags, best_models, best_fit

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
        
    
    def run(self, X_for_test = None, y_for_test = None, monitor_fun = None, get_fitness = False, initial_bags: List[Bag]|None=None) -> List[BaggingModel]:
        T = self.T0
        
        if initial_bags is None:
            bags, models, fitness = self.get_initial_population()
        else:
            bags = initial_bags.copy()
            models = create_models(bags)
            fitness = self.calculate_fitness(models)
            
        
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
        