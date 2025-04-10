import math
from typing import List, Literal, Tuple
from Bagging import Bag, BaggingModel, create_models, create_bags, evaluate, predict
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
                 T0: float, alpha: float, cooling_method: Literal['linear', 'geometric', 'logarithmic'], max_iterations: int, n_trees: int,
                 feature_mutation_chance: float = 0.1, test_split_amount: int = 10,
                 ):
        self.T0 = T0
        self.n_trees = n_trees
        self.max_iterations = max_iterations
        self.feature_mutation_chance = feature_mutation_chance
        self.test_split_amount = test_split_amount
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        self.X_train = X_train
        self.y_train = y_train
        self.sub_groups_X_test = np.array_split(X_test, self.test_split_amount)
        self.sub_groups_y_test = np.array_split(y_test, self.test_split_amount)
        self.features = X.shape[1]
        self.alpha = alpha
        self.cooling_method = cooling_method
        
    def disagreement_measure(self, models: List[BaggingModel], X_test: np.ndarray) -> float:
        disagreement_sum = 0.0
        pop_predictions = predict(X_test, models)
        
        for model in models:
            model_predictions = model.model.predict(X_test[:,model.bag.features])
            disagreement_sum += np.sum(pop_predictions != model_predictions) / len(model_predictions)
        return disagreement_sum / len(models)
        
    def calculate_fitness(self, models: List[BaggingModel]) -> float:
        acc_sum = 0
        disagreement_sum = 0
        for i in range(self.test_split_amount):
            sub_X, sub_y = self.sub_groups_X_test[i], self.sub_groups_y_test[i]
            
            acc_sum += evaluate(X=sub_X, y=sub_y, models=models)
            disagreement_sum += self.disagreement_measure(models, sub_X)
            
        accuracy = acc_sum / self.test_split_amount
        disagreement = disagreement_sum / self.test_split_amount
        
        alpha = 0.9
        fitness = (alpha * accuracy) + ((1-alpha) * disagreement)
        return fitness
    
    def get_neighbors(self, population: List[Bag]) -> List[BaggingModel]:
        bags = [single.copy() for single in population]
        for single in bags:
            is_feature_mutation = random.random() < self.feature_mutation_chance
            
            if is_feature_mutation:
                mutation_point = random.randint(0, len(single.features) - 1)
                new_feature = None
                while new_feature is None or new_feature in single.features:
                    new_feature = random.randint(0, self.features - 1)
                single.features[mutation_point] = new_feature
            else:            
                mutation_point = random.randint(0, len(single.X_bin) - 1)
                single.X_bin[mutation_point] = not single.X_bin[mutation_point]
        return bags
    
    def get_initial_population(self) -> Tuple[List[Bag], List[BaggingModel], float]:
        amount = self.n_trees * 10
        bags = create_bags(self.X_train, amount)
        models = create_models(self.X_train, self.y_train, bags)
        
        mean_accuracy = []
        for X_test, y_test in zip(self.sub_groups_X_test, self.sub_groups_y_test):
            predictions_per_model = [model.model.predict(X_test[:,model.bag.features]) for model in models]
            accuracy_per_model = [accuracy_score(y_test, pred) for pred in predictions_per_model]
            mean_accuracy.append(accuracy_per_model)
        
        accuracy_per_model = np.mean(mean_accuracy, axis=0)
        sorted_res = sorted(zip(bags, models, accuracy_per_model), key=lambda x: x[2], reverse=True)
        best = sorted_res[:self.n_trees]
        best_bags, best_models, best_accuracy = zip(*best)
        fitness = self.calculate_fitness(best_models)
        
        return list(best_bags), list(best_models), fitness

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
        
    
    def run(self, X_for_test = None, y_for_test = None, monitor_fun = None) -> List[BaggingModel]:
        T = self.T0
        
        bags, models, fitness = self.get_initial_population()
        
        best_models = models.copy()
        best_fitness = fitness
        
        iteration = 1
        
        while T > 1e-10 and iteration <= self.max_iterations:
            new_bags = self.get_neighbors(bags)
            models = create_models(self.X_train, self.y_train, new_bags)
            new_fitness = self.calculate_fitness(models)
            
            if X_for_test is not None and y_for_test is not None:
                accuracy = evaluate(X_for_test, y_for_test, models)
            else:
                accuracy = None
            
            if monitor_fun is not None:
                monitor_fun(iteration, T, best_fitness, fitness, new_fitness, accuracy)
    
            if best_fitness < new_fitness:
                best_models = models.copy()
                best_fitness = new_fitness
    
            if fitness < new_fitness:
                fitness = new_fitness
                bags = new_bags.copy()
            elif fitness == new_fitness:
                pass
            else:
                threshold = random.random()
                prob = self.calculate_probability(new_fitness, fitness, T)
                if prob > threshold:
                    fitness = new_fitness
                    bags = new_bags.copy()
                    
            T = self.calculate_temperature(self.cooling_method, T, iteration)
            iteration += 1
        
        return best_models
        