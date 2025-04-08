from typing import List
from Bagging import Bag, BaggingModel, create_models, create_bags, evaluate, predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import random


class BaggingGA:
    def __init__(self,
                 X: np.ndarray, y: np.ndarray,
                 n_trees: int,
                 max_iterations: int, mutation_rate: float, crossover_rate: float, population_size: int|None = None):
        self.X = X
        self.y = y
        self.n_trees = n_trees
        self.max_iterations = max_iterations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population_size = population_size if population_size else n_trees

    def disagreement_measure(self, models: List[BaggingModel], X_test: np.ndarray) -> List[float]:
        res = []
        pop_predictions = predict(X_test, models)
        
        for model in models:
            model_predictions = model.model.predict(X_test[:,model.bag.features])
            disagreement = np.sum(pop_predictions != model_predictions) / len(model_predictions)
            res.append(disagreement)
        return res

    def calculate_fitness(self, models: List[BaggingModel], X_test: np.ndarray, y_test: np.ndarray) -> List[float]:
        predictions_per_model = [model.model.predict(X_test[:,model.bag.features]) for model in models]
        accuracy_per_model = [accuracy_score(y_test, pred) for pred in predictions_per_model]
        disagreement_per_model = self.disagreement_measure(models, X_test)

        alpha = 0.9
        
        fitness_per_model = [ 
            accuracy * alpha + disagreement * (1 - alpha)
            for accuracy, disagreement in zip(accuracy_per_model, disagreement_per_model)
        ]
        return fitness_per_model
    
    def selection(self, population: List[Bag], fitness: List[float], X_train: np.ndarray) -> List[Bag]:
        sum_fitness = sum(fitness)
        selection_probs = [f / sum_fitness for f in fitness]
        selected_indices = np.random.choice(range(len(population)), size=self.population_size, p=selection_probs, replace=True)
        pop = np.array(population)[selected_indices]
        return pop.tolist()        

    def mutate(self, population: List[Bag]) -> List[BaggingModel]:
        for single in population:
            if random.random() >= self.mutation_rate:
                continue
            selected = np.random.choice(range(len(single.X_bin)), size=int(len(single.X_bin) * 0.01), replace=False)
            single.X_bin[selected] = np.random.randint(0, 2, size=len(selected))
        return population
    
    def crossover(self, population: List[Bag]) -> List[Bag]:
        for i in range(0, len(population), 2):
            if random.random() >= self.crossover_rate:
                continue
            crossover_point = random.randint(0, len(population[0].X_bin) - 1)
            tmp = population[i].X_bin[:crossover_point].copy()
            population[i].X_bin[:crossover_point] = population[i + 1].X_bin[:crossover_point]
            population[i + 1].X_bin[:crossover_point] = tmp
        return population
    
    def get_n_models(self, models: List[BaggingModel], fitness_per_model: List[float]) -> List[BaggingModel]:
        sorted_models = sorted(zip(models, fitness_per_model), key=lambda x: x[1], reverse=True)
        best = sorted_models[:self.n_trees]
        return [model for model, _ in best]
        
    def run(self, X_for_test, y_for_test) -> List[BaggingModel]:
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, shuffle=True)
        
        population = create_bags(X_train, self.population_size)
        
        best_models = None
        best_fitness = None
        iteration = 0

        while iteration < self.max_iterations:
            models = create_models(X_train, y_train, population)
            fitness = self.calculate_fitness(models, X_test, y_test)
            fitness_pointer = np.median(fitness)

            if(X_for_test is not None and y_for_test is not None):
                n_models = self.get_n_models(models, fitness)
                accuracy = evaluate(X=X_for_test, y=y_for_test, models=n_models)
            else:
                accuracy = None

            if best_fitness is None or fitness_pointer > best_fitness:
                best_fitness = fitness_pointer
                best_models = models
                
            print(f"Iteration {iteration}, Best fitness: {best_fitness:.3f}, Fitness: {fitness_pointer:.3f}, Accuracy: {accuracy:.3f}")
            
            population = self.selection(population, fitness, X_train)
            #population = self.crossover(population)
            #population = self.mutate(population)

            iteration += 1

        n_best_models = self.get_n_models(best_models, fitness)
        return n_best_models
    