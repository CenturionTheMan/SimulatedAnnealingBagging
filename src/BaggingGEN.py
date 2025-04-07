from itertools import combinations
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import random
from typing import Tuple
from dataclasses import dataclass
from typing import List, Dict, Any
from Bagging import Bag, BaggingModel, create_bags, create_models, get_accuracy, get_accuracy_and_predictions

    
class BaggingGEN:
    def __init__(self,
                 X: np.ndarray, y: np.ndarray,
                  n_trees: int, max_iterations: int, mutation_rate: float, crossover_rate: float, population_size: int|None = None
                 ):
        self.X = X
        self.y = y
        self.max_iterations = max_iterations
        self.population_size = population_size if population_size is not None else n_trees
        self.n_trees = n_trees
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        

    def mutate(self, population: List[Bag], mutate_rate:float, X: np.ndarray, y: np.ndarray) -> None:
        for bag in population:
            if random.random() > mutate_rate:
                continue
            
            x_length = len(bag.X)
            swap_amount = int(x_length * 0.001)
            if swap_amount == 0:
                swap_amount = 1
                
            for _ in range(swap_amount):
                is_add = True if len(bag.X) <=2 else random.choice([True, False])
                
                if is_add:
                    index_to_add = np.random.randint(0, len(X))
                    tmpX = X[index_to_add][bag.features].reshape(1, -1)
                    tmpy = y[index_to_add]
                    bag.X = np.append(bag.X, tmpX, axis=0)
                    bag.y = np.append(bag.y, tmpy)
                else:
                    index_to_remove = np.random.randint(0, len(bag.X))
                    bag.X = np.delete(bag.X, index_to_remove, axis=0)
                    bag.y = np.delete(bag.y, index_to_remove)
    
    def crossover(self, population: List[Bag], crossover_rate: float) -> List[Bag]:
        new_population = []
        
        for i in range(0, len(population), 2):
            if i + 1 >= len(population):
                new_population.append(population[i])
                continue
            
            parent1 = population[i]
            parent2 = population[i + 1]

            if random.random() < crossover_rate:
                p1_half1 = np.random.choice(range(len(parent1.X)), size=int(len(parent1.X)/2), replace=False)
                p1_half2 = np.setdiff1d(range(len(parent1.X)), p1_half1)

                p2_half1 = np.random.choice(range(len(parent2.X)), size=int(len(parent2.X)/2), replace=False)
                p2_half2 = np.setdiff1d(range(len(parent2.X)), p2_half1)
                
                child1 = Bag(X=np.vstack((parent1.X[p1_half1], parent2.X[p2_half2])), y=np.concatenate((parent1.y[p1_half1], parent2.y[p2_half2])), features=parent1.features)
                child2 = Bag(X=np.vstack((parent2.X[p2_half1], parent1.X[p1_half2])), y=np.concatenate((parent2.y[p2_half1], parent1.y[p1_half2])), features=parent2.features)
                
                new_population.append(child1)
                new_population.append(child2)
            else:
                new_population.append(parent1)
                new_population.append(parent2)
                
        return new_population
    
    
    
    def pairwise_disagreement(self, predictions):
        """
        Diversity measure: average disagreement between all pairs of classifiers
        """
        disagreement = 0
        pairs = list(combinations(predictions, 2))
        for pred1, pred2 in pairs:
            disagreement += np.mean(pred1 != pred2)
        return disagreement / len(pairs)
    
    def evaluate_fitness(self, population: List[Bag], X_test: np.ndarray, y_test: np.ndarray) -> Tuple[List[float], List[BaggingModel]]:
        models = create_models(bags=population, n_trees=self.population_size)
        
        predictions_per_model = [model.model.predict(X_test[:,model.features]) for model in models]
        #accuracy_per_model = [accuracy_score(y_test, pred) for pred in predictions_per_model]
        div = [self.pairwise_disagreement(predictions) for predictions in predictions_per_model]
        return div, models
        # return accuracy_per_model, models        
    
    
    def selection(self, population: List[Bag], fitness: List[float]) -> List[Bag]:
        # tournament selection
        selected = []
        for _ in range(len(population)):
            competitors = random.sample(list(zip(population, fitness)), k=3)
            winner = max(competitors, key=lambda x: x[1])[0]
            selected.append(winner)
        
        return selected  
    
    def run_genetic_algorithm(self) -> Tuple[List[BaggingModel], List[BaggingModel], float]:
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.1, shuffle=True)
        population = create_bags(X=X_train, y=y_train, n_bags=self.population_size, with_replacement=False)
        initial_population = population.copy()
        
        best_models = None
        best_models_fitness = None
        best_fitness = 0.0
        
        iteration = 0
        
        while iteration < self.max_iterations:
            _, X_test_sub, _, y_test_sub = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)
            fitness_per_model, models = self.evaluate_fitness(population, X_test_sub, y_test_sub)
            fitness_mean = np.mean(fitness_per_model)

            if best_fitness < fitness_mean:
                best_fitness = fitness_mean
                best_models_fitness = fitness_per_model.copy()
                best_models = models.copy()
            
            print(f"Iteration: {iteration}, Best fitness: {best_fitness:.3f}, Current fitness: {fitness_mean:.3f}")
            
            selected_population = self.selection(population, fitness_per_model)
            self.mutate(selected_population, self.mutation_rate, X_train, y_train)
            population = self.crossover(selected_population, self.crossover_rate)
            
            iteration += 1
            
        
        fitness_per_model, initial_models = self.evaluate_fitness(initial_population, X_test, y_test)
        
        return self.get_n_best_models(best_models, best_models_fitness, self.n_trees), self.get_n_best_models(initial_models, fitness_per_model, self.n_trees), best_fitness
    
    
    def get_n_best_models(self, models: List[BaggingModel], fitness_per_model: List[float], n: int) -> List[BaggingModel]:
        sorted_models = sorted(zip(models, fitness_per_model), key=lambda x: x[1], reverse=True)
        return [model for model, _ in sorted_models[:n]]