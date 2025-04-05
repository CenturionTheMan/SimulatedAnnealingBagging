from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import random
from typing import Tuple
from dataclasses import dataclass
from typing import List, Dict, Any
from Bagging import Bag, create_bags, create_models, get_accuracy, get_single_model_accuracy

    
class BaggingGEN:
    def __init__(self,
                 X: np.ndarray, y: np.ndarray,
                  n_trees: int, max_iterations: int, mutation_rate: float, crossover_rate: float,
                 ):
        self.X = X
        self.y = y
        self.max_iterations = max_iterations
        self.population_size = n_trees
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
    
    
    def mutate(self, population: List[Bag], mutate_rate:float, X: np.ndarray, y: np.ndarray) -> None:
        for bag in population:
            if random.random() > mutate_rate:
                continue
            
            swap_amount = int(len(bag.X) * 0.01)
            if swap_amount == 0:
                swap_amount = 1
        
            is_add = random.choice([True, False])
            
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
            
            if random.random() < crossover_rate:
                parent1 = population[i]
                parent2 = population[i + 1]
                
                shorter_len = min(len(parent1.X), len(parent2.X))
                crossover_point = random.randint(1, shorter_len - 1)
                parent1.X, parent2.X = parent1.X[:crossover_point], parent2.X[crossover_point:]
                parent1.y, parent2.y = parent1.y[:crossover_point], parent2.y[crossover_point:]
                child1 = Bag(X=np.vstack((parent1.X, parent2.X)), y=np.concatenate((parent1.y, parent2.y)), features=parent1.features)
                child2 = Bag(X=np.vstack((parent2.X, parent1.X)), y=np.concatenate((parent2.y, parent1.y)), features=parent2.features)
                new_population.append(child1)
                new_population.append(child2)
            else:
                new_population.append(parent1)
                new_population.append(parent2)
        
        return new_population
    
    def evaluate_population_fitness(self, population: List[Bag], X_test: np.ndarray, y_test: np.ndarray) -> List[float]:
        models = create_models(bags=population, n_trees=self.population_size)
        evaluations = [get_single_model_accuracy(model, X_test[:][model.features], y_test) for model in models]
        return evaluations
    
    def selection(self, population: List[Bag], fitness: List[float]) -> List[Bag]:
        # tournament selection
        selected = []
        for _ in range(len(population)):
            competitors = random.sample(list(zip(population, fitness)), k=3)
            winner = max(competitors, key=lambda x: x[1])[0]
            selected.append(winner)
        
        return selected    
        
        
    def run_genetic_algorithm(self) -> List[DecisionTreeClassifier]:
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
        population = create_bags(X=X_train, y=y_train, n_bags=self.population_size, with_replacement=False)

        best_avg_fitness = 0
        best_population = None

        iteration = 0
        
        while iteration < self.max_iterations:
            fitnesses = self.evaluate_population_fitness(population, X_test, y_test)
            avg_fitness = np.mean(fitnesses)            

            print(f"Iteration: {iteration}, Avg Fitness: {avg_fitness:.4f}")

            if avg_fitness > best_avg_fitness:
                best_avg_fitness = avg_fitness
                best_population = population.copy()

            
            population = self.selection(population, fitnesses)
            population = self.crossover(population, self.crossover_rate)
            self.mutate(population, self.mutation_rate, X_train, y_train)
            iteration += 1

        models = create_models(bags=best_population, n_trees=self.n_trees)
        return models        
        
                    