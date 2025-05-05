from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import random
from raw_python.DatasetsHandle import get_dataset
from raw_python.BaggingSA import BaggingSA
from typing import Literal, Tuple
import sklearn
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from raw_python.Bagging import create_models, create_bags, evaluate, evaluate_stats, predict


k_cross = 5
reps = 5
n_trees_arr = [10, 20, 30, 40, 50]
datasets = ['digits', 'wine', 'breast_cancer', 'pima']

bagging_sa_params = {
    'wine' : {
        'T0': 2,
        'cooling_method': 'geometric',
        'alpha': 0.995,
        'max_iterations': 2000,
        'feature_mutation_chance': 0.25,
        'test_split_amount': 5,
        'beta': 0.75,
        'gamma': 0.25,
        'delta': 0.1,
        'epsilon': 0.05
    },
    'breast_cancer' : {
        'T0': 2,
        'cooling_method': 'geometric',
        'alpha': 0.995,
        'max_iterations': 2000,
        'feature_mutation_chance': 0.25,
        'test_split_amount': 5,
        'beta': 0.75,
        'gamma': 0.25,
        'delta': 0.1,
        'epsilon': 0.05   
    },
    'pima' : {
        'T0': 2,
        'cooling_method': 'geometric',
        'alpha': 0.995,
        'max_iterations': 2000,
        'feature_mutation_chance': 0.25,
        'test_split_amount': 5,
        'beta': 0.75,
        'gamma': 0.25,
        'delta': 0.1,
        'epsilon': 0.05  
    },
    'digits' : {
        'T0': 2,
        'cooling_method': 'geometric',
        'alpha': 0.995,
        'max_iterations': 2000,
        'feature_mutation_chance': 0.25,
        'test_split_amount': 5,
        'beta': 0.75,
        'gamma': 0.25,
        'delta': 0.1,
        'epsilon': 0.05     
    }
}

seed = 42
np.random.seed(seed)
random.seed(seed)
  
    
def evaluate_bagging_sa(X_train, y_train, X_test, y_test, n_trees: int, params: dict) -> float:
    T0 = params['T0']
    cooling_method = params['cooling_method']
    alpha = params['alpha']
    max_iterations = params['max_iterations']
    feature_mutation_chance = params['feature_mutation_chance']
    test_split_amount = params['test_split_amount']
    beta = params['beta']
    gamma = params['gamma']
    delta = params['delta']
    epsilon = params['epsilon']
    bagging_sa = BaggingSA(X=X_train, y=y_train,
                            T0=T0, cooling_method=cooling_method, alpha=alpha, max_iterations=max_iterations, n_trees=n_trees,
                            feature_mutation_chance=feature_mutation_chance, test_split_amount=test_split_amount,
                            beta=beta, gamma=gamma, delta=delta, epsilon=epsilon)
    models = bagging_sa.run()
    metrics = evaluate_stats(X=X_test, y=y_test, models=models)
    return metrics


def evaluate_bagging(X_train, y_train, X_test, y_test, n_trees: int) -> float:
    bags = create_bags(X_train, y_train, bags_amount=n_trees)
    models = create_models(bags=bags)
    metrics = evaluate_stats(X=X_test, y=y_test, models=models)
    return metrics  

print(f"Start at {pd.Timestamp.now()}")
for dataset in datasets:
    result = []
    X, y = get_dataset(dataset)       
    
    random_indices = np.arange(X.shape[0])
    np.random.shuffle(random_indices)
    X = X[random_indices]
    y = y[random_indices]
    
    sub_groups_X = np.array_split(np.array(X), k_cross)
    sub_groups_y = np.array_split(np.array(y), k_cross) 
         
    for n_tree in n_trees_arr:
        for rep in range(reps):
            for k in range(k_cross):
                X_train = np.concatenate(sub_groups_X[:k] + sub_groups_X[k+1:])
                y_train = np.concatenate(sub_groups_y[:k] + sub_groups_y[k+1:])
                X_test = sub_groups_X[k]
                y_test = sub_groups_y[k]
                pars = bagging_sa_params[dataset]
                
                metrics_bagging = evaluate_bagging(X_train, y_train, X_test, y_test, n_trees=n_tree)
                metrics_baggingSA = evaluate_bagging_sa(X_train, y_train, X_test, y_test, n_trees=n_tree, params=pars)
                
                print(f"    Dataset: {dataset}, n_trees: {n_tree}, rep: {rep}, k: {k+1}/{k_cross} >> Bagging: {metrics_bagging['accuracy']:.3f}, BaggingSA: {metrics_baggingSA['accuracy']:.3f}")
                
                result.append([
                    dataset, n_tree, rep, k+1, 
                    metrics_bagging['accuracy'], metrics_bagging['precision'], metrics_bagging['recall'], metrics_bagging['f1'],
                    metrics_baggingSA['accuracy'], metrics_baggingSA['precision'], metrics_baggingSA['recall'], metrics_baggingSA['f1']
                ])
                
                df = pd.DataFrame(result, columns=[
                    "Dataset",
                    "nTrees",
                    "Rep",
                    "K",
                    "BaggingAccuracy",
                    "BaggingPrecision",
                    "BaggingRecall",
                    "BaggingF1",
                    "SAAccuracy",
                    "SAPrecision",
                    "SARecall",
                    "SAF1"
                ])
                
                df.to_csv(f'./../res/metrics_{dataset}.csv', index=False)        
print(f"End at {pd.Timestamp.now()}")