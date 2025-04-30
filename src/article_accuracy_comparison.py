from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import random
from raw_python.BaggingSA import BaggingSA
from typing import Literal, Tuple
import sklearn
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from raw_python.Bagging import create_models, create_bags, evaluate, predict

seed = 42

k_cross = 5
reps = 5
n_trees = [10, 20, 30, 40, 50]
datasets = ['digits', 'wine', 'breast_cancer', 'pima']

bagging_sa_params = {
    'wine' : {
        'T0': 2,
        'cooling_method': 'geometric',
        'alpha': 0.995,
        'max_iterations': 2000,
        'feature_mutation_chance': 0.25,
        'test_split_amount': 5,
        'theta': 0.85,
        'beta': 0.1,
        'gamma': 0.05,
    },
    'breast_cancer' : {
        'T0': 2,
        'cooling_method': 'geometric',
        'alpha': 0.995,
        'max_iterations': 2000,
        'feature_mutation_chance': 0.25,
        'test_split_amount': 5,
        'theta': 0.85,
        'beta': 0.1,
        'gamma': 0.05,        
    },
    'pima' : {
        'T0': 2,
        'cooling_method': 'geometric',
        'alpha': 0.995,
        'max_iterations': 2000,
        'feature_mutation_chance': 0.25,
        'test_split_amount': 5,
        'theta': 0.85,
        'beta': 0.1,
        'gamma': 0.05,         
    },
    'digits' : {
        'T0': 2,
        'cooling_method': 'geometric',
        'alpha': 0.995,
        'max_iterations': 2000,
        'feature_mutation_chance': 0.25,
        'test_split_amount': 5,
        'theta': 0.85,
        'beta': 0.1,
        'gamma': 0.05,         
    }
}

np.random.seed(seed)
random.seed(seed)

def get_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    if dataset_name == 'digits':
        data = sklearn.datasets.load_digits()
        X = data.data
        y = data.target
        
    elif dataset_name == 'wine':
        data = sklearn.datasets.load_wine()
        X = data.data
        y = data.target
    
    elif dataset_name == 'breast_cancer':
        data = sklearn.datasets.load_breast_cancer()
        X = data.data
        y = data.target
        
    elif dataset_name == 'pima':
        data = pd.read_csv("./../datasets/pima.csv")
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
    
    else:
        raise ValueError("Unsupported dataset")
    return X, y


def evaluate_rf(X_train, y_train, X_test, y_test, n_trees: int) -> float:
    model = RandomForestClassifier(n_estimators=n_trees, random_state=seed)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

def evaluate_bagging_custom(X_train, y_train, X_test, y_test, n_trees: int) -> float:
    bags = create_bags(X_train, y_train, bags_amount=n_trees)
    models = create_models(bags=bags)
    accuracy = evaluate(X=X_test, y=y_test, models=models)
    return accuracy
  
    
def evaluate_bagging_sa(X_train, y_train, X_test, y_test, n_trees: int, params: dict) -> float:
    T0 = params['T0']
    cooling_method = params['cooling_method']
    alpha = params['alpha']
    max_iterations = params['max_iterations']
    feature_mutation_chance = params['feature_mutation_chance']
    test_split_amount = params['test_split_amount']
    theta = params['theta']
    beta = params['beta']
    gamma = params['gamma']
    bagging_sa = BaggingSA(X=X_train, y=y_train,
                            T0=T0, cooling_method=cooling_method, alpha=alpha, max_iterations=max_iterations, n_trees=n_trees,
                            feature_mutation_chance=feature_mutation_chance, test_split_amount=test_split_amount,
                            theta=theta, beta=beta, gamma=gamma)
    models = bagging_sa.run()
    accuracy = evaluate(X=X_test, y=y_test, models=models)
    return accuracy
    
    
def evaluate_decision_tree(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def evaluate_bagging(X_train, y_train, X_test, y_test, n_trees: int) -> float:
    model = BaggingClassifier(n_estimators=n_trees, random_state=seed)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return accuracy    

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
         
    for n_tree in n_trees:
        for rep in range(reps):
            for k in range(k_cross):
                X_train = np.concatenate(sub_groups_X[:k] + sub_groups_X[k+1:])
                y_train = np.concatenate(sub_groups_y[:k] + sub_groups_y[k+1:])
                X_test = sub_groups_X[k]
                y_test = sub_groups_y[k]
                pars = bagging_sa_params[dataset]
                
                dt_acc= evaluate_decision_tree(X_train, y_train, X_test, y_test)
                bagging_acc = evaluate_bagging(X_train, y_train, X_test, y_test, n_trees=n_tree)
                rf_acc = evaluate_rf(X_train, y_train, X_test, y_test, n_trees=n_tree)
                bagging_custom_acc = evaluate_bagging_custom(X_train, y_train, X_test, y_test, n_trees=n_tree)
                bagging_sa_acc = evaluate_bagging_sa(X_train, y_train, X_test, y_test, n_trees=n_tree, params=pars)
                
                print(f"    Dataset: {dataset}, n_trees: {n_tree}, rep: {rep}, k: {k+1}/{k_cross} >> DT: {dt_acc:.3f}, Bagging: {bagging_acc:.3f}, RF: {rf_acc:.3f}, BaggingCustom: {bagging_custom_acc:.3f}, BaggingSA: {bagging_sa_acc:.3f}")
                
                result.append([
                    dataset, n_tree, rep, k+1, dt_acc, bagging_acc, rf_acc, bagging_custom_acc, bagging_sa_acc
                ])
                
                df = pd.DataFrame(result, columns=[
                    "Dataset",
                    "nTrees",
                    "Rep",
                    "K",
                    "DT",
                    "Bagging",
                    "RF",
                    "BaggingCustom",
                    "BaggingSA"
                ])
                
                df.to_csv(f'./../res/accuracy_comparison_{dataset}.csv', index=False)        
print(f"End at {pd.Timestamp.now()}")