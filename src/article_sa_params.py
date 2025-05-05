#!/usr/bin/env python
# coding: utf-8

# In[1]:


from raw_python.Bagging import create_models, create_bags, evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import random
from raw_python.BaggingSA import BaggingSA
from typing import Literal, Tuple
from raw_python.Bagging import predict
import sklearn
from scipy.stats import spearmanr, kendalltau, pearsonr
from raw_python.DatasetsHandle import get_dataset

seed = 42
np.random.seed(seed)
random.seed(seed)


# In[2]:


k_cross = 5
# reps = 5
reps = 2

datasets = ['digits','wine', 'breast_cancer', 'pima']
par =  {
    'T0': 2,
    'cooling_method': 'geometric',
    'alpha': 0.995,
    'max_iterations': 2000,
    'feature_mutation_chance': 0.25,
    'test_split_amount': 5,
    'n_trees': 10,
    'theta': None,
    'beta': None,
    'gamma': None,
}

# theta_r = [0.25, 0.50, 0.75, 1.0]
# beta_r = [0.25, 0.50, 0.75, 1.0]
# gamma_r  = [0.25, 0.50, 0.75, 1.0]

theta_r = [0.25, 0.50, 0.75]
beta_r = [0.25, 0.50, 0.75]
gamma_r  = [0.25, 0.50, 0.75]


# In[ ]:


def evaluate_bagging_sa(X_train, y_train, X_test, y_test, params, theta, beta, gamma) -> Tuple[float, int, int]: 
    bagging_sa = BaggingSA(X=X_train, y=y_train, T0=params['T0'], alpha=params['alpha'], 
                           cooling_method=params['cooling_method'], max_iterations=params['max_iterations'],
                           n_trees=params['n_trees'], feature_mutation_chance=params['feature_mutation_chance'],
                            test_split_amount=params['test_split_amount'], theta=theta,
                            beta=beta, gamma=gamma)
    models, fitness = bagging_sa.run(monitor_fun=fun_monitor, get_fitness=True, X_for_test=X_test, y_for_test=y_test)
    accuracy = evaluate(X=X_test, y=y_test, models=models)
    return bagging_sa, accuracy, fitness

def fun_monitor(iteration, T, best_fitness, fitness, new_fitness, accuracy):
    fits.append(new_fitness)
    accs.append(accuracy)

    # if iteration % 100 == 0:
    #     print(f"    Iteration: {iteration}, T: {T:.2f}, Best fitness: {best_fitness:.4f}")

fits = []
accs = []
result = []
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

    for theta in theta_r:
        for beta in beta_r:
            for gamma in gamma_r:
                for k in range(k_cross):
                    for rep in range(reps):
                        print(f"[Dataset: {dataset}, K: {k+1}/{k_cross}, Rep: {rep+1}/{reps}, Theta: {theta}, Beta: {beta}, Gamma: {gamma}]")

                        if k_cross == 1:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
                        else:
                            X_train = np.concatenate(sub_groups_X[:k] + sub_groups_X[k+1:])
                            y_train = np.concatenate(sub_groups_y[:k] + sub_groups_y[k+1:])
                            X_test = sub_groups_X[k]
                            y_test = sub_groups_y[k]

                        fits = []
                        accs = []

                        bagging_sa, accuracy, fitness = evaluate_bagging_sa(X_train, y_train, X_test, y_test, par, theta, beta, gamma)

                        spearman_corr, spearman_p = spearmanr(fits, accs)

                        result.append([dataset, k+1, rep+1, fitness, accuracy, spearman_corr, spearman_p, theta, beta, gamma])

                        df = pd.DataFrame(result, columns=['dataset', 'k_cross', 'rep', 'fitness', 'accuracy', 'spearman_corr', 'spearman_p', 'theta', 'beta', 'gamma'])

                        df.to_csv("./../res/bagging_sa_params.csv", index=False)
                        print(f"    Accuracy: {accuracy:.4f}")

