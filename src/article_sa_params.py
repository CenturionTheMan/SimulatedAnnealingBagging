from raw_python.Bagging import create_models, create_bags, evaluate, evaluate_stats
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
from itertools import product

seed = 42
np.random.seed(seed)
random.seed(seed)


# In[2]:


k_cross = 4
reps = 3

datasets = ['digits','wine', 'breast_cancer', 'pima', 'obesity', 'students_dropout']
par =  {
    'T0': 2,
    'cooling_method': 'geometric',
    'alpha': 0.995,
    'max_iterations': 2000,
    'feature_mutation_chance': 0.25,
    'validation_split_amount': 5,
    'n_trees': 10,
    'beta': None,
    'gamma': None,
    'delta': None,
}

beta_r = [0.25, 0.50, 0.75]     # accuracy
gamma_r  = [0.25, 0.50, 0.75]   # disagreement
delta_r = [0.05, 0.1]       # noise


# In[ ]:


def evaluate_bagging_sa(X_train, y_train, X_test, y_test, params, beta, gamma, delta):
    bagging_sa = BaggingSA(X=X_train, y=y_train, T0=params['T0'], alpha=params['alpha'], 
                           cooling_method=params['cooling_method'], max_iterations=params['max_iterations'],
                           n_trees=params['n_trees'], feature_mutation_chance=params['feature_mutation_chance'],
                            validation_split_amount=params['validation_split_amount'],
                            beta=beta, gamma=gamma, delta=delta)
    models, fitness = bagging_sa.run(monitor_fun=fun_monitor, get_fitness=True, X_for_test=X_test, y_for_test=y_test)
    metrics = evaluate_stats(X=X_test, y=y_test, models=models)
    return bagging_sa, fitness, metrics

def fun_monitor(iteration, T, best_fitness, fitness, new_fitness, accuracy):
    fits.append(new_fitness)
    accs.append(accuracy)

    # if iteration % 100 == 0:
    #     print(f"    Iteration: {iteration}, T: {T:.2f}, Best fitness: {best_fitness:.4f}")

fits = []
accs = []
result = []

greeks_permutations = list(product(beta_r, gamma_r, delta_r))
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

    for beta, gamma, delta, epsilon in greeks_permutations:
        for k in range(k_cross):
            for rep in range(reps):
                print(f"[Dataset: {dataset}, K: {k+1}/{k_cross}, Rep: {rep+1}/{reps}, Beta: {beta}, Gamma: {gamma}, Delta: {delta}, Epsilon: {epsilon}]")

                if k_cross == 1:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
                else:
                    X_train = np.concatenate(sub_groups_X[:k] + sub_groups_X[k+1:])
                    y_train = np.concatenate(sub_groups_y[:k] + sub_groups_y[k+1:])
                    X_test = sub_groups_X[k]
                    y_test = sub_groups_y[k]

                fits = []
                accs = []

                bagging_sa, fitness, metrics = evaluate_bagging_sa(X_train, y_train, X_test, y_test, par, beta, gamma, delta, epsilon)

                spearman_corr, spearman_p = spearmanr(fits, accs)


                result.append([dataset, k+1, rep+1, 
                               bagging_sa.beta, bagging_sa.gamma, bagging_sa.delta, bagging_sa.epsilon,
                               fitness, spearman_corr, spearman_p, metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']])

                df = pd.DataFrame(result, columns=['dataset', 'fold', 'rep', 
                                                   'beta', 'gamma', 'delta', 'epsilon',
                                                   'fitness', 'spearman_corr', 'spearman_p',
                                                   'accuracy', 'precision', 'recall', 'f1'])

                df.to_csv(f"./../res/params_{dataset}.csv", index=False)
                print(f"    Fitness={fitness:.3f}, Spearman Correlation={spearman_corr:.3f}, Spearman p-value={spearman_p:.3f}, Accuracy={metrics['accuracy']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
print(f"End at {pd.Timestamp.now()}")

