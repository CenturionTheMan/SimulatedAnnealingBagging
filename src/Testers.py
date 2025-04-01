import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from Bagging import create_bags, create_models, get_accuracy
from BaggingSA import BaggingSA


def test_method(method_to_run, X_train, X_test, y_train, y_test, reps: int):
    results = []
    method_name = method_to_run.__name__

    for i in range(reps):
        res = method_to_run(X_train, X_test, y_train, y_test)
        print(f"[{method_name}] iteration {i + 1} completed")
        results.append([i + 1, res])
    results_df = pd.DataFrame(results, columns=["Iteration", "Result"])
    results_df.to_csv(f"./../res/accuracy_{method_name}.csv", index=False)
    print()
            
    
def dt(X_train, X_test, y_train, y_test) -> float:
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

def bagging(X_train, X_test, y_train, y_test) -> float:
    bags = create_bags(X_train, y_train, n_bags=10, with_replacement=False)
    models = create_models(bags=bags, n_trees=10)
    return get_accuracy(models=models, X=X_test, y=y_test)

def bagging_sa(X_train, X_test, y_train, y_test) -> float:
    bagging_sa = BaggingSA(X=X_train, y=y_train, bags_with_replacement=False,
                            T0=1.0, alpha=0.95, max_iterations=100, n_trees=10)
    models = bagging_sa.run_simulated_annealing()
    return get_accuracy(models=models, X=X_test, y=y_test)