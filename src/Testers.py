from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from Bagging import create_bags, create_models, get_accuracy
from BaggingSA import BaggingSA

class Tester:
    """docstring for Tester."""
    def __init__(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                 trees_n: int, T0: float, alpha: float, max_iterations: int,
                 reps: int, dir_to_save: str = "./../res/"):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.reps = reps
        self.dir_to_save = dir_to_save
        self.trees_n = trees_n
        self.T0 = T0
        self.alpha = alpha
        self.max_iterations = max_iterations
        
    def test_methods(self, reps: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        reps = reps if reps is not None else self.reps
        print("Testing Decision Tree")
        res1 = self.test_dt(reps=reps, save_path=f"{self.dir_to_save}accuracy_dt_{self.trees_n}.csv")
        print("Testing Bagging")
        res2 = self.test_bagging(reps=reps, save_path=f"{self.dir_to_save}accuracy_bagging_{self.trees_n}.csv")
        print("Testing Bagging SA")
        res3 = self.test_bagging_sa(reps=reps, save_path=f"{self.dir_to_save}accuracy_bagging_sa_{self.trees_n}.csv")
        return res1, res2, res3
    
        
    def test_dt(self, reps: int | None = None, save_path: str | None = None) -> pd.DataFrame:
        reps = reps if reps is not None else self.reps
        
        def method_to_run():
            clf = DecisionTreeClassifier()
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)
            return accuracy_score(self.y_test, y_pred)
        res = test_method(method_to_run, args=None, reps=reps, save_path=save_path)
        return res
        

    def test_bagging(self, reps: int | None = None, save_path: str | None = None) -> pd.DataFrame:
        reps = reps if reps is not None else self.reps
        def method_to_run():
            bags = create_bags(self.X_train, self.y_train, n_bags=self.trees_n, with_replacement=False)
            models = create_models(bags=bags, n_trees=self.trees_n)
            return get_accuracy(models=models, X=self.X_test, y=self.y_test)
        return test_method(method_to_run, args=None, reps=reps, save_path=save_path)

    def test_bagging_sa(self, reps: int | None = None, save_path: str | None = None) -> pd.DataFrame:
        reps = reps if reps is not None else self.reps
        
        def method_to_run():
            bagging_sa = BaggingSA(X=self.X_train, y=self.y_train, bags_with_replacement=False,
                                T0=self.T0, alpha=self.alpha, max_iterations=self.max_iterations, n_trees=self.trees_n)
            best_models, _, _ = bagging_sa.run_simulated_annealing()
            return get_accuracy(models=best_models, X=self.X_test, y=self.y_test)
        return test_method(method_to_run, args=None, reps=reps, save_path=save_path)     
        
        

def test_method(method_to_run, args, reps: int, save_path: str | None) -> pd.DataFrame:
    results = []
    for i in range(reps):
        res = method_to_run(args) if args is not None else method_to_run()
        print(f"Iteration {i + 1} completed")
        results.append([i + 1, res])
    print()
    results_df = pd.DataFrame(results, columns=["Iteration", "Accuracy"])
    if save_path is not None:
        results_df.to_csv(save_path, index=False)
    return results_df
            
    
