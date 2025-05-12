from typing import List, Literal, Tuple
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer


def get_dataset(dataset_name: Literal['digits', 'wine', 'breast_cancer', 'pima', 'users_vs_bots', 'students_dropout']) -> Tuple[np.ndarray, np.ndarray]:
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
        
    elif dataset_name == 'users_vs_bots':
        df = pd.read_csv("./../datasets/bots_vs_users.csv")
        df.dropna(axis=1, how='all', inplace=True)
        df.fillna(-1, inplace=True)
        df.replace('Unknown', -1, inplace=True)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        imputer = SimpleImputer(strategy='mean')
        X = df.drop('target', axis=1)
        X = imputer.fit_transform(X)
        y = df['target'].values
        
    elif dataset_name == 'students_dropout':
        ds = pd.read_csv("./../datasets/students_dropout.csv", sep=';')
        label_encoder = LabelEncoder()
        ds["Target"] = label_encoder.fit_transform(ds["Target"])
        X = ds.iloc[:,:-1].values
        sc = StandardScaler()
        X = sc.fit_transform(X)
        y = ds.iloc[:,-1].values
    
    else:
        raise ValueError("Unsupported dataset")
    return X, y