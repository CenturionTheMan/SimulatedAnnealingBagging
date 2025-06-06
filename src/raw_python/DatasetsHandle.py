from typing import List, Literal, Tuple
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer


def get_dataset(dataset_name: Literal['digits', 'wine', 'breast_cancer', 'pima', 'users_vs_bots', 'students_dropout', 'obesity']) -> Tuple[np.ndarray, np.ndarray]:
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

        #print labels with mapped values
        print(label_encoder.classes_)
        print(label_encoder.transform(label_encoder.classes_))
        
        
        X = ds.iloc[:,:-1].values
        sc = StandardScaler()
        X = sc.fit_transform(X)
        y = ds.iloc[:,-1].values
        
    elif dataset_name == 'obesity':
        df = pd.read_csv("./../datasets/obesity.csv")
        y_tmp = df["NObeyesdad"]
        X_tmp = df.drop("NObeyesdad", axis=1)
        X_encoded = X_tmp.copy()
        label_encoder = LabelEncoder()
        for col in X_encoded.select_dtypes(include=['object']).columns:
            X_encoded[col] = label_encoder.fit_transform(X_encoded[col])
        y_encoded = label_encoder.fit_transform(y_tmp)
        #print labels with mapped values
        print(label_encoder.classes_)
        print(label_encoder.transform(label_encoder.classes_))
        y = y_encoded
        X = X_encoded.values        
    
    else:
        raise ValueError("Unsupported dataset")
    return X, y