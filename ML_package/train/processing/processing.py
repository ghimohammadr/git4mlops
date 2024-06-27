from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from train.config import config
from sklearn.base import BaseEstimator, TransformerMixin

class normalization(BaseEstimator, TransformerMixin):
    def __init__(self, variable=None):
        self.variable  = variable

    def fit(self, X=None):
        return self
    
    def transform(self, X):
        if self.variable == 1:
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        elif self.variable == 2:
            X = X
        return X

class Drop_Features(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop = None):
        self.variables_to_drop = variables_to_drop

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X = X.drop(columns = self.variables_to_drop)
        return X

class Add_Delay(BaseEstimator, TransformerMixin):
    def __init__(self, Delay = None):
        self.Delay = Delay

    def fit(self, X, y):
        X2, y2 = [], []
        for k in range(len(X)-self.Delay-1):
            a = X[k:(k+self.Delay), :]
            flatted = a.flatten()
            self.X2.append(flatted) 
            self.y2.append(y[k + self.Delay]) 
        return self
    
    def transform(self, X, y):
        X = self.X2
        X = self.y2

        return X, y