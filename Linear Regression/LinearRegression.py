import numpy as np
class LinearRegression:
    def __init__(self, lr = 0.001 , n_iters =30):
        self.lr = lr
        self.n_iters = n_iters
        self.weight = None
        self.bias = None
    

    def fit(self,X,y):
        n_samples , n_features = X.shape
        # import pdb
        # pdb.set_trace()
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            
            y_pred = np.dot(X, self.weights) + self.bias
            
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

    

            if _ % 10 == 0:
                print(f"Iteration {_}, db = {db}")


            self.weights = self.weights - self.lr * dw
            
            self.bias = self.bias - self.lr * db

            # if np.isnan(self.weights).any() or np.isnan(self.bias):
            #     print("NaN encountered!")
                # break

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred


