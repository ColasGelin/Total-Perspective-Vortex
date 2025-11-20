import numpy as np

class LDA:
    def __init__(self, shrinkage=1e-4):
        self.shrinkage = shrinkage
        self.w = None # The direction that best separates the classes w = S^(-1) @ (mu1 - mu0)
        self.b = None 
    
    def fit(self, X, y):
        # Get class labels
        classes = np.unique(y)
        
        # Get center point of each class
        mu0 = X[y == classes[0]].mean(axis=0)
        mu1 = X[y == classes[1]].mean(axis=0)
        # (mu0 - mu1) ← between-class variance
        
        # Pooled covariance matrix # ← within-class variance
        S = np.cov(X.T)
        # Make S Invertible (If one of the covariance matrix eigenvalues is 0, S is not invertible)
        S += self.shrinkage * np.eye(X.shape[1])
        # Invert S
        S_inv = np.linalg.inv(S)
        
        # LDA weights self.w = [a1, a2, a3, ..., an] or w = S^(-1) @ (mu1 - mu0)
        self.w = S_inv @ (mu1 - mu0) 
        # Make sure the decision boundary is in the middle of the two classes
        self.b = -0.5 * (mu1 + mu0) @ self.w
        self.classes = classes
        return self
    
    def predict(self, X):
        # score = w1*x1 + w2*x2 + w3*x3 + ... + wn*xn + b
        scores = X @ self.w + self.b
        # Return class based on sign of score
        return np.where(scores > 0, self.classes[1], self.classes[0])