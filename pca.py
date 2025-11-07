import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class PCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=0.90):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.eig_vals = None
        self.eig_vecs = None
        
    def fit(self, X, y=None):
        # Flatten
        n_epochs, n_channels, n_times = X.shape
        X_flat = X.reshape(n_epochs, n_channels * n_times)
        
        # Center the data (substract mean)
        self.mean_ = np.mean(X_flat, axis=0)
        X_centered = X_flat - self.mean_
        
        # Covariance matrix (features x features)
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Get Components by eigen decomposition
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
        
        # Sort in descending order
        sorted_idx = np.argsort(eig_vals)[::-1]
        self.eig_vals = eig_vals[sorted_idx]
        self.eig_vecs = eig_vecs[:, sorted_idx]
        
        # Filter out near-zero eigenvalues
        valid_idx = self.eig_vals > 1e-10
        self.eig_vals = self.eig_vals[valid_idx]
        self.eig_vecs = self.eig_vecs[:, valid_idx]
        
        self.compute_explained_variance_ratio()
        self.select_components()
        
        return self
    
    def transform(self, X):
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("You must fit PCAFeatureExtractor before executing the transform function")

        n_epochs, n_channels, n_times = X.shape
        X_flat = X.reshape(n_epochs, n_channels * n_times)
        
        # Center the data
        X_centered = X_flat - self.mean_
        
        # Project onto principal components
        X_projected = np.dot(X_centered, self.components_)
        
        return X_projected
    
    def compute_explained_variance_ratio(self):
        total_var = np.sum(self.eig_vals)
        self.explained_variance_ratio_ = self.eig_vals / total_var

    def select_components(self):
        if isinstance(self.n_components, float) and 0 < self.n_components < 1:
            cumulative_var = np.cumsum(self.explained_variance_ratio_)
            n_keep = np.searchsorted(cumulative_var, self.n_components) + 1
        else:
            n_keep = int(self.n_components)

        self.n_selected_ = n_keep
        self.components_ = self.eig_vecs[:, :n_keep]
        self.explained_variance_ = self.eig_vals[:n_keep]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:n_keep]