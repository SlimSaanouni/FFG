import numpy as np


class CorrelatedSimulator:
    """Gère la corrélation entre les différents actifs"""
    
    def __init__(self, correlation_matrix):
        self.correlation = correlation_matrix
        self.cholesky = np.linalg.cholesky(correlation_matrix)
        
    def correlate_paths(self, rates_randoms, equity_randoms, real_estate_randoms):
        """Applique la corrélation via Cholesky"""
        n_paths, n_steps = rates_randoms.shape
        
        correlated = np.zeros((3, n_paths, n_steps))
        
        for t in range(n_steps):
            uncorrelated = np.array([
                rates_randoms[:, t],
                equity_randoms[:, t],
                real_estate_randoms[:, t]
            ])
            
            # Application de Cholesky
            correlated[:, :, t] = self.cholesky @ uncorrelated
            
        return correlated[0], correlated[1], correlated[2]
