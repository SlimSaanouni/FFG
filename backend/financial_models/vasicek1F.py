import numpy as np
from scipy.optimize import minimize

# ============================
# MODÈLES DE TAUX D'INTÉRÊT
# ============================

class VasicekModel1F:
    """Modèle de Vasicek à 1 facteur: dr(t) = a(b - r(t))dt + σ dW(t)"""
    
    def __init__(self):
        self.a = None  # Vitesse de retour à la moyenne
        self.b = None  # Niveau de long terme
        self.sigma = None  # Volatilité
        self.r0 = None  # Taux initial
        
    def calibrate(self, curve_df):
        """
        Calibration du modèle Vasicek 1F sur la courbe des taux
        Minimisation de l'écart entre prix ZC théoriques et observés
        """
        maturities = curve_df['Maturity'].values
        rates = curve_df['Base'].values / 100  # Conversion en décimal
        self.r0 = rates[0]
        
        # Prix zéro-coupon observés
        zcb_market = np.exp(-rates * maturities)
        
        def zcb_price(T, a, b, sigma, r0):
            """Prix d'un zéro-coupon dans le modèle Vasicek"""
            B = (1 - np.exp(-a * T)) / a
            A = np.exp((b - sigma**2/(2*a**2)) * (B - T) - sigma**2/(4*a) * B**2)
            return A * np.exp(-B * r0)
        
        def objective(params):
            a, b, sigma = params
            if a <= 0 or sigma <= 0:
                return 1e10
            zcb_model = np.array([zcb_price(T, a, b, sigma, self.r0) for T in maturities])
            return np.sum((zcb_model - zcb_market)**2)
        
        # Optimisation avec contraintes raisonnables
        result = minimize(
            objective, 
            x0=[0.1, 0.03, 0.01],
            bounds=[(0.01, 2), (0.001, 0.1), (0.001, 0.1)],
            method='L-BFGS-B'
        )
        
        self.a, self.b, self.sigma = result.x
        return self
    
    def simulate(self, n_paths, T, dt=1/252):
        """Simulation des trajectoires de taux"""
        n_steps = int(T / dt)
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.r0
        
        sqrt_dt = np.sqrt(dt)
        for t in range(n_steps):
            dW = np.random.randn(n_paths) * sqrt_dt
            paths[:, t+1] = paths[:, t] + self.a * (self.b - paths[:, t]) * dt + self.sigma * dW
            
        return paths
