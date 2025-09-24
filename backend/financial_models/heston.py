import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm


class HestonModel:
    """Modèle de Heston avec volatilité stochastique"""
    
    def __init__(self):
        self.S0 = None
        self.v0 = None  # Variance initiale
        self.kappa = None  # Vitesse de retour à la moyenne
        self.theta = None  # Variance long terme
        self.xi = None  # Vol de la vol
        self.rho = None  # Corrélation
        
    def calibrate(self, options_df, curve_df):
        """Calibration du modèle de Heston par optimisation globale"""
        self.S0 = options_df['S'].iloc[0]
        
        def heston_call_price(S0, K, r, T, v0, kappa, theta, xi, rho):
            """Prix d'un call dans le modèle de Heston (approximation)"""
            # Utilisation de l'approximation de Carr-Madan ou formule semi-fermée
            # Ici, approximation simplifiée pour la pédagogie
            sigma_avg = np.sqrt(v0 * (1 - np.exp(-kappa * T)) / (kappa * T) + 
                              theta * (1 - (1 - np.exp(-kappa * T)) / (kappa * T)))
            
            # Ajustement pour la corrélation et vol de vol
            adjustment = 1 + 0.25 * xi * rho * np.sqrt(T)
            sigma_eff = sigma_avg * adjustment
            
            # Prix Black-Scholes ajusté
            d1 = (np.log(S0/K) + (r + 0.5*sigma_eff**2)*T) / (sigma_eff*np.sqrt(T))
            d2 = d1 - sigma_eff*np.sqrt(T)
            return S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        
        def objective(params):
            v0, kappa, theta, xi, rho = params
            if v0 <= 0 or kappa <= 0 or theta <= 0 or xi <= 0 or abs(rho) >= 1:
                return 1e10
            
            total_error = 0
            for _, row in options_df.iterrows():
                model_price = heston_call_price(
                    self.S0, row['K'], row['r'], row['T'],
                    v0, kappa, theta, xi, rho
                )
                total_error += (model_price - row['market_price'])**2
            return total_error
        
        # Optimisation globale pour éviter les minima locaux
        result = differential_evolution(
            objective,
            bounds=[(0.01, 0.5), (0.1, 5), (0.01, 0.5), (0.1, 2), (-0.9, 0.9)],
            maxiter=100,
            popsize=15
        )
        
        self.v0, self.kappa, self.theta, self.xi, self.rho = result.x
        return self
    
    def simulate(self, n_paths, T, dt=1/252):
        """Simulation du modèle de Heston avec schéma d'Euler"""
        n_steps = int(T / dt)
        S_paths = np.zeros((n_paths, n_steps + 1))
        v_paths = np.zeros((n_paths, n_steps + 1))
        
        S_paths[:, 0] = self.S0
        v_paths[:, 0] = self.v0
        
        sqrt_dt = np.sqrt(dt)
        
        for t in range(n_steps):
            z1 = np.random.randn(n_paths)
            z2 = np.random.randn(n_paths)
            
            dW_S = z1 * sqrt_dt
            dW_v = (self.rho * z1 + np.sqrt(1 - self.rho**2) * z2) * sqrt_dt
            
            # Schéma de Milstein pour la variance (évite les valeurs négatives)
            v_paths[:, t+1] = np.abs(v_paths[:, t] + 
                                    self.kappa * (self.theta - v_paths[:, t]) * dt +
                                    self.xi * np.sqrt(np.abs(v_paths[:, t])) * dW_v)
            
            # Prix de l'actif
            S_paths[:, t+1] = S_paths[:, t] * np.exp(-0.5 * v_paths[:, t] * dt + 
                                                    np.sqrt(v_paths[:, t]) * dW_S)
        
        return S_paths