import numpy as np
from scipy.optimize import minimize


class VasicekModel2F:
    """Modèle de Vasicek à 2 facteurs pour plus de flexibilité"""
    
    def __init__(self):
        self.a1 = None
        self.a2 = None
        self.sigma1 = None
        self.sigma2 = None
        self.rho = None
        self.x0 = None
        self.y0 = None
        
    def calibrate(self, curve_df):
        """Calibration du modèle Vasicek 2F"""
        maturities = curve_df['Maturity'].values
        rates = curve_df['Base'].values / 100
        r0 = rates[0]
        
        # Prix ZC observés
        zcb_market = np.exp(-rates * maturities)
        
        def zcb_price_2f(T, a1, a2, sigma1, sigma2, rho, x0, y0):
            """Prix ZC dans le modèle 2 facteurs"""
            B1 = (1 - np.exp(-a1 * T)) / a1
            B2 = (1 - np.exp(-a2 * T)) / a2
            
            var_term = (sigma1**2 / (2*a1**3)) * (2*a1*T - 3 + 4*np.exp(-a1*T) - np.exp(-2*a1*T))
            var_term += (sigma2**2 / (2*a2**3)) * (2*a2*T - 3 + 4*np.exp(-a2*T) - np.exp(-2*a2*T))
            var_term += 2 * rho * sigma1 * sigma2 / (a1 * a2 * (a1 + a2)) * (
                T + (np.exp(-a1*T) - 1)/a1 + (np.exp(-a2*T) - 1)/a2 - 
                (np.exp(-(a1+a2)*T) - 1)/(a1+a2)
            )
            
            A = np.exp(var_term / 2)
            return A * np.exp(-B1 * x0 - B2 * y0)
        
        def objective(params):
            a1, a2, sigma1, sigma2, rho = params
            if a1 <= 0 or a2 <= 0 or sigma1 <= 0 or sigma2 <= 0 or abs(rho) >= 1:
                return 1e10
            x0 = r0 / 2
            y0 = r0 / 2
            zcb_model = np.array([zcb_price_2f(T, a1, a2, sigma1, sigma2, rho, x0, y0) 
                                 for T in maturities])
            return np.sum((zcb_model - zcb_market)**2)
        
        result = minimize(
            objective,
            x0=[0.5, 0.3, 0.01, 0.01, 0.3],
            bounds=[(0.01, 2), (0.01, 2), (0.001, 0.1), (0.001, 0.1), (-0.99, 0.99)],
            method='L-BFGS-B'
        )
        
        self.a1, self.a2, self.sigma1, self.sigma2, self.rho = result.x
        self.x0 = r0 / 2
        self.y0 = r0 / 2
        return self
    
    def simulate(self, n_paths, T, dt=1/252):
        """Simulation des trajectoires pour le modèle 2F"""
        n_steps = int(T / dt)
        x_paths = np.zeros((n_paths, n_steps + 1))
        y_paths = np.zeros((n_paths, n_steps + 1))
        x_paths[:, 0] = self.x0
        y_paths[:, 0] = self.y0
        
        sqrt_dt = np.sqrt(dt)
        for t in range(n_steps):
            z1 = np.random.randn(n_paths)
            z2 = np.random.randn(n_paths)
            dW1 = z1 * sqrt_dt
            dW2 = (self.rho * z1 + np.sqrt(1 - self.rho**2) * z2) * sqrt_dt
            
            x_paths[:, t+1] = x_paths[:, t] - self.a1 * x_paths[:, t] * dt + self.sigma1 * dW1
            y_paths[:, t+1] = y_paths[:, t] - self.a2 * y_paths[:, t] * dt + self.sigma2 * dW2
        
        return x_paths + y_paths
