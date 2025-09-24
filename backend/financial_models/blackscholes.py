import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.interpolate import interp1d


class BlackScholesModel:
    """Modèle de Black-Scholes: dS/S = r dt + σ dW"""
    
    def __init__(self):
        self.S0 = None
        self.sigma = None
        self.r_func = None  # Fonction d'interpolation des taux
        
    def bs_price(self, S, K, r, T, sigma):
        """Prix d'un call européen selon Black-Scholes"""
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    
    def calibrate(self, options_df, curve_df):
        """Calibration de la volatilité implicite moyenne"""
        self.S0 = options_df['S'].iloc[0]
        
        # Interpolation de la courbe des taux
        maturities = curve_df['Maturity'].values
        rates = curve_df['Base'].values / 100
        self.r_func = interp1d(maturities, rates, kind='linear', fill_value='extrapolate')
        
        def objective(sigma):
            total_error = 0
            for _, row in options_df.iterrows():
                model_price = self.bs_price(row['S'], row['K'], row['r'], row['T'], sigma[0])
                total_error += (model_price - row['market_price'])**2
            return total_error
        
        result = minimize(objective, x0=[0.2], bounds=[(0.01, 1.0)], method='L-BFGS-B')
        self.sigma = result.x[0]
        return self
    
    def simulate(self, n_paths, T, dt=1/252):
        """Simulation GBM avec taux déterministe"""
        n_steps = int(T / dt)
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0
        
        sqrt_dt = np.sqrt(dt)
        times = np.linspace(0, T, n_steps + 1)
        
        for t in range(n_steps):
            r_t = self.r_func(times[t])
            dW = np.random.randn(n_paths) * sqrt_dt
            paths[:, t+1] = paths[:, t] * np.exp((r_t - 0.5*self.sigma**2)*dt + self.sigma*dW)
            
        return paths