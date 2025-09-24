import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class DupireModel:
    """Modèle à volatilité locale de Dupire"""
    
    def __init__(self):
        self.S0 = None
        self.local_vol_grid = None
        self.K_grid = None
        self.T_grid = None
        
    def calibrate(self, options_df, curve_df):
        """Calibration de la surface de volatilité locale"""
        self.S0 = options_df['S'].iloc[0]
        
        # Création de la grille de volatilité locale
        strikes = options_df['K'].unique()
        maturities = options_df['T'].unique()
        
        # Matrice de volatilité implicite
        impl_vol_surface = np.zeros((len(maturities), len(strikes)))
        
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                opt = options_df[(options_df['T'] == T) & (options_df['K'] == K)]
                if not opt.empty:
                    # Calibration de la vol implicite pour ce point
                    market_price = opt['market_price'].iloc[0]
                    r = opt['r'].iloc[0]
                    
                    def obj(vol):
                        bs_price = self.bs_call(self.S0, K, r, T, vol)
                        return (bs_price - market_price)**2
                    
                    res = minimize(obj, x0=[0.2], bounds=[(0.01, 1.0)], method='L-BFGS-B')
                    impl_vol_surface[i, j] = res.x[0]
        
        # Formule de Dupire pour la volatilité locale
        # σ_loc²(K,T) = (∂C/∂T) / (0.5 * K² * ∂²C/∂K²)
        # Approximation simplifiée ici
        self.local_vol_grid = impl_vol_surface
        self.K_grid = strikes
        self.T_grid = maturities
        return self
    
    def bs_call(self, S, K, r, T, sigma):
        """Prix BS pour la calibration"""
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    
    def get_local_vol(self, S, t):
        """Interpolation de la volatilité locale"""
        # Interpolation bilinéaire simplifiée
        if t > self.T_grid[-1]:
            t = self.T_grid[-1]
        if S < self.K_grid[0]:
            S = self.K_grid[0]
        if S > self.K_grid[-1]:
            S = self.K_grid[-1]
            
        t_idx = np.searchsorted(self.T_grid, t)
        k_idx = np.searchsorted(self.K_grid, S)
        
        if t_idx >= len(self.T_grid):
            t_idx = len(self.T_grid) - 1
        if k_idx >= len(self.K_grid):
            k_idx = len(self.K_grid) - 1
            
        return self.local_vol_grid[t_idx, k_idx]
    
    def simulate(self, n_paths, T, dt=1/252):
        """Simulation avec volatilité locale"""
        n_steps = int(T / dt)
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0
        
        sqrt_dt = np.sqrt(dt)
        
        for t in range(n_steps):
            current_time = t * dt
            for i in range(n_paths):
                local_vol = self.get_local_vol(paths[i, t], current_time)
                dW = np.random.randn() * sqrt_dt
                paths[i, t+1] = paths[i, t] * (1 + local_vol * dW)
        
        return paths