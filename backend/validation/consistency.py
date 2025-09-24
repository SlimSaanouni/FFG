import numpy as np


class MarketConsistencyTests:
    """Tests de cohérence avec le marché"""
    
    @staticmethod
    def test_martingale(paths, r, dt=1/252):
        """Test de la propriété de martingale sous Q"""
        n_steps = paths.shape[1] - 1
        discounted = np.zeros_like(paths)
        
        for t in range(n_steps + 1):
            discount_factor = np.exp(-r * t * dt)
            discounted[:, t] = paths[:, t] * discount_factor
            
        # Vérification E[S(t)/B(t)] = S(0)/B(0)
        initial_value = discounted[:, 0].mean()
        final_value = discounted[:, -1].mean()
        
        relative_error = abs(final_value - initial_value) / initial_value
        return {
            'test': 'Martingale',
            'passed': relative_error < 0.05,
            'error': relative_error,
            'initial': initial_value,
            'final': final_value
        }
    
    @staticmethod
    def test_option_prices(model, options_df):
        """Vérification de la reproduction des prix d'options"""
        errors = []
        
        for _, opt in options_df.iterrows():
            if hasattr(model, 'bs_price'):
                model_price = model.bs_price(
                    opt['S'], opt['K'], opt['r'], opt['T'], model.sigma
                )
            else:
                # Pour Heston/Dupire, utiliser une méthode Monte Carlo
                n_mc = 10000
                paths = model.simulate(n_mc, opt['T'])
                payoffs = np.maximum(paths[:, -1] - opt['K'], 0)
                model_price = np.exp(-opt['r'] * opt['T']) * payoffs.mean()
                
            rel_error = abs(model_price - opt['market_price']) / opt['market_price']
            errors.append(rel_error)
            
        mean_error = np.mean(errors)
        return {
            'test': 'Option Pricing',
            'passed': mean_error < 0.10,
            'mean_error': mean_error,
            'max_error': np.max(errors)
        }
    
    @staticmethod
    def test_correlation(rates_paths, equity_paths, real_estate_paths, target_corr):
        """Test de la corrélation empirique vs théorique"""
        # Calcul des rendements
        rates_returns = np.diff(rates_paths, axis=1) / rates_paths[:, :-1]
        equity_returns = np.diff(equity_paths, axis=1) / equity_paths[:, :-1]
        re_returns = np.diff(real_estate_paths, axis=1) / real_estate_paths[:, :-1]
        
        # Corrélations empiriques
        emp_corr = np.corrcoef([
            rates_returns.flatten(),
            equity_returns.flatten(),
            re_returns.flatten()
        ])
        
        # Comparaison avec la cible
        corr_error = np.abs(emp_corr - target_corr).max()
        
        return {
            'test': 'Correlation',
            'passed': corr_error < 0.1,
            'max_error': corr_error,
            'empirical_corr': emp_corr,
            'target_corr': target_corr
        }