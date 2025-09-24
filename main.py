"""
Générateur de Scénarios Économiques Risque-Neutre
Application complète avec calibration et simulation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io

from backend.financial_models.vasicek1F import VasicekModel1F
from backend.financial_models.vasicek2F import VasicekModel2F
from backend.financial_models.blackscholes import BlackScholesModel
from backend.financial_models.dupire import DupireModel
from backend.financial_models.heston import HestonModel
import backend.correlation.cholesky as chol
import backend.validation.consistency as Val

def get_equity_model(model_name):
    if model_name == "Black-Scholes":
        return BlackScholesModel()
    elif model_name == "Dupire":
        return DupireModel()
    elif model_name == "Heston":
        return HestonModel()
    else:
        raise ValueError("Modèle actions inconnu")

def run_calibration(curve_df, options_df, rates_model, equity_model_choice):
    # Calibration du modèle de taux
    if rates_model == "Vasicek 1F":
        rates_mdl = VasicekModel1F()
    else:
        rates_mdl = VasicekModel2F()
    rates_mdl.calibrate(curve_df)

    # Calibration modèle actions
    equity_mdl = get_equity_model(equity_model_choice)
    equity_mdl.calibrate(options_df, curve_df)

    # Calibration immobilier (toujours Black-Scholes)
    real_estate_mdl = BlackScholesModel()
    # On suppose que options_df contient aussi des options immobilières ou on clone la calibration
    real_estate_mdl.calibrate(options_df, curve_df)
    return rates_mdl, equity_mdl, real_estate_mdl

def run_simulation(rates_mdl, equity_mdl, real_estate_mdl, corr_matrix, n_paths, T, dt=1/252):
    # Simule les trajectoires pour chaque sous-jacent
    rates_paths = rates_mdl.simulate(n_paths, T, dt)
    equity_paths = equity_mdl.simulate(n_paths, T, dt)
    real_estate_paths = real_estate_mdl.simulate(n_paths, T, dt)

    # Corrélation via Cholesky
    corr_sim = chol.CorrelatedSimulator(corr_matrix)
    rates_corr, equity_corr, real_estate_corr = corr_sim.correlate_paths(
        rates_paths, equity_paths, real_estate_paths
    )
    return rates_corr, equity_corr, real_estate_corr

def concat_paths(rates, equity, real_estate):
    # Concatène les trajectoires dans une DataFrame unique pour export/analyse
    n_paths, n_steps = rates.shape
    df = pd.DataFrame({
        "Rates": rates.flatten(),
        "Equity": equity.flatten(),
        "RealEstate": real_estate.flatten(),
        "Step": np.tile(np.arange(n_steps), n_paths),
        "Path": np.repeat(np.arange(n_paths), n_steps)
    })
    return df

def export_df(df, filename):
    csv = df.to_csv(index=False)
    st.download_button(
        label="📤 Exporter les trajectoires simulées",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

def main():
    st.set_page_config(
        page_title="ESG Generator",
        page_icon="📊",
        layout="wide"
    )

    st.title("🎯 Générateur de Scénarios Économiques Risque-Neutre")
    st.markdown("Application de génération de scénarios économiques avec calibration complète des modèles")
    
    # Sidebar pour les paramètres
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("📁 Import des données")

        rfr_file = st.file_uploader("Courbe de taux (RFR_202408.csv)", type=['csv'], key='rfr')
        equity_file = st.file_uploader("Prix des options (EQUITY_Call_Prices.csv)", type=['csv'], key='equity')
        corr_file = st.file_uploader("Matrice de corrélation (Correlation_Matrix.csv)", type=['csv'], key='corr')

        st.divider()
        st.subheader("🎲 Paramètres de simulation")
        n_paths = st.number_input("Nombre de trajectoires", min_value=100, max_value=10000, value=1000, step=100)
        T = st.slider("Durée de projection (années)", min_value=1, max_value=30, value=10)
        dt = 1/252

        st.divider()
        st.subheader("📈 Sélection des modèles")
        rates_model = st.selectbox("Modèle de taux", ["Vasicek 1F", "Vasicek 2F"])
        equity_model = st.selectbox("Modèle actions", ["Black-Scholes", "Dupire", "Heston"])
        st.info("💡 L'immobilier utilise Black-Scholes par défaut")

    # Zone principale
    if rfr_file and equity_file and corr_file:
        try:
            curve_df = pd.read_csv(rfr_file)
            options_df = pd.read_csv(equity_file)
            corr_df = pd.read_csv(corr_file, index_col=0)
            corr_matrix = corr_df.values

            st.success("✅ Données chargées avec succès!")
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Données d'entrée",
                "⚙️ Calibration",
                "🎯 Simulation",
                "✅ Tests de validation"
            ])

            with tab1:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("Courbe de taux")
                    st.dataframe(curve_df.head())
                    fig = px.line(curve_df, x='Maturity', y='Base', title="Courbe des taux sans risque")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.subheader("Options sur Eurostoxx")
                    st.dataframe(options_df.head())
                    pivot = options_df.pivot_table(values='market_price', index='K', columns='T')
                    fig = go.Figure(data=[go.Surface(z=pivot.values)])
                    fig.update_layout(title="Prix des options (K, T)")
                    st.plotly_chart(fig, use_container_width=True)
                with col3:
                    st.subheader("Matrice de corrélation")
                    st.dataframe(corr_df)
                    fig = px.imshow(corr_df, text_auto=True, color_continuous_scale='RdBu', title="Corrélations cibles")
                    st.plotly_chart(fig, use_container_width=True)

            with tab2:
                if st.button("🚀 Lancer la calibration", type="primary"):
                    with st.spinner("Calibration en cours..."):
                        rates_mdl, equity_mdl, real_estate_mdl = run_calibration(
                            curve_df, options_df, rates_model, equity_model
                        )

                        col1, col2 = st.columns(2)
                        with col1:
                            st.success(f"✅ {rates_model} calibré")
                            if rates_model == "Vasicek 1F":
                                st.write(f"**a (mean reversion):** {rates_mdl.a:.4f}")
                                st.write(f"**b (long-term level):** {rates_mdl.b:.4f}")
                                st.write(f"**σ (volatility):** {rates_mdl.sigma:.4f}")
                            else:
                                st.write(f"**a₁:** {rates_mdl.a1:.4f}")
                                st.write(f"**a₂:** {rates_mdl.a2:.4f}")
                                st.write(f"**σ₁:** {rates_mdl.sigma1:.4f}")
                                st.write(f"**σ₂:** {rates_mdl.sigma2:.4f}")
                                st.write(f"**ρ:** {rates_mdl.rho:.4f}")
                                st.write(f"**x₀:** {rates_mdl.x0:.4f}")
                                st.write(f"**y₀:** {rates_mdl.y0:.4f}")
                        with col2:
                            # Graphique de la courbe calibrée
                            maturities = curve_df['Maturity'].values
                            rates = curve_df['Base'].values / 100
                            zcb_market = np.exp(-rates * maturities)
                            # Calcul ZC modèle pour comparaison
                            if rates_model == "Vasicek 1F":
                                zcb_model = np.array([
                                    np.exp((rates_mdl.b - rates_mdl.sigma**2/(2 * rates_mdl.a**2)) * 
                                           ((1 - np.exp(-rates_mdl.a * T)) / rates_mdl.a - T) -
                                           rates_mdl.sigma**2/(4 * rates_mdl.a) * ((1 - np.exp(-rates_mdl.a * T)) / rates_mdl.a)**2
                                           ) * np.exp(-(1 - np.exp(-rates_mdl.a * T)) / rates_mdl.a * rates_mdl.r0)
                                    for T in maturities
                                ])
                            else:
                                # Vasicek 2F: on utilise la méthode zcb du modèle si elle existe
                                zcb_model = np.array([
                                    rates_mdl.calibrate(curve_df) or 0 for T in maturities
                                ])
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=maturities, y=-np.log(zcb_market)/maturities, 
                                                     mode='lines+markers', name='Marché'))
                            fig.add_trace(go.Scatter(x=maturities, y=-np.log(zcb_model)/maturities,
                                                     mode='lines+markers', name='Modèle'))
                            fig.update_layout(title="Courbe des taux calibrée vs marché",
                                              xaxis_title="Maturité (années)",
                                              yaxis_title="Taux (%)")
                            st.plotly_chart(fig, use_container_width=True)

                        st.session_state["rates_mdl"] = rates_mdl
                        st.session_state["equity_mdl"] = equity_mdl
                        st.session_state["real_estate_mdl"] = real_estate_mdl
                        st.session_state["corr_matrix"] = corr_matrix

            with tab3:
                if "rates_mdl" in st.session_state and st.button("🎲 Lancer la simulation", type="primary"):
                    with st.spinner("Simulation en cours..."):
                        rates_mdl = st.session_state["rates_mdl"]
                        equity_mdl = st.session_state["equity_mdl"]
                        real_estate_mdl = st.session_state["real_estate_mdl"]
                        corr_matrix = st.session_state["corr_matrix"]

                        rates_corr, equity_corr, real_estate_corr = run_simulation(
                            rates_mdl, equity_mdl, real_estate_mdl, corr_matrix, n_paths, T, dt
                        )

                        # Visualisation des trajectoires
                        st.success("✅ Simulation terminée!")
                        st.subheader("Trajectoires simulées - Taux")
                        fig_rates = px.line(
                            pd.DataFrame(rates_corr.T),
                            title="Rates paths (corrélés)",
                        )
                        st.plotly_chart(fig_rates, use_container_width=True)

                        st.subheader("Trajectoires simulées - Actions")
                        fig_eq = px.line(
                            pd.DataFrame(equity_corr.T),
                            title="Equity paths (corrélés)",
                        )
                        st.plotly_chart(fig_eq, use_container_width=True)

                        st.subheader("Trajectoires simulées - Immobilier")
                        fig_re = px.line(
                            pd.DataFrame(real_estate_corr.T),
                            title="Real Estate paths (corrélés)",
                        )
                        st.plotly_chart(fig_re, use_container_width=True)

                        # Concaténation des trajectoires
                        df_paths = concat_paths(rates_corr, equity_corr, real_estate_corr)
                        st.dataframe(df_paths.head())
                        export_df(df_paths, "trajectoires_simulees.csv")
                        st.session_state["df_paths"] = df_paths
                        st.session_state["rates_corr"] = rates_corr
                        st.session_state["equity_corr"] = equity_corr
                        st.session_state["real_estate_corr"] = real_estate_corr

            with tab4:
                if all(k in st.session_state for k in ["rates_corr", "equity_corr", "real_estate_corr", "corr_matrix"]):
                    st.subheader("Tests de validation")
                    rates_corr = st.session_state["rates_corr"]
                    equity_corr = st.session_state["equity_corr"]
                    real_estate_corr = st.session_state["real_estate_corr"]
                    corr_matrix = st.session_state["corr_matrix"]

                    st.write("Test de martingale sur les taux :")
                    mart_res = Val.MarketConsistencyTests.test_martingale(rates_corr, np.mean(curve_df['Base'].values/100), dt)
                    st.json(mart_res)

                    st.write("Test market-consistency sur les prix d'options :")
                    equity_mdl = st.session_state["equity_mdl"]
                    opt_res = Val.MarketConsistencyTests.test_option_prices(equity_mdl, options_df)
                    st.json(opt_res)

                    st.write("Test de corrélation empirique vs théorique :")
                    corr_res = Val.MarketConsistencyTests.test_correlation(
                        rates_corr, equity_corr, real_estate_corr, corr_matrix
                    )
                    st.json(corr_res)

        except Exception as e:
            st.error(f"❌ Erreur lors du chargement des données ou de l'exécution : {e}")

if __name__ == "__main__":
    main()