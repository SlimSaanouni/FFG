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
import backend.correlation.cholesky as chol
import backend.validation.consistency as Val


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
        
        # Upload des fichiers
        st.subheader("📁 Import des données")
        
        rfr_file = st.file_uploader(
            "Courbe de taux (RFR_202408.csv)",
            type=['csv'],
            key='rfr'
        )
        
        equity_file = st.file_uploader(
            "Prix des options (EQUITY_Call_Prices.csv)",
            type=['csv'],
            key='equity'
        )
        
        corr_file = st.file_uploader(
            "Matrice de corrélation (Correlation_Matrix.csv)",
            type=['csv'],
            key='corr'
        )
        
        st.divider()
        
        # Paramètres de simulation
        st.subheader("🎲 Paramètres de simulation")
        n_paths = st.number_input(
            "Nombre de trajectoires",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100
        )
        
        T = st.slider(
            "Durée de projection (années)",
            min_value=1,
            max_value=30,
            value=10
        )
        
        st.divider()
        
        # Choix des modèles
        st.subheader("📈 Sélection des modèles")
        
        rates_model = st.selectbox(
            "Modèle de taux",
            ["Vasicek 1F", "Vasicek 2F"]
        )
        
        equity_model = st.selectbox(
            "Modèle actions",
            ["Black-Scholes", "Dupire", "Heston"]
        )
        
        st.info("💡 L'immobilier utilise Black-Scholes par défaut")
    
    # Zone principale
    if rfr_file and equity_file and corr_file:
        # Chargement des données
        try:
            curve_df = pd.read_csv(rfr_file)
            options_df = pd.read_csv(equity_file)
            corr_df = pd.read_csv(corr_file, index_col=0)
            
            st.success("✅ Données chargées avec succès!")
            
            # Tabs pour l'organisation
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
                    
                    fig = px.line(
                        curve_df,
                        x='Maturity',
                        y='Base',
                        title="Courbe des taux sans risque"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Options sur Eurostoxx")
                    st.dataframe(options_df.head())
                    
                    # Surface de volatilité implicite
                    pivot = options_df.pivot_table(
                        values='market_price',
                        index='K',
                        columns='T'
                    )
                    fig = go.Figure(data=[go.Surface(z=pivot.values)])
                    fig.update_layout(title="Prix des options (K, T)")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col3:
                    st.subheader("Matrice de corrélation")
                    st.dataframe(corr_df)
                    
                    fig = px.imshow(
                        corr_df,
                        text_auto=True,
                        color_continuous_scale='RdBu',
                        title="Corrélations cibles"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                if st.button("🚀 Lancer la calibration", type="primary"):
                    with st.spinner("Calibration en cours..."):
                        
                        # Calibration des taux
                        st.subheader("📉 Calibration du modèle de taux")
                        if rates_model == "Vasicek 1F":
                            rates_mdl = VasicekModel1F()
                        else:
                            rates_mdl = VasicekModel2F()
                        
                        rates_mdl.calibrate(curve_df)
                        
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
                            zcb_model = rates_mdl.zcb(maturities)
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=maturities, y=-np.log(zcb_market)/maturities, 
                                                     mode='lines+markers', name='Marché'))
                            fig.add_trace(go.Scatter(x=maturities, y=-np.log(zcb_model)/maturities,
                                                     mode='lines+markers', name='Modèle'))
                            fig.update_layout(title="Courbe des taux calibrée vs marché",
                                              xaxis_title="Maturité (années)",
                                              yaxis_title="Taux (%)")
                            st.plotly_chart(fig, use_container_width=True)
                            
        except:
            st.error("❌ Erreur lors du chargement des données. Vérifiez les fichiers.")
