import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import cvxpy as cp
from scipy.optimize import minimize_scalar

# Sidebar for user inputs
st.sidebar.header("Portfolio Settings")
assets = {
    "Stocks": ["SPY", "EFA", "VWO"],
    "Corporate Bonds": ["LQD", "HYG"],
    "Sovereign Bonds": ["TLT", "BNDX"],
    "Commodities": ["GLD", "USO"]
}
selected_assets = st.sidebar.multiselect("Select Assets", options=sum(assets.values(), []), default=sum(assets.values(), []))
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-10-01"))  # Adjusted to avoid future data

# Fetch data with robust MultiIndex handling
@st.cache_data
def get_data(tickers, start, end):
    try:
        # Download data
        raw_data = yf.download(tickers, start=start, end=end)
        # Extract Adj Close and handle MultiIndex
        if isinstance(raw_data.columns, pd.MultiIndex):
            adj_close = raw_data.xs("Adj Close", axis=1, level=1)
        else:
            adj_close = raw_data["Adj Close"]
        return adj_close.dropna()
    except Exception as e:
        st.error(f"Failed to fetch data: {str(e)}")
        return pd.DataFrame()

if st.sidebar.button("Optimize Portfolio"):
    if not selected_assets:
        st.error("Please select at least one asset.")
    else:
        data = get_data(selected_assets, start_date, end_date)
        if data.empty:
            st.error("No data available for the selected assets and date range. Please adjust your selection.")
        else:
            returns = data.pct_change().dropna()
            
            # Calculate expected returns and covariance (annualized)
            mu = returns.mean() * 252
            S = returns.cov() * 252
            n = len(selected_assets)
            S_np = S.to_numpy()
            mu_np = mu.to_numpy()
            
            # Function to solve for weights given rho
            def solve_with_rho(rho):
                w = cp.Variable(n)
                objective = cp.Minimize(cp.quad_form(w, S_np) - rho * cp.sum(cp.log(w)))
                constraints = [cp.sum(w) == 1, w >= 1e-6]
                prob = cp.Problem(objective, constraints)
                try:
                    prob.solve(solver=cp.ECOS)
                    if prob.status == cp.OPTIMAL:
                        return w.value
                    else:
                        return None
                except:
                    return None
            
            # Function to compute variance of risk contributions
            def get_rc_var(rho):
                w = solve_with_rho(rho)
                if w is None or np.any(w < 0):
                    return np.inf
                var = w @ S_np @ w
                sigma = np.sqrt(var)
                MRC = S_np @ w
                RC = w * MRC / sigma
                return np.var(RC)
            
            # Optimize rho to minimize variance of RC
            res = minimize_scalar(get_rc_var, bounds=(1e-6, 1e-1), method='bounded', tol=1e-5)
            best_rho = res.x
            weights = solve_with_rho(best_rho)
            
            if weights is None:
                st.error("Optimization failed. Try adjusting parameters.")
            else:
                # Clean weights (set very small to zero)
                weights = np.where(np.abs(weights) < 1e-4, 0, weights)
                weights /= np.sum(weights)  # Renormalize if needed
                
                # Calculate portfolio metrics
                port_var = weights @ S_np @ weights
                sigma = np.sqrt(port_var)
                MRC = S_np @ weights
                risk_contrib = weights * MRC / sigma
                total_risk = np.sum(risk_contrib)
                risk_contrib_pct = (risk_contrib / total_risk) * 100
                
                # Display results
                st.header("Optimized Portfolio")
                st.write("Weights:")
                weight_series = pd.Series(weights, index=selected_assets)
                st.write(weight_series)
                
                st.write("Risk Contributions (%):")
                rc_series = pd.Series(risk_contrib_pct, index=selected_assets)
                st.write(rc_series)
                
                # Visualization
                fig = go.Figure(data=[
                    go.Pie(labels=selected_assets, values=weights, hole=0.3)
                ])
                fig.update_layout(title="Portfolio Allocation")
                st.plotly_chart(fig)
                
                fig2 = go.Figure(data=[
                    go.Bar(x=selected_assets, y=risk_contrib_pct)
                ])
                fig2.update_layout(title="Risk Contributions")
                st.plotly_chart(fig2)
                
                # Performance metrics
                exp_ret = np.dot(mu_np, weights)
                sharpe = exp_ret / sigma if sigma > 0 else 0
                st.write(f"Expected Annual Return: {exp_ret:.2%}")
                st.write(f"Annual Volatility: {sigma:.2%}")
                st.write(f"Sharpe Ratio: {sharpe:.2f}")

# Add some interactivity and documentation
st.sidebar.markdown("### Instructions")
st.sidebar.write("1. Select assets and date range.")
st.sidebar.write("2. Click 'Optimize Portfolio' to see results.")
st.sidebar.write("3. Explore the visualizations and metrics.")
