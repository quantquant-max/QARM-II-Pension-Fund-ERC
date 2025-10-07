import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import cvxpy as cp
from scipy.optimize import minimize_scalar
from datetime import datetime

# Custom styling for a professional look
st.set_page_config(page_title="Pension Fund Optimizer", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stSidebar {
        background-color: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stHeader {
        color: #2c3e50;
        font-size: 32px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for user inputs with enhanced labels
st.sidebar.title("Pension Fund Optimizer Settings")
assets = {
    "US Stocks": [
        "SPY",  # S&P 500
        "MDY",  # S&P MidCap 400
        "IJR",  # S&P SmallCap 600
        "XLK",  # Technology Select Sector
        "XLF",  # Financial Select Sector
    ],
    "International Stocks": [
        "EFA",  # EAFE (Europe, Australasia, Far East)
        "VWO",  # Emerging Markets
        "EWJ",  # Japan
        "EEM",  # Emerging Markets (alternative)
    ],
    "Corporate Bonds": [
        "LQD",  # iShares iBoxx $ Inv Grade Corp Bond
        "HYG",  # High Yield Corporate Bond
        "VCIT", # Intermediate Corp Bond
        "JNK",  # High Yield Bond
    ],
    "Sovereign Bonds": [
        "TLT",  # Long-Term Treasury
        "BNDX", # Total International Bond
        "TIP",  # TIPS (Inflation-Protected)
        "BWX",  # International Treasury Bond
    ],
    "Commodities": [
        "GLD",  # Gold
        "SLV",  # Silver
        "USO",  # Oil
        "DBC",  # Broad Commodity
    ],
    "REITs": [
        "VNQ",  # Vanguard Real Estate ETF
        "RWR",  # Wilshire US REIT
        "SCHH", # Schwab US REIT
    ],
    "Other": [
        "HEFA", # MSCI EAFE Hedged
        "EMGF", # MSCI Emerging Markets Hedged
        "XLE",  # Energy Select Sector
    ]
}
selected_assets = st.sidebar.multiselect(
    "Select Asset Classes to Include",
    options=sum(assets.values(), []),
    default=sum(assets.values(), []),
    help="Choose a diverse mix of assets for your pension fund portfolio to balance risk and return."
)
start_date = st.sidebar.date_input(
    "Start Date",
    pd.to_datetime("2020-01-01"),
    help="Set the starting date for historical data analysis."
)
end_date = st.sidebar.date_input(
    "End Date",
    pd.to_datetime(datetime.now().date() - pd.Timedelta(days=1)),  # Use yesterday
    help="Set the ending date for historical data (up to yesterday)."
)

# Fetch data with focus on Close only
@st.cache_data
def get_data(tickers, start, end):
    try:
        raw_data = yf.download(tickers, start=start, end=end)
        if raw_data.empty:
            raise ValueError("No data returned for the given tickers and date range.")
        
        if isinstance(raw_data.columns, pd.MultiIndex):
            close_data = raw_data.xs("Close", axis=1, level=0)
        else:
            close_data = raw_data["Close"]
        
        if not all(t in close_data.columns for t in tickers):
            missing = [t for t in tickers if t not in close_data.columns]
            raise ValueError(f"Missing data for tickers: {missing}")
        
        return close_data.dropna()
    except Exception as e:
        st.error(f"Data fetch failed: {str(e)}")
        return pd.DataFrame()

# Optimize portfolio
if st.sidebar.button("Optimize My Portfolio"):
    if not selected_assets:
        st.error("Please select at least one asset to proceed.")
    else:
        with st.spinner("Calculating your optimal portfolio..."):
            data = get_data(selected_assets, start_date, end_date)
            if data.empty:
                st.error("No data available for the selected assets and date range. Please adjust your selection.")
            else:
                returns = data.pct_change().dropna()
                
                if len(returns) < len(selected_assets) + 1:
                    st.error("Insufficient data points for optimization. Please extend the date range.")
                else:
                    mu = returns.mean() * 252
                    S = returns.cov() * 252
                    n = len(selected_assets)
                    S_np = S.to_numpy()
                    mu_np = mu.to_numpy()
                    
                    # Regularize covariance matrix if needed
                    min_eig = np.min(np.linalg.eigvals(S_np))
                    if min_eig < 0:
                        S_np += np.eye(n) * abs(min_eig) * 1.01
                    
                    def solve_with_rho(rho):
                        w = cp.Variable(n)
                        objective = cp.Minimize(cp.quad_form(w, S_np) - rho * cp.sum(cp.log(w)))
                        constraints = [cp.sum(w) == 1, w >= 1e-6]
                        prob = cp.Problem(objective, constraints)
                        prob.solve(solver=cp.ECOS)
                        if prob.status == "optimal":
                            return w.value
                        return None
                    
                    def get_rc_var(rho):
                        w = solve_with_rho(rho)
                        if w is None:
                            return np.inf
                        var = w @ S_np @ w
                        sigma = np.sqrt(var)
                        MRC = S_np @ w
                        RC = w * MRC / sigma
                        return np.var(RC)
                    
                    res = minimize_scalar(get_rc_var, bounds=(1e-6, 1e-1), method='bounded', tol=1e-5)
                    best_rho = res.x
                    weights = solve_with_rho(best_rho)
                    
                    if weights is None:
                        st.error("Optimization failed. Please try a different date range or fewer assets.")
                    else:
                        weights = np.where(np.abs(weights) < 1e-4, 0, weights)
                        weights /= np.sum(weights)
                        
                        port_var = weights @ S_np @ weights
                        sigma = np.sqrt(port_var)
                        MRC = S_np @ weights
                        risk_contrib = weights * MRC / sigma
                        total_risk = np.sum(risk_contrib)
                        risk_contrib_pct = (risk_contrib / total_risk) * 100
                        
                        # Display results in a clean layout
                        st.title("Your Optimized Pension Fund Portfolio")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Allocation Weights")
                            st.table(pd.DataFrame({
                                "Asset": selected_assets,
                                "Weight (%)": [w * 100 for w in weights]
                            }).set_index("Asset").round(2))
                        with col2:
                            st.subheader("Risk Contributions")
                            st.table(pd.DataFrame({
                                "Asset": selected_assets,
                                "Contribution (%)": risk_contrib_pct.round(2)
                            }).set_index("Asset"))
                        
                        # Visualizations
                        fig = go.Figure(data=[go.Pie(labels=selected_assets, values=weights * 100, hole=0.3)])
                        fig.update_layout(title="Portfolio Allocation", title_x=0.5)
                        st.plotly_chart(fig)
                        
                        fig2 = go.Figure(data=[go.Bar(x=selected_assets, y=risk_contrib_pct)])
                        fig2.update_layout(title="Risk Contributions", title_x=0.5, xaxis_title="Assets", yaxis_title="Percentage")
                        st.plotly_chart(fig2)
                        
                        # Performance metrics
                        st.subheader("Performance Metrics")
                        col3, col4, col5 = st.columns(3)
                        col3.metric("Expected Annual Return", f"{np.dot(mu_np, weights) * 100:.2f}%")
                        col4.metric("Annual Volatility", f"{sigma * 100:.2f}%")
                        col5.metric("Sharpe Ratio", f"{np.dot(mu_np, weights) / sigma if sigma > 0 else 0:.2f}")

# Add welcome and instructions
st.sidebar.markdown("### How to Use")
st.sidebar.write("""
- **Select Assets**: Choose a diverse mix of assets for your portfolio to balance risk and return.
- **Set Date Range**: Adjust the start and end dates to analyze historical performance.
- **Optimize**: Click 'Optimize My Portfolio' to generate your results.
- **Explore**: Review weights, risk contributions, and performance metrics visually.
""")
st.sidebar.markdown("---")
st.sidebar.write("Built for your pension fund success! 🎉")
