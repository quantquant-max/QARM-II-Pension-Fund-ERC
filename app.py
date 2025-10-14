import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import cvxpy as cp
from scipy.optimize import minimize_scalar
from datetime import datetime

# Custom styling for black and white theme (remove logo-specific CSS)
st.set_page_config(page_title="Pension Fund Optimizer", layout="wide")
st.logo("ERC Portfolio.png")  # Add this line for the top-left logo

st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #111111;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #ffffff;
        color: #000000;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #dddddd;
    }
    .stHeader {
        color: #ffffff;
        font-size: 32px;
        font-weight: bold;
    }
    .stExpander {
        background-color: #222222;
        color: #ffffff;
    }
    .stMultiSelect [data-testid=stMarkdownContainer] {
        color: #ffffff;
    }
    .stPlotlyChart {
        background-color: #000000;
    }
    /* Ensure date input labels are white */
    .stDateInput label {
        color: #ffffff !important;
    }
    /* Ensure table text is white */
    .stTable {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Display logo
st.markdown(
    """
    <div class="logo-container">
        <img class="logo-img" src="ERC Portfolio.png" alt="ERC Portfolio Logo">
    </div>
    """,
    unsafe_allow_html=True
)

# Asset categories with expanders
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

# Tabs
tab1, tab2, tab3 = st.tabs(["Asset Selection", "Portfolio Results", "About Us"])

with tab1:
    st.title("Asset Selection")
    
    # Collect selected assets from all categories
    for category, tickers in assets.items():
        with st.expander(category, expanded=False):
            st.multiselect(
                f"Select {category} Assets",
                options=tickers,
                help=f"Choose assets from {category} for your portfolio.",
                key=category
            )
    
    start_date = st.date_input(
        "Start Date",
        pd.to_datetime("2020-01-01"),
        help="Set the starting date for historical data analysis.",
        key="start_date"
    )
    end_date = st.date_input(
        "End Date",
        pd.to_datetime(datetime.now().date() - pd.Timedelta(days=1)),  # Use yesterday
        help="Set the ending date for historical data (up to yesterday).",
        key="end_date"
    )
    
    st.markdown("### How to Use")
    st.write("""
    - **Select Assets**: Expand categories to choose assets for your portfolio.
    - **Set Date Range**: Adjust the start and end dates to analyze historical performance.
    - **Optimize**: Click 'Optimize My Portfolio' to generate your results.
    - **Explore**: Review weights, risk contributions, and performance metrics visually in the Portfolio Results tab.
    """)
    st.write("Built for your pension fund success! 🎉")
    
    if st.button("Optimize My Portfolio"):
        selected_assets = []
        for category in assets:
            selected_assets.extend(st.session_state.get(category, []))
        
        if not selected_assets:
            st.error("Please select at least one asset to proceed.")
        else:
            with st.spinner("Calculating your optimal portfolio..."):
                data = get_data(selected_assets, st.session_state.start_date, st.session_state.end_date)
                bench_data = get_data(["SPY"], st.session_state.start_date, st.session_state.end_date)
                if data.empty or bench_data.empty:
                    st.error("No data available for the selected assets and date range. Please adjust your selection.")
                else:
                    returns = data.pct_change().dropna()
                    bench_returns = bench_data.pct_change().dropna().squeeze()
                    
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
                            
                            # Calculate cumulative returns
                            port_returns = returns.dot(weights)
                            cum_port = (1 + port_returns).cumprod()
                            cum_bench = (1 + bench_returns).cumprod()
                            
                            # Store results in session state
                            st.session_state.results = {
                                "selected_assets": selected_assets,
                                "weights": weights,
                                "risk_contrib_pct": risk_contrib_pct,
                                "expected_return": np.dot(mu_np, weights) * 100,
                                "volatility": sigma * 100,
                                "sharpe": np.dot(mu_np, weights) / sigma if sigma > 0 else 0,
                                "cum_port": cum_port,
                                "cum_bench": cum_bench
                            }
                            st.success("Optimization complete! Check the Portfolio Results tab.")

with tab2:
    st.title("Portfolio Results")
    if "results" in st.session_state:
        results = st.session_state.results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Allocation Weights")
            st.table(pd.DataFrame({
                "Asset": results["selected_assets"],
                "Weight (%)": [w * 100 for w in results["weights"]]
            }).set_index("Asset").round(2))
        with col2:
            st.subheader("Risk Contributions")
            st.table(pd.DataFrame({
                "Asset": results["selected_assets"],
                "Contribution (%)": results["risk_contrib_pct"].round(2)
            }).set_index("Asset"))
        
        # Visualizations
        fig = go.Figure(data=[go.Pie(labels=results["selected_assets"], values=results["weights"] * 100, hole=0.3)])
        fig.update_layout(title="Portfolio Allocation", title_x=0.5, paper_bgcolor="#000000", font_color="#ffffff")
        st.plotly_chart(fig)
        
        fig2 = go.Figure(data=[go.Bar(x=results["selected_assets"], y=results["risk_contrib_pct"])])
        fig2.update_layout(title="Risk Contributions", title_x=0.5, xaxis_title="Assets", yaxis_title="Percentage", paper_bgcolor="#000000", font_color="#ffffff")
        st.plotly_chart(fig2)
        
        # Performance metrics
        st.subheader("Performance Metrics")
        col3, col4, col5 = st.columns(3)
        col3.metric("Expected Annual Return", f"{results['expected_return']:.2f}%")
        col4.metric("Annual Volatility", f"{results['volatility']:.2f}%")
        col5.metric("Sharpe Ratio", f"{results['sharpe']:.2f}")
        
        # Comparison chart
        st.subheader("Cumulative Returns Comparison")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=results["cum_port"].index, y=results["cum_port"], mode='lines', name='Portfolio'))
        fig3.add_trace(go.Scatter(x=results["cum_bench"].index, y=results["cum_bench"], mode='lines', name='Benchmark (SPY)'))
        fig3.update_layout(title="Cumulative Returns", title_x=0.5, xaxis_title="Date", yaxis_title="Cumulative Return", paper_bgcolor="#000000", font_color="#ffffff")
        st.plotly_chart(fig3)
    else:
        st.info("Please select assets and optimize in the Asset Selection tab.")

with tab3:
    st.title("About Us")
    st.write("""
    Welcome to the Pension Fund Optimizer!
    
    We are a dedicated team of financial experts and developers passionate about helping individuals and institutions optimize their pension funds for maximum efficiency and risk management.
    
    Our tool uses advanced optimization techniques, specifically Equal Risk Contribution (ERC), to create balanced portfolios that aim to equalize the risk contributions from each asset.
    
    Built with Streamlit and powered by open-source libraries, this app provides an intuitive interface for selecting assets, analyzing historical data, and visualizing results.
    
    If you have any questions or feedback, feel free to reach out at support@pensionoptimizer.com.
    
    Thank you for using our tool! 🎉
    """)
