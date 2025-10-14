import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import cvxpy as cp
from scipy.optimize import minimize_scalar
from datetime import datetime, timedelta

# Custom styling for black and white theme with header and logo adjustments
st.set_page_config(page_title="Pension Fund Optimizer", layout="wide")
st.logo("ERC Portfolio.png")  # Logo in top-left

st.markdown(
    """
    <style>
    :root {
        --primary-color: #f0f0f0;
    }
    .stApp {
        background-color: #000000;
        color: #f0f0f0;
        font-family: 'Times New Roman', serif;
    }
    .stSidebar {
        background-color: #111111;
        color: #f0f0f0;
        font-family: 'Times New Roman', serif;
    }
    .stButton>button {
        background-color: #f0f0f0;
        color: #000000;
        border-radius: 8px;
        padding: 10px 20px;
        font-family: 'Times New Roman', serif;
    }
    .stButton>button:hover {
        background-color: #dddddd;
    }
    .stHeader {
        color: #f0f0f0;
        font-size: 32px;
        font-weight: bold;
        font-family: 'Times New Roman', serif;
    }
    .stExpander {
        background-color: #222222;
        color: #f0f0f0;
        font-family: 'Times New Roman', serif;
    }
    .stMultiSelect [data-testid=stMarkdownContainer] {
        color: #f0f0f0;
        font-family: 'Times New Roman', serif;
    }
    .stPlotlyChart {
        background-color: #000000;
    }
    /* Ensure date input labels are white */
    .stDateInput label {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    /* Ensure table text is white */
    .stTable {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    table {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    th, td {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    /* Style metrics */
    .stMetric {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    .stMetric label {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    .stMetricValue {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    [data-testid="stMetric"] {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    [data-testid="stMetricLabel"] {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    [data-testid="stMetricValue"] {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    div[data-testid="metric-container"] {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    div[data-testid="metric-container"] p {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    div[data-testid="metric-container"] div {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    /* Make top banner (header) black */
    header {
        background-color: #000000 !important;
    }
    /* Make logo 4x bigger (default ~24px, so 96px height) */
    header img {
        height: 60px !important;
        width: auto !important;
    }
    /* Override error messages to use white instead of red */
    div[data-testid="stAlert"] {
        background-color: #111111 !important;
        color: #f0f0f0 !important;
        border-color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    div.kind-error {
        background-color: #111111 !important;
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    </style>
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
        "LQD", # iShares iBoxx $ Inv Grade Corp Bond
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
                # Fetch data starting 1 year before start_date for initial estimation
                lookback_start = st.session_state.start_date - timedelta(days=365)
                data = get_data(selected_assets, lookback_start, st.session_state.end_date)
                bench_data = get_data(["SPY"], lookback_start, st.session_state.end_date)
                if data.empty or bench_data.empty:
                    st.error("No data available for the selected assets and date range. Please adjust your selection.")
                else:
                    returns = data.pct_change().dropna()
                    bench_returns = bench_data.pct_change().dropna().squeeze()
                    
                    # Filter to user-selected period
                    returns = returns.loc[st.session_state.start_date:st.session_state.end_date]
                    bench_returns = bench_returns.loc[st.session_state.start_date:st.session_state.end_date]
                    
                    if len(returns) < 252:
                        st.error("Insufficient data points for optimization. Please extend the date range.")
                    else:
                        # Transaction cost rate (0.1%)
                        tc_rate = 0.001
                        
                        # Find annual rebalance dates: first trading day of each year within the range
                        date_range = pd.date_range(start=st.session_state.start_date, end=st.session_state.end_date, freq='YS')
                        rebalance_dates = []
                        for d in date_range:
                            # Find the first date with data on or after d
                            candidates = returns.index[returns.index >= d]
                            if not candidates.empty:
                                rebalance_dates.append(candidates[0])
                        
                        # Add the end date if not already included
                        if returns.index[-1] not in rebalance_dates:
                            rebalance_dates.append(returns.index[-1])
                        
                        # Initialize
                        n = len(selected_assets)
                        previous_weights = np.zeros(n)
                        port_returns = pd.Series(index=returns.index, dtype=float)
                        weights_over_time = {}
                        total_tc = 0.0
                        
                        for i in range(len(rebalance_dates) - 1):
                            rebal_date = rebalance_dates[i]
                            next_rebal_date = rebalance_dates[i+1]
                            
                            # Use last 252 days before rebal_date for estimation
                            est_end = rebal_date - pd.Timedelta(days=1)
                            est_start = est_end - pd.Timedelta(days=365)
                            est_returns = returns.loc[est_start:est_end]
                            
                            if len(est_returns) < n + 1:
                                st.error("Insufficient estimation data for a rebalance period.")
                                break
                            
                            mu_est = est_returns.mean() * 252
                            S_est = est_returns.cov() * 252
                            S_np = S_est.to_numpy()
                            
                            # Regularize
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
                                st.error("Optimization failed for a period.")
                                break
                            
                            weights = np.where(np.abs(weights) < 1e-4, 0, weights)
                            weights /= np.sum(weights)
                            
                            # Transaction cost
                            delta = weights - previous_weights
                            turnover = np.sum(np.abs(delta)) / 2
                            cost = turnover * tc_rate
                            total_tc += cost
                            
                            # Portfolio returns for the period, subtract cost at rebalance
                            period_returns = returns.loc[rebal_date:next_rebal_date].iloc[:-1]  # Up to but not including next rebal
                            period_port_ret = period_returns.dot(weights)
                            period_port_ret.iloc[0] -= cost  # Subtract cost on rebalance day
                            port_returns.loc[period_returns.index] = period_port_ret
                            
                            # Save weights
                            weights_over_time[rebal_date] = weights
                            previous_weights = weights
                        
                        # Last period to end
                        if 'weights' in locals():
                            last_period_returns = returns.loc[rebalance_dates[-1]:]
                            if not last_period_returns.empty:
                                port_returns.loc[last_period_returns.index] = last_period_returns.dot(weights)
                        
                        # Drop NaN if any
                        port_returns = port_returns.dropna()
                        
                        # Performance metrics
                        ann_return = port_returns.mean() * 252
                        ann_vol = port_returns.std() * np.sqrt(252)
                        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
                        
                        # Cumulative
                        cum_port = (1 + port_returns).cumprod()
                        cum_bench = (1 + bench_returns).cumprod()
                        
                        # Weights graph
                        weights_df = pd.DataFrame(weights_over_time, index=selected_assets).T * 100
                        fig_weights = go.Figure()
                        for asset in selected_assets:
                            fig_weights.add_trace(go.Bar(x=weights_df.index, y=weights_df[asset], name=asset))
                        fig_weights.update_layout(barmode='stack', title=dict(text="Weights Over Time (%)", font=dict(color="#f0f0f0", family="Times New Roman")), paper_bgcolor="#000000", font_color="#f0f0f0", font_family="Times New Roman")
                        fig_weights.update_xaxes(title_font_color="#f0f0f0", tickfont_color="#f0f0f0", title_font_family="Times New Roman", tickfont_family="Times New Roman")
                        fig_weights.update_yaxes(title_font_color="#f0f0f0", tickfont_color="#f0f0f0", title_font_family="Times New Roman", tickfont_family="Times New Roman")
                        fig_weights.update_layout(legend=dict(font=dict(color="#f0f0f0", family="Times New Roman")))
                        
                        # Store results
                        st.session_state.results = {
                            "selected_assets": selected_assets,
                            "weights": weights,  # Last weights
                            "risk_contrib_pct": risk_contrib_pct,  # Last, or compute average? Keep last for simplicity
                            "expected_return": ann_return * 100,
                            "volatility": ann_vol * 100,
                            "sharpe": sharpe,
                            "cum_port": cum_port,
                            "cum_bench": cum_bench,
                            "total_tc": total_tc * 100,
                            "fig_weights": fig_weights
                        }
                        st.success("Optimization complete! Check the Portfolio Results tab.")

with tab2:
    st.title("Portfolio Results")
    if "results" in st.session_state:
        results = st.session_state.results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Allocation Weights (Latest)")
            st.table(pd.DataFrame({
                "Asset": results["selected_assets"],
                "Weight (%)": [w * 100 for w in results["weights"]]
            }).set_index("Asset").round(2))
        with col2:
            st.subheader("Risk Contributions (Latest)")
            st.table(pd.DataFrame({
                "Asset": results["selected_assets"],
                "Contribution (%)": results["risk_contrib_pct"].round(2)
            }).set_index("Asset"))
        
        # Visualizations
        fig = go.Figure(data=[go.Pie(labels=results["selected_assets"], values=results["weights"] * 100, hole=0.3, textfont=dict(color="#f0f0f0", family="Times New Roman"))])
        fig.update_layout(title=dict(text="Portfolio Allocation", font=dict(color="#f0f0f0", family="Times New Roman")), title_x=0.5, paper_bgcolor="#000000", font_color="#f0f0f0", font_family="Times New Roman")
        fig.update_traces(textfont_color="#f0f0f0")
        st.plotly_chart(fig)
        
        fig2 = go.Figure(data=[go.Bar(x=results["selected_assets"], y=results["risk_contrib_pct"])])
        fig2.update_layout(title=dict(text="Risk Contributions", font=dict(color="#f0f0f0", family="Times New Roman")), title_x=0.5, xaxis_title="Assets", yaxis_title="Percentage", paper_bgcolor="#000000", font_color="#f0f0f0", font_family="Times New Roman")
        fig2.update_xaxes(title_font_color="#f0f0f0", tickfont_color="#f0f0f0", title_font_family="Times New Roman", tickfont_family="Times New Roman")
        fig2.update_yaxes(title_font_color="#f0f0f0", tickfont_color="#f0f0f0", title_font_family="Times New Roman", tickfont_family="Times New Roman")
        fig2.update_layout(legend=dict(font=dict(color="#f0f0f0", family="Times New Roman")))
        st.plotly_chart(fig2)
        
        # Performance metrics
        st.subheader("Performance Metrics")
        col3, col4, col5, col6 = st.columns(4)
        col3.metric("Expected Annual Return", f"{results['expected_return']:.2f}%")
        col4.metric("Annual Volatility", f"{results['volatility']:.2f}%")
        col5.metric("Sharpe Ratio", f"{results['sharpe']:.2f}")
        col6.metric("Total Transaction Costs", f"{results['total_tc']:.2f}%")
        
        # Weights over time
        st.subheader("Weights Changes Over Time")
        st.plotly_chart(results["fig_weights"])
        
        # Comparison chart
        st.subheader("Cumulative Returns Comparison")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=results["cum_port"].index, y=results["cum_port"], mode='lines', name='Portfolio', line=dict(color='blue')))
        fig3.add_trace(go.Scatter(x=results["cum_bench"].index, y=results["cum_bench"], mode='lines', name='Benchmark (SPY)', line=dict(color='red')))
        fig3.update_layout(title=dict(text="Cumulative Returns", font=dict(color="#f0f0f0", family="Times New Roman")), title_x=0.5, xaxis_title="Date", yaxis_title="Cumulative Return", paper_bgcolor="#000000", plot_bgcolor="#000000", font_color="#f0f0f0", font_family="Times New Roman")
        fig3.update_xaxes(title_font_color="#f0f0f0", tickfont_color="#f0f0f0", title_font_family="Times New Roman", tickfont_family="Times New Roman")
        fig3.update_yaxes(title_font_color="#f0f0f0", tickfont_color="#f0f0f0", title_font_family="Times New Roman", tickfont_family="Times New Roman")
        fig3.update_layout(legend=dict(font=dict(color="#f0f0f0", family="Times New Roman")))
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
