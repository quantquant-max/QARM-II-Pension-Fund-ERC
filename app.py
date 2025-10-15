import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import cvxpy as cp
from scipy.optimize import minimize_scalar
from datetime import datetime, timedelta
from fpdf import FPDF
import io
from sklearn.covariance import LedoitWolf

# Custom styling with pop-up and HELP icon
st.set_page_config(page_title="Pension Fund Optimizer", layout="wide")
st.logo("ERC Portfolio.png")

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
    .stDateInput label {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
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
    header {
        background-color: #000000 !important;
        position: relative;
    }
    header img {
        height: 60px !important;
        width: auto !important;
    }
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
    /* HELP icon in top-right */
    .help-icon {
        position: absolute;
        top: 10px;
        right: 20px;
        font-size: 24px;
        color: #f0f0f0;
        cursor: pointer;
        z-index: 1000;
    }
    /* Pop-up modal */
    .modal {
        display: none;
        position: fixed;
        z-index: 1001;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0,0,0,0.5);
    }
    .modal-content {
        background-color: #222222;
        color: #f0f0f0;
        margin: 15% auto;
        padding: 20px;
        border: 1px solid #f0f0f0;
        width: 70%;
        max-width: 600px;
        border-radius: 8px;
        font-family: 'Times New Roman', serif;
    }
    .close {
        color: #f0f0f0;
        float: right;
        font-size: 28px;
        font-weight: bold;
        cursor: pointer;
    }
    .close:hover {
        color: #dddddd;
    }
    #howToUseModal {
        display: none;
    }
    </style>
    <div class="help-icon" onclick="document.getElementById('howToUseModal').style.display='block'">HELP</div>
    <div id="howToUseModal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="document.getElementById('howToUseModal').style.display='none'">&times;</span>
            <h2>How to Use</h2>
            <p>
            - <b>Set Date Range</b>: Select and confirm the start and end month/year for historical performance analysis.<br>
            - <b>Select Assets</b>: Choose US stocks from the list of stocks with data in the selected range.<br>
            - <b>Rebalance Frequency</b>: Choose quarterly, semi-annually, or annually.<br>
            - <b>Optimize</b>: Click 'Optimize My Portfolio' to generate your results.<br>
            - <b>Explore</b>: Review weights, risk contributions, and performance metrics in the Portfolio Results tab.
            </p>
        </div>
    </div>
    <script>
        // Show modal on first load if not shown before
        if (!sessionStorage.getItem('modalShown')) {
            document.getElementById('howToUseModal').style.display = 'block';
            sessionStorage.setItem('modalShown', 'true');
        }
    </script>
    """,
    unsafe_allow_html=True
)

# Data loading functions
def load_custom_data():
    try:
        df = pd.read_csv("Stock_Returns_With_Names_post2000.csv")
        df.set_index('COMNAM', inplace=True)
        df.columns = pd.to_datetime(df.columns.str.split(' ').str[0])
        df = df.transpose()
        return df
    except Exception as e:
        return pd.DataFrame()

def get_data(tickers, start, end, custom_data):
    try:
        data = custom_data.loc[start:end, tickers]
        data = data.dropna()
        return data
    except Exception as e:
        return pd.DataFrame()

# Cache valid stocks computation
@st.cache_data
def get_valid_stocks(_custom_data, start_date, end_date):
    try:
        period_data = _custom_data.loc[start_date:end_date]
        valid_stocks = [col for col in _custom_data.columns if period_data[col].notna().any()]
        return valid_stocks
    except Exception:
        return []

# Optimization function
def perform_optimization(selected_assets, start_date, end_date, rebalance_freq, custom_data):
    try:
        lookback_start = start_date - timedelta(days=365)
        data = get_data(selected_assets, lookback_start, end_date, custom_data)
        bench_data = get_data(["Value Weighted Benchmark", "Equally Weighted Benchmark"], lookback_start, end_date, custom_data)
        
        if data.empty or bench_data.empty:
            st.error("No data available for the selected assets and date range.")
            return None
        
        returns = data
        value_weighted_returns = bench_data["Value Weighted Benchmark"]
        equally_weighted_returns = bench_data["Equally Weighted Benchmark"]
        
        period_returns = returns.loc[start_date:end_date]
        period_value_weighted = value_weighted_returns.loc[start_date:end_date]
        period_equally_weighted = equally_weighted_returns.loc[start_date:end_date]
        
        if len(period_returns) < 2:
            st.error("Insufficient data points for the selected period.")
            return None
        
        tc_rate = 0.001
        freq_map = {
            'Quarterly': 'QS',
            'Semi-Annually': '6MS',
            'Annually': 'YS'
        }
        freq = freq_map[rebalance_freq]
        
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        rebalance_dates = []
        for d in date_range:
            candidates = returns.index[returns.index >= d]
            if not candidates.empty:
                rebalance_dates.append(candidates[0])
        
        if returns.index[-1] not in rebalance_dates and returns.index[-1] > rebalance_dates[-1]:
            rebalance_dates.append(returns.index[-1])
        
        n = len(selected_assets)
        previous_weights = np.ones(n) / n
        port_returns = pd.Series(index=period_returns.index, dtype=float)
        weights_over_time = {}
        total_tc = 0.0
        
        rebal_date = rebalance_dates[0]
        est_end = rebal_date - pd.Timedelta(days=1)
        est_start = max(returns.index[0], est_end - pd.Timedelta(days=365))
        est_returns = returns.loc[est_start:est_end]
        
        if len(est_returns) < n + 1:
            st.error("Insufficient initial estimation data for the first rebalance period.")
            return None
        else:
            mu = est_returns.mean() * 252
            lw = LedoitWolf().fit(est_returns)
            S_np = lw.covariance_ * 252
            mu_np = mu.to_numpy()
            
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
                st.error("Initial optimization failed. Please try a different date range or fewer assets.")
            else:
                weights = np.where(np.abs(weights) < 1e-4, 0, weights)
                weights /= np.sum(weights)
                
                turnover = np.sum(np.abs(weights - previous_weights)) / 2
                cost = turnover * tc_rate
                total_tc += cost
                
                previous_weights = weights
                weights_over_time[rebal_date] = weights
        
        for i in range(1, len(rebalance_dates)):
            rebal_date = rebalance_dates[i]
            prev_rebal_date = rebalance_dates[i-1]
            
            period_returns_slice = period_returns.loc[prev_rebal_date:rebal_date - pd.Timedelta(days=1)]
            if not period_returns_slice.empty:
                period_port_ret = period_returns_slice.dot(previous_weights)
                port_returns.loc[period_returns_slice.index] = period_port_ret
            
            est_end = rebal_date - pd.Timedelta(days=1)
            est_start = max(returns.index[0], est_end - pd.Timedelta(days=365))
            est_returns = returns.loc[est_start:est_end]
            
            if len(est_returns) < n + 1:
                st.warning(f"Insufficient data for rebalance at {rebal_date}. Keeping previous weights.")
                weights = previous_weights
                cost = 0
            else:
                mu = est_returns.mean() * 252
                lw = LedoitWolf().fit(est_returns)
                S_np = lw.covariance_ * 252
                mu_np = mu.to_numpy()
                
                res = minimize_scalar(get_rc_var, bounds=(1e-6, 1e-1), method='bounded', tol=1e-5)
                best_rho = res.x
                weights = solve_with_rho(best_rho)
                
                if weights is None:
                    st.warning(f"Optimization failed for rebalance at {rebal_date}. Keeping previous weights.")
                    weights = previous_weights
                    cost = 0
                else:
                    weights = np.where(np.abs(weights) < 1e-4, 0, weights)
                    weights /= np.sum(weights)
                    
                    delta = weights - previous_weights
                    turnover = np.sum(np.abs(delta)) / 2
                    cost = turnover * tc_rate
                    total_tc += cost
            
            weights_over_time[rebal_date] = weights
            previous_weights = weights
        
        last_rebal_date = rebalance_dates[-1]
        last_period_returns = period_returns.loc[last_rebal_date:]
        if not last_period_returns.empty:
            last_port_ret = last_period_returns.dot(previous_weights)
            port_returns.loc[last_period_returns.index] = last_port_ret
        
        port_returns = port_returns.dropna()
        
        cum_port = (1 + port_returns).cumprod()
        cum_value_weighted = (1 + period_value_weighted).cumprod()
        cum_equally_weighted = (1 + period_equally_weighted).cumprod()
        
        ann_return = port_returns.mean() * 252
        ann_vol = port_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        est_returns = returns.iloc[-252:]
        if len(est_returns) >= n + 1:
            lw = LedoitWolf().fit(est_returns)
            S_np = lw.covariance_ * 252
            port_var = weights @ S_np @ weights
            sigma = np.sqrt(port_var)
            MRC = S_np @ weights
            risk_contrib = weights * MRC / sigma
            total_risk = np.sum(risk_contrib)
            risk_contrib_pct = (risk_contrib / total_risk) * 100 if total_risk > 0 else np.zeros(n)
        else:
            risk_contrib_pct = np.ones(n) / n * 100
        
        weights_df = pd.DataFrame(weights_over_time, index=selected_assets).T * 100
        frames = []
        dates = sorted(weights_df.index)
        for i in range(len(dates)):
            frame_data = []
            for asset in selected_assets:
                frame_data.append(go.Bar(x=selected_assets, y=weights_df.iloc[i], name=asset))
            frames.append(go.Frame(data=frame_data, name=str(dates[i])))
        
        fig_weights = go.Figure(data=[go.Bar(x=selected_assets, y=weights_df.iloc[0], name=asset) for asset in selected_assets],
                                layout=go.Layout(updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])],
                                                 transition_duration=500))
        fig_weights.frames = frames
        fig_weights.update_layout(title=dict(text="Weights Evolution Over Time (%)", font=dict(color="#f0f0f0", family="Times New Roman")), paper_bgcolor="#000000", font_color="#f0f0f0", font_family="Times New Roman")
        fig_weights.update_xaxes(title_font_color="#f0f0f0", tickfont_color="#f0f0f0", title_font_family="Times New Roman", tickfont_family="Times New Roman")
        fig_weights.update_yaxes(title_font_color="#f0f0f0", tickfont_color="#f0f0f0", title_font_family="Times New Roman", tickfont_family="Times New Roman")
        fig_weights.update_layout(legend=dict(font=dict(color="#f0f0f0", family="Times New Roman")))
        
        results = {
            "selected_assets": selected_assets,
            "weights": weights,
            "risk_contrib_pct": risk_contrib_pct,
            "expected_return": ann_return * 100,
            "volatility": ann_vol * 100,
            "sharpe": sharpe,
            "cum_port": cum_port,
            "cum_value_weighted": cum_value_weighted,
            "cum_equally_weighted": cum_equally_weighted,
            "total_tc": total_tc * 100,
            "fig_weights": fig_weights,
            "port_returns": port_returns,
            "weights_df": weights_df
        }
        return results
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        return None

# Visualization functions
def create_pie_chart(assets, values):
    fig = go.Figure(data=[go.Pie(labels=assets, values=values, hole=0.3, textfont=dict(color="#f0f0f0", family="Times New Roman"))])
    fig.update_layout(title=dict(text="Portfolio Allocation", font=dict(color="#f0f0f0", family="Times New Roman")), title_x=0.5, paper_bgcolor="#000000", font_color="#f0f0f0", font_family="Times New Roman")
    fig.update_traces(textfont_color="#f0f0f0")
    return fig

def create_bar_chart(assets, values):
    fig = go.Figure(data=[go.Bar(x=assets, y=values)])
    fig.update_layout(title=dict(text="Risk Contributions", font=dict(color="#f0f0f0", family="Times New Roman")), title_x=0.5, xaxis_title="Assets", yaxis_title="Percentage", paper_bgcolor="#000000", font_color="#f0f0f0", font_family="Times New Roman")
    fig.update_xaxes(title_font_color="#f0f0f0", tickfont_color="#f0f0f0", title_font_family="Times New Roman", tickfont_family="Times New Roman")
    fig.update_yaxes(title_font_color="#f0f0f0", tickfont_color="#f0f0f0", title_font_family="Times New Roman", tickfont_family="Times New Roman")
    fig.update_layout(legend=dict(font=dict(color="#f0f0f0", family="Times New Roman")))
    return fig

def create_line_chart(cum_port, cum_value_weighted, cum_equally_weighted):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_port.index, y=cum_port, mode='lines', name='Portfolio', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=cum_value_weighted.index, y=cum_value_weighted, mode='lines', name='Value Weighted Benchmark', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=cum_equally_weighted.index, y=cum_equally_weighted, mode='lines', name='Equally Weighted Benchmark', line=dict(color='red')))
    fig.update_layout(title=dict(text="Cumulative Returns", font=dict(color="#f0f0f0", family="Times New Roman")), title_x=0.5, xaxis_title="Date", yaxis_title="Cumulative Return", paper_bgcolor="#000000", plot_bgcolor="#000000", font_color="#f0f0f0", font_family="Times New Roman")
    fig.update_xaxes(title_font_color="#f0f0f0", tickfont_color="#f0f0f0", title_font_family="Times New Roman", tickfont_family="Times New Roman")
    fig.update_yaxes(title_font_color="#f0f0f0", tickfont_color="#f0f0f0", title_font_family="Times New Roman", tickfont_family="Times New Roman")
    fig.update_layout(legend=dict(font=dict(color="#f0f0f0", family="Times New Roman")))
    return fig

# Export functions
def export_csv(weights_df, filename):
    csv = weights_df.to_csv()
    st.download_button(label="Download Weights History as CSV", data=csv, file_name=filename, mime="text/csv")

def export_pdf(results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Times", size=12)
    pdf.cell(200, 10, txt="Portfolio Results", ln=1, align='C')
    pdf.cell(200, 10, txt=f"Expected Annual Return: {results['expected_return']:.2f}%", ln=1)
    pdf.cell(200, 10, txt=f"Annual Volatility: {results['volatility']:.2f}%", ln=1)
    pdf.cell(200, 10, txt=f"Sharpe Ratio: {results['sharpe']:.2f}", ln=1)
    pdf.cell(200, 10, txt=f"Total Transaction Costs: {results['total_tc']:.2f}%", ln=1)
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    st.download_button(label="Download Report as PDF", data=pdf_buffer, file_name="portfolio_report.pdf", mime="application/pdf")

# Tabs
tab1, tab2, tab3 = st.tabs(["Asset Selection", "Portfolio Results", "About Us"])

with tab1:
    st.title("Asset Selection")
    custom_data = load_custom_data()
    if custom_data.empty:
        st.error("Failed to load the custom dataset. Ensure 'Stock_Returns_With_Names_post2000.csv' is in the root directory.")
    else:
        min_date = custom_data.index.min().date()
        max_date = custom_data.index.max().date()
        
        # Generate month/year options
        month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        date_options = []
        date_to_str = {}
        for year in range(min_date.year, max_date.year + 1):
            for month in range(1, 13):
                if year == min_date.year and month < min_date.month:
                    continue
                if year == max_date.year and month > max_date.month:
                    break
                date_str = f"{month_names[month-1]} {year}"
                date_options.append(date_str)
                date_to_str[date_str] = (year, month)
        
        # Initialize session state
        if 'dates_confirmed' not in st.session_state:
            st.session_state.dates_confirmed = False
        if 'start_date' not in st.session_state:
            st.session_state.start_date = None
        if 'end_date' not in st.session_state:
            st.session_state.end_date = None
        if 'valid_stocks' not in st.session_state:
            st.session_state.valid_stocks = []
        
        # Date inputs
        st.markdown("### Select Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date_str = st.selectbox(
                "Start Month/Year",
                options=date_options,
                index=0,
                key="start_date_str"
            )
        with col2:
            end_date_str = st.selectbox(
                "End Month/Year",
                options=date_options,
                index=len(date_options)-1,
                key="end_date_str"
            )
        
        # Convert selected strings to dates
        try:
            start_year, start_month = date_to_str[start_date_str]
            end_year, end_month = date_to_str[end_date_str]
            start_date = datetime(start_year, start_month, 1).date()
            end_date = (datetime(end_year, end_month, 1) + pd.offsets.MonthEnd(0)).date()
            
            if start_date > end_date:
                st.error("Start date must be before end date.")
            elif end_date > max_date or start_date < min_date:
                st.error(f"Dates must be within data range: {min_date} to {max_date}.")
            else:
                if st.button("Confirm Dates"):
                    st.session_state.start_date = start_date
                    st.session_state.end_date = end_date
                    st.session_state.valid_stocks = get_valid_stocks(custom_data, start_date, end_date)
                    st.session_state.dates_confirmed = True
                    st.success("Dates confirmed! Please select assets and rebalance frequency below.")
        except Exception:
            st.error("Invalid date selection. Please choose valid month and year.")
        
        # Show stock selection and rebalance frequency only after dates are confirmed
        if st.session_state.dates_confirmed:
            st.markdown("### Select Assets and Rebalance Frequency")
            selected_assets = st.multiselect(
                "Select US Stocks",
                options=st.session_state.valid_stocks,
                key="us_stocks"
            )
            
            rebalance_freq = st.selectbox(
                "Rebalance Frequency",
                options=['Quarterly', 'Semi-Annually', 'Annually'],
                index=2
            )
            
            if st.button("Optimize My Portfolio"):
                if not selected_assets:
                    st.error("Please select at least one asset to proceed.")
                else:
                    with st.spinner("Calculating..."):
                        results = perform_optimization(selected_assets, st.session_state.start_date, st.session_state.end_date, rebalance_freq, custom_data)
                        if results:
                            st.session_state.results = results
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
        
        st.plotly_chart(create_pie_chart(results["selected_assets"], results["weights"] * 100), use_container_width=True)
        
        st.plotly_chart(create_bar_chart(results["selected_assets"], results["risk_contrib_pct"]), use_container_width=True)
        
        st.subheader("Performance Metrics")
        col3, col4, col5, col6 = st.columns(4)
        col3.metric("Expected Annual Return", f"{results['expected_return']:.2f}%")
        col4.metric("Annual Volatility", f"{results['volatility']:.2f}%")
        col5.metric("Sharpe Ratio", f"{results['sharpe']:.2f}")
        col6.metric("Total Transaction Costs", f"{results['total_tc']:.2f}%")
        
        st.subheader("Weights Evolution Over Time")
        st.plotly_chart(results["fig_weights"], use_container_width=True)
        
        st.subheader("Cumulative Returns Comparison")
        st.plotly_chart(create_line_chart(results["cum_port"], results["cum_value_weighted"], results["cum_equally_weighted"]), use_container_width=True)
        
        st.subheader("Export Results")
        export_csv(results["weights_df"], "weights_history.csv")
        export_pdf(results)
    else:
        st.info("Please select assets and optimize in the Asset Selection tab.")

with tab3:
    st.title("About Us")
    st.write("""
    Welcome to the Pension Fund Optimizer!
    
    We are a dedicated team of financial experts and developers passionate about helping individuals and institutions optimize their pension funds for maximum efficiency and risk management.
    
    Our tool uses advanced optimization techniques, specifically Dynamic Equal Risk Contribution (ERC) with annual rebalancing, to create balanced portfolios that aim to equalize the risk contributions from each asset over time.
    
    Built with Streamlit and powered by open-source libraries, this app provides an intuitive interface for selecting assets, analyzing historical data, and visualizing results.
    
    If you have any questions or feedback, feel free to reach out at support@pensionoptimizer.com.
    
    Thank you for using our tool! 🎉
    """)
