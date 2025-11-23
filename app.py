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

# Custom styling
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
    </style>
    """,
    unsafe_allow_html=True
)

# Data loading functions
@st.cache_data
def load_custom_data():
    comp = pd.read_parquet("compustat_git.parquet")
    etf = pd.read_parquet("etf_git.parquet")

    
    comp = comp[["date", "company_name", "monthly_return"]].copy()
    comp["date"] = pd.to_datetime(comp["date"])

    comp = comp.rename(columns={
        "company_name": "asset",  
        "monthly_return": "ret"
    })

    etf = etf.rename(columns={
        "ETF": "asset",
        "return_monthly": "ret"     
    })

    etf = etf[["date", "asset", "ret"]].copy()
    etf["date"] = pd.to_datetime(etf["date"])

    returns_long = pd.concat([comp, etf], ignore_index=True)

    returns_wide = (
        returns_long
        .pivot(index="date", columns="asset", values="ret")
        .sort_index()
    )
    returns_wide.index = pd.to_datetime(returns_wide.index)

    return returns_wide


def get_data(tickers, start, end, custom_data):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    if custom_data.empty:
        return pd.DataFrame()

    missing = [t for t in tickers if t not in custom_data.columns]

    if len(missing) > 0:
        st.warning(f"⚠️ The following assets do not exist in the database: {missing}")
        tickers = [t for t in tickers if t in custom_data.columns]

    if len(tickers) == 0:
        return pd.DataFrame()

    data = custom_data.loc[start:end, tickers]

    data = data.sort_index()

    return data


@st.cache_data
def get_valid_assets(custom_data, start_date, end_date):

    start_date = pd.to_datetime(start_date)
    end_date   = pd.to_datetime(end_date)

    comp = pd.read_parquet("compustat_git.parquet")
    etf  = pd.read_parquet("etf_git.parquet")

    comp_assets = sorted(comp["company_name"].unique())
    etf_assets  = sorted(etf["ETF"].unique())

    subset = custom_data.loc[start_date:end_date]

    available_assets = subset.columns[subset.notna().any()].tolist()

    # 3. Intersection pour filtrer
    valid_stocks = sorted(list(set(comp_assets) & set(available_assets)))
    valid_etfs   = sorted(list(set(etf_assets)  & set(available_assets)))

    return {
        "stocks": valid_stocks,
        "etfs": valid_etfs
    }
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize_scalar
from sklearn.covariance import LedoitWolf
import streamlit as st


def get_common_start_date(custom_data: pd.DataFrame,
                          selected_assets: list[str],
                          user_start_date) -> pd.Timestamp:
    user_start_date = pd.to_datetime(user_start_date)

    missing = [a for a in selected_assets if a not in custom_data.columns]
    if missing:
        st.error(f"The selected assets are not available in the database : {missing}")
        return None


    first_valid = custom_data[selected_assets].apply(lambda col: col.first_valid_index())


    common_start = first_valid.max()

    if pd.isna(common_start):
        st.error("No common valid date found for the selected assets.")
        return None


    if common_start > user_start_date:
        st.warning(
            f"⚠️ The chosen start date ({user_start_date.date()}) "
            f"is not available for the selected assets.\n\n"
            f"➡️ The optimisation will start on{common_start.date()}**, "
            f"which is the first date where all return series are available."
        )

    return max(common_start, user_start_date)


def compute_rebalance_indices(dates: pd.DatetimeIndex, freq_label: str) -> list[int]:

    if freq_label == "Quarterly":
        step = 3
    elif freq_label == "Semi-Annually":
        step = 6
    elif freq_label == "Annually":
        step = 12
    else:
        raise ValueError(f"Unknown frequency : {freq_label}")

    n = len(dates)
    idxs = list(range(0, n, step))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)  # on force un dernier rebalance à la fin

    return idxs


def solve_erc_weights(cov_matrix: np.ndarray) -> np.ndarray:
    n = cov_matrix.shape[0]

    def solve_with_rho(rho: float):
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cov_matrix) - rho * cp.sum(cp.log(w)))
        constraints = [cp.sum(w) == 1, w >= 1e-6]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        if prob.status == "optimal":
            return np.array(w.value).flatten()
        return None

    def rc_variance(rho: float) -> float:
        w = solve_with_rho(rho)
        if w is None:
            return np.inf
        var = w @ cov_matrix @ w
        sigma = np.sqrt(var)
        if sigma <= 0:
            return np.inf
        mrc = cov_matrix @ w
        rc = w * mrc / sigma
        return np.var(rc)

    res = minimize_scalar(
        rc_variance,
        bounds=(1e-6, 1e-1),
        method="bounded",
        tol=1e-5
    )
    best_rho = res.x
    w_star = solve_with_rho(best_rho)
    if w_star is None:
        raise RuntimeError("ERC Optimisation Failed (No optimal solution has been found).")

    w_star = np.where(np.abs(w_star) < 1e-6, 0, w_star)
    w_star = np.clip(w_star, 0, None)
    if w_star.sum() <= 0:
        raise RuntimeError("Non-valid ERC Solution (sum of weights <= 0).")
    w_star /= w_star.sum()
    return w_star


def perform_optimization(
    selected_assets: list[str],
    user_start_date,
    end_date,
    rebalance_freq: str,
    custom_data: pd.DataFrame,
    lookback_months: int = 36,
    ann_factor: int = 12,
    tc_rate: float = 0.001,
):

    try:

        user_start_date = pd.to_datetime(user_start_date)
        end_date = pd.to_datetime(end_date)

        if custom_data.empty:
            st.error("Les données de marché sont vides.")
            return None

        common_start = get_common_start_date(custom_data, selected_assets, user_start_date)
        if common_start is None:
            return None

        returns_all = custom_data[selected_assets].sort_index()

        returns_all = returns_all.loc[common_start:end_date]

        if returns_all.shape[0] < lookback_months + 2:
            st.error(
                f"Not enough data history to estimate the covariance on {lookback_months} months "
                f"with a study period until {end_date.date()}."
            )
            return None

        period_returns = returns_all.copy()
        period_dates = period_returns.index

        rebalance_indices = compute_rebalance_indices(period_dates, rebalance_freq)

        n = len(selected_assets)
        previous_weights = np.zeros(n)
        port_returns = pd.Series(index=period_dates, dtype=float)
        weights_over_time = {}
        total_tc = 0.0

        full_dates = returns_all.index

        for j, reb_idx in enumerate(rebalance_indices):
            rebal_date = period_dates[reb_idx]

            global_reb_pos = full_dates.get_loc(rebal_date)
            start_pos = max(0, global_reb_pos - lookback_months)
            est_window = returns_all.iloc[start_pos:global_reb_pos]

            est_window = est_window.dropna(how="all")
            est_window = est_window.dropna(how="any")

            if est_window.shape[0] < n + 1:
                st.error(
                    f"Not enough proper data to estimate covariance "
                    f"before rebalancing date {rebal_date.date()}."
                )
                return None

            # Estimation de la covariance (Ledoit-Wolf)
            lw = LedoitWolf().fit(est_window.values)
            cov = lw.covariance_ * ann_factor

            try:
                weights = solve_erc_weights(cov)
            except Exception as e:
                st.error(f"ERC Optimisation failed on {rebal_date.date()} : {e}")
                return None

            turnover = np.sum(np.abs(weights - previous_weights)) / 2
            cost = turnover * tc_rate
            total_tc += cost

            previous_weights = weights.copy()
            weights_over_time[rebal_date] = weights

            if j == len(rebalance_indices) - 1:
                # Dernier rebalance : jusqu'à la fin
                start_slice = reb_idx
                end_slice = len(period_dates)
            else:
                start_slice = reb_idx
                end_slice = rebalance_indices[j + 1]

            sub_ret = period_returns.iloc[start_slice:end_slice].fillna(0.0)
            if not sub_ret.empty:
                port_ret = sub_ret.values @ weights
                port_returns.iloc[start_slice:end_slice] = port_ret

        port_returns = port_returns.dropna()
        if port_returns.empty:
            st.error("The portfolio's return series is empty after backtesting.")
            return None

        cum_port = (1 + port_returns).cumprod()

        ann_return = port_returns.mean() * ann_factor
        ann_vol = port_returns.std() * np.sqrt(ann_factor)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        port_var = weights @ cov @ weights
        sigma_p = np.sqrt(port_var)
        mrc = cov @ weights
        rc = weights * mrc / sigma_p  # contributions absolues
        total_risk = rc.sum()
        if total_risk <= 0:
            risk_contrib_pct = np.zeros_like(rc)
        else:
            risk_contrib_pct = rc / total_risk * 100.0

        weights_df = (
            pd.DataFrame(weights_over_time, index=selected_assets)
            .T
            .sort_index()
        ) 

        corr_matrix = est_window.corr()

        results = {
            "selected_assets": selected_assets,
            "weights": weights,
            "risk_contrib_abs": rc,
            "risk_contrib_pct": risk_contrib_pct,
            "expected_return": ann_return * 100,   # en %
            "volatility": ann_vol * 100,          # en %
            "sharpe": sharpe,
            "port_returns": port_returns,
            "cum_port": cum_port,
            "total_tc": total_tc * 100,           # en %
            "weights_df": weights_df,
            "corr_matrix": corr_matrix,
        }
        return results

    except Exception as e:
        st.error(f"Erreur dans l'optimisation : {e}")
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
tab0, tab1, tab2, tab3 = st.tabs(["How to Use", "Asset Selection", "Portfolio Results", "About Us"])

with tab0:
    st.title("How to Use")
    st.write("""
    - **Set Date Range**: Select and confirm the start and end month/year for historical performance analysis.
    - **Select Assets**: Choose US stocks from the list of stocks with data in the selected range. Only stocks listed on or before the start date are available.
    - **Rebalance Frequency**: Choose quarterly, semi-annually, or annually.
    - **Optimize**: Click 'Optimize My Portfolio' to generate your results.
    - **Explore**: Review weights, risk contributions, and performance metrics in the Portfolio Results tab.
    """)

with tab1:
    st.title("Asset Selection")

    # -------------------------------
    # 1) Chargement des données
    # -------------------------------
    custom_data = load_custom_data()
    if custom_data.empty:
        st.error("Failed to load dataset.")
        st.stop()

    # Déterminer les dates min/max disponibles dans les données
    min_date = custom_data.index.min().date()
    max_date = custom_data.index.max().date()

    # -------------------------------
    # 2) Sélection période utilisateur
    # -------------------------------
    st.markdown("### Select Date Range")

    col1, col2 = st.columns(2)
    with col1:
        start_date_user = st.date_input("📅 Start Date", value=min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date_user = st.date_input("📅 End Date", value=max_date, min_value=min_date, max_value=max_date)

    if start_date_user > end_date_user:
        st.error("Start date must be before end date.")
        st.stop()

    # -------------------------------
    # 3) Obtenir via cache les listes d'actifs (stocks, etfs, all)
    # -------------------------------
    stocks, etfs, all_assets = get_valid_assets(custom_data)

    # -------------------------------
    # 4) Sélection des actifs
    # -------------------------------
    st.markdown("### Choose Your Assets")

    col1, col2 = st.columns(2)
    with col1:
        selected_stocks = st.multiselect("Stocks", options=stocks)
    with col2:
        selected_etfs = st.multiselect("ETFs", options=etfs)

    selected_assets = selected_stocks + selected_etfs

    if not selected_assets:
        st.info("Select at least one stock or ETF to proceed.")
        st.stop()

    # -------------------------------
    # 5) Vérifier si toutes les séries commencent après la date utilisateur
    # -------------------------------
    asset_first_dates = {
        a: custom_data[a].first_valid_index().date() for a in selected_assets
    }
    common_start = max(asset_first_dates.values())

    if common_start > start_date_user:
        st.warning(
            f"Some assets do not have data at your chosen start date. "
            f"Optimization will start at **{common_start}** instead of **{start_date_user}**."
        )

    # -------------------------------
    # 6) Fréquence de rebalancement
    # -------------------------------
    rebalance_freq = st.selectbox(
        "Rebalance Frequency",
        options=["Quarterly", "Semi-Annually", "Annually"],
        index=2
    )

    # -------------------------------
    # 7) Bouton d'optimisation
    # -------------------------------
    if st.button("Optimize My Portfolio"):
        with st.spinner("Running optimization..."):
            results = perform_optimization(
                selected_assets=selected_assets,
                start_date_user=start_date_user,
                end_date_user=end_date_user,
                rebalance_freq=rebalance_freq,
                custom_data=custom_data,
            )
            if results is not None:
                st.session_state.results = results
                st.success("Optimization complete! See Portfolio Results tab.")


with tab2:
    st.title("Portfolio Results")

    if "results" not in st.session_state:
        st.info("Please run an optimization first.")
    else:
        results = st.session_state.results

        st.subheader("Final Weights")
        st.write(
            pd.DataFrame({
                "Asset": results["selected_assets"],
                "Weight": results["weights"]
            }).set_index("Asset")
        )

        st.subheader("Risk Contributions (%)")
        st.write(
            pd.DataFrame({
                "Asset": results["selected_assets"],
                "RC %": results["risk_contrib_pct"]
            }).set_index("Asset")
        )

        st.subheader("Performance metrics")
        st.write(f"Expected annual return: **{results['expected_return']:.2f}%**")
        st.write(f"Annual volatility: **{results['volatility']:.2f}%**")
        st.write(f"Sharpe ratio: **{results['sharpe']:.2f}**")
        st.write(f"Total transaction costs: **{results['total_tc']:.2f}%**")

        st.subheader("Cumulative Performance")
        st.line_chart(results["cum_port"])

        st.subheader("Correlation Matrix")
        st.dataframe(results["corr_matrix"])


st.markdown("<br>", unsafe_allow_html=True)

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

    st.markdown("---")
    st.markdown("## 👥 Meet the Team")
    st.markdown("<br>", unsafe_allow_html=True)

    team = [
        {
            "name": "Lucas Jaccard",
            "role": "Frontend Developer",
            "desc": "Lucas designs the app’s visual experience, combining clarity, interactivity, and elegance to make financial analysis more accessible.",
            "photo": "https://raw.githubusercontent.com/quantquant-max/QARM-II-Pension-Fund-ERC/main/team_photos/Lucas_JACCARD.JPG"
        },
        {
            "name": "Audrey Champion",
            "role": "Financial Engineer",
            "desc": "Audrey focuses on translating theory into practice, helping design the pension fund strategy and ensuring academic rigor in implementation.",
            "photo": "https://raw.githubusercontent.com/quantquant-max/QARM-II-Pension-Fund-ERC/main/team_photos/Audrey_CHAMPION.JPG"
        },
        {
            "name": "Arda Budak",
            "role": "Quantitative Analyst",
            "desc": "Arda applies quantitative methods and stochastic simulations to enhance risk control and portfolio diversification within the project.",
            "photo": "https://raw.githubusercontent.com/quantquant-max/QARM-II-Pension-Fund-ERC/main/team_photos/Arda_BUDAK.JPG"
        },
        {
            "name": "Rihem Rhaiem",
            "role": "Data Scientist",
            "desc": "Rihem specializes in financial data analytics and portfolio optimization models, contributing quantitative insight to the ERC framework.",
            "photo": "https://raw.githubusercontent.com/quantquant-max/QARM-II-Pension-Fund-ERC/main/team_photos/Rihem_RHAIEM.JPG"
        },
        {
            "name": "Edward Arion",
            "role": "Backend Developer",
            "desc": "Edward ensures computational stability and performance, integrating optimization algorithms efficiently within the Streamlit app.",
            "photo": "https://raw.githubusercontent.com/quantquant-max/QARM-II-Pension-Fund-ERC/main/team_photos/Edward_ARION.JPG"
        },
    ]

    # Display team members
    cols = st.columns(len(team))
    for i, member in enumerate(team):
        with cols[i]:
            st.image(member["photo"], width=150)
            st.markdown(f"### {member['name']}")
            st.markdown(f"**{member['role']}**")
            st.write(member["desc"])

