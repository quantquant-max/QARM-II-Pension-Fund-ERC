import pandas as pd
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize_scalar
from sklearn.covariance import LedoitWolf
from datetime import timedelta
import streamlit as st
from utils.data_loader import get_data
import plotly.graph_objects as go

def perform_optimization(selected_assets, start_date, end_date, rebalance_freq, custom_data):  # Removed base_currency
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
        
        # Removed multi-currency block entirely
        
        period_returns = returns.loc[start_date:end_date]
        period_value_weighted = value_weighted_returns.loc[start_date:end_date]
        period_equally_weighted = equally_weighted_returns.loc[start_date:end_date]
        
        if len(period_returns) < 2:
            st.error("Insufficient data points for the selected period.")
            return None
        
        # Transaction cost rate (0.1%)
        tc_rate = 0.001
        
        # Rebalance frequency mapping
        freq_map = {
            'Quarterly': 'QS',
            'Semi-Annually': '6MS',
            'Annually': 'YS'
        }
        freq = freq_map[rebalance_freq]
        
        # Find rebalance dates
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        rebalance_dates = []
        for d in date_range:
            candidates = returns.index[returns.index >= d]
            if not candidates.empty:
                rebalance_dates.append(candidates[0])
        
        # Add the end date if not already included
        if returns.index[-1] not in rebalance_dates and returns.index[-1] > rebalance_dates[-1]:
            rebalance_dates.append(returns.index[-1])
        
        n = len(selected_assets)
        previous_weights = np.ones(n) / n  # Initial equal
        port_returns = pd.Series(index=period_returns.index, dtype=float)
        weights_over_time = {}
        total_tc = 0.0
        
        # Initial optimization
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
                return None
            else:
                weights = np.where(np.abs(weights) < 1e-4, 0, weights)
                weights /= np.sum(weights)
                
                # Initial transaction
                turnover = np.sum(np.abs(weights - previous_weights)) / 2
                cost = turnover * tc_rate
                total_tc += cost
                
                previous_weights = weights
                weights_over_time[rebal_date] = weights
        
        # Subsequent rebalances
        for i in range(1, len(rebalance_dates)):
            rebal_date = rebalance_dates[i]
            prev_rebal_date = rebalance_dates[i-1]
            
            # Period returns
            period_returns_slice = period_returns.loc[prev_rebal_date:rebal_date - pd.Timedelta(days=1)]
            if not period_returns_slice.empty:
                period_port_ret = period_returns_slice.dot(previous_weights)
                port_returns.loc[period_returns_slice.index] = period_port_ret
            
            # Rebalance
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
        
        # Last period
        last_rebal_date = rebalance_dates[-1]
        last_period_returns = period_returns.loc[last_rebal_date:]
        if not last_period_returns.empty:
            last_port_ret = last_period_returns.dot(previous_weights)
            port_returns.loc[last_period_returns.index] = last_port_ret
        
        # Drop NaN
        port_returns = port_returns.dropna()
        
        # Cumulative
        cum_port = (1 + port_returns).cumprod()
        cum_value_weighted = (1 + period_value_weighted).cumprod()
        cum_equally_weighted = (1 + period_equally_weighted).cumprod()
        
        # Metrics
        ann_return = port_returns.mean() * 252
        ann_vol = port_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Risk contrib using last year
        est_returns = returns.iloc[-252:]
        if len(est_returns) >= n + 1:
            lw = LedoitWolf().fit(est_returns)
            S_np = lw.covariance_ * 252
            port_var = weights @ S_np @ weights  # Use last weights
            sigma = np.sqrt(port_var)
            MRC = S_np @ weights
            risk_contrib = weights * MRC / sigma
            total_risk = np.sum(risk_contrib)
            risk_contrib_pct = (risk_contrib / total_risk) * 100 if total_risk > 0 else np.zeros(n)
        else:
            risk_contrib_pct = np.ones(n) / n * 100
        
        # Weights animation
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
        
        # Store results
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
