import streamlit as st
from utils.visualizations import create_pie_chart, create_bar_chart, create_line_chart
from utils.exports import export_csv, export_pdf

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
    
    # Performance metrics
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
    
    # Export features
    st.subheader("Export Results")
    export_csv(results["weights_df"], "weights_history.csv")
    export_pdf(results)
else:
    st.info("Please optimize a portfolio in the Asset Selection page.")
