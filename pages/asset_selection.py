import streamlit as st
from utils.data_loader import load_custom_data, get_data
from utils.optimizer import perform_optimization
from datetime import datetime, timedelta

st.title("Asset Selection")

# Load data
custom_data = load_custom_data()
if custom_data.empty:
    st.error("Failed to load the custom dataset.")
else:
    min_date = custom_data.index.min().date()
    max_date = custom_data.index.max().date()
    
    start_date = st.date_input(
        "Start Date",
        min_date,
        min_value=min_date,
        max_value=max_date,
        key="start_date"
    )
    end_date = st.date_input(
        "End Date",
        max_date,
        min_value=min_date,
        max_value=max_date,
        key="end_date"
    )
    
    if start_date and end_date:
        period_data = custom_data.loc[start_date:end_date]
        complete_stocks = [col for col in custom_data.columns if period_data[col].notna().all()]
        
        selected_assets = st.multiselect(
            "Select US Stocks",
            options=complete_stocks,
            key="us_stocks"
        )
    
    rebalance_freq = st.selectbox(
        "Rebalance Frequency",
        options=['Quarterly', 'Semi-Annually', 'Annually'],
        index=2
    )
    
    base_currency = st.selectbox(
        "Base Currency",
        options=['USD', 'EUR', 'GBP'],
        index=0
    )
    
    # How to Use section (paste from your original)
    st.markdown("### How to Use")
    st.write("""
    # ... (paste your How to Use text here)
    """)
    
    if st.button("Optimize My Portfolio"):
        if not selected_assets:
            st.error("Please select at least one asset to proceed.")
        else:
            with st.spinner("Calculating..."):
                results = perform_optimization(selected_assets, start_date, end_date, rebalance_freq, base_currency, custom_data)
                if results:
                    st.session_state.results = results
                    st.success("Optimization complete! Go to Portfolio Results page.")
