import streamlit as st

# Custom styling (keep your existing CSS here - paste the full <style> block from your original code)
st.markdown(
    """
    <style>
    # ... (paste your full CSS here from the original app.py)
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="Pension Fund Optimizer", layout="wide")
st.logo("ERC Portfolio.png")
