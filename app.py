import streamlit as st

# Custom styling (full CSS block)
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

st.set_page_config(page_title="Pension Fund Optimizer", layout="wide")
st.logo("ERC Portfolio.png")
