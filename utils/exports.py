import streamlit as st
from fpdf import FPDF
import io

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
