import plotly.graph_objects as go

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
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=cum_port.index, y=cum_port, mode='lines', name='Portfolio', line=dict(color='blue')))
    fig3.add_trace(go.Scatter(x=cum_value_weighted.index, y=cum_value_weighted, mode='lines', name='Value Weighted Benchmark', line=dict(color='green')))
    fig3.add_trace(go.Scatter(x=cum_equally_weighted.index, y=cum_equally_weighted, mode='lines', name='Equally Weighted Benchmark', line=dict(color='red')))
    fig3.update_layout(title=dict(text="Cumulative Returns", font=dict(color="#f0f0f0", family="Times New Roman")), title_x=0.5, xaxis_title="Date", yaxis_title="Cumulative Return", paper_bgcolor="#000000", plot_bgcolor="#000000", font_color="#f0f0f0", font_family="Times New Roman")
    fig3.update_xaxes(title_font_color="#f0f0f0", tickfont_color="#f0f0f0", title_font_family="Times New Roman", tickfont_family="Times New Roman")
    fig3.update_yaxes(title_font_color="#f0f0f0", tickfont_color="#f0f0f0", title_font_family="Times New Roman", tickfont_family="Times New Roman")
    fig3.update_layout(legend=dict(font=dict(color="#f0f0f0", family="Times New Roman")))
    return fig3
