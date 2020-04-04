import dash
from Dashboard.layout import main_layout

app = dash.Dash('COVID19 dash app')
app.layout = main_layout
