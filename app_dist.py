import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Load the CSV file
data_file = './results/HIS17-two-classes/ranked/rank_encoded.csv'
df = pd.read_csv(data_file)

# Create the Dash app
app = dash.Dash(__name__)
app.title = "Feature Distribution Viewer"

# Layout of the app
app.layout = html.Div([
    html.H1("Statistical Distribution of Features", style={"textAlign": "center"}),

    html.Div([
        html.Label("Select a Feature:"),
        dcc.Dropdown(
            id='feature-dropdown',
            options=[{'label': col, 'value': col} for col in df.columns],
            placeholder="Choose a feature",
            style={"width": "50%"}
        ),
    ], style={"padding": "10px"}),

    dcc.Graph(id='distribution-plot'),
])

# Callback to update the plot based on the selected feature
@app.callback(
    Output('distribution-plot', 'figure'),
    [Input('feature-dropdown', 'value')]
)
def update_distribution(selected_feature):
    if selected_feature is None:
        return px.histogram(title="Select a feature to view its distribution")

    # Create the distribution plot
    fig = px.histogram(
        df,
        x=selected_feature,
        title=f"Distribution of {selected_feature}",
        nbins=30,
        marginal="box",  # Adds a box plot to show statistical details
    )
    fig.update_layout(xaxis_title=selected_feature, yaxis_title="Count")
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
