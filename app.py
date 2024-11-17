import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

# Initialize the Dash app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the app
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("LLM ICL Pipeline Front End", className="text-center text-primary mb-4"),
                width=12
            )
        ),
        dbc.Row(
            dbc.Col(
                dbc.Textarea(
                    id='prompt-input',
                    placeholder='Enter your prompt here...',
                    style={'width': '100%', 'height': 100},
                ),
                width=12
            )
        ),
        dbc.Row(
            dbc.Col(
                dbc.Button("Submit", id='submit-button', color='primary', className="mt-3"),
                width=12,
                className="text-center"
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Div(id='output-area', className="mt-4", style={'whiteSpace': 'pre-wrap'}),
                width=12
            )
        )
    ],
    fluid=True,
    className="p-4"
)

# Define the callback to handle button clicks
@app.callback(
    Output('output-area', 'children'),
    Input('submit-button', 'n_clicks'),
    State('prompt-input', 'value')
)
def update_output(n_clicks, prompt):
    if n_clicks is None:
        return "Please enter a prompt and click Submit."
    elif not prompt:
        return "Prompt is empty. Please enter a valid prompt."
    else:
        # Mock response for testing purposes
        mock_response = f"Mock Response for the prompt:\n\"{prompt}\""
        return mock_response

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
