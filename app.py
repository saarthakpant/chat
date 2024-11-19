# app.py

import dash
from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import dash_dangerously_set_inner_html
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
import os
from dotenv import load_dotenv
import json
import markdown2

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Database connection parameters
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# Initialize the Dash app with Bootstrap and custom CSS
app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.4.0/styles/github.min.css",
    ],
    external_scripts=[
        "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.4.0/highlight.min.js"
    ]
)

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Custom CSS for chat bubbles
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .chat-container {
                height: 60vh;
                overflow-y: auto;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
            }
            .message {
                margin: 10px 0;
                padding: 10px 15px;
                border-radius: 15px;
                max-width: 80%;
                word-wrap: break-word;
            }
            .user-message {
                background: #007bff;
                color: white;
                margin-left: auto;
            }
            .assistant-message {
                background: white;
                color: black;
                margin-right: auto;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            pre {
                background-color: #f6f8fa;
                padding: 16px;
                border-radius: 6px;
                overflow-x: auto;
            }
            code {
                font-family: monospace;
                background-color: #f6f8fa;
                padding: 2px 4px;
                border-radius: 3px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <script>
            document.addEventListener('DOMContentLoaded', (event) => {
                hljs.highlightAll();
            });
        </script>
    </body>
</html>
'''

# Define the layout of the app
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("AI Chat Assistant", className="text-center text-primary mb-4"),
                width=12
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Div(id='chat-container', className="chat-container"),
                width=12
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Textarea(
                        id='prompt-input',
                        placeholder='Type your message here...',
                        style={'width': '100%', 'height': 80},
                    ),
                    width=10
                ),
                dbc.Col(
                    dbc.Button(
                        "Send", 
                        id='submit-button', 
                        color='primary', 
                        className="h-100 w-100"
                    ),
                    width=2
                ),
            ],
            className="mt-3"
        ),
        dcc.Store(id='chat-history', data=[]),
    ],
    fluid=True,
    className="p-4"
)

def create_message_div(text, is_user=False):
    message_class = "message user-message" if is_user else "message assistant-message"
    if not is_user:
        # Convert markdown to HTML for assistant messages
        html_content = markdown2.markdown(
            text,
            extras=["fenced-code-blocks", "tables", "break-on-newline"]
        )
        return html.Div(
            dash_dangerously_set_inner_html.DangerouslySetInnerHTML(html_content),
            className=message_class
        )
    return html.Div(text, className=message_class)

@app.callback(
    [Output('chat-container', 'children'),
     Output('chat-history', 'data'),
     Output('prompt-input', 'value')],
    [Input('submit-button', 'n_clicks'),
     Input('prompt-input', 'n_submit')],
    [State('prompt-input', 'value'),
     State('chat-history', 'data')],
    prevent_initial_call=True
)
def update_chat(n_clicks, n_submit, prompt, chat_history):
    if not callback_context.triggered or (prompt is None or prompt.strip() == ""):
        return dash.no_update, dash.no_update, dash.no_update

    if chat_history is None:
        chat_history = []

    try:
        # Generate embedding for the user prompt
        prompt_embedding = model.encode(prompt, convert_to_tensor=False)

        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        register_vector(conn)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Perform similarity search
        cursor.execute("""
            SELECT 
                dialogue_id, 
                speaker, 
                utterance, 
                embedding
            FROM conversation_turns
            ORDER BY embedding <=> %s
            LIMIT 3;
        """, (prompt_embedding,))

        similar_dialogues = cursor.fetchall()
        cursor.close()
        conn.close()

        context = ""
        if similar_dialogues:
            for dialogue in similar_dialogues:
                speaker = "User" if dialogue['speaker'] == 1 else "Assistant"
                utterance = dialogue['utterance']
                context += f"{speaker}: {utterance}\n"

        openai_prompt = f"{context}\nUser: {prompt}\nAssistant:"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Format your responses using markdown when appropriate."},
                {"role": "user", "content": openai_prompt}
            ],
            max_tokens=150,
            temperature=0.7,
        )

        assistant_response = response.choices[0].message.content.strip()

        # Update chat history
        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": assistant_response})

        # Create message components
        message_components = []
        for message in chat_history:
            is_user = message["role"] == "user"
            message_components.append(create_message_div(message["content"], is_user))

        return message_components, chat_history, ""

    except Exception as e:
        error_message = f"❌ An error occurred: {str(e)}"
        chat_history.append({"role": "assistant", "content": error_message})
        message_components = [create_message_div(msg["content"], msg["role"] == "user") for msg in chat_history]
        return message_components, chat_history, ""

if __name__ == '__main__':
    app.run_server(debug=True)