import dash
from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import dash_dangerously_set_inner_html
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
import json
import markdown2
import torch
from groq import Groq
import logging
import numpy as np
import faiss

# Load environment variables
load_dotenv()

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Database connection parameters
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# Load the sentence transformer model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Load embeddings and Faiss index
EMBEDDINGS_PATH = "dialogue_embeddings.npy"
DIALOGUE_IDS_PATH = "dialogue_ids.json"
FAISS_INDEX_PATH = "faiss_index.index"

# Load dialogue IDs
with open(DIALOGUE_IDS_PATH, 'r', encoding='utf-8') as f:
    dialogue_ids = json.load(f)

# Load Faiss index
embeddings = np.load(EMBEDDINGS_PATH).astype('float32')  # For dimension info
dimension = embeddings.shape[1]
faiss_index = faiss.read_index(FAISS_INDEX_PATH)

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.4.0/styles/github.min.css",
    ],
    external_scripts=[
        "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.4.0/highlight.min.js"
    ],
    suppress_callback_exceptions=True
)

# Setup logging
logging.basicConfig(
    filename='chat_app.log',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

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
            body {
                background-color: #343541;
                color: #FFFFFF;
                margin: 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            }
            .chat-container {
                height: calc(100vh - 180px);
                overflow-y: auto;
                padding: 0;
                background: #343541;
                scroll-behavior: smooth;
            }
            .message-wrapper {
                display: flex;
                padding: 24px 0;
                border-bottom: 1px solid rgba(255,255,255,0.1);
                margin: 0;
                width: 100%;
            }
            .message-wrapper.user {
                background-color: #343541;
            }
            .message-wrapper.assistant {
                background-color: #444654;
            }
            .message {
                max-width: 800px;
                width: 800px;
                margin: 0 auto;
                padding: 0 20px;
                line-height: 1.6;
                display: flex;
                gap: 20px;
                align-items: flex-start;
            }
            .message-content {
                flex-grow: 1;
            }
            .user-message, .assistant-message {
                background: none;
                color: #FFFFFF;
                border-radius: 0;
            }
            pre {
                background-color: #1a1b26;
                padding: 16px;
                border-radius: 6px;
                overflow-x: auto;
                margin: 10px 0;
            }
            code {
                font-family: 'Söhne Mono', Monaco, 'Andale Mono', 'Ubuntu Mono', monospace;
                color: #e9ecef;
                font-size: 14px;
            }
            .input-container {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                padding: 20px;
                background: #343541;
                border-top: 1px solid rgba(255,255,255,0.1);
            }
            .input-wrapper {
                max-width: 800px;
                margin: 0 auto;
                position: relative;
            }
            .prompt-input {
                width: 100%;
                padding: 16px 45px 16px 15px;
                border-radius: 6px;
                border: 1px solid rgba(255,255,255,0.2);
                background-color: #40414f;
                color: white;
                font-size: 1rem;
                resize: none;
                outline: none;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            .prompt-input:focus {
                border-color: #10a37f;
            }
            .send-button {
                position: absolute;
                right: 10px;
                bottom: 12px;
                background: transparent;
                border: none;
                color: #fff;
                cursor: pointer;
                padding: 5px;
                border-radius: 4px;
                font-size: 1.2rem;
                opacity: 0.8;
                transition: opacity 0.2s;
            }
            .send-button:hover {
                opacity: 1;
            }
            .avatar {
                width: 30px;
                height: 30px;
                border-radius: 2px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-size: 14px;
            }
            .user-avatar {
                background-color: #5436DA;
            }
            .assistant-avatar {
                background-color: #10a37f;
            }
            .hljs {
                background: #1a1b26;
                color: #e9ecef;
            }
            .context-container {
                max-width: 800px;
                margin: 20px auto;
                background-color: #444654;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            }
            .context-container h2 {
                margin-top: 0;
                color: #10a37f;
                font-size: 1.2rem;
                font-weight: 600;
                margin-bottom: 16px;
            }
            p {
                margin: 0 0 10px 0;
            }
            ul, ol {
                margin: 0;
                padding-left: 20px;
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
                document.querySelectorAll('pre code').forEach((el) => {
                    hljs.highlightElement(el);
                });
            });
        </script>
    </body>
</html>
'''

# Define the app layout
app.layout = html.Div([
    # Chat container
    html.Div(
        id='chat-container',
        className="chat-container",
        children=[]
    ),

    # Context display container
    html.Div(
        id='context-container',
        className="context-container",
        children=[]
    ),
    
    # Input container
    html.Div([
        html.Div([
            # Text input
            dbc.Textarea(
                id='prompt-input',
                placeholder='Send a message...',
                className="prompt-input",
                style={'height': '50px'},
                n_submit=0
            ),
            # Submit button
            html.Button(
                "➤",
                id='submit-button',
                className="send-button",
                n_clicks=0
            ),
        ], className="input-wrapper")
    ], className="input-container"),
    
    # Store component for chat history with initial structure
    dcc.Store(id='chat-history', data={"history": [], "pending_info": {}}),
    dcc.Store(id='context-data', data={})
])

# Function to create message divs
def create_message_div(text, is_user=False):
    """Create a message div with proper formatting"""
    wrapper_class = "message-wrapper user" if is_user else "message-wrapper assistant"
    avatar_class = "avatar user-avatar" if is_user else "avatar assistant-avatar"
    avatar_text = "U" if is_user else "A"

    # Render markdown for assistant messages
    if not is_user:
        html_content = markdown2.markdown(
            text,
            extras=["fenced-code-blocks", "tables", "break-on-newline"]
        )
        message_content = html.Div(
            dash_dangerously_set_inner_html.DangerouslySetInnerHTML(html_content),
            className="message-content"
        )
    else:
        message_content = html.Div(text, className="message-content")

    return html.Div([
        html.Div([
            html.Div(avatar_text, className=avatar_class),
            message_content
        ], className="message")
    ], className=wrapper_class)

# Function to create context display
def create_context_div(context_data):
    """Create a div to display the fetched context"""
    if not context_data:
        return html.Div()

    dialogues = context_data.get('dialogues', [])
    turns = context_data.get('turns', [])

    context_markdown = ""

    # Display dialogues
    for dialogue in dialogues:
        context_markdown += f"**Dialogue ID:** {dialogue['dialogue_id']}\n"
        context_markdown += f"**Scenario Category:** {dialogue.get('scenario_category', '')}\n"
        context_markdown += f"**Generated Scenario:** {dialogue.get('generated_scenario', '')}\n"
        context_markdown += f"**Resolution Status:** {dialogue.get('resolution_status', '')}\n\n"

    # Display turns
    for turn in turns:
        context_markdown += f"**Turn {turn['turn_number']}**\n"
        context_markdown += f"- **User:** {turn['utterance']}\n"
        if turn['assistant_response']:
            context_markdown += f"- **Assistant:** {turn['assistant_response']}\n"
        context_markdown += "\n"

    html_content = markdown2.markdown(
        context_markdown,
        extras=["fenced-code-blocks", "tables", "break-on-newline"]
    )

    return html.Div([
        html.H2("Context Retrieved"),
        dash_dangerously_set_inner_html.DangerouslySetInnerHTML(html_content)
    ], className="context-container")

# Function to perform similarity search using Faiss
def find_similar_dialogues(prompt_embedding, top_k=3):
    prompt_embedding = np.array(prompt_embedding).astype('float32').reshape(1, -1)
    distances, indices = faiss_index.search(prompt_embedding, top_k)
    similar_dialogue_ids = [dialogue_ids[i] for i in indices[0]]
    return similar_dialogue_ids

# Function to retrieve dialogues from database using dialogue IDs
def get_dialogues_by_ids(dialogue_ids_list):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        placeholders = ','.join(['%s'] * len(dialogue_ids_list))
        query = f"""
            SELECT 
                dialogue_id,
                scenario_category,
                generated_scenario,
                resolution_status,
                time_slot,
                regions,
                num_lines,
                user_emotions,
                assistant_emotions,
                embedding
            FROM dialogues
            WHERE dialogue_id IN ({placeholders});
        """
        cursor.execute(query, dialogue_ids_list)
        dialogues = cursor.fetchall()

        # Retrieve turns for these dialogues
        query_turns = f"""
            SELECT
                dialogue_id,
                turn_number,
                utterance,
                intent,
                assistant_response,
                embedding
            FROM dialogue_turns
            WHERE dialogue_id IN ({placeholders})
            ORDER BY dialogue_id, turn_number;
        """
        cursor.execute(query_turns, dialogue_ids_list)
        turns = cursor.fetchall()

        cursor.close()
        conn.close()

        return {'dialogues': dialogues, 'turns': turns}

    except Exception as e:
        logging.error(f"Error retrieving dialogues: {e}")
        return None
    
# Callback to handle chat updates
@app.callback(
    [Output('chat-container', 'children'),
     Output('chat-history', 'data'),
     Output('prompt-input', 'value'),
     Output('context-container', 'children')],
    [Input('submit-button', 'n_clicks'),
     Input('prompt-input', 'n_submit')],
    [State('prompt-input', 'value'),
     State('chat-history', 'data'),
     State('context-data', 'data')],
    prevent_initial_call=True
)
def update_chat(n_clicks, n_submit, prompt, chat_history, context_data):
    if not callback_context.triggered or (prompt is None or prompt.strip() == ""):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Ensure chat_history has the correct structure
    if not isinstance(chat_history, dict):
        chat_history = {"history": [], "pending_info": {}}
    else:
        if 'history' not in chat_history:
            chat_history['history'] = []
        if 'pending_info' not in chat_history:
            chat_history['pending_info'] = {}

    try:
        # Add user message to chat history
        chat_history['history'].append({"role": "user", "content": prompt})

        # Generate embedding for the user prompt
        with torch.no_grad():
            prompt_embedding = model.encode(prompt, convert_to_numpy=True)

        # Find similar dialogues using Faiss
        similar_dialogue_ids = find_similar_dialogues(prompt_embedding, top_k=3)

        # Retrieve dialogues and turns from the database
        search_results = get_dialogues_by_ids(similar_dialogue_ids)

        # Log the context retrieval
        logging.info(f"Retrieved context: {search_results}")

        # Update context data store
        context_data = search_results

        # Prepare system prompt based on retrieved context
        if search_results:
            dialogues = search_results['dialogues']
            turns = search_results['turns']

            # Construct scenario context
            scenario_context = ""
            if dialogues:
                for dialogue in dialogues:
                    scenario_context += f"**Dialogue ID:** {dialogue['dialogue_id']}\n"
                    scenario_context += f"**Scenario Category:** {dialogue['scenario_category']}\n"
                    scenario_context += f"**Generated Scenario:** {dialogue['generated_scenario']}\n"
                    scenario_context += f"**Resolution Status:** {dialogue['resolution_status']}\n\n"

            # Construct similar conversation examples
            similar_conversations = ""
            for turn in turns:
                speaker = "User" if turn['turn_number'] % 2 == 1 else "Assistant"
                similar_conversations += f"**{speaker}:** {turn['utterance']}\n"
                if turn['assistant_response']:
                    similar_conversations += f"**Assistant:** {turn['assistant_response']}\n\n"

            # Integrate the base prompt into the system prompt
            base_prompt = """
You are a friendly and professional help desk service representative. You work at the company's IT Support Center. Your responsibilities include:

- Addressing user queries with patience and clarity
- Providing step-by-step solutions when needed
- Following up to ensure issues are resolved
- Using appropriate technical terminology while remaining accessible
- Maintaining a professional yet approachable tone

Guidelines for your responses:
- Always greet users professionally
- Ask clarifying questions when needed
- Break down complex solutions into manageable steps
- Use markdown formatting for instructions and code snippets
- Conclude by asking if there's anything else you can help with
- Keep responses concise but thorough

Remember to:
- Show empathy for user frustrations
- Validate concerns before providing solutions
- Use bullet points and numbered lists for clarity
- Highlight important warnings or notes
- Provide relevant documentation links when appropriate

**Contextual Information:**

{scenario_context}

**Similar Conversations for In-Context Learning:**

{similar_conversations}
"""
            # Format the system prompt with scenario context and similar conversations
            system_prompt = base_prompt.format(
                scenario_context=scenario_context,
                similar_conversations=similar_conversations
            )
        else:
            # Integrate the base prompt with default contextual information
            base_prompt = """
You are a friendly and professional help desk service representative. You work at the company's IT Support Center. Your responsibilities include:

- Addressing user queries with patience and clarity
- Providing step-by-step solutions when needed
- Following up to ensure issues are resolved
- Using appropriate technical terminology while remaining accessible
- Maintaining a professional yet approachable tone

Guidelines for your responses:
- Always greet users professionally
- Ask clarifying questions when needed
- Break down complex solutions into manageable steps
- Use markdown formatting for instructions and code snippets
- Conclude by asking if there's anything else you can help with
- Keep responses concise but thorough

Remember to:
- Show empathy for user frustrations
- Validate concerns before providing solutions
- Use bullet points and numbered lists for clarity
- Highlight important warnings or notes
- Provide relevant documentation links when appropriate

**Contextual Information:**

No relevant context found for the current query.

**Similar Conversations for In-Context Learning:**

None
"""
            system_prompt = base_prompt

        # Construct messages for LLM
        messages = [
            {"role": "system", "content": system_prompt}
        ] + chat_history['history']

        # Generate Groq response
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            # model="mixtral-8x7b-32768",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            top_p=1,
            stream=False
        )

        assistant_response = response.choices[0].message.content.strip()
        chat_history['history'].append({"role": "assistant", "content": assistant_response})

        # Create message components for the chat interface
        message_components = [
            create_message_div(msg["content"], msg["role"] == "user")
            for msg in chat_history['history']
        ]

        # Create context component
        context_component = create_context_div(context_data)

        return message_components, chat_history, "", context_component

    except Exception as e:
        logging.error(f"Error in update_chat: {e}")
        error_message = f"❌ An error occurred: {str(e)}"
        chat_history['history'].append({"role": "assistant", "content": error_message})
        # Create message components for the chat interface
        message_components = [
            create_message_div(msg["content"], msg["role"] == "user")
            for msg in chat_history['history']
        ]
        return message_components, chat_history, "", html.Div()

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)