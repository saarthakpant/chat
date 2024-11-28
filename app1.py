import dash
from dash import html, dcc, Input, Output, State
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
import logging
import numpy as np
import faiss
from groq import Groq  # Ensure you have the groq package installed

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
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css",
    ],
    suppress_callback_exceptions=True
)

# Setup logging
logging.basicConfig(
    filename='chat_app.log',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Custom CSS to mimic OpenAI ChatGPT UI
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
                color: #d1d5db;
                font-family: "Segoe UI", sans-serif;
                height: 100vh;
                margin: 0;
                padding: 0;
                overflow: hidden;
            }
            .app-container {
                display: flex;
                flex-direction: column;
                height: 100vh;
            }
            .main-container {
                flex: 1;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            .chat-container {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
            }
            .context-container {
                overflow-y: auto;
                max-height: 150px; /* Adjust the height as needed */
                padding: 20px;
                background-color: #2A2B32;
            }
            .message-row {
                display: flex;
                align-items: flex-start;
                margin-bottom: 20px;
            }
            .message-row.user {
                justify-content: flex-end;
            }
            .message-row.assistant {
                justify-content: flex-start;
            }
            .message {
                max-width: 80%;
                padding: 15px;
                border-radius: 8px;
                line-height: 1.5;
                position: relative;
                white-space: pre-wrap;
                word-wrap: break-word;
                overflow-wrap: break-word;
                font-size: 16px;
            }
            .message.user {
                background-color: #202123;
                color: #fff;
            }
            .message.assistant {
                background-color: #444654;
                color: #fff;
            }
            .input-container {
                padding: 10px 20px;
                background-color: #343541;
                border-top: 1px solid #444654;
            }
            .input-area {
                display: flex;
                align-items: center;
                max-width: 800px;
                margin: 0 auto;
            }
            .prompt-input {
                flex: 1;
                padding: 12px 15px;
                border-radius: 5px;
                border: none;
                background-color: #40414F;
                color: #fff;
                font-size: 16px;
                resize: none;
                outline: none;
            }
            .prompt-input::placeholder {
                color: #8e8ea0;
            }
            .send-button {
                background: none;
                border: none;
                color: #fff;
                font-size: 24px;
                margin-left: 10px;
                cursor: pointer;
                outline: none;
            }
            .send-button:hover {
                color: #19C37D;
            }
            .assistant .avatar {
                margin-right: 10px;
            }
            .user .avatar {
                display: none;
            }
            .avatar {
                width: 40px;
                height: 40px;
                background-color: #19C37D;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #fff;
                font-size: 20px;
                flex-shrink: 0;
            }
            /* Scrollbar styling */
            ::-webkit-scrollbar {
                width: 8px;
            }
            ::-webkit-scrollbar-track {
                background: #2A2B32;
            }
            ::-webkit-scrollbar-thumb {
                background-color: #888;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #555;
            }
            /* Code block styling */
            pre {
                background-color: #202123;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
            }
            code {
                font-family: "Fira Code", monospace;
                color: #d1d5db;
            }
            /* Context container styling */
            .context-container h2 {
                margin-top: 0;
                margin-bottom: 15px;
                color: #19C37D;
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
    </body>
</html>
'''

# Define the app layout
app.layout = html.Div([
    html.Div([
        html.Div([
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
        ], className="main-container"),
        # Input container
        html.Div([
            html.Div([
                # Text input
                dcc.Textarea(
                    id='prompt-input',
                    placeholder='Send a message...',
                    className="prompt-input",
                    rows=1
                ),
                # Submit button
                html.Button(
                    html.I(className="fas fa-paper-plane"),
                    id='submit-button',
                    className="send-button",
                    n_clicks=0
                ),
            ], className="input-area")
        ], className="input-container"),
    ], className="app-container"),
    # Store component for chat history with initial structure
    dcc.Store(id='chat-history', data={"history": [], "pending_info": {}}),
    dcc.Store(id='context-data', data={})
])

# Function to create message divs
def create_message_div(text, is_user=False):
    """Create a message div with proper formatting"""
    message_row_class = "message-row user" if is_user else "message-row assistant"
    message_class = "message user" if is_user else "message assistant"

    avatar = html.Div("A", className="avatar") if not is_user else html.Div()

    # Render markdown for assistant messages
    if not is_user:
        try:
            html_content = markdown2.markdown(
                text,
                extras=["fenced-code-blocks", "tables", "break-on-newline"]
            )
        except Exception as e:
            logging.error(f"Error rendering markdown: {e}")
            # Escape the text to prevent code execution
            import html
            safe_text = html.escape(text)
            html_content = f"<pre>{safe_text}</pre>"
        message_body = html.Div(
            dash_dangerously_set_inner_html.DangerouslySetInnerHTML(html_content),
            className=message_class
        )
    else:
        message_body = html.Div(text, className=message_class)

    return html.Div([
        avatar,
        message_body
    ], className=message_row_class)

# Function to create context display
def create_context_div(context_data):
    """Create a div to display the fetched context"""
    if not context_data or not context_data.get('dialogues'):
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
        html.H2("Retrieved Context"),
        dash_dangerously_set_inner_html.DangerouslySetInnerHTML(html_content)
    ])

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

        # Build the placeholders string
        placeholders = ', '.join(['%s'] * len(dialogue_ids_list))

        logging.info(f"Dialogue IDs being queried: {dialogue_ids_list}")

        query = f"""
            SELECT 
                dialogue_id,
                scenario_category,
                generated_scenario,
                resolution_status
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
                assistant_response
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
        return {'dialogues': [], 'turns': []}

# Callback to handle chat updates
@app.callback(
    [Output('chat-container', 'children'),
     Output('chat-history', 'data'),
     Output('prompt-input', 'value'),
     Output('context-container', 'children')],
    Input('submit-button', 'n_clicks'),
    [State('prompt-input', 'value'),
     State('chat-history', 'data'),
     State('context-data', 'data')],
    prevent_initial_call=True
)
def update_chat(n_clicks, prompt, chat_history, context_data):
    if n_clicks is None or (prompt is None or prompt.strip() == ""):
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
        logging.info(f"Similar Dialogue IDs: {similar_dialogue_ids}")

        # Retrieve dialogues and turns from the database
        search_results = get_dialogues_by_ids(similar_dialogue_ids)
        logging.info(f"Search Results: {search_results}")

        # Update context data store
        context_data = search_results

        # Log the retrieved context
        logging.info("Retrieved Context:")
        logging.info(json.dumps(context_data, indent=2))

        # Print the retrieved context to the console
        print("Retrieved Context:")
        print(json.dumps(context_data, indent=2))

        # Prepare system prompt based on retrieved context
        if search_results and search_results['dialogues']:
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

        # Generate assistant response using Groq client
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            top_p=1,
            stream=False
        )

        assistant_response = response.choices[0].message.content.strip()
        logging.info(f"Assistant response: {assistant_response}")

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
        error_message = f"An error occurred: {str(e)}"
        chat_history['history'].append({"role": "assistant", "content": error_message})
        # Create message components for the chat interface
        message_components = [
            create_message_div(msg["content"], msg["role"] == "user")
            for msg in chat_history['history']
        ]
        return message_components, chat_history, "", html.Div()

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
