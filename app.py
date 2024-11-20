import dash
from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import dash_dangerously_set_inner_html
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
import os
from dotenv import load_dotenv
import json
import markdown2
import torch
from groq import Groq

# Load environment variables
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Database connection parameters
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

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

# Load the sentence transformer model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

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
            }
            .chat-container {
                height: calc(100vh - 180px);
                overflow-y: auto;
                padding: 20px;
                background: #343541;
            }
            .message-wrapper {
                display: flex;
                padding: 20px 0;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }
            .message-wrapper.user {
                background-color: #444654;
            }
            .message-wrapper.context {
                background-color: #2c2c3a;
            }
            .message {
                max-width: 800px;
                margin: 0 auto;
                width: 100%;
                padding: 0 20px;
                line-height: 1.5;
            }
            .user-message {
                color: #FFFFFF;
            }
            .assistant-message {
                color: #FFFFFF;
            }
            .context-message {
                color: #a8a8b4;
                font-size: 0.9em;
            }
            pre {
                background-color: #1a1b26;
                padding: 16px;
                border-radius: 6px;
                overflow-x: auto;
                margin: 10px 0;
            }
            code {
                font-family: monospace;
                color: #e9ecef;
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
                padding: 12px 45px 12px 15px;
                border-radius: 6px;
                border: 1px solid rgba(255,255,255,0.2);
                background-color: #40414f;
                color: white;
                font-size: 1rem;
                resize: none;
                outline: none;
            }
            .prompt-input:focus {
                border-color: rgba(255,255,255,0.4);
            }
            .send-button {
                position: absolute;
                right: 10px;
                bottom: 10px;
                background: transparent;
                border: none;
                color: #fff;
                cursor: pointer;
                padding: 5px;
                border-radius: 4px;
            }
            .send-button:hover {
                background: rgba(255,255,255,0.1);
            }
            .avatar {
                width: 30px;
                height: 30px;
                margin-right: 15px;
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
                background-color: #19C37D;
            }
            .context-avatar {
                background-color: #4b4b5b;
            }
            .hljs {
                background: #1a1b26;
                color: #e9ecef;
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
    
    # Store component for chat history
    dcc.Store(id='chat-history', data=[])
])

def get_relevant_context(cursor, prompt_embedding, prompt):
    """Enhanced context retrieval with hierarchical search"""
    
    # Convert numpy array to list and format for PostgreSQL vector
    embedding_list = prompt_embedding.tolist()
    vector_str = f"[{','.join(map(str, embedding_list))}]"
    
    # First: Find most relevant dialogues based on scenario and embedding similarity
    cursor.execute("""
        SELECT 
            d.dialogue_id,
            d.scenario_category,
            d.services,
            d.generated_scenario,
            d.embedding <=> %s::vector as dialogue_similarity
        FROM dialogues d
        WHERE d.embedding <=> %s::vector < 0.8
        ORDER BY dialogue_similarity
        LIMIT 3;
    """, (vector_str, vector_str))
    
    relevant_dialogues = cursor.fetchall()
    if not relevant_dialogues:
        return None

    # Second: Get relevant turns from these dialogues
    dialogue_ids = [d['dialogue_id'] for d in relevant_dialogues]
    placeholders = ','.join(['%s'] * len(dialogue_ids))
    
    cursor.execute(f"""
        WITH ranked_turns AS (
            SELECT 
                dt.*,
                dm.user_emotions,
                dm.assistant_emotions,
                dt.embedding <=> %s::vector as turn_similarity,
                ROW_NUMBER() OVER (
                    PARTITION BY dt.dialogue_id 
                    ORDER BY dt.embedding <=> %s::vector
                ) as rank
            FROM dialogue_turns dt
            JOIN dialogue_metadata dm ON dt.dialogue_id = dm.dialogue_id
            WHERE dt.dialogue_id IN ({placeholders})
            AND dt.embedding <=> %s::vector < 0.8
        )
        SELECT *
        FROM ranked_turns
        WHERE rank <= 2
        ORDER BY turn_similarity
        LIMIT 6;
    """, (vector_str, vector_str, *dialogue_ids, vector_str))

    relevant_turns = cursor.fetchall()

    # Structure the context
    context = {
        'dialogues': [dict(d) for d in relevant_dialogues],
        'turns': [dict(t) for t in relevant_turns]
    }

    return context

def create_message_div(text, is_user=False, is_context=False):
    """Create a message div with proper formatting"""
    avatar_content = "U" if is_user else ("C" if is_context else "A")
    avatar_class = "user-avatar" if is_user else ("context-avatar" if is_context else "assistant-avatar")
    wrapper_class = f"message-wrapper {'user' if is_user else 'context' if is_context else ''}"
    
    if not is_user:
        html_content = markdown2.markdown(
            text,
            extras=["fenced-code-blocks", "tables", "break-on-newline"]
        )
        message = html.Div(
            dash_dangerously_set_inner_html.DangerouslySetInnerHTML(html_content),
            className=f"message {'context-message' if is_context else 'assistant-message'}"
        )
    else:
        message = html.Div(text, className="message user-message")
    
    return html.Div([
        html.Div(avatar_content, className=f"avatar {avatar_class}"),
        message
    ], className=wrapper_class)

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
        with torch.no_grad():
            prompt_embedding = model.encode(prompt, convert_to_tensor=False)

        # Connect to database and get context
        conn = psycopg2.connect(**DB_CONFIG)
        register_vector(conn)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Get relevant context using enhanced search
        search_results = get_relevant_context(cursor, prompt_embedding, prompt)
        cursor.close()
        conn.close()

        # Debug print to show context being fetched
        print("\n=== Context Retrieved ===")
        if search_results:
            print("\nRelevant Dialogues:")
            for dialogue in search_results['dialogues']:
                print(f"\nCategory: {dialogue['scenario_category']}")
                print(f"Services: {', '.join(dialogue['services']) if dialogue['services'] else 'None'}")
                print(f"Scenario: {dialogue['generated_scenario']}")
                print(f"Similarity Score: {dialogue['dialogue_similarity']:.4f}")

            print("\nRelevant Turns:")
            for turn in search_results['turns']:
                print(f"\nTurn {turn['turn_number']}:")
                print(f"Speaker: {'User' if turn['turn_number'] % 2 == 1 else 'Assistant'}")
                print(f"Intent: {turn['intent']}")
                print(f"Utterance: {turn['utterance']}")
                if turn['assistant_response']:
                    print(f"Response: {turn['assistant_response']}")
                if 'turn_similarity' in turn:
                    print(f"Similarity Score: {turn['turn_similarity']:.4f}")
                if 'user_emotions' in turn:
                    print(f"User Emotions: {', '.join(turn['user_emotions']) if turn['user_emotions'] else 'None'}")
                if 'assistant_emotions' in turn:
                    print(f"Assistant Emotions: {', '.join(turn['assistant_emotions']) if turn['assistant_emotions'] else 'None'}")

            # Add context to chat history
            context_message = "**Similar Conversations:**\n\n"
            for turn in search_results['turns']:
                speaker = "User" if turn['turn_number'] % 2 == 1 else "Assistant"
                context_message += f"**{speaker}**: {turn['utterance']}\n"
                if turn['assistant_response']:
                    context_message += f"**Response**: {turn['assistant_response']}\n"
                context_message += "\n"

            chat_history.append({"role": "context", "content": context_message})
        else:
            print("No relevant context found")
        print("\n=== End Context ===\n")

        # Add user message to chat history
        chat_history.append({"role": "user", "content": prompt})

        # Prepare ICL examples from search results
        if search_results:
            icl_examples = []
            for turn in search_results['turns']:
                if turn['turn_number'] % 2 == 1:  # User message
                    icl_examples.append({
                        "role": "user",
                        "content": turn['utterance']
                    })
                    if turn['assistant_response']:
                        icl_examples.append({
                            "role": "assistant",
                            "content": turn['assistant_response']
                        })

            # Prepare scenario context
            scenario_context = ""
            if search_results['dialogues']:
                dialogue = search_results['dialogues'][0]
                scenario_context = f"""
Current Context:
- Category: {dialogue['scenario_category']}
- Services: {', '.join(dialogue['services']) if dialogue['services'] else 'None'}
- Scenario: {dialogue['generated_scenario']}

Similar conversation examples for reference:"""

            # Construct messages for LLM
            messages = [
                {"role": "system", "content": f"""You are a friendly and professional help desk service representative. You work at the company's IT Support Center. Your responsibilities include:

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

{scenario_context}"""},
                *icl_examples,  # Include example conversations
                {"role": "user", "content": prompt}
            ]
        else:
            # If no context found, use basic system message
            messages = [
                {"role": "system", "content": """You are a friendly and professional help desk service representative. You work at the company's IT Support Center. Your responsibilities include:

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
- Provide relevant documentation links when appropriate"""},
                {"role": "user", "content": prompt}
            ]

        # Generate Groq response
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            top_p=1,
            stream=False
        )

        assistant_response = response.choices[0].message.content.strip()
        chat_history.append({"role": "assistant", "content": assistant_response})

        # Replace this section
        message_components = [
            create_message_div(msg["content"], msg["role"] == "user", msg["role"] == "context")
            for msg in chat_history
        ]

        return message_components, chat_history, ""

    except Exception as e:
        error_message = f"❌ An error occurred: {str(e)}"
        chat_history.append({"role": "assistant", "content": error_message})
        # And replace this section
        message_components = [
            create_message_div(msg["content"], msg["role"] == "user", msg["role"] == "context")
            for msg in chat_history
        ]
        return message_components, chat_history, ""

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)



    # Generate OpenAI response
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": """You are a friendly and professional help desk service representative. Your name is Alex and you work at the company's IT Support Center. Your responsibilities include:

# - Addressing user queries with patience and clarity
# - Providing step-by-step solutions when needed
# - Following up to ensure issues are resolved
# - Using appropriate technical terminology while remaining accessible
# - Maintaining a professional yet approachable tone

# Guidelines for your responses:
# - Always greet users professionally
# - Ask clarifying questions when needed
# - Break down complex solutions into manageable steps
# - Use markdown formatting for instructions and code snippets
# - Conclude by asking if there's anything else you can help with
# - Keep responses concise but thorough

# Remember to:
# - Show empathy for user frustrations
# - Validate concerns before providing solutions
# - Use bullet points and numbered lists for clarity
# - Highlight important warnings or notes
# - Provide relevant documentation links when appropriate"""},
#                 {"role": "user", "content": f"{openai_context}\nUser: {prompt}"}
#             ],
#             max_tokens=500,
#             temperature=0.7,
#         )

#         assistant_response = response.choices[0].message.content.strip()
#         chat_history.append({"role": "assistant", "content": assistant_response})


 # except Exception as e:
    #     error_message = f"❌ An error occurred: {str(e)}"
    #     chat_history.append({"role": "assistant", "content": error_message})
    #     message_components = [
    #             create_message_div(msg["content"], msg["role"] == "user", msg["role"] == "context") 
    #             for msg in chat_history
    #         ]
    #     return message_components, chat_history, ""

    # except Exception as e:
    #     error_message = f"❌ An error occurred: {str(e)}"
    #     chat_history.append({"role": "assistant", "content": error_message})
    #     message_components = [
    #         create_message_div(msg["content"], msg["role"] == "user", msg["role"] == "context") 
    #         for msg in chat_history
    #     ]
    #     return message_components, chat_history, ""
