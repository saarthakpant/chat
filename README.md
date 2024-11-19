# AI Chat Assistant with RAG Pipeline

A chatbot application that uses a Retrieval-Augmented Generation (RAG) pipeline with PostgreSQL vector storage and GPT-4.


## PostgreSQL Setup

1. Install PostgreSQL:
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install postgresql postgresql-contrib

   # macOS (using Homebrew)
   brew install postgresql
   ```

2. Start PostgreSQL service:
   ```bash
   # Ubuntu/Debian
   sudo systemctl start postgresql
   sudo systemctl enable postgresql

   # macOS
   brew services start postgresql
   ```

3. Install pgvector extension:
   ```bash
   # Ubuntu/Debian
   sudo apt install postgresql-server-dev-13  # replace 13 with your PostgreSQL version
   git clone https://github.com/pgvector/pgvector.git
   cd pgvector
   make
   sudo make install
   ```

4. Create database and user:
   ```bash
   # Access PostgreSQL prompt
   sudo -u postgres psql

   # Create database and user (in psql prompt)
   CREATE DATABASE your_database_name;
   CREATE USER your_username WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE your_database_name TO your_username;
   ```

5. Enable pgvector and create table:
   ```sql
   # Connect to your database
   \c your_database_name

   # Enable vector extension
   CREATE EXTENSION vector;

   # Create conversation table
   CREATE TABLE conversation_turns (
       dialogue_id SERIAL PRIMARY KEY,
       speaker INTEGER NOT NULL,  -- 1 for user, 2 for assistant
       utterance TEXT NOT NULL,
       embedding vector(384)      -- Dimension matches the SentenceTransformer model
   );
   ```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   DB_NAME=your_database_name
   DB_USER=your_username
   DB_PASSWORD=your_password
   DB_HOST=localhost
   DB_PORT=5432
   ```

## Running the Application

1. Ensure PostgreSQL is running:
   ```bash
   # Check status (Ubuntu/Debian)
   sudo systemctl status postgresql

   # Check status (macOS)
   brew services list
   ```

2. Start the application:
   ```bash
   python app.py
   ```

3. Open your browser and navigate to `http://localhost:8050`

## Dependencies

Install all dependencies:
```bash
pip install dash dash-bootstrap-components dash-mantine-components dash-dangerously-set-inner-html openai sentence-transformers psycopg2-binary pgvector python-dotenv markdown2
```

## Troubleshooting

1. If pgvector installation fails:
   ```bash
   # Ubuntu/Debian
   sudo apt install build-essential
   sudo apt install postgresql-server-dev-all
   ```

2. If PostgreSQL connection fails:
   ```bash
   # Check PostgreSQL is running
   sudo systemctl status postgresql

   # Check PostgreSQL logs
   sudo tail -f /var/log/postgresql/postgresql-13-main.log
   ```

3. Common permission issues:
   ```bash
   # Grant necessary permissions
   sudo -u postgres psql -c "ALTER USER your_username CREATEDB;"
   sudo -u postgres psql -c "ALTER USER your_username WITH SUPERUSER;"
   ```

## Project Structure

```
.
├── app.py          # Main application file
├── requirements.txt # Python dependencies
├── .env            # Environment variables
└── README.md       # This file
```

