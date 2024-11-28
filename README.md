# AI Chat Assistant with RAG Pipeline

A sophisticated chatbot application that leverages Retrieval-Augmented Generation (RAG) with PostgreSQL vector storage and GPT-4. The application features a ChatGPT-like interface and semantic search capabilities for context-aware responses.

## 🌟 Features

### Core Functionality
- **RAG Pipeline Integration**: Combines retrieval-based and generative approaches for more accurate responses
- **Vector Similarity Search**: Utilizes pgvector for efficient semantic search
- **GPT-4 Integration**: Leverages OpenAI's most advanced language model
- **Conversation Memory**: Stores and retrieves relevant conversation history

### User Interface
- **ChatGPT-like Design**: Modern, responsive interface matching OpenAI's design
- **Markdown Support**: Rich text formatting including:
  - Code blocks with syntax highlighting
  - Tables
  - Lists
  - Inline formatting
- **Real-time Updates**: Immediate response rendering
- **Mobile-Responsive**: Adapts to different screen sizes

### Technical Features
- **Vector Embeddings**: Uses SentenceTransformers for text embedding
- **PostgreSQL Integration**: Efficient vector storage and retrieval
- **Async Support**: Handles multiple requests efficiently
- **Environment Configuration**: Flexible deployment settings

## 🚀 Prerequisites

### Software Requirements
- Python 3.8 or higher
- PostgreSQL 13 or higher

### API Keys
- OpenAI API key with GPT-4 access

### System Requirements
- Minimum 4GB RAM
- 2GB free disk space
- Internet connection for API calls

## 💾 PostgreSQL Setup

### 1. Installation

#### Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Install PostgreSQL and required tools
sudo apt install postgresql postgresql-contrib postgresql-server-dev-all build-essential

# Verify installation
psql --version
```

#### macOS
```bash
# Using Homebrew
brew install postgresql

# Start PostgreSQL service
brew services start postgresql
```

#### Windows
- Download installer from [PostgreSQL Official Website](https://www.postgresql.org/download/windows/)
- Run installer and follow wizard
- Add PostgreSQL bin directory to system PATH

### 2. pgvector Extension Setup

```bash
# Clone pgvector repository
git clone https://github.com/pgvector/pgvector.git
cd pgvector

# Build and install
make
sudo make install

# Verify installation
psql -d postgres -c "CREATE EXTENSION vector;" 2>/dev/null && echo "pgvector installed successfully"
```

### 3. Database Configuration

```bash
# Access PostgreSQL prompt
sudo -u postgres psql

# Create database and user
CREATE DATABASE chatbot_db;
CREATE USER chatbot_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE chatbot_db TO chatbot_user;

# Connect to the new database
\c chatbot_db

# Enable vector extension
CREATE EXTENSION vector;

# Create conversation table
CREATE TABLE conversation_turns (
    dialogue_id SERIAL PRIMARY KEY,
    speaker INTEGER NOT NULL,  -- 1 for user, 2 for assistant
    utterance TEXT NOT NULL,
    embedding vector(384),     -- Dimension matches SentenceTransformer model
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

# Create index for vector similarity search
CREATE INDEX ON conversation_turns USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

## 🛠️ Installation

### 1. Repository Setup
```bash
# Clone repository
git clone <repository-url>
cd <repository-name>

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Unix/macOS
source venv/bin/activate
```

### 2. Dependencies Installation
```bash
# Update pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Verify installations
python -c "import dash; import openai; import sentence_transformers; print('Dependencies installed successfully')"
```

### 3. Environment Configuration

Create a `.env` file in the project root:
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4  # or gpt-4-turbo-preview

# Database Configuration
DB_NAME=chatbot_db
DB_USER=chatbot_user
DB_PASSWORD=your_secure_password
DB_HOST=localhost
DB_PORT=5432

# Application Configuration
DEBUG_MODE=True
LOG_LEVEL=INFO
MAX_TOKENS=150
TEMPERATURE=0.7
```

## 🚀 Running the Application

### 1. Pre-run Checks
```bash
# Verify PostgreSQL is running
sudo systemctl status postgresql  # Linux
brew services list               # macOS

# Check database connection
psql -h localhost -U chatbot_user -d chatbot_db -c "\conninfo"
```

### 2. Start the Application
```bash
# Development mode
python app.py

```

### 3. Access the Application
- Open browser and navigate to `http://localhost:8050`
- Default port can be changed in the application configuration

## 🔧 Troubleshooting

### Common Issues and Solutions

#### PostgreSQL Connection Issues
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# View PostgreSQL logs
sudo tail -f /var/log/postgresql/postgresql-13-main.log

# Verify connection settings
psql -h localhost -U chatbot_user -d chatbot_db
```

#### Permission Issues
```bash
# Grant necessary permissions
sudo -u postgres psql -c "ALTER USER chatbot_user CREATEDB;"
sudo -u postgres psql -c "ALTER USER chatbot_user WITH SUPERUSER;"
```

#### pgvector Installation Problems
```bash
# Install development tools
sudo apt install build-essential postgresql-server-dev-all

# Rebuild and install pgvector
cd pgvector
make clean
make
sudo make install
```

## 📁 Project Structure
```
.
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables
├── assets/              # Static assets
│   ├── styles.css       # Custom CSS
│   └── scripts.js       # Custom JavaScript
├── components/          # Reusable UI components
├── utils/              # Utility functions
├── database/           # Database operations
└── README.md           # This file
```

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and test
4. Commit changes: `git commit -m 'Add amazing feature'`
5. Push to branch: `git push origin feature/amazing-feature`
6. Submit Pull Request

### Code Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Add comments for complex logic
- Include docstrings for functions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-4 API
- Sentence Transformers team
- pgvector contributors
- Dash and Plotly team