import json
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import warnings
import logging
import torch
from tqdm import tqdm
import numpy as np
import time

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables
load_dotenv()

# Configuration
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'
DATASET_PATH = "Touse.json"
BATCH_SIZE = 100  # Adjust based on memory constraints
EMBEDDING_DIM = 384
MAX_RETRIES = 3   # Number of retries for failed operations

# Logging setup
logging.basicConfig(
    filename='import_dataset.log',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def create_tables(conn):
    """Create optimized tables for the dialogue system with pgvector embeddings"""
    try:
        with conn.cursor() as cur:
            # Enable required extensions
            cur.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE EXTENSION IF NOT EXISTS pg_trgm;
            """)

            # Drop existing tables if they exist to avoid conflicts
            cur.execute("""
                DROP TABLE IF EXISTS dialogue_emotions CASCADE;
                DROP TABLE IF EXISTS dialogue_services CASCADE;
                DROP TABLE IF EXISTS dialogue_regions CASCADE;
                DROP TABLE IF EXISTS dialogue_time_slots CASCADE;
                DROP TABLE IF EXISTS dialogue_turns CASCADE;
                DROP TABLE IF EXISTS dialogues CASCADE;
            """)

            # Create dialogues table with additional columns
            cur.execute("""
                CREATE TABLE dialogues (
                    id SERIAL PRIMARY KEY,
                    dialogue_id TEXT UNIQUE NOT NULL,
                    scenario_category TEXT,
                    generated_scenario TEXT,
                    resolution_status TEXT,
                    num_lines INTEGER,
                    time_slot JSONB,
                    regions TEXT[],
                    user_emotions TEXT[],
                    assistant_emotions TEXT[],
                    embedding VECTOR({dim}),
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
            """.format(dim=EMBEDDING_DIM))

            # Create turns table
            cur.execute("""
                CREATE TABLE dialogue_turns (
                    id SERIAL PRIMARY KEY,
                    dialogue_id TEXT REFERENCES dialogues(dialogue_id) ON DELETE CASCADE,
                    turn_number INTEGER,
                    utterance TEXT,
                    intent TEXT,
                    assistant_response TEXT,
                    embedding VECTOR({dim})
                );
            """.format(dim=EMBEDDING_DIM))

            # Create services table
            cur.execute("""
                CREATE TABLE dialogue_services (
                    id SERIAL PRIMARY KEY,
                    dialogue_id TEXT REFERENCES dialogues(dialogue_id) ON DELETE CASCADE,
                    service TEXT
                );
            """)

            # Indexes for optimized queries
            cur.execute("""
                CREATE INDEX idx_dialogues_scenario_category ON dialogues(scenario_category);
                CREATE INDEX idx_dialogues_resolution_status ON dialogues(resolution_status);
                CREATE INDEX idx_dialogues_embedding ON dialogues USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
                CREATE INDEX idx_dialogue_turns_intent ON dialogue_turns(intent);
                CREATE INDEX idx_dialogue_turns_utterance_trgm ON dialogue_turns USING gin(utterance gin_trgm_ops);
                CREATE INDEX idx_dialogue_turns_embedding ON dialogue_turns USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
            """)
            
            conn.commit()
            print("✅ Created database schema successfully")
            logging.info("Created database schema successfully")
    except Exception as e:
        conn.rollback()
        print(f"❌ Failed to create schema: {e}")
        logging.error(f"Failed to create schema: {e}")
        raise

def generate_embeddings(texts, model, batch_size=32):
    """Generate embeddings with batching and progress bar"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            batch_embeddings = model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True
            )
        embeddings.extend(batch_embeddings)
        
        if (i // batch_size) % 10 == 0:  # Clear cache periodically
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return np.array(embeddings)

def process_dialogue(dialogue, model):
    """Process a single dialogue and return structured data"""
    dialogue_id = dialogue['dialogue_id']
    
    # Generate embedding for the entire dialogue context
    dialogue_text = " ".join([turn['utterance'] for turn in dialogue['turns']])
    dialogue_embedding = model.encode([dialogue_text])[0]
    
    # Process turns and generate embeddings
    turns_data = []
    utterances = [turn['utterance'] for turn in dialogue['turns']]
    turn_embeddings = generate_embeddings(utterances, model, batch_size=32)
    
    for turn, embedding in zip(dialogue['turns'], turn_embeddings):
        turns_data.append({
            'dialogue_id': dialogue_id,
            'turn_number': turn['turn_number'],
            'utterance': turn['utterance'],
            'intent': turn['intent'],
            'assistant_response': turn['assistant_response'],
            'embedding': embedding.tolist()
        })
    
    # Structure dialogue data with additional columns
    dialogue_data = {
        'dialogue_id': dialogue_id,
        'scenario_category': dialogue['scenario_category'],
        'generated_scenario': dialogue['generated_scenario'],
        'resolution_status': dialogue['resolution_status'],
        'num_lines': dialogue['num_lines'],
        'time_slot': json.dumps({
            'start_hour': dialogue['time_slot'][0],
            'end_hour': dialogue['time_slot'][1],
            'period': dialogue['time_slot'][2]
        }) if 'time_slot' in dialogue else None,
        'regions': dialogue.get('regions', []),
        'user_emotions': dialogue.get('user_emotions', []),
        'assistant_emotions': dialogue.get('assistant_emotions', []),
        'embedding': dialogue_embedding.tolist()
    }
    
    # Structure services data
    services_data = [{'dialogue_id': dialogue_id, 'service': service} for service in dialogue.get('services', [])]
    
    return dialogue_data, turns_data, services_data

def batch_insert(conn, table_name, columns, data):
    """Insert data in batches with retries"""
    if not data:
        return
        
    for attempt in range(MAX_RETRIES):
        try:
            with conn.cursor() as cur:
                placeholders = ','.join(['%s'] * len(columns))
                columns_formatted = ', '.join(columns)
                sql = f"INSERT INTO {table_name} ({columns_formatted}) VALUES %s ON CONFLICT DO NOTHING"
                values = [tuple(row[col] for col in columns) for row in data]
                execute_values(cur, sql, values)
            conn.commit()
            return
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            logging.warning(f"Retry {attempt + 1} for table {table_name} failed: {e}")
            time.sleep(1)
            conn.rollback()

def process_large_dataset(data, model, conn):
    """Process large datasets with better progress tracking and error handling"""
    total_dialogues = len(data)
    processed_count = 0
    failed_dialogues = []
    
    print(f"🚀 Starting to process {total_dialogues} dialogues...")
    
    # Process in smaller chunks
    for chunk_start in range(0, total_dialogues, BATCH_SIZE):
        chunk_end = min(chunk_start + BATCH_SIZE, total_dialogues)
        chunk = data[chunk_start:chunk_end]
        
        dialogues_data = []
        turns_data = []
        services_data = []
        
        # Process each dialogue in the chunk
        for dialogue in tqdm(chunk, desc=f"Processing dialogues {chunk_start+1}-{chunk_end}"):
            try:
                dialogue_info, turns_info, services_info = process_dialogue(dialogue, model)
                dialogues_data.append(dialogue_info)
                turns_data.extend(turns_info)
                services_data.extend(services_info)
            except Exception as e:
                failed_dialogues.append((dialogue.get('dialogue_id', 'Unknown'), str(e)))
                logging.error(f"Failed to process dialogue {dialogue.get('dialogue_id', 'Unknown')}: {e}")
                continue
        
        try:
            # Insert dialogues
            batch_insert(conn, 'dialogues',
                        ['dialogue_id', 'scenario_category', 'generated_scenario', 'resolution_status', 
                         'num_lines', 'time_slot', 'regions', 'user_emotions', 'assistant_emotions', 'embedding'],
                        dialogues_data)
            
            # Insert turns
            batch_insert(conn, 'dialogue_turns',
                        ['dialogue_id', 'turn_number', 'utterance', 'intent', 
                         'assistant_response', 'embedding'],
                        turns_data)
            
            # Insert services
            batch_insert(conn, 'dialogue_services',
                        ['dialogue_id', 'service'],
                        services_data)
            
        except Exception as e:
            logging.error(f"Failed to insert batch {chunk_start}-{chunk_end}: {e}")
            failed_dialogues.extend([(d['dialogue_id'], str(e)) for d in dialogues_data])
            continue
        
        processed_count += len(chunk)
        print(f"✅ Processed {processed_count}/{total_dialogues} dialogues")
        
        # Clear memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return processed_count, failed_dialogues

def main():
    print("🚀 Starting data import process...")
    
    # Load dataset
    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} dialogues from dataset")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        logging.error(f"Failed to load dataset: {e}")
        return

    # Initialize model
    try:
        model = SentenceTransformer(TRANSFORMER_MODEL)
        print("✅ Initialized SentenceTransformer model")
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        logging.error(f"Failed to initialize model: {e}")
        return

    # Connect to database
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        register_vector(conn)
        print("✅ Connected to database")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        logging.error(f"Failed to connect to database: {e}")
        return

    try:
        # Create tables
        create_tables(conn)
        
        # Process the dataset
        processed_count, failed_dialogues = process_large_dataset(data, model, conn)
        
        # Report results
        print("\n=== Import Summary ===")
        print(f"Total dialogues: {len(data)}")
        print(f"Successfully processed: {processed_count}")
        print(f"Failed dialogues: {len(failed_dialogues)}")
        
        if failed_dialogues:
            print("\nFailed dialogues:")
            for dialogue_id, error in failed_dialogues:
                print(f"- {dialogue_id}: {error}")
            
            # Log failed dialogues
            with open('failed_dialogues.log', 'w', encoding='utf-8') as f:
                json.dump(failed_dialogues, f, indent=2)
        
        print("\n✅ Data import completed")
        logging.info("Data import completed successfully")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Import failed: {e}")
        logging.error(f"Import failed: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    main()
