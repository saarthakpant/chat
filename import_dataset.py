import json
import psycopg2
from psycopg2.extras import execute_values
import spacy
import os
import sys
from dotenv import load_dotenv
import warnings
import logging
from sentence_transformers import SentenceTransformer
import torch

# Suppress FutureWarning from PyTorch temporarily
warnings.filterwarnings("ignore", category=FutureWarning, module='thinc.shims.pytorch')

# =============================================================================
# Configuration Variables
# =============================================================================

# Load environment variables from .env file
load_dotenv()

# Database connection parameters
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'  # or another sentence-transformer model of your choice

# Path to your dataset
DATASET_PATH = "dataset1.json"  # Ensure your dataset1.json is in the same directory

# spaCy model
SPACY_MODEL = "en_core_web_trf"

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    filename='import_dataset.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# =============================================================================
# Function Definitions
# =============================================================================

def load_dataset(path):
    """
    Load dataset from a JSON file.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded dataset from {path}.")
        logging.info(f"Loaded dataset from {path}.")
        print(f"Dataset type: {type(data)}")
        if isinstance(data, dict):
            print(f"Dataset keys: {list(data.keys())}")
            logging.info(f"Dataset keys: {list(data.keys())}")
        elif isinstance(data, list):
            print(f"Number of dialogues: {len(data)}")
            logging.info(f"Number of dialogues: {len(data)}")
            if len(data) > 0:
                print(f"First dialogue keys: {list(data[0].keys())}")
                logging.info(f"First dialogue keys: {list(data[0].keys())}")
        return data
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        logging.error(f"Failed to load dataset: {e}")
        sys.exit(1)

def transform_data(data):
    """
    Transform dataset into a list of tuples for insertion.
    Assumes that data is a list of dialogues.
    Each dialogue contains 'dialogue_id', 'services', and 'turns'.
    Each turn contains 'turn_id', 'speaker', 'utterance', 'frames', 'dialogue_acts'.
    """
    try:
        if not isinstance(data, list):
            raise TypeError("Data is not a list.")
        
        required_dialogue_keys = ['dialogue_id', 'services', 'turns']
        required_turn_keys = ['turn_id', 'speaker', 'utterance', 'frames', 'dialogue_acts']
        
        records = []
        missing_keys_count = 0
        for dialogue in data:
            # Validate dialogue keys
            for key in required_dialogue_keys:
                if key not in dialogue:
                    logging.warning(f"Missing key in dialogue: {key}")
                    print(f"⚠️ Missing key in dialogue: {key}")
                    continue  # Skip this dialogue
            
            dialogue_id = dialogue.get('dialogue_id', 'unknown_dialogue_id')
            services = json.dumps(dialogue.get('services', []))
            
            turns = dialogue.get('turns', [])
            if not isinstance(turns, list):
                logging.warning(f"'turns' should be a list in dialogue {dialogue_id}. Skipping this dialogue.")
                print(f"⚠️ 'turns' should be a list in dialogue {dialogue_id}. Skipping this dialogue.")
                continue  # Skip this dialogue
            
            for turn in turns:
                # Validate turn keys
                missing_turn_keys = [key for key in required_turn_keys if key not in turn]
                if missing_turn_keys:
                    missing_keys_count += 1
                    logging.warning(f"Missing keys {missing_turn_keys} in turn of dialogue {dialogue_id}. Using default values.")
                    print(f"⚠️ Missing keys {missing_turn_keys} in turn of dialogue {dialogue_id}. Using default values.")
                
                turn_id_original = turn.get('turn_id', 'unknown_turn_id')
                turn_id = f"{dialogue_id}_{turn_id_original}"  # Ensure unique turn_id across dialogues
                
                # Convert speaker to integer
                speaker_str = str(turn.get('speaker', '')).upper()
                speaker = 0  # default value
                if speaker_str == 'USER':
                    speaker = 1
                elif speaker_str in ['SYSTEM', 'ASSISTANT']:
                    speaker = 0
                else:
                    try:
                        speaker = int(speaker_str)
                    except (ValueError, TypeError):
                        logging.warning(f"Invalid speaker value '{speaker_str}' in dialogue {dialogue_id}. Using default value 0.")
                        print(f"⚠️ Invalid speaker value '{speaker_str}' in dialogue {dialogue_id}. Using default value 0.")
                
                utterance = turn.get('utterance', '')
                frames = json.dumps(turn.get('frames', {}))
                dialogue_acts = json.dumps(turn.get('dialogue_acts', {}))
                
                records.append((turn_id, dialogue_id, services, speaker, utterance, frames, dialogue_acts))
        
        print(f"✅ Transformed {len(records)} records from {len(data)} dialogues.")
        logging.info(f"Transformed {len(records)} records from {len(data)} dialogues.")
        if missing_keys_count > 0:
            print(f"⚠️ Total turns with missing keys: {missing_keys_count}")
            logging.warning(f"Total turns with missing keys: {missing_keys_count}")
        return records
    except KeyError as e:
        print(f"❌ Missing expected key in dataset: {e}")
        logging.error(f"Missing expected key in dataset: {e}")
        sys.exit(1)
    except TypeError as e:
        print(f"❌ Type error: {e}")
        logging.error(f"Type error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to transform data: {e}")
        logging.error(f"Failed to transform data: {e}")
        sys.exit(1)

def generate_embeddings(utterances, model):
    """
    Generate vector embeddings for a list of utterances using sentence-transformers.
    """
    try:
        # Move model to GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        # Generate embeddings in batches
        embeddings = model.encode(
            utterances,
            batch_size=32,  # Adjust batch size based on your GPU memory
            show_progress_bar=True,
            device=device
        )
        
        print(f"✅ Generated embeddings for utterances using {device}.")
        logging.info(f"Generated embeddings for utterances using {device}.")
        return embeddings
    except Exception as e:
        print(f"❌ Failed to generate embeddings: {e}")
        logging.error(f"Failed to generate embeddings: {e}")
        sys.exit(1)

def insert_data(conn, records, embeddings):
    """
    Insert data into the conversation_turns table.
    """
    try:
        with conn.cursor() as cur:
            # Prepare the SQL statement
            sql = """
                INSERT INTO conversation_turns 
                (turn_id, dialogue_id, services, speaker, utterance, frames, dialogue_acts, embedding)
                VALUES %s
                ON CONFLICT (turn_id) DO NOTHING
            """
            # Combine records with embeddings
            data_to_insert = [
                (
                    record[0],  # turn_id
                    record[1],  # dialogue_id
                    record[2],  # services
                    record[3],  # speaker
                    record[4],  # utterance
                    record[5],  # frames
                    record[6],  # dialogue_acts
                    embeddings[i].tolist()  # embedding
                )
                for i, record in enumerate(records)
            ]
            # Execute batch insert
            execute_values(cur, sql, data_to_insert)
            conn.commit()
            print(f"✅ Inserted {len(data_to_insert)} records into the database.")
            logging.info(f"Inserted {len(data_to_insert)} records into the database.")
    except Exception as e:
        conn.rollback()
        print(f"❌ Failed to insert data: {e}")
        logging.error(f"Failed to insert data: {e}")
        sys.exit(1)

def create_table_if_not_exists(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    turn_id TEXT PRIMARY KEY,
                    dialogue_id TEXT NOT NULL,
                    services JSONB,
                    speaker INTEGER,
                    utterance TEXT,
                    frames JSONB,
                    dialogue_acts JSONB,
                    embedding FLOAT[]
                );
            """)
            conn.commit()
            print("✅ Ensured conversation_turns table exists.")
            logging.info("Ensured conversation_turns table exists.")
    except Exception as e:
        conn.rollback()
        print(f"❌ Failed to create table: {e}")
        logging.error(f"Failed to create table: {e}")
        sys.exit(1)

# =============================================================================
# Main Execution
# =============================================================================

def main():
    # Validate environment variables
    missing_vars = []
    for var in ["DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT"]:
        if not globals()[var]:
            missing_vars.append(var)
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        logging.error(f"Missing environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    # Load dataset
    data = load_dataset(DATASET_PATH)

    # Transform data
    records = transform_data(data)

    # Check if any records to process
    if not records:
        print("⚠️ No records to insert.")
        logging.warning("No records to insert.")
        sys.exit(0)

    # Extract utterances for embedding
    utterances = [record[4] for record in records]

    # Initialize transformer model
    try:
        model = SentenceTransformer(TRANSFORMER_MODEL)
        print(f"✅ Loaded Sentence Transformer model '{TRANSFORMER_MODEL}'.")
        logging.info(f"Loaded Sentence Transformer model '{TRANSFORMER_MODEL}'.")
    except Exception as e:
        print(f"❌ Failed to load Sentence Transformer model '{TRANSFORMER_MODEL}': {e}")
        logging.error(f"Failed to load Sentence Transformer model '{TRANSFORMER_MODEL}': {e}")
        sys.exit(1)

    # Generate embeddings
    embeddings = generate_embeddings(utterances, model)

    # Database operations
    conn = None
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        print("✅ Connected to the PostgreSQL database.")
        logging.info("Connected to the PostgreSQL database.")
        
        # Create table if needed
        create_table_if_not_exists(conn)
        
        # Insert data
        insert_data(conn, records, embeddings)
        
    except Exception as e:
        print(f"❌ Database operation failed: {e}")
        logging.error(f"Database operation failed: {e}")
        if conn:
            try:
                conn.rollback()
            except:
                pass
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            print("✅ Database connection closed.")
            logging.info("Database connection closed.")

if __name__ == "__main__":
    main()
