import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import warnings
import logging
import torch
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables
load_dotenv()

# Configuration
TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'
DATASET_PATH = "dataset_1k.json"
EMBEDDINGS_PATH = "dialogue_embeddings.npy"
DIALOGUE_IDS_PATH = "dialogue_ids.json"
EMBEDDING_DIM = 384

# Logging setup
logging.basicConfig(
    filename='import_embeddings.log',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

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

def main():
    print("🚀 Starting embedding generation process...")
    
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

    dialogue_texts = []
    dialogue_ids = []

    # Prepare dialogues for embedding
    for dialogue in data:
        dialogue_id = dialogue['dialogue_id']
        # Concatenate all user utterances and assistant responses into a single text
        dialogue_text = " ".join([
            turn['utterance'] + " " + (turn.get('assistant_response', '') or '')
            for turn in dialogue['turns']
        ])
        dialogue_texts.append(dialogue_text)
        dialogue_ids.append(dialogue_id)

    # Generate embeddings
    embeddings = generate_embeddings(dialogue_texts, model, batch_size=32)

    # Save embeddings and dialogue IDs
    try:
        np.save(EMBEDDINGS_PATH, embeddings)
        with open(DIALOGUE_IDS_PATH, 'w', encoding='utf-8') as f:
            json.dump(dialogue_ids, f)
        print("✅ Saved embeddings and dialogue IDs")
    except Exception as e:
        print(f"❌ Failed to save embeddings or dialogue IDs: {e}")
        logging.error(f"Failed to save embeddings or dialogue IDs: {e}")

    print("\n✅ Embedding generation completed")
    logging.info("Embedding generation completed successfully")

if __name__ == "__main__":
    main()
