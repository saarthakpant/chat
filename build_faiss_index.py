import numpy as np
import faiss

EMBEDDINGS_PATH = "dialogue_embeddings.npy"
FAISS_INDEX_PATH = "faiss_index.index"

def build_faiss_index():
    # Load embeddings
    embeddings = np.load(EMBEDDINGS_PATH).astype('float32')  # Faiss requires float32

    # Build the index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save the index
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("✅ Faiss index built and saved.")

if __name__ == "__main__":
    build_faiss_index()
