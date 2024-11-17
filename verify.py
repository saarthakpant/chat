import spacy

# Load the transformer-based model
nlp = spacy.load("en_core_web_trf")

# Process a sample text
doc = nlp("This is a test sentence.")

# Access the vector
print(doc.vector)
print(doc.vector.shape)
