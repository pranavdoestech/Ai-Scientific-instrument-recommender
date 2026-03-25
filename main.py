# AI Research Assistant (Offline LLM Simulation)
# This is a beginner-to-intermediate level project you can upload to GitHub
# It simulates a scientific instrument recommendation system using simple logic + embeddings

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample database of instruments
instruments = [
    {
        "name": "Scanning Electron Microscope (SEM)",
        "description": "Used for high-resolution surface imaging of solid samples"
    },
    {
        "name": "X-Ray Diffraction (XRD)",
        "description": "Used for crystal structure analysis of materials"
    },
    {
        "name": "Gas Chromatography (GC)",
        "description": "Used for separating and analyzing volatile compounds"
    }
]

# Encode descriptions
instrument_embeddings = model.encode([inst['description'] for inst in instruments])

# Function to recommend instrument
def recommend_instrument(query):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, instrument_embeddings)[0]
    best_idx = np.argmax(similarities)

    return {
        "recommended_instrument": instruments[best_idx]['name'],
        "confidence_score": float(similarities[best_idx])
    }

# Example usage
if __name__ == "__main__":
    user_query = input("Enter your research goal: ")
    result = recommend_instrument(user_query)

    print("\nRecommended Instrument:", result['recommended_instrument'])
    print("Confidence Score:", result['confidence_score'])

# NEXT STEPS (mention this in README):
# - Add more instruments
# - Add reasoning explanation
# - Convert into API using Flask/FastAPI
# - Add offline LLM (like LLaMA via Ollama)
