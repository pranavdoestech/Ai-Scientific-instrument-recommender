from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

instruments = [
    {
        "name": "Scanning Electron Microscope (SEM)",
        "description": "High-resolution surface imaging of solid samples",
        "reason": "Best for surface morphology and nano-scale imaging"
    },
    {
        "name": "X-Ray Diffraction (XRD)",
        "description": "Crystal structure analysis of materials",
        "reason": "Ideal for phase identification and crystallography"
    },
    {
        "name": "Gas Chromatography (GC)",
        "description": "Analysis of volatile chemical compounds",
        "reason": "Used for separating and analyzing chemical mixtures"
    }
]

embeddings = model.encode([inst['description'] for inst in instruments])

def recommend(query):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, embeddings)[0]
    idx = np.argmax(scores)

    result = instruments[idx]

    return {
        "instrument": result["name"],
        "confidence": float(scores[idx]),
        "reasoning": result["reason"],
        "alternative": instruments[(idx+1) % len(instruments)]["name"]
    }

if __name__ == "__main__":
    query = input("Enter research objective: ")
    res = recommend(query)

    print("\nRecommended Instrument:", res["instrument"])
    print("Reason:", res["reasoning"])
    print("Alternative Option:", res["alternative"])
    print("Confidence Score:", res["confidence"])
