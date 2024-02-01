import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

if __name__ == "__main__":
    model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    embedding = model.encode("Departure date?")
    index = faiss.read_index("new_new_indices_with_mappings/prefix_index.index")
    with open("new_new_indices_with_mappings/ngram_mappings.pkl", "rb") as f:
        ngram_map = pickle.load(f)
    suffix_embs = np.load("new_new_indices_with_mappings/suffix_embeddings.npy")
    d, i = index.search(embedding.reshape(1,-1),3)
    key = i[0][0]
    print("suffix:", ngram_map[key])
    print("suffix embedding:", suffix_embs[key])
