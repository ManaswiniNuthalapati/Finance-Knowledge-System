import pandas as pd
from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

data = pd.read_csv("financial_data.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

data_vectors = model.encode(data["content"])

@app.get("/")
def home():
    return {"message":"Smart AI Financial Recommendation Running 🚀"}


@app.get("/recommend")
def recommend(query:str):

    query_vec = model.encode([query])

    scores = cosine_similarity(query_vec, data_vectors).flatten()

    best_index = scores.argmax()

    row = data.iloc[best_index]

    return {
        "query":query,
        "best_match":row["category"],
        "title":row["title"],
        "content":row["content"],
        "match_score":float(scores[best_index])
    }
