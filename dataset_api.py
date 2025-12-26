import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

data = pd.read_csv("financial_data.csv")

@app.get("/")
def home():
    return {"message":"Dataset Based Financial Recommendation Running 🚀"}

@app.get("/recommend")
def recommend(topic:str):

    topic = topic.lower()

    result = data[data["category"] == topic]

    if result.empty:
        return {
            "status":"failed",
            "message":"Use stocks / crypto / banking / mutual funds"
        }

    row = result.iloc[0]

    return {
        "title":row["title"],
        "category":row["category"],
        "content":row["content"]
    }
