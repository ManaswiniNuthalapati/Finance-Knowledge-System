import os
import requests
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

load_dotenv()

# ------------ Load Dataset ------------
data = pd.read_csv("financial_data.csv")
data["text"] = data["title"] + " " + data["content"]

# ------------ Embeddings ------------
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(data["text"])

# ------------ AI Summarizer ------------
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)


# ------------ Internet Summary (Wikipedia API) ------------
def web_summary(query):
    try:
        topic = query.replace(" ", "%20")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"

        headers = {
            "User-Agent": "Financial-Knowledge-App/1.0 (student-project)"
        }

        res = requests.get(url, headers=headers, timeout=10)

        if res.status_code == 200:
            data = res.json()

            if "extract" in data and data["extract"]:
                text = data["extract"]

                if len(text) > 400:
                    summary = summarizer(
                        text,
                        max_length=150,
                        min_length=70,
                        do_sample=False
                    )
                    return summary[0]["summary_text"]

                return text

        return "This is an important financial topic. Explore the reference links below to learn more."

    except Exception as e:
        print("Internet Error:", e)
        return "Internet explanation unavailable due to network or API limits."


# ------------ Recommendation Logic ------------
def recommend(query):
    query_lower = query.lower()

    user_vector = model.encode([query_lower])
    similarity_scores = cosine_similarity(user_vector, vectors).flatten()

    results = []
    for index, score in enumerate(similarity_scores):
        category = data.iloc[index]["category"]

        if category in query_lower:
            score += 0.5

        results.append((index, score))

    results = sorted(results, key=lambda x: x[1], reverse=True)[:3]

    print("\n===== AI Financial Recommendation System =====\n")

    for index, score in results:
        print(f"Title: {data.iloc[index]['title']}")
        print(f"Category: {data.iloc[index]['category']}")
        print(f"Match Score: {round(float(score),2)}")
        print(f"Preview: {data.iloc[index]['content'][:120]}...\n")

        print("📌 Useful Knowledge:")
        print(web_summary(query))
        print()

        print("📚 References:")
        print(f"https://www.investopedia.com/search?q={query}")
        print(f"https://en.wikipedia.org/wiki/{query.replace(' ','_')}")
        print("https://www.rbi.org.in/")
        print("https://www.nseindia.com/\n")

        print("🎥 Learn with Videos:")
        print(f"https://www.youtube.com/results?search_query=introduction+to+{query}")
        print(f"https://www.youtube.com/results?search_query={query}+explained")
        print(f"https://www.youtube.com/results?search_query={query}+course")
        print("\n---------------------------------------------\n")


# ------------ Run Program ------------
user_input = input("\nEnter your interest (stocks / crypto / banking / mutual funds): ")
recommend(user_input)
