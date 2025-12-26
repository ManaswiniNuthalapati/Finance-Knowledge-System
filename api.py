from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import requests
import os
from groq import Groq

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------- Wikipedia Internet Fetch --------
def fetch_wikipedia(topic):
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
        res = requests.get(url, timeout=10)

        if res.status_code == 200:
            data = res.json()
            return data.get("extract", "No Wikipedia data available.")
        return "Wikipedia information not available."
    except:
        return "Internet unavailable or blocked."


# -------- Groq AI Generator --------
def ai_generate(topic, wiki_text):

    prompt = f"""
You are an expert financial teacher.
Explain the topic: {topic}

Return output ONLY IN CLEAN HTML FORMAT.
No markdown. No ** symbols. No stars.

Structure exactly like this:

<h2>Understanding {topic}</h2>

<section>
<h3>1️⃣ What it is</h3>
<p>Clear beginner explanation</p>
</section>

<section>
<h3>2️⃣ Why it is Important</h3>
<ul>
<li>Point 1</li>
<li>Point 2</li>
<li>Point 3</li>
</ul>
</section>

<section>
<h3>3️⃣ Where it is Used</h3>
<ul>
<li>Use case 1</li>
<li>Use case 2</li>
<li>Use case 3</li>
</ul>
</section>

<section>
<h3>4️⃣ How to Start (Step by Step)</h3>
<ol>
<li>Step 1</li>
<li>Step 2</li>
<li>Step 3</li>
</ol>
</section>

<section>
<h3>5️⃣ Key Benefits</h3>
<ul>
<li>Benefit 1</li>
<li>Benefit 2</li>
</ul>
</section>

<section>
<h3>6️⃣ Risks & Mistakes to Avoid</h3>
<ul>
<li>Risk 1</li>
<li>Risk 2</li>
</ul>
</section>

<section>
<h3>7️⃣ Real World Examples</h3>
<ul>
<li>Example 1</li>
<li>Example 2</li>
</ul>
</section>

<section>
<h3>8️⃣ Beginner Roadmap</h3>
<ol>
<li>Step</li>
<li>Step</li>
</ol>
</section>

<section>
<h3>9️⃣ Key Terms Explained</h3>
<ul>
<li>Term - Meaning</li>
</ul>
</section>

<section>
<h3>🔟 Short Conclusion</h3>
<p>Short and clear closing</p>
</section>

Also use this Wikipedia information as reference but rewrite simply:
{wiki_text}

Do not write anything outside HTML.
"""


    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


@app.get("/")
def home():
    return {"message": "Financial Knowledge Groq AI Running 🚀"}


@app.get("/learn")
def learn(topic: str):

    topic = topic.lower()

    wiki_summary = fetch_wikipedia(topic)
    ai_response = ai_generate(topic, wiki_summary)

    topic_links = {
        "stocks": {
            "articles": [
                "https://www.investopedia.com/terms/s/stock.asp",
                "https://groww.in/p/stock-market",
                "https://www.nerdwallet.com/article/investing/stocks",
                "https://www.moneycontrol.com/stocksmarketsindia/"
            ],
            "videos": [
                "https://www.youtube.com/watch?v=p7HKvqRI_Bo",
                "https://www.youtube.com/watch?v=ywbv_6RZQY8",
                "https://www.youtube.com/watch?v=ZCFkWDdmXG8"
            ]
        },

        "crypto": {
            "articles": [
                "https://www.investopedia.com/cryptocurrency-4689743",
                "https://coinmarketcap.com/alexandria/",
                "https://www.binance.com/en/learn",
            ],
            "videos": [
                "https://www.youtube.com/watch?v=SSo_EIwHSd4",
                "https://www.youtube.com/watch?v=bBC-nXj3Ng4"
            ]
        },

        "banking": {
            "articles": [
                "https://www.investopedia.com/terms/b/banking.asp",
                "https://www.rbi.org.in/",
                "https://www.worldbank.org/en/home"
            ],
            "videos": [
                "https://www.youtube.com/watch?v=B9c2tgR_iqI",
                "https://www.youtube.com/watch?v=s8MCNjYgS3Q"
            ]
        },

        "mutual funds": {
            "articles": [
                "https://www.investopedia.com/terms/m/mutualfund.asp",
                "https://groww.in/mutual-funds",
                "https://www.morningstar.com/"
            ],
            "videos": [
                "https://www.youtube.com/watch?v=3UFkZo9B5OU",
                "https://www.youtube.com/watch?v=z3OWJxsNU5g"
            ]
        }
    }

    selected_links = topic_links.get(topic, {
        "articles": [
            f"https://www.investopedia.com/search?q={topic}",
            f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        ],
        "videos":[
            f"https://www.youtube.com/results?search_query={topic}+explained"
        ]
    })

    return {
        "topic": topic,
        "internet_summary": wiki_summary,
        "ai_detailed_explanation": ai_response,
        "articles": selected_links["articles"],
        "videos": selected_links["videos"]
    }
