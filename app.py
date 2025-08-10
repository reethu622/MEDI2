import os
from flask import Flask, request, jsonify, render_template
import requests
import google.generativeai as genai

# Load environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_SEARCH_KEY = os.getenv("GOOGLE_SEARCH_KEY")
GOOGLE_SEARCH_CX = os.getenv("GOOGLE_SEARCH_CX")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)

# Search function using Google Custom Search API
def google_search(query, num_results=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_SEARCH_KEY,
        "cx": GOOGLE_SEARCH_CX,
        "q": query
    }
    res = requests.get(url, params=params)
    res.raise_for_status()
    data = res.json()
    results = []
    if "items" in data:
        for item in data["items"][:num_results]:
            results.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet")
            })
    return results

# Get trusted medical answer from Gemini
def get_medical_answer(question, chat_history):
    trusted_sites = [
        "site:mayoclinic.org",
        "site:webmd.com",
        "site:nih.gov",
        "site:who.int",
        "site:cdc.gov",
        "site:clevelandclinic.org"
    ]
    search_query = f"{question} {' OR '.join(trusted_sites)}"

    search_results = google_search(search_query, num_results=5)

    # Prepare context for Gemini
    sources_text = "\n".join(
        [f"[{i+1}] {r['title']} - {r['snippet']}" for i, r in enumerate(search_results)]
    )

    prompt = f"""
You are Medibot, a helpful and polite medical assistant.
If user uses abusive language, politely warn them.
Always keep conversation context from earlier messages.
When user refers to 'it' or 'those', resolve from previous history.

Question: {question}

Relevant medical info from trusted sources:
{sources_text}

Instructions:
1. Answer in clear, plain language.
2. Base your answer on the trusted sources above.
3. Cite sources as [1], [2], etc. at the end of relevant sentences.
4. If unsure, say so and recommend consulting a doctor.
"""

    # Include chat history for context
    messages = [{"role": m["role"], "parts": [m["content"]]} for m in chat_history]
    messages.append({"role": "user", "parts": [prompt]})

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(messages)

    return response.text.strip(), search_results

@app.route("/")
def index():
    return render_template("medibot.html")

@app.route("/api/v1/search_answer", methods=["POST"])
def search_answer():
    data = request.json
    messages = data.get("messages", [])
    if not messages:
        return jsonify({"answer": "No question provided.", "sources": []})

    # Last user message
    question = messages[-1]["content"]
    answer, sources = get_medical_answer(question, messages[:-1])
    return jsonify({"answer": answer, "sources": sources})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7000))
    app.run(host="0.0.0.0", port=port)
