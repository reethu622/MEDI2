import os
import requests
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="static")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# ✅ Trusted sources including Wikipedia
trusted_sites = [
    "site:mayoclinic.org",
    "site:webmd.com",
    "site:nih.gov",
    "site:who.int",
    "site:health.harvard.edu",
    "site:cdc.gov",
    "site:clevelandclinic.org",
    "site:medlineplus.gov",
    "site:wikipedia.org"
]

def google_search(query):
    """Searches Google Custom Search restricted to trusted sources."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def get_gemini_answer(question, context):
    """Calls Gemini API to generate answer based on context."""
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    payload = {
        "contents": [{
            "parts": [{"text": f"Answer the following medical question:\n\nQuestion: {question}\n\nSources:\n{context}"}]
        }]
    }
    resp = requests.post(url, headers=headers, params=params, json=payload)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return "Sorry, I couldn't generate an answer."

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "medibot.html")

@app.route("/api/v1/search_answer", methods=["POST"])
def search_answer():
    data = request.json
    if not data or "messages" not in data:
        return jsonify({"error": "Invalid request"}), 400

    user_question = ""
    for msg in data["messages"]:
        if msg["role"] == "user":
            user_question = msg["content"]

    if not user_question:
        return jsonify({"error": "No question found"}), 400

    # Build query restricted to trusted sites
    query = f"{user_question} {' OR '.join(trusted_sites)}"

    try:
        # Google Custom Search
        search_results = google_search(query)
        items = search_results.get("items", [])
        sources = [{"title": i["title"], "link": i["link"]} for i in items[:5]]
        context_text = "\n".join([f"{i['title']}: {i['link']}" for i in sources])

        # Gemini AI Answer
        answer = get_gemini_answer(user_question, context_text)

        return jsonify({"answer": answer, "sources": sources})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


