import os
from flask import Flask, request, jsonify, send_from_directory
from better_profanity import profanity
from flask_cors import CORS
import requests
from google import genai  # your import style

app = Flask(__name__)
CORS(app)

# Load profanity filter words
profanity.load_censor_words()

# Configure Google Generative AI (Gemini)
GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY") or "YOUR_GENAI_API_KEY"
client = genai.Client(api_key=GENAI_API_KEY)  # create client instance

# Google Custom Search API setup
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "YOUR_GOOGLE_API_KEY"
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID") or "YOUR_CUSTOM_SEARCH_ENGINE_ID"

def google_search(query, num_results=3):
    """Perform Google Custom Search and return title/link list"""
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": num_results,
    }
    resp = requests.get(search_url, params=params)
    results = []
    if resp.status_code == 200:
        data = resp.json()
        items = data.get("items", [])
        for item in items:
            results.append({"title": item.get("title"), "link": item.get("link")})
    return results

def generate_gemini_response(prompt):
    """Use Google Gemini (Generative AI) to generate a response."""
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print("Gemini API error:", e)
        return "Sorry, I am having trouble generating a response right now."

@app.route("/")
def index():
    return send_from_directory("static", "medibot.html")

@app.route("/api/v1/search_answer", methods=["POST"])
def search_answer():
    data = request.get_json()
    messages = data.get("messages", [])
    user_message = messages[-1]["content"] if messages else ""

    # Profanity filter
    if profanity.contains_profanity(user_message):
        return jsonify({
            "answer": "Please avoid using inappropriate language. How can I assist you politely?",
            "sources": []
        })

    # Greeting detection
    greetings = ["hi", "hello", "hey", "hola", "howdy"]
    if any(user_message.lower().startswith(greet) for greet in greetings):
        return jsonify({
            "answer": "Hi! 👋 How can I help you today?",
            "sources": []
        })

    # Keywords to choose Gemini AI for explanations
    ai_keywords = ["type", "explain", "definition", "meaning", "what is", "how does", "describe", "tell me about"]
    if any(kw in user_message.lower() for kw in ai_keywords):
        answer = generate_gemini_response(user_message)
        return jsonify({
            "answer": answer,
            "sources": []
        })

    # Otherwise, fallback to Google Search results
    sources = google_search(user_message)
    answer = "Here are some results I found related to your question."
    return jsonify({
        "answer": answer,
        "sources": sources
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

