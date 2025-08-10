import os
from flask import Flask, request, jsonify, send_from_directory
from better_profanity import profanity
from flask_cors import CORS
import requests
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# Load API keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_SEARCH_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_SEARCH_CX")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")
if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    raise RuntimeError("Google API key and CSE ID must be set")

# Configure Google Gemini (Generative AI)
genai.configure(api_key=GEMINI_API_KEY)

# Load profanity filter
profanity.load_censor_words()

def google_search(query, num_results=3):
    """Perform Google Custom Search and return list of results with title/link."""
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
            results.append({
                "title": item.get("title"),
                "link": item.get("link")
            })
    return results

def generate_gemini_response(prompt):
    """Use Google Gemini to generate AI response."""
    try:
        response = genai.generate_text(
            model="gemini-2.0-flash",
            prompt=prompt,
            temperature=0.7,
            max_output_tokens=300
        )
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "Sorry, I am having trouble generating a response right now."

@app.route("/")
def index():
    return send_from_directory("static", "medibot.html")  # your frontend file

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

    # Keywords for AI explanations
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





