import os
import re
import requests
from flask import Flask, request, jsonify, send_from_directory
from google import genai

app = Flask(__name__)

# Initialize Google GenAI client
client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

# Google Custom Search API config - set these environment variables
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_CX = os.getenv("GOOGLE_SEARCH_CX")

GREETING_KEYWORDS = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
ABUSIVE_WORDS = ["idiot", "stupid", "dumb", "hate", "damn"]  # expand as needed

def is_greeting(text):
    text = text.lower()
    return any(greet in text for greet in GREETING_KEYWORDS)

def is_abusive(text):
    text = text.lower()
    return any(word in text for word in ABUSIVE_WORDS)

def resolve_reference(messages):
    last_msg = messages[-1]["content"].lower()
    if any(phrase in last_msg for phrase in ["types of it", "type of it", "elaborate on it"]):
        # Find last medical topic (naive approach: last user message before this one)
        for msg in reversed(messages[:-1]):
            if msg["role"] == "user":
                return msg["content"]
    return None

def google_search(query, num_results=3):
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_CX:
        return []

    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_SEARCH_API_KEY,
        "cx": GOOGLE_SEARCH_CX,
        "q": query,
        "num": num_results,
    }

    try:
        resp = requests.get(search_url, params=params)
        resp.raise_for_status()
        results = resp.json().get("items", [])
        sources = []
        for item in results:
            title = item.get("title")
            link = item.get("link")
            if title and link:
                sources.append({"title": title, "link": link})
        return sources
    except Exception as e:
        print("Google Search API error:", e)
        return []

@app.route("/api/v1/search_answer", methods=["POST"])
def search_answer():
    data = request.get_json()
    messages = data.get("messages", [])

    if not messages:
        return jsonify({"answer": "Please say something!", "sources": []})

    user_message = messages[-1]["content"].strip()

    if is_abusive(user_message):
        return jsonify({
            "answer": "I’m here to help you respectfully. Please avoid using abusive language.",
            "sources": []
        })

    if is_greeting(user_message):
        return jsonify({
            "answer": "Hi again! 👋 How can I help you today?",
            "sources": []
        })

    referenced_topic = resolve_reference(messages)
    prompt_text = user_message
    if referenced_topic:
        prompt_text = f"{user_message}. Context: {referenced_topic}"

    # Call GenAI for answer
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt_text,
        )
        answer_text = getattr(response, "text", None) or "Sorry, I couldn't find an answer."
    except Exception as e:
        print("GenAI error:", e)
        answer_text = "Sorry, I encountered an error while trying to answer."

    # Use Google Custom Search for sources
    sources = google_search(prompt_text)

    return jsonify({
        "answer": answer_text,
        "sources": sources,
    })

@app.route("/")
def index():
    return send_from_directory("static", "medibot.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)






