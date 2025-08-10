import os
import re
import requests
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Load API keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GOOGLE_SEARCH_KEY = os.getenv("GOOGLE_SEARCH_KEY", "")
GOOGLE_SEARCH_CX = os.getenv("GOOGLE_SEARCH_CX", "")

if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY environment variable")

genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__, static_folder="static")
CORS(app)

# List of abusive or negative words to detect
ABUSIVE_WORDS = [
    "stupid", "idiot", "hate", "dumb", "kill", "shut up", "hate you", "fool", "moron"
]

# Basic offline medical FAQ fallback
MEDICAL_FAQ = {
    "fever symptoms": "Common symptoms include high temperature, sweating, chills, headache, and muscle aches.",
    "cold symptoms": "Sneezing, runny or stuffy nose, sore throat, coughing, mild headache, and fatigue.",
    "covid symptoms": "Fever, dry cough, tiredness, loss of taste or smell, shortness of breath."
}

def contains_abusive(text):
    text_lower = text.lower()
    for word in ABUSIVE_WORDS:
        if word in text_lower:
            return True
    return False

def google_search_with_citations(query):
    if not GOOGLE_SEARCH_KEY or not GOOGLE_SEARCH_CX:
        return [], ""

    params = {
        "key": GOOGLE_SEARCH_KEY,
        "cx": GOOGLE_SEARCH_CX,
        "q": query,
        "num": 3,
    }
    try:
        response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Google Search API error: {e}")
        return [], ""

    results = []
    formatted_results = ""
    for i, item in enumerate(data.get("items", []), 1):
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        results.append({"title": title, "snippet": snippet, "link": link})
        formatted_results += f"{i}. {title}\n{snippet}\nSource: {link}\n\n"
    return results, formatted_results

@app.route("/api/v1/search_answer", methods=["POST"])
def search_answer():
    data = request.get_json()

    messages = data.get("messages")
    if not messages or not isinstance(messages, list):
        return jsonify({"answer": "Please provide conversation history as a list of messages.", "sources": []})

    # Check for abusive words in user messages
    for msg in messages:
        if msg.get("role") == "user" and contains_abusive(msg.get("content", "")):
            polite_response = (
                "I'm here to help you respectfully. "
                "Please avoid using offensive language. How can I assist you with your medical questions?"
            )
            return jsonify({"answer": polite_response, "sources": []})

    # Get latest user message
    latest_user_message = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            latest_user_message = msg.get("content", "").strip()
            break

    if not latest_user_message:
        return jsonify({"answer": "No user message found in conversation.", "sources": []})

    # Run Google Custom Search for the latest question
    results, formatted_results = google_search_with_citations(latest_user_message)

    # Prepare prompt with search results and conversation history
    prompt = (
        "You are a helpful and knowledgeable medical assistant. "
        "Use the following web search results to answer the user's question accurately and politely. "
        "Cite your sources with numbers like [1], [2], etc.\n\n"
        f"{formatted_results}\n"
        "Conversation:\n"
    )
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        prompt += f"{role}: {msg['content']}\n"
    prompt += "Assistant:"

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_text(
            prompt=prompt,
            temperature=0.3,
            max_output_tokens=512,
        )
        answer = response.text.strip()
        return jsonify({"answer": answer, "sources": results})
    except Exception as e:
        print(f"Gemini error: {e}")

    # Fallback to offline FAQ
    for key, answer in MEDICAL_FAQ.items():
        if key in latest_user_message.lower():
            return jsonify({"answer": answer, "sources": []})

    return jsonify({"answer": "I don't know. Please consult a medical professional.", "sources": []})

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "medibot.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7000))
    app.run(host="0.0.0.0", port=port)

