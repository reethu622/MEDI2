import os
import requests
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Load API keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GOOGLE_SEARCH_KEY = os.getenv("GOOGLE_SEARCH_KEY", "")
GOOGLE_SEARCH_CX = os.getenv("GOOGLE_SEARCH_CX", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__, static_folder="static")
CORS(app)

# Simple offline fallback FAQ
MEDICAL_FAQ = {
    "fever symptoms": "Common symptoms include high temperature, sweating, chills, headache, and muscle aches.",
    "cold symptoms": "Sneezing, runny or stuffy nose, sore throat, coughing, mild headache, and fatigue.",
    "covid symptoms": "Fever, dry cough, tiredness, loss of taste or smell, shortness of breath."
}

def google_search_with_citations(query):
    if not GOOGLE_SEARCH_KEY or not GOOGLE_SEARCH_CX:
        return [], ""

    params = {
        "key": GOOGLE_SEARCH_KEY,
        "cx": GOOGLE_SEARCH_CX,
        "q": query,
        "num": 5
    }
    try:
        r = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Google Search API error: {e}")
        return [], ""

    results = []
    formatted_results = ""
    for i, item in enumerate(data.get("items", []), start=1):
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

    latest_user_message = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            latest_user_message = msg.get("content", "").strip()
            break
    if not latest_user_message:
        return jsonify({"answer": "No user message found in conversation.", "sources": []})

    results, formatted_results = google_search_with_citations(latest_user_message)

    system_prompt = (
        "You are a helpful and knowledgeable medical assistant. "
        "Answer the user's questions based on the following web search results. "
        "Cite your sources with numbers like [1], [2], etc.\n\n"
        f"{formatted_results}\n"
    )

    conversation_text = system_prompt + "\nConversation:\n"
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_text += f"{role}: {msg['content']}\n"
    conversation_text += "Assistant:"

    if GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_text(conversation_text)
            answer = resp.text
            return jsonify({"answer": answer, "sources": results})
        except Exception as e:
            return jsonify({"answer": f"Gemini error: {e}", "sources": []})

    # Offline FAQ fallback
    for key, answer in MEDICAL_FAQ.items():
        if key in latest_user_message.lower():
            return jsonify({"answer": answer, "sources": []})

    return jsonify({"answer": "I don't know. Please consult a medical professional.", "sources": []})

# Add this route to serve your frontend at /
@app.route("/")
def index():
    return send_from_directory("static", "medibot.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7000))
    app.run(host="0.0.0.0", port=port, debug=True)




