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

# Configure Gemini API key
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__, static_folder="static")
CORS(app)

# Basic offline medical FAQ fallback
MEDICAL_FAQ = {
    "fever symptoms": "Common symptoms include high temperature, sweating, chills, headache, and muscle aches.",
    "cold symptoms": "Sneezing, runny or stuffy nose, sore throat, coughing, mild headache, and fatigue.",
    "covid symptoms": "Fever, dry cough, tiredness, loss of taste or smell, shortness of breath."
}

# Set of abusive or negative keywords to detect
ABUSIVE_WORDS = set([
    "stupid", "idiot", "dumb", "hate", "shut up", "useless", "kill", "hate you", "moron",
    "fool", "crap", "bastard", "suck"
])

def contains_abusive(text):
    text_lower = text.lower()
    return any(word in text_lower for word in ABUSIVE_WORDS)

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

    # Get latest user message
    latest_user_message = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            latest_user_message = msg.get("content", "").strip()
            break

    if not latest_user_message:
        return jsonify({"answer": "No user message found in conversation.", "sources": []})

    # Check for abusive/negative language
    if contains_abusive(latest_user_message):
        polite_response = "I'm here to help and want to keep our conversation respectful. How can I assist you today?"
        return jsonify({"answer": polite_response, "sources": []})

    # Detect if user wants elaboration (e.g., "elaborate", "explain more", "tell me more")
    wants_elaboration = bool(re.search(r"\b(elaborate|explain more|tell me more|expand)\b", latest_user_message.lower()))

    # Prepare conversation memory/history for the model
    # We'll pass the last 10 messages for context
    conversation_history = messages[-10:]

    # Run Google Search on latest user question unless it's just a greeting or elaboration
    do_search = True
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
    if latest_user_message.lower() in greetings or wants_elaboration:
        do_search = False

    results, formatted_results = [], ""
    if do_search:
        results, formatted_results = google_search_with_citations(latest_user_message)

    # Construct prompt for Gemini model
    system_prompt = (
        "You are a helpful, honest, and knowledgeable medical assistant. "
        "Avoid hallucinations and do not provide information unless you are confident it is accurate. "
        "If you are unsure, politely suggest consulting a healthcare professional. "
        "Answer the user's questions based on the following web search results, citing your sources with numbers like [1], [2], etc.\n\n"
        f"{formatted_results}\n"
    )

    # Build conversation text for Gemini prompt
    conversation_text = system_prompt + "Conversation:\n"
    for msg in conversation_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_text += f"{role}: {msg['content']}\n"

    # If user wants elaboration, add a hint for Gemini to expand the last answer
    if wants_elaboration:
        conversation_text += "Assistant: Please elaborate on the previous response.\n"
    else:
        conversation_text += "Assistant:"

    # Call Gemini API
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate(prompt=conversation_text)
        answer = response.text.strip()
        if not answer:
            answer = "I don't know. Please consult a medical professional."
        return jsonify({"answer": answer, "sources": results})
    except Exception as e:
        print(f"Gemini error: {e}")
        # Fallback to offline FAQ
        for key, faq_answer in MEDICAL_FAQ.items():
            if key in latest_user_message.lower():
                return jsonify({"answer": faq_answer, "sources": []})
        return jsonify({"answer": f"Gemini error: {e}", "sources": []})

@app.route("/")
def index():
    return send_from_directory("static", "medibot.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7000))
    app.run(host="0.0.0.0", port=port)





