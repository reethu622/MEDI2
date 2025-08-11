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

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__, static_folder="static")
CORS(app)

ABUSIVE_WORDS = ["idiot", "stupid", "dumb", "hate", "shut up", "fool", "damn", "bastard", "crap"]

def contains_abuse(text):
    text = text.lower()
    return any(word in text for word in ABUSIVE_WORDS)

def google_search(query, num_results=5):
    if not GOOGLE_SEARCH_KEY or not GOOGLE_SEARCH_CX:
        return []
    params = {
        "key": GOOGLE_SEARCH_KEY,
        "cx": GOOGLE_SEARCH_CX,
        "q": query,
        "num": num_results,
    }
    try:
        r = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Google Search API error: {e}")
        return []

    results = []
    for item in data.get("items", []):
        link = item.get("link", "")
        # Optionally verify link is reachable (skip if slow)
        results.append({
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": link
        })
    return results

def extract_last_medical_topic(messages):
    if not GEMINI_API_KEY:
        return None
    try:
        convo_text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
        prompt = (
            "Identify the most recent medical topic, disease, symptom, or treatment mentioned by the user. "
            "Reply with only that term.\n\n" + convo_text
        )
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        print(f"Gemini topic extraction error: {e}")
        return None

def rewrite_query(query, last_topic):
    if not last_topic:
        return query
    pronouns = ["it", "those", "these", "that", "them",
                "what about that", "and that", "more about it",
                "about that", "tell me about it"]
    pattern = re.compile(r"\b(" + "|".join(re.escape(p) for p in pronouns) + r")\b", flags=re.IGNORECASE)
    return pattern.sub(last_topic, query)

def generate_answer(messages, search_results):
    sources_text = ""
    for idx, res in enumerate(search_results, start=1):
        sources_text += f"[{idx}] {res['title']}\n{res['snippet']}\nLink: {res['link']}\n\n"

    system_prompt = (
        "You are Medibot, a medical assistant. Use ONLY the provided sources to answer the user's question clearly and briefly. "
        "Cite sources like [1], [2], etc. If information is not in the sources, provide a brief, medically accurate explanation from your knowledge. "
        "Avoid disclaimers unless safety-critical.\n\n"
        f"Sources:\n{sources_text}\n"
    )

    try:
        convo_text = system_prompt + "Conversation:\n"
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            convo_text += f"{role}: {msg['content']}\n"
        convo_text += "Assistant:"

        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(convo_text)
        return resp.text.strip()
    except Exception as e:
        return f"Error generating answer: {e}"

@app.route("/api/v1/search_answer", methods=["POST"])
def search_answer():
    data = request.get_json()
    messages = data.get("messages")
    if not messages or not isinstance(messages, list):
        return jsonify({"answer": "Please send a list of conversation messages.", "sources": []})

    latest_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), None)
    if not latest_user_msg:
        return jsonify({"answer": "No user message found.", "sources": []})

    # Greet first user message
    if len(messages) == 1 and messages[0]["role"] == "user":
        return jsonify({"answer": "Hello! 👋 I'm Medibot, your Medi Assistant. How may I help you today?", "sources": []})

    # Abusive filter
    if contains_abuse(latest_user_msg):
        return jsonify({"answer": "Please keep the conversation respectful. I'm here to help with medical questions.", "sources": []})

    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if latest_user_msg.lower() in greetings:
        return jsonify({"answer": "Hello! 👋 I'm Medibot, your Medi Assistant. How may I help you today?", "sources": []})

    thanks = ["thanks", "thank you", "thx", "ty"]
    if latest_user_msg.lower() in thanks:
        return jsonify({"answer": "You're welcome! 😊 Happy to help.", "sources": []})

    last_topic = extract_last_medical_topic(messages)
    search_query = rewrite_query(latest_user_msg, last_topic)

    search_results = google_search(search_query, num_results=5)

    answer = generate_answer(messages, search_results)

    return jsonify({"answer": answer, "sources": search_results})

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "medibot.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7000))
    app.run(host="0.0.0.0", port=port)


















