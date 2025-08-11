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

# Basic abusive words list
ABUSIVE_WORDS = ["idiot", "stupid", "dumb", "hate", "shut up", "fool", "damn", "bastard", "crap"]

def contains_abuse(text):
    """Check for abusive words."""
    text = text.lower()
    return any(word in text for word in ABUSIVE_WORDS)

def google_search_with_citations(query, num_results=5):
    """Google Custom Search with working link verification."""
    if not GOOGLE_SEARCH_KEY or not GOOGLE_SEARCH_CX:
        return [], ""

    params = {
        "key": GOOGLE_SEARCH_KEY,
        "cx": GOOGLE_SEARCH_CX,
        "q": query,
        "num": num_results
    }
    try:
        r = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Google Search API error: {e}")
        return [], ""

    results = []
    for item in data.get("items", []):
        link = item.get("link", "")
        try:
            head_req = requests.head(link, timeout=3)
            if head_req.status_code >= 400:
                continue
        except:
            continue

        results.append({
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": link
        })

    return results, ""

def is_answer_incomplete(answer_text, user_query):
    """Detect incomplete answers."""
    answer_lower = answer_text.lower()
    if any(phrase in answer_lower for phrase in ["sorry", "don't know", "cannot find", "need more information"]):
        return True

    question_keywords = ["type", "types", "explain", "list", "what are", "different kinds", "kinds"]
    if any(word in user_query.lower() for word in question_keywords):
        if "type" not in answer_lower and "kind" not in answer_lower and "explain" not in answer_lower:
            return True

    return False

def extract_last_medical_topic_with_gemini(messages):
    """Ask Gemini to find the last medical topic from conversation."""
    if not GEMINI_API_KEY:
        return None
    try:
        conversation_text = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages
        )
        prompt = (
            "From the conversation below, identify the most recent medical topic, disease, "
            "condition, symptom, or treatment mentioned by the user. Reply with only that term, nothing else.\n\n"
            f"{conversation_text}"
        )
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        print(f"Gemini topic extraction error: {e}")
        return None

def rewrite_query(query, last_topic, pronouns=None):
    """Replace pronouns/vague phrases with last medical topic."""
    if not last_topic:
        return query
    if pronouns is None:
        pronouns = ["it", "those", "these", "that", "them",
                    "what about that", "and that", "more about it",
                    "about that", "tell me about it"]
    pattern = re.compile(r"\b(" + "|".join(re.escape(p) for p in pronouns) + r")\b", flags=re.IGNORECASE)
    return pattern.sub(last_topic, query)

def generate_answer_with_sources(messages, results):
    """Generate an answer using Gemini."""
    formatted_results_text = ""
    for idx, item in enumerate(results, start=1):
        formatted_results_text += f"[{idx}] {item['title']}\n{item['snippet']}\nSource: {item['link']}\n\n"

    system_prompt = (
        "You are a helpful and knowledgeable medical assistant chatbot. "
        "When the user uses vague pronouns, infer they mean the most recent medical topic from the conversation. "
        "Always answer based on these search results. "
        "If unsure, politely say you don't know and recommend consulting a healthcare professional. "
        "Cite sources with [1], [2], etc.\n\n"
        f"{formatted_results_text}\n"
    )

    try:
        conversation_text = system_prompt + "\nConversation:\n"
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n"
        conversation_text += "Assistant:"
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(conversation_text)
        return resp.text
    except Exception as e:
        return f"Gemini error: {e}"

@app.route("/api/v1/search_answer", methods=["POST"])
def search_answer():
    data = request.get_json()
    messages = data.get("messages")
    if not messages or not isinstance(messages, list):
        return jsonify({"answer": "Please provide conversation history as a list of messages.", "sources": []})

    latest_user_message = next((msg.get("content", "").strip() for msg in reversed(messages) if msg.get("role") == "user"), None)
    if not latest_user_message:
        return jsonify({"answer": "No user message found in conversation.", "sources": []})

    # Greet if first interaction
    if len(messages) == 1 and messages[0]["role"] == "user":
        return jsonify({
            "answer": "Hello! 👋 I'm Medibot, your Medi Assistant. How may I help you today?",
            "sources": []
        })

    # Abusive filter
    if contains_abuse(latest_user_message):
        return jsonify({
            "answer": "I am here to help with medical questions. Please keep the conversation respectful.",
            "sources": []
        })

    # Greetings
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if latest_user_message.lower() in greetings:
        return jsonify({
            "answer": "Hello! 👋 I'm Medibot, your Medi Assistant. How may I help you today?",
            "sources": []
        })

    # Thanks handling
    thanks_keywords = ["thanks", "thank you", "thx", "ty"]
    if any(latest_user_message.lower() == t for t in thanks_keywords):
        return jsonify({
            "answer": "You're welcome! 😊 Always happy to help with your medical questions.",
            "sources": []
        })

    # Last topic detection
    last_topic = extract_last_medical_topic_with_gemini(messages)
    search_query = rewrite_query(latest_user_message, last_topic)

    # Google search
    results, _ = google_search_with_citations(search_query, num_results=5)
    answer = generate_answer_with_sources(messages, results)

    # If incomplete, broader search
    if is_answer_incomplete(answer, latest_user_message):
        fallback_results, _ = google_search_with_citations(search_query, num_results=15)
        answer = generate_answer_with_sources(messages, fallback_results)
        return jsonify({"answer": answer, "sources": fallback_results})

    return jsonify({"answer": answer, "sources": results})

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "medibot.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7000))
    app.run(host="0.0.0.0", port=port)
















