import os
import re
import requests
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GOOGLE_SEARCH_KEY = os.getenv("GOOGLE_SEARCH_KEY", "")
GOOGLE_SEARCH_CX_RESTRICTED = os.getenv("GOOGLE_SEARCH_CX_RESTRICTED", "")
GOOGLE_SEARCH_CX_BROAD = os.getenv("GOOGLE_SEARCH_CX_BROAD", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)
CORS(app)

ABUSIVE_WORDS = ["idiot", "stupid", "dumb", "hate", "shut up", "fool", "damn", "bastard", "crap"]

def contains_abuse(text):
    text = text.lower()
    return any(word in text for word in ABUSIVE_WORDS)

def google_search(query, cx_key, num_results=5):
    if not GOOGLE_SEARCH_KEY or not cx_key:
        return []

    params = {
        "key": GOOGLE_SEARCH_KEY,
        "cx": cx_key,
        "q": query,
        "num": num_results,
    }
    try:
        r = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", "")
            })
        return results
    except Exception as e:
        print(f"Google search error: {e}")
        return []

def extract_main_topic(text):
    """Extract main medical term or keyword from user query to handle 'types of it' kind of queries"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    stopwords = {"what", "is", "the", "of", "in", "and", "about", "tell", "me", "a", "an", "are", "types", "type"}
    words = [w for w in text.split() if w not in stopwords]
    if words:
        return words[-1]
    return None

last_topic = None

def generate_answer_with_gemini(messages, results):
    if not GEMINI_API_KEY:
        # If no Gemini, fallback to simple snippet concat answer
        if not results:
            return "I couldn't find relevant information in the sources. Please consult a healthcare professional."
        snippets = []
        for idx, item in enumerate(results, start=1):
            snippets.append(f"{item['snippet']} [{idx}]")
        return " ".join(snippets)

    # Format sources for Gemini prompt
    formatted_sources = ""
    for idx, item in enumerate(results, start=1):
        formatted_sources += f"[{idx}] {item['title']}\n{item['snippet']}\nSource: {item['link']}\n\n"

    system_prompt = (
        "You are a helpful and knowledgeable medical assistant chatbot. "
        "Answer the user's questions using ONLY the following search results as your sources. "
        "If the answer is unclear or incomplete, say you do not know and advise consulting a healthcare professional. "
        "Cite your sources in brackets like [1], [2], etc.\n\n"
        f"{formatted_sources}"
    )

    conversation_text = system_prompt + "\nConversation:\n"
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_text += f"{role}: {msg['content']}\n"
    conversation_text += "Assistant:"

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(conversation_text)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        # Fallback simple answer if Gemini fails
        if not results:
            return "I couldn't find relevant information and had an internal error. Please consult a healthcare professional."
        snippets = []
        for idx, item in enumerate(results, start=1):
            snippets.append(f"{item['snippet']} [{idx}]")
        return " ".join(snippets)

@app.route("/api/v1/search_answer", methods=["POST"])
def search_answer():
    global last_topic

    data = request.get_json()
    messages = data.get("messages", [])

    if not messages or not isinstance(messages, list):
        return jsonify({"answer": "Please send conversation messages in the correct format.", "sources": []})

    # Get latest user message content
    latest_user_msg = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            latest_user_msg = msg.get("content", "").strip()
            break

    if not latest_user_msg:
        return jsonify({"answer": "No user message found.", "sources": []})

    # Handle abusive words
    if contains_abuse(latest_user_msg):
        return jsonify({"answer": "Please keep the conversation respectful.", "sources": []})

    # Greetings
    if latest_user_msg.lower() in ["hi", "hello", "hey"]:
        return jsonify({
            "answer": "Hello! 👋 I'm Medibot, your Medi Assistant. How may I help you today?",
            "sources": []
        })

    # Thanks
    if latest_user_msg.lower() in ["thanks", "thank you", "thx", "ty"]:
        return jsonify({
            "answer": "You're welcome! 😊 Happy to help.",
            "sources": []
        })

    # Detect if user is asking about vague "types of it" or similar
    vague_type_phrases = ["types of it", "type of it", "what are the types", "types of that", "types"]
    is_vague_type_query = any(phrase in latest_user_msg.lower() for phrase in vague_type_phrases)

    if is_vague_type_query and last_topic:
        search_query = f"types of {last_topic}"
    else:
        search_query = latest_user_msg
        main_topic = extract_main_topic(latest_user_msg)
        if main_topic:
            last_topic = main_topic

    # Try restricted Google search first
    results = google_search(search_query, GOOGLE_SEARCH_CX_RESTRICTED, num_results=5)

    # If no results, fallback to broad search
    if not results:
        results = google_search(search_query, GOOGLE_SEARCH_CX_BROAD, num_results=5)

    answer = generate_answer_with_gemini(messages, results)

    # Prepare clickable sources for frontend
    sources = []
    for idx, item in enumerate(results, start=1):
        # Simple HTML links — your frontend should render these safely!
        sources.append(f"[{idx}] <a href='{item['link']}' target='_blank' rel='noopener noreferrer'>{item['title']}</a>")

    return jsonify({"answer": answer, "sources": sources})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7000))
    app.run(host="0.0.0.0", port=port)
