import os
import requests
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("medibot")

app = Flask(__name__, static_folder="static")
CORS(app)

# Environment variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")  # restricted medical CSE
GOOGLE_CSE_ID_PUBLIC = os.environ.get("GOOGLE_CSE_ID_PUBLIC")  # fallback public CSE (optional)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GOOGLE_API_KEY or not GOOGLE_CSE_ID or not GEMINI_API_KEY:
    log.warning("Missing GOOGLE_API_KEY, GOOGLE_CSE_ID, or GEMINI_API_KEY. Expect errors.")

# Trusted medical sites for Google Custom Search filtering
TRUSTED_SITES = [
    "site:mayoclinic.org",
    "site:webmd.com",
    "site:nih.gov",
    "site:who.int",
    "site:cdc.gov",
    "site:clevelandclinic.org",
    "site:medlineplus.gov",
    "site:health.harvard.edu",
    "site:wikipedia.org"
]

conversation_history: List[Dict[str, str]] = []

def google_search(query: str, cx_id: str, num: int = 5) -> List[Dict]:
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": cx_id,
        "q": query,
        "num": num
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("items", [])
    except Exception:
        log.exception("Google search failed")
        return []

def build_search_query(user_text: str) -> str:
    site_filter = " OR ".join(TRUSTED_SITES)
    return f"{user_text} {site_filter}"

def call_gemini(prompt_text: str) -> str:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    body = {
        "prompt": {
            "text": prompt_text
        },
        "temperature": 0.2,
        "maxOutputTokens": 800
    }
    try:
        resp = requests.post(url, headers=headers, params=params, json=body, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        # Adjust this based on actual response structure
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        log.exception("Gemini API call failed")
        return ""

SYSTEM_PROMPT = """You are Medibot, a professional and cautious medical assistant.
Rules:
- ONLY provide medical/health info using the given search results.
- Cite sources inline as [1], [2].
- If info not found in results, say so and advise consulting a professional.
- Refuse abusive, illegal, or unsafe queries politely.
- Keep answers concise and factual."""

def build_prompt(user_question: str, search_items: List[Dict], history: List[Dict]) -> str:
    snippets = []
    for i, item in enumerate(search_items[:6]):
        title = item.get("title", "No title")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        snippets.append(f"[{i+1}] {title}\n{snippet}\n{link}")

    search_text = "\n\n".join(snippets) if snippets else "No search results available."

    history_text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in history[-10:]) or "(no context)"

    prompt = (
        SYSTEM_PROMPT + "\n\n"
        "Conversation history:\n" + history_text + "\n\n"
        "Search results:\n" + search_text + "\n\n"
        f"User question: {user_question}\n\n"
        "Answer ONLY using the above search results. Cite sources inline."
    )
    return prompt

def is_greeting(text: str) -> bool:
    greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
    return text.lower().strip() in greetings

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "medibot.html")

@app.route("/api/v1/search_answer", methods=["POST"])
def search_answer():
    global conversation_history

    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"answer": "Invalid JSON.", "sources": []}), 400

    messages = data.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return jsonify({"answer": "Please send conversation messages (list).", "sources": []}), 400

    latest_user = None
    for msg in reversed(messages):
        if msg.get("role") == "user" and msg.get("content"):
            latest_user = msg.get("content").strip()
            break

    if not latest_user:
        return jsonify({"answer": "No user question found.", "sources": []}), 400

    # If greeting detected, reply with greeting and do not search
    if is_greeting(latest_user):
        greeting_reply = "Hello! 👋 I'm Medibot, your Medi Assistant. How may I help you today?"
        conversation_history.append({"role": "user", "content": latest_user})
        conversation_history.append({"role": "assistant", "content": greeting_reply})
        return jsonify({"answer": greeting_reply, "sources": []})

    # Append user message to history
    conversation_history.append({"role": "user", "content": latest_user})

    query_restricted = build_search_query(latest_user)

    items = []
    if GOOGLE_API_KEY and GOOGLE_CSE_ID:
        items = google_search(query_restricted, GOOGLE_CSE_ID)

    # Fallback to public CSE if no results
    if (not items) and GOOGLE_CSE_ID_PUBLIC:
        items = google_search(latest_user, GOOGLE_CSE_ID_PUBLIC)

    if not items:
        fallback_msg = "Sorry, I couldn't find reliable sources for that question. Please try rephrasing or consult a healthcare professional."
        conversation_history.append({"role": "assistant", "content": fallback_msg})
        return jsonify({"answer": fallback_msg, "sources": []})

    sources = [{"title": it.get("title", "")[:200], "link": it.get("link", "")} for it in items[:6]]

    prompt = build_prompt(latest_user, items, conversation_history)
    gemini_answer = call_gemini(prompt)

    if not gemini_answer:
        snippets = [it.get("snippet", "") for it in items[:3] if it.get("snippet")]
        fallback_answer = " ".join(snippets) or "Sorry, I couldn't find an answer."
        conversation_history.append({"role": "assistant", "content": fallback_answer})
        return jsonify({"answer": fallback_answer, "sources": sources})

    conversation_history.append({"role": "assistant", "content": gemini_answer})

    return jsonify({"answer": gemini_answer, "sources": sources})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)




