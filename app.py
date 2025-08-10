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

# Environment variables (set these in Railway or your environment)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")  # restricted medical CSE
GOOGLE_CSE_ID_PUBLIC = os.environ.get("GOOGLE_CSE_ID_PUBLIC")  # fallback public CSE (optional)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GOOGLE_API_KEY or not GOOGLE_CSE_ID or not GEMINI_API_KEY:
    log.warning("One or more required environment variables are missing. "
                "Expect errors if you haven't set GOOGLE_API_KEY, GOOGLE_CSE_ID, or GEMINI_API_KEY.")

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
    except Exception as e:
        log.exception("Google search failed")
        return []


def build_search_query(user_text: str) -> str:
    site_filter = " OR ".join(TRUSTED_SITES)
    return f"{user_text} {site_filter}"


def call_gemini(prompt_text: str) -> str:
    """Call Gemini generative AI with gemini-2.5-flash-lite model."""
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateText"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GEMINI_API_KEY}"
    }
    body = {
        "prompt": {
            "text": prompt_text
        },
        "temperature": 0.2,
        "maxOutputTokens": 800
    }
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        # The generated text is inside 'candidates'[0]['output']['content']
        return data["candidates"][0]["output"]["content"]
    except Exception as e:
        log.exception("Gemini call failed")
        return ""


SYSTEM_PROMPT = """You are Medibot, a professional, cautious and accurate medical assistant.
Rules (very important):
- ONLY provide medical/health-related information: conditions, symptoms, causes, tests, treatments, prevention, and when to seek care.
- Use the provided search results (listed below) as the ONLY sources of factual information. Cite them inline with numbers like [1], [2].
- If the information is not present in the provided sources, say you don't have enough information and advise consulting a healthcare professional.
- If the user asks something clearly non-medical or abusive/illegal/dangerous (e.g., instructions for wrongdoing, explicit sexual content, self-harm steps), refuse politely and redirect to safe resources or encourage seeking professional help.
- Resolve ambiguous pronouns ("it", "those", "that") using the conversation history provided.
- Keep answers concise (3-6 short paragraphs) and factual. At the end, list the sources used with numbers and links.
"""


def build_prompt(user_question: str, search_items: List[Dict], history: List[Dict]) -> str:
    history_text_lines = []
    last_n = 10
    for turn in history[-last_n:]:
        role = turn.get("role")
        content = turn.get("content", "")
        if role and content:
            history_text_lines.append(f"{role.title()}: {content}")
    history_text = "\n".join(history_text_lines) if history_text_lines else "(no previous context)"

    snippets = []
    for i, item in enumerate(search_items[:6]):
        title = item.get("title", "No title")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        snippets.append(f"[{i+1}] {title}\n{snippet}\n{link}")

    search_text = "\n\n".join(snippets) if snippets else "No search results available."

    prompt = (
        SYSTEM_PROMPT + "\n\n"
        "Conversation history:\n" + history_text + "\n\n"
        "Search results (trusted sources):\n" + search_text + "\n\n"
        f"User question: {user_question}\n\n"
        "Using ONLY the above search results and the conversation context, answer the user's question. "
        "Cite sources inline using [1], [2], etc., and then list the sources used with their links at the end."
    )
    return prompt


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "medibot.html")


@app.route("/api/v1/search_answer", methods=["POST"])
def search_answer():
    global conversation_history

    try:
        data = request.get_json(force=True)
    except Exception as e:
        log.exception("Bad JSON in request")
        return jsonify({"answer": "Invalid request JSON.", "sources": []}), 400

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

    # Greet only once at the very first message
    greetings = ["hi", "hello", "hey"]
    if latest_user.lower() in greetings and len(conversation_history) == 0:
        greeting_msg = "Hello! 👋 I'm Medibot, your Medi Assistant. How may I help you today?"
        conversation_history.append({"role": "assistant", "content": greeting_msg})
        return jsonify({"answer": greeting_msg, "sources": []})

    conversation_history.append({"role": "user", "content": latest_user})

    query_restricted = build_search_query(latest_user)

    items = []
    if GOOGLE_API_KEY and GOOGLE_CSE_ID:
        items = google_search(query_restricted, GOOGLE_CSE_ID)

    if (not items) and GOOGLE_CSE_ID_PUBLIC:
        items = google_search(latest_user, GOOGLE_CSE_ID_PUBLIC)

    if not items:
        fallback_msg = "Sorry, I couldn't find reliable sources for that question. Please try rephrasing or consult a healthcare professional."
        conversation_history.append({"role": "assistant", "content": fallback_msg})
        return jsonify({"answer": fallback_msg, "sources": []})

    sources = []
    for it in items[:6]:
        sources.append({
            "title": it.get("title", "")[:200],
            "link": it.get("link", "")
        })

    prompt = build_prompt(latest_user, items, conversation_history)
    gemini_answer = call_gemini(prompt)

    if not gemini_answer:
        log.warning("Gemini returned empty — falling back to snippets summary")
        snippets = [it.get("snippet", "") for it in items[:3] if it.get("snippet")]
        fallback_answer = " ".join(snippets) or "Sorry, I couldn't find an answer."
        conversation_history.append({"role": "assistant", "content": fallback_answer})
        return jsonify({"answer": fallback_answer, "sources": sources})

    conversation_history.append({"role": "assistant", "content": gemini_answer})

    return jsonify({"answer": gemini_answer, "sources": sources})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)





