import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

# Load API keys from Railway variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Expanded list of reliable medical sources
SEARCH_SITES = (
    "site:mayoclinic.org OR site:webmd.com OR site:nih.gov OR "
    "site:who.int OR site:cdc.gov OR site:clevelandclinic.org OR site:medlineplus.gov"
)

# Keep conversation context in memory
conversation_history = []


def google_search(query):
    """Perform a Google Custom Search limited to our trusted medical sites."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": f"{query} {SEARCH_SITES}",
        "num": 5
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    results = []
    if "items" in data:
        for item in data["items"]:
            results.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet")
            })
    return results


def generate_answer(prompt):
    """Generate answer using Gemini API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    resp = requests.post(url, headers=headers, json=body)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return "Sorry, I couldn't process the answer."


@app.route("/")
def index():
    # Serve HTML file from static folder
    return send_from_directory("static", "medibot.html")


@app.route("/api/v1/search_answer", methods=["POST"])
def search_answer():
    global conversation_history
    data = request.get_json()
    messages = data.get("messages", [])

    if not messages:
        return jsonify({"answer": "No question received.", "sources": []})

    # Extract the latest user question
    latest_message = messages[-1]["content"]

    # Handle pronouns like "it" or "that"
    if latest_message.lower() in ["it", "that", "they", "those"] or latest_message.lower().startswith(("what are the types", "tell me more")):
        # Find the last medical topic mentioned in the conversation
        for msg in reversed(conversation_history):
            if msg["role"] == "user" and not any(word in msg["content"].lower() for word in ["it", "that", "they", "those"]):
                latest_message = latest_message.replace("it", msg["content"])
                break

    # Add latest user question to conversation history
    conversation_history.append({"role": "user", "content": latest_message})

    # Search Google Custom Search for reliable info
    search_results = google_search(latest_message)

    # Prepare search summary for Gemini
    search_text = "\n".join([f"{item['title']} - {item['snippet']}" for item in search_results])

    # Ask Gemini to answer based on the search results
    prompt = (
        f"You are Medibot, a helpful medical assistant.\n\n"
        f"User asked: {latest_message}\n\n"
        f"Here are some relevant sources:\n{search_text}\n\n"
        "Please give a concise, factual, and easy-to-understand answer for the user."
    )
    answer = generate_answer(prompt)

    # Add bot answer to conversation history
    conversation_history.append({"role": "assistant", "content": answer})

    return jsonify({"answer": answer, "sources": search_results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

