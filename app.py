import os
import re
import requests
import openai
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Load API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GOOGLE_SEARCH_KEY = os.getenv("GOOGLE_SEARCH_KEY", "")
GOOGLE_SEARCH_CX = os.getenv("GOOGLE_SEARCH_CX", "")

# Configure OpenAI and Gemini if keys are present
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
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

# Basic abusive word list (expand as needed)
ABUSIVE_WORDS = ["idiot", "stupid", "dumb", "hate", "shut up", "fool", "damn", "bastard", "crap"]

def contains_abuse(text):
    text = text.lower()
    for word in ABUSIVE_WORDS:
        if word in text:
            return True
    return False

def google_search_with_citations(query):
    """Perform Google Custom Search and return results with formatted citations."""
    if not GOOGLE_SEARCH_KEY or not GOOGLE_SEARCH_CX:
        return [], ""  # Skip search if keys missing

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

    # Check for abusive content in latest message
    if contains_abuse(latest_user_message):
        polite_response = (
            "I am here to help with medical questions. "
            "Please keep the conversation respectful. How can I assist you today?"
        )
        return jsonify({"answer": polite_response, "sources": []})

    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if latest_user_message.lower() in greetings:
        greeting_reply = "Hi! How may I help you with your medical questions today?"
        return jsonify({"answer": greeting_reply, "sources": []})

    # Run Google Search on latest user question for grounding
    results, formatted_results = google_search_with_citations(latest_user_message)

    # Strong system prompt to force citation from results only
    system_prompt = (
        "You are a helpful medical assistant chatbot. Answer the user's medical question ONLY using the numbered search results below. "
        "Cite your sources explicitly using [1], [2], etc., corresponding to the numbers of the search results. "
        "If the answer cannot be found in the search results, say you don't know and recommend consulting a healthcare professional.\n\n"
        f"{formatted_results}\n"
    )

    # Prepare messages for OpenAI
    openai_messages = [{"role": "system", "content": system_prompt}]
    openai_messages.extend(messages)

    if OPENAI_API_KEY:
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=openai_messages,
                temperature=0.3,
            )
            answer = resp.choices[0].message["content"]

            # Extract cited source numbers from answer
            cited_nums = set(int(n) for n in re.findall(r"\[(\d+)\]", answer))
            filtered_sources = [results[i-1] for i in cited_nums if 0 < i <= len(results)]

            return jsonify({"answer": answer, "sources": filtered_sources})
        except Exception as e:
            if "quota" not in str(e).lower():
                return jsonify({"answer": f"OpenAI error: {e}", "sources": []})
            print("⚠ OpenAI quota exceeded, switching to Gemini...")

    # Use Gemini if OpenAI fails or quota exceeded
    if GEMINI_API_KEY:
        try:
            conversation_text = system_prompt + "\nConversation:\n"
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation_text += f"{role}: {msg['content']}\n"
            conversation_text += "Assistant:"

            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(conversation_text)
            answer = resp.text

            cited_nums = set(int(n) for n in re.findall(r"\[(\d+)\]", answer))
            filtered_sources = [results[i-1] for i in cited_nums if 0 < i <= len(results)]

            return jsonify({"answer": answer, "sources": filtered_sources})
        except Exception as e:
            return jsonify({"answer": f"Gemini error: {e}", "sources": []})

    # Fallback offline FAQ (optional)
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











