import os
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

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__, static_folder="static")
CORS(app)

ABUSIVE_WORDS = ["idiot", "stupid", "dumb", "hate", "shut up", "fool", "damn", "bastard", "crap"]

def contains_abuse(text):
    text = text.lower()
    return any(word in text for word in ABUSIVE_WORDS)

def google_search_with_citations(query, num_results=5):
    """
    Perform Google Custom Search and return a list of dicts:
    [{title, snippet, link}, ...]
    """
    if not GOOGLE_SEARCH_KEY or not GOOGLE_SEARCH_CX:
        return []

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
        return []

    results = []
    for item in data.get("items", []):
        results.append({
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": item.get("link", "")
        })
    return results

def is_answer_incomplete(answer_text, user_query):
    """
    Basic check if the answer is incomplete or uncertain.
    """
    answer_lower = answer_text.lower()
    if any(phrase in answer_lower for phrase in ["sorry", "don't know", "cannot find", "need more information"]):
        return True

    question_keywords = ["type", "types", "explain", "list", "what are", "different kinds", "kinds"]
    if any(word in user_query.lower() for word in question_keywords):
        if all(x not in answer_lower for x in ["type", "kind", "explain"]):
            return True

    return False

def generate_answer_with_sources(messages, results):
    """
    Generate answer from the LLM with web search results context and
    with pronoun resolution system prompt to track context.
    """
    # Format the web results with numbering to allow citation [1], [2], etc.
    formatted_results_text = ""
    for idx, item in enumerate(results, start=1):
        formatted_results_text += f"[{idx}] {item['title']}\n{item['snippet']}\nSource: {item['link']}\n\n"

    system_prompt = (
        "You are Medibot, a helpful and knowledgeable medical assistant chatbot. "
        "When the user uses pronouns like 'it', 'those', 'these', or says 'explain that', "
        "infer that they mean the most recent medical topic or condition discussed earlier in the conversation. "
        "Always keep track of conversational context carefully and do not guess unrelated topics. "
        "Answer the user's questions based on the following web search results. "
        "If you cannot find a clear answer, politely say you don't know and recommend consulting a healthcare professional. "
        "Cite your sources using the numbers [1], [2], etc. as given below.\n\n"
        f"{formatted_results_text}\n"
    )

    # Prepare full messages list for LLM
    llm_messages = [{"role": "system", "content": system_prompt}] + messages

    # Try OpenAI GPT first
    if OPENAI_API_KEY:
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=llm_messages,
                temperature=0.3,
            )
            answer = resp.choices[0].message["content"].strip()
            return answer
        except Exception as e:
            print(f"OpenAI error: {e}")
            if "quota" not in str(e).lower():
                return f"OpenAI error: {e}"

    # Fallback to Gemini
    if GEMINI_API_KEY:
        try:
            conversation_text = system_prompt + "\nConversation:\n"
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation_text += f"{role}: {msg['content']}\n"
            conversation_text += "Assistant:"
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(conversation_text)
            return resp.text.strip()
        except Exception as e:
            print(f"Gemini error: {e}")
            return f"Gemini error: {e}"

    # No LLM keys present
    return "I don't know. Please consult a medical professional."

@app.route("/api/v1/search_answer", methods=["POST"])
def search_answer():
    data = request.get_json()
    messages = data.get("messages")
    if not messages or not isinstance(messages, list):
        return jsonify({"answer": "Please provide conversation history as a list of messages.", "sources": []})

    # Extract latest user query for search and abuse check
    latest_user_message = next((msg["content"].strip() for msg in reversed(messages) if msg.get("role") == "user"), None)
    if not latest_user_message:
        return jsonify({"answer": "No user message found in conversation.", "sources": []})

    if contains_abuse(latest_user_message):
        return jsonify({
            "answer": "I am here to help with medical questions. Please keep the conversation respectful. How can I assist you today?",
            "sources": []
        })

    # Quick greeting responses
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if latest_user_message.lower() in greetings:
        return jsonify({"answer": "Hi! How may I help you with your medical questions today?", "sources": []})

    # Step 1: Limited Google Custom Search (5 results)
    results = google_search_with_citations(latest_user_message, num_results=5)

    # Step 2: Generate answer using those results
    answer = generate_answer_with_sources(messages, results)

    # Step 3: If answer incomplete, fallback to broader search (15 results)
    if is_answer_incomplete(answer, latest_user_message):
        fallback_results = google_search_with_citations(latest_user_message, num_results=15)
        answer = generate_answer_with_sources(messages, fallback_results)
        return jsonify({"answer": answer, "sources": fallback_results})

    # Step 4: Return answer and sources
    return jsonify({"answer": answer, "sources": results})

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "medibot.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7000))
    app.run(host="0.0.0.0", port=port)













