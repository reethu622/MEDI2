import os
from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# Load API key from environment variable
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GENAI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

# Configure the Google Generative AI client
genai.configure(api_key=GEMINI_API_KEY)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        response = genai.generate_text(
            model="gemini-2.0-flash",
            prompt=prompt,
            temperature=0.7,
            max_output_tokens=300
        )
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": f"Failed to generate response: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)




