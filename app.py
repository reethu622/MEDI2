import os
import google.generativeai as genai

GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY") or "YOUR_API_KEY"
genai.configure(api_key=GENAI_API_KEY)

def generate_gemini_response(prompt):
    try:
        response = genai.chat.completions.create(
            model="models/gemini-1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_output_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Gemini error:", e)
        return "Sorry, I am having trouble generating a response right now."

print(generate_gemini_response("What is diabetes?"))


