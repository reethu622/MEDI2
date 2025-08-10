import google.generativeai as genai

genai.configure(api_key=GENAI_API_KEY)

response = genai.generate_text(
    model="gemini-2.0-flash",
    prompt="What is diabetes?",
    temperature=0.7,
    max_output_tokens=300
)

print(response.text)



