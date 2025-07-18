import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load your API key from .env if needed
if os.path.exists('.env'):
    load_dotenv('.env')
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
else:
    GEMINI_API_KEY = None

if not GEMINI_API_KEY:
    raise ValueError("No Gemini API key found!")

genai.configure(api_key=GEMINI_API_KEY)

print("Available Gemini models:")
for m in genai.list_models():
    print(m.name)
