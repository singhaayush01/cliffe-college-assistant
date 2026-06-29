import os
from dotenv import load_dotenv
from google import genai

# Load your key
load_dotenv()
key = os.getenv("GOOGLE_API_KEY")

if not key:
    print("❌ Error: No GOOGLE_API_KEY found in .env file")
    exit()

print(f"🔑 Testing Key: {key[:5]}... (Hidden)")

try:
    client = genai.Client(api_key=key)
    print("📡 Sending a test request to gemini-2.5-flash...")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Reply with the single word: OK",
    )

    print(f"✅ Success! Model replied: {response.text.strip()}")
    print("👉 Your key works. Use model='gemini-2.5-flash' in main.py and ingest.py.")

except Exception as e:
    print(f"❌ Connection Failed: {e}")
    print("   Check that GOOGLE_API_KEY is correct and the Generative Language API is enabled for your project.")