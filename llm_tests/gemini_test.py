import os
import google.generativeai as genai
from google.api_core import exceptions

# Try to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed. Relying on system environment variables.")

def test_gemini_key():
    # 1. Get the Key
    # Make sure your .env file has: GOOGLE_API_KEY=AIza...
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("❌ Error: GOOGLE_API_KEY is not found in environment variables.")
        print("   Make sure you have a .env file with GOOGLE_API_KEY=AIzaSy...")
        return

    print(f"🔑 Key found: {api_key[:8]}...****")
    print("⏳ Attempting to connect to Google Gemini...")

    # 2. Configure the SDK
    genai.configure(api_key=api_key)

    try:
        # 3. Initialize a simple model (Gemini 1.5 Flash is fast and free-tier eligible)
        model = genai.GenerativeModel('gemini-flash-latest')

        # 4. Send a test message
        response = model.generate_content("Say 'Pong' if you can hear me.")

        # 5. Check response
        if response.text:
            print("\n✅ SUCCESS! The Gemini API key is working.")
            print(f"   Response from Gemini: \"{response.text.strip()}\"")
        else:
            print("\n⚠️  Connected, but received an empty response.")

    except exceptions.InvalidArgument:
        print("\n❌ Invalid API Key: The key provided is incorrect.")
        print("   Action: Check your .env file or generate a new key at aistudio.google.com")

    except exceptions.PermissionDenied:
        print("\n❌ Permission Denied: The key may lack permissions or be restricted.")
        print("   Action: Check API key restrictions in Google Cloud Console.")

    except exceptions.ResourceExhausted:
        print("\n❌ Quota Exceeded: You have hit the rate limit or free tier limit.")
        print("   Action: Wait a minute or check your billing status.")

    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")

if __name__ == "__main__":
    test_gemini_key()