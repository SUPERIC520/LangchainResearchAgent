import os
import sys
from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError

# Try to load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed. Relying on system environment variables.")

def test_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ Error: OPENAI_API_KEY is not found in environment variables.")
        print("   Make sure you have a .env file with OPENAI_API_KEY=sk-...")
        return

    print(f"🔑 Key found: {api_key[:8]}...****")
    print("⏳ Attempting to connect to OpenAI...")

    client = OpenAI(api_key=api_key)

    try:
        # We use a tiny model and token count to minimize cost/latency
        response = client.chat.completions.create(
            model="gpt-4o-mini", # or "gpt-4o-mini"
            messages=[
                {"role": "user", "content": "Say 'Pong' if you hear me."}
            ],
            max_tokens=5
        )
        
        reply = response.choices[0].message.content
        print("\n✅ SUCCESS! The API key is working.")
        print(f"   Response from OpenAI: \"{reply}\"")

    except AuthenticationError:
        print("\n❌ Authentication Error: Your API key is invalid.")
        print("   Action: Check for typos in your .env file or generate a new key.")

    except RateLimitError as e:
        print("\n❌ Rate Limit / Quota Error:")
        print(f"   {e}")
        print("\n   Action: This usually means you have run out of credits ($0.00 balance).")
        print("   Go to https://platform.openai.com/settings/organization/billing/overview to add funds.")

    except APIConnectionError:
        print("\n❌ Connection Error: Could not reach OpenAI servers.")
        print("   Action: Check your internet connection or VPN/Firewall settings.")

    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")

if __name__ == "__main__":
    test_api_key()