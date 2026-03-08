import os
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Important ---
GEMINI_KEY = "GOOGLE API KEY"
MODEL_NAME = "gemini-pro"

print("--- Starting Direct API Test ---")

try:
    # Initialize the AI model directly
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, api_key=GEMINI_KEY)
    print("--- Model Initialized Successfully ---")

    # Send a simple test prompt
    print("Sending a test prompt to the AI...")
    response = llm.invoke("Hello, what is your name?")

    print("\n--- AI Responded Successfully! ---")
    print("Response:")
    print(response.content)

except Exception as e:
    print("\n--- TEST FAILED ---")
    print("An error occurred:")
    print(e)
