"""
Find available music generation models on Hugging Face.
"""

import requests
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("HUGGINGFACE_API_KEY")

# Popular music generation models to test
models_to_test = [
    "facebook/musicgen-small",
    "facebook/musicgen-medium",
    "facebook/musicgen-large",
    "declare-lab/musicgen-small",
    "declare-lab/musicgen-medium",
    "openai/whisper-small",
    "stabilityai/stable-audio-open-1.0",
    "google/musiclm",
]

print("[TESTING] Hugging Face Models")
print("=" * 60)

for model in models_to_test:
    try:
        url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {api_key}"}

        # Send minimal request to check if model exists
        response = requests.post(
            url,
            headers=headers,
            json={"inputs": "test"},
            timeout=5
        )

        status = response.status_code
        available = status != 404

        print(f"[{status}] {model}")
        if status == 200:
            print(f"     -> SUCCESS - Model is available!")
        elif status == 503:
            print(f"     -> Model loading (try later)")
        elif status == 401:
            print(f"     -> Invalid API key")
        else:
            print(f"     -> Error: {response.reason}")

    except Exception as e:
        print(f"[ERR] {model} - {str(e)}")

print("=" * 60)
print("\nTip: Copy working model name and update api_client.py")
