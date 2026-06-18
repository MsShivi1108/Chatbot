import os
from dotenv import load_dotenv
import requests

load_dotenv()

API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.environ['API_KEY']}",
}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

response = query({
    "messages": [
        {
            "role": "user",
            "content": "What is the Hugging face?"
        }
    ],
    "model": "zai-org/GLM-5.1:together"
})

print(response["choices"][0]["message"]["content"])