import requests
import json
import time

url = "http://127.0.0.1:5000/chat"
payload = {
    "message": "Write a python function to add two numbers.",
    "mode": "mcts"
}
headers = {'Content-Type': 'application/json'}

print("Sending request to MCTS endpoint...")
for i in range(5):
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        break
    except Exception as e:
        print(f"Attempt {i+1} failed: {e}")
        time.sleep(2)
