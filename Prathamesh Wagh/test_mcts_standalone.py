from agent.mcts import MCTSSearch
from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == "__main__":
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in .env")
        exit(1)

    print("Initializing MCTS...")
    mcts = MCTSSearch(max_simulations=3, exploration_constant=1.0)
    
    prompt = "Write a Python function to check for prime numbers."
    print(f"Running MCTS for prompt: '{prompt}'")
    
    best_code = mcts.search(prompt)
    
    print("\n--- Best Code Found ---")
    print(best_code)
    print("-----------------------")
