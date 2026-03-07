import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI

class Evaluator:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            temperature=0.0 
        )

    def evaluate(self, code: str, requirements: str) -> float:
        """
        Evaluates the given code against the requirements using an LLM.
        Returns a score between 0.0 and 1.0.
        """
        prompt = f"""You are an expert code reviewer.
        Your task is to evaluate the following code snippet based on the user's requirements.
        
        Requirements:
        {requirements}
        
        Code:
        ```
        {code}
        ```
        
        Evaluate the code based on:
        1. Correctness (Does it solve the problem?)
        2. Efficiency (Time and Space complexity)
        3. Readability (Clean, commented, Pythonic)
        4. Robustness (Edge cases handling)

        Return ONLY a JSON object with the following structure:
        {{
            "score": <float between 0.0 and 1.0>,
            "reasoning": "<brief explanation of the score>"
        }}
        Do not output markdown code blocks for the JSON. Just the JSON string.
        """
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Simple regex to extract JSON if the LLM adds extra text or code blocks
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                import json
                result = json.loads(match.group(0))
                return float(result.get("score", 0.0))
            else:
                print("Evaluator Parse Error: No JSON found")
                return 0.0
                
        except Exception as e:
            print(f"Evaluator Error: {e}")
            return 0.0
