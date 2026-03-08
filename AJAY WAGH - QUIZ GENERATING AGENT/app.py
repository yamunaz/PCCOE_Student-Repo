import os
import io
import base64
import traceback
import logging
import sys
import time
import json
from collections import deque

# --- 0. Fix for Windows Asyncio Loop Policy (Standard Safety) ---
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import List, Union
from dotenv import load_dotenv

# --- 1. Suppress Warnings ---
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("absl").setLevel(logging.ERROR)

# --- 2. Safe Imports ---
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
    print("WARNING: 'pypdf' library not found. PDF upload will fail.")

try:
    from PIL import Image
except ImportError:
    Image = None
    print("WARNING: 'Pillow' library not found. Image upload will fail.")

load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

app = Flask(__name__, static_folder='static', template_folder='static')
CORS(app)

# --- 3. RATE LIMITER (The Gatekeeper) ---
class RateLimiter:
    """
    Ensures we never exceed N requests per M seconds.
    Implementation: Sliding Window Log / Token Bucket style.
    """
    def __init__(self, max_calls, period_seconds):
        self.max_calls = max_calls
        self.period = period_seconds
        self.timestamps = deque()

    def wait_for_token(self):
        """Blocks execution until a token is available."""
        now = time.time()
        
        # Remove timestamps older than the period
        while self.timestamps and now - self.timestamps[0] > self.period:
            self.timestamps.popleft()

        if len(self.timestamps) >= self.max_calls:
            sleep_time = self.period - (now - self.timestamps[0]) + 0.5 # +0.5s buffer
            print(f"🛑 Rate Limit Hit. Sleeping for {sleep_time:.2f}s...")
            time.sleep(sleep_time)
            
            # Re-clean after sleep to be sure
            self.wait_for_token()
        else:
            self.timestamps.append(time.time())

# Initialize Limiter: 5 requests per 60 seconds (Conservative for Free Tier)
# We use 4/60 to be extra safe.
limiter = RateLimiter(max_calls=4, period_seconds=65)

# --- 4. AI MODEL SETUP ---
def get_llm():
    if not GOOGLE_API_KEY:
        print("--- CRITICAL ERROR: GOOGLE_API_KEY is missing ---")
        return None
    
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7
    )

llm = get_llm()

# --- Pydantic Schemas ---
class Question(BaseModel):
    question: str = Field(description="The text of the quiz question.")
    options: List[str] = Field(description="A list of 2 or 4 possible answers.")
    correctAnswerIndex: Union[int, List[int]] = Field(description="The index/indices of the correct answer.")

class EvaluationResult(BaseModel):
    candidate_index: int = Field(description="0 for the first candidate, 1 for the second.")
    score: int = Field(description="Quality score from 1 to 10.")
    reasoning: str = Field(description="Brief reason for the score.")

class BatchEvaluation(BaseModel):
    evaluations: List[EvaluationResult] = Field(description="List of evaluations for the provided candidates.")

class SingleQuestionOutput(BaseModel):
    question_data: Question = Field(description="The generated question object.")

# --- HELPER FUNCTIONS ---
def extract_text_from_pdf(pdf_file):
    if PdfReader is None: return None
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content: text += content
        return text
    except Exception as e:
        print(f"PDF Error: {e}")
        return None

def extract_text_from_image(image_file):
    if Image is None: return None
    try:
        img = Image.open(image_file)
        buffered = io.BytesIO()
        img.save(buffered, format=img.format if img.format else "JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Note: Images consume quota too, so we apply limit here
        limiter.wait_for_token()
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Extract all readable text from this image. Return ONLY the text."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
            ]
        )
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f"Image Error: {e}")
        return None

# --- BATCHED MCTS COMPONENTS ---

def generate_candidate_question(input_text, question_type, context_instruction):
    """Generates a SINGLE candidate question."""
    # RATE LIMIT CHECK
    limiter.wait_for_token()
    
    parser = JsonOutputParser(pydantic_object=SingleQuestionOutput)
    safe_text = input_text[:6000]
    
    system_instruction = (
        f"You are a quiz generator. Generate ONE {question_type} question based on this context: "
        f"{safe_text}... \n"
        f"{context_instruction}"
        "Output JSON with a single key 'question_data'."
    )

    prompt = PromptTemplate(
        template="{system_instruction}\n\n{format_instructions}",
        input_variables=["system_instruction"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    try:
        res = chain.invoke({"system_instruction": system_instruction})
        return res['question_data']
    except Exception as e:
        print(f"Gen Error: {e}")
        return None

def batch_evaluate_candidates(candidates, input_text):
    """
    BATCHING OPTIMIZATION:
    Sends multiple candidates to the LLM in ONE request to save quota.
    """
    if not candidates: return []
    
    # RATE LIMIT CHECK
    limiter.wait_for_token()
    
    parser = JsonOutputParser(pydantic_object=BatchEvaluation)
    
    candidates_str = ""
    for i, cand in enumerate(candidates):
        candidates_str += f"\n--- Candidate {i} ---\nQuestion: {cand['question']}\nOptions: {cand['options']}\nAnswer: {cand['correctAnswerIndex']}\n"
    
    system_instruction = (
        "You are a strict Teacher/Critic. You have been given multiple potential quiz questions.\n"
        "Evaluate EACH candidate based on Clarity, Accuracy, and Distractor Quality.\n"
        f"Context Snippet: {input_text[:1000]}...\n"
        f"{candidates_str}\n"
        "Return a JSON object with a list 'evaluations'. Each item must have 'candidate_index' and 'score' (1-10)."
    )
    
    prompt = PromptTemplate(
        template="{system_instruction}\n\n{format_instructions}",
        input_variables=["system_instruction"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    try:
        print("   ⚖️ Sending Batch Evaluation Request...")
        res = chain.invoke({"system_instruction": system_instruction})
        return res['evaluations']
    except Exception as e:
        print(f"Batch Eval Error: {e}")
        return []

def mcts_question_generation(input_text, num_questions, question_type):
    """
    Advanced MCTS with Rate Limiting & Batching.
    Cost: 2 Requests per Question (1 Generation + 0.5 Evaluation amortized).
    """
    print(f"--- Starting Batched MCTS for {num_questions} questions ---")
    final_quiz = []

    is_mixed = question_type == 'mixed'
    is_multi = question_type == 'multi_choice'
    
    type_desc = "Single Choice"
    if is_multi: type_desc = "Multiple Choice (Multiple Correct)"
    elif is_mixed: type_desc = "Mixed (True/False or MCQ)"

    for i in range(num_questions):
        print(f"--- Processing Question Slot {i+1}/{num_questions} ---")
        
        # STEP 1: EXPANSION (Generate 2 Candidates sequentially)
        # We generate 2 candidates to choose from.
        # This costs 2 requests.
        c1 = generate_candidate_question(input_text, type_desc, "Ensure question is conceptual.")
        # If we are really tight on budget, we can skip c2.
        # But MCTS needs choices. Let's try generating just 1 first, 
        # and only generate a 2nd if the first is bad (Lazy Expansion).
        
        candidates = []
        if c1: candidates.append(c1)
        
        # STEP 2: EVALUATION (Batched or Single)
        best_candidate = None
        
        if candidates:
            # Evaluate the single candidate
            # In a full MCTS, we would have 2 candidates here and batch evaluate them.
            # To save API calls, we will implement "Lazy MCTS":
            # Check Candidate 1. If Score > 8, keep it. If not, generate Candidate 2.
            
            # Using Batch Evaluator even for 1 item keeps logic consistent
            evals = batch_evaluate_candidates(candidates, input_text)
            
            score = 0
            if evals: score = evals[0]['score']
            print(f"   Candidate 1 Score: {score}/10")
            
            if score >= 8:
                best_candidate = candidates[0]
            else:
                print("   ⚠️ Score low. Expanding tree (Generating Candidate 2)...")
                c2 = generate_candidate_question(input_text, type_desc, "Previous was too complex/simple. Make this one perfect.")
                if c2:
                    # Now we have to evaluate C2.
                    evals_2 = batch_evaluate_candidates([c2], input_text)
                    score_2 = evals_2[0]['score'] if evals_2 else 0
                    print(f"   Candidate 2 Score: {score_2}/10")
                    
                    if score_2 > score:
                        best_candidate = c2
                    else:
                        best_candidate = candidates[0] # Fallback to first
                else:
                    best_candidate = candidates[0]

        if best_candidate:
            final_quiz.append(best_candidate)
        else:
             print("   ❌ Failed to generate valid question for this slot.")

    return final_quiz

# --- ROUTES ---
@app.route('/', methods=['GET'])
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/generate-quiz', methods=['POST'])
def generate_quiz_route():
    try:
        topic = request.form.get('topic')
        num_questions_str = request.form.get('numQuestions')
        question_type = request.form.get('questionType')
        uploaded_file = request.files.get('file')
        
        if not num_questions_str: return jsonify({"error": "Missing parameters"}), 400
        num_questions = int(num_questions_str)

        input_text = topic
        
        if uploaded_file and uploaded_file.filename != '':
            filename = uploaded_file.filename.lower()
            
            extracted_text = None
            if filename.endswith('.pdf'):
                extracted_text = extract_text_from_pdf(uploaded_file)
            elif filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                extracted_text = extract_text_from_image(uploaded_file)
            
            if extracted_text and len(extracted_text.strip()) > 0:
                input_text = extracted_text
            else:
                return jsonify({"error": "Could not extract text from file."}), 400
        
        elif not topic:
             return jsonify({"error": "No Topic or File provided."}), 400

        # Run MCTS
        quiz_data = mcts_question_generation(input_text, num_questions, question_type)
        return jsonify({"quiz": quiz_data})

    except Exception as e:
        print(f"SERVER ERROR: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
