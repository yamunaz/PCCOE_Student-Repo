import os
import io
import base64
import traceback
import logging
import asyncio
import sys
import random
from concurrent.futures import ThreadPoolExecutor

# --- 0. Fix for Windows Asyncio Loop Policy ---
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import List, Union, Optional
from dotenv import load_dotenv

# --- 1. Suppress Warnings ---
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("absl").setLevel(logging.ERROR)

# --- 2. Safe Import of Libraries ---
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

# --- 3. AI MODEL SETUP ---
def get_llm():
    if not GOOGLE_API_KEY:
        print("--- CRITICAL ERROR: GOOGLE_API_KEY is missing in .env file ---")
        return None
    
    return ChatGoogleGenerativeAI(
        model="gemini-3.0-flash", # Updated to faster model if available, else use 1.5-flash
        google_api_key=GOOGLE_API_KEY,
        temperature=0.8 # Slightly higher temp for diverse MCTS generation
    )

llm = get_llm()

# --- Pydantic Schemas ---
class Question(BaseModel):
    question: str = Field(description="The text of the quiz question.")
    options: List[str] = Field(description="A list of 2 or 4 possible answers.")
    correctAnswerIndex: Union[int, List[int]] = Field(description="The index/indices of the correct answer.")

class QuestionCritique(BaseModel):
    score: int = Field(description="Quality score from 1 to 10.")
    reasoning: str = Field(description="Why this score was given.")
    is_valid: bool = Field(description="True if the question is factually correct and clear.")

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

# --- MCTS / CRITIC COMPONENTS ---

def generate_candidate_question(input_text, question_type, context_instruction):
    """Generates a SINGLE candidate question."""
    parser = JsonOutputParser(pydantic_object=SingleQuestionOutput)
    
    system_instruction = (
        f"You are a quiz generator. Generate ONE {question_type} question based on this context: "
        f"{input_text[:5000]}... \n" # Limit context for speed
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
        return None

def evaluate_candidate(question_obj, input_text):
    """The Critic Agent: Scores the question."""
    if not question_obj: return 0
    
    parser = JsonOutputParser(pydantic_object=QuestionCritique)
    
    question_str = f"Question: {question_obj['question']}\nOptions: {question_obj['options']}\nAnswer Index: {question_obj['correctAnswerIndex']}"
    
    system_instruction = (
        "You are a strict Teacher/Critic. Evaluate this quiz question based on:\n"
        "1. Clarity (Is it easy to understand?)\n"
        "2. Accuracy (Is the answer correct based on general knowledge or context?)\n"
        "3. Distractor Quality (Are wrong answers plausible?)\n"
        f"Context Snippet: {input_text[:1000]}...\n\n"
        f"{question_str}\n\n"
        "Give a score (1-10) and valid boolean."
    )
    
    prompt = PromptTemplate(
        template="{system_instruction}\n\n{format_instructions}",
        input_variables=["system_instruction"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    try:
        res = chain.invoke({"system_instruction": system_instruction})
        return res['score'] if res['is_valid'] else 0
    except:
        return 0

def mcts_question_generation(input_text, num_questions, question_type):
    """
    Simulated MCTS Loop:
    1. Expansion: Generate 2 candidates for each slot.
    2. Simulation: Critic evaluates them.
    3. Selection: Pick the best one.
    """
    print(f"--- Starting MCTS Quiz Gen for {num_questions} questions ---")
    final_quiz = []

    # Configure type descriptions
    is_mixed = question_type == 'mixed'
    is_multi = question_type == 'multi_choice'
    
    type_desc = "Single Choice"
    if is_multi: type_desc = "Multiple Choice (Multiple Correct)"
    elif is_mixed: type_desc = "Mixed (True/False or MCQ)"

    # We will generate questions sequentially to maintain context flow, 
    # but use MCTS for quality control on each question.
    
    for i in range(num_questions):
        print(f"--- Generating Question {i+1}/{num_questions} ---")
        
        # MCTS STEP 1: EXPANSION (Generate 2 Candidates)
        # We run this in parallel to save time
        candidates = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            # We vary the prompt slightly for diversity
            futures = [
                executor.submit(generate_candidate_question, input_text, type_desc, "Ensure question is conceptual."),
                executor.submit(generate_candidate_question, input_text, type_desc, "Ensure question tests factual details.")
            ]
            for f in futures:
                res = f.result()
                if res: candidates.append(res)
        
        if not candidates:
            print("Fallback: Gen failed for this slot.")
            continue

        # MCTS STEP 2: SIMULATION/EVALUATION (Score Candidates)
        best_candidate = None
        best_score = -1
        
        for cand in candidates:
            score = evaluate_candidate(cand, input_text)
            print(f"   Candidate Score: {score}/10")
            if score > best_score:
                best_score = score
                best_candidate = cand
        
        # MCTS STEP 3: SELECTION
        if best_candidate and best_score >= 5:
            final_quiz.append(best_candidate)
        elif candidates:
            # Fallback if scores are low, take the first one
            final_quiz.append(candidates[0])

    return final_quiz

# --- ROUTES ---
@app.route('/', methods=['GET'])
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/generate-quiz', methods=['POST'])
def generate_quiz_route():
    try:
        topic = request.form.get('topic')
        num_questions = int(request.form.get('numQuestions'))
        question_type = request.form.get('questionType')
        uploaded_file = request.files.get('file')
        
        input_text = topic
        
        if uploaded_file and uploaded_file.filename != '':
            filename = uploaded_file.filename.lower()
            print(f"Processing File: {filename}")
            
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

        # Trigger MCTS Logic
        quiz_data = mcts_question_generation(input_text, num_questions, question_type)
        
        return jsonify({"quiz": quiz_data})

    except Exception as e:
        print(f"SERVER ERROR: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)