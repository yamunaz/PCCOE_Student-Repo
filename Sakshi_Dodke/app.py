import os
import time
import random
import math
import json
import cv2
import numpy as np
from PIL import Image
import io
import base64
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template
import requests
from datetime import datetime

# Configuration
OLLAMA_URL = "http://localhost:11435/api/generate"
VISION_MODEL = "llava:7b"  # Vision model for image analysis
TEXT_MODEL = "llama3.2:latest"
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'mkv'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB

# Create Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['SECRET_KEY'] = 'multimodal-mcts-secret'

# Create uploads directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_session_directory():
    """Create a temporary directory for session files"""
    session_id = f"session_{int(time.time())}_{random.randint(1000, 9999)}"
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir, session_id

def image_to_base64(image_path):
    """Convert image to base64 string"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large (max 512px on longest side)
            max_size = 512
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return img_str
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

def analyze_image_with_vision(image_path):
    """Analyze image content using vision model"""
    try:
        # Convert image to base64
        img_base64 = image_to_base64(image_path)
        if not img_base64:
            return "Unable to process image"
        
        # Prepare prompt for vision model
        prompt = """Describe this image in detail. Include:
        1. Main objects and subjects
        2. Colors and visual elements
        3. Any text visible in the image
        4. Overall scene and context
        
        Provide a comprehensive description:"""
        
        # Call vision model (llava)
        payload = {
            "model": VISION_MODEL,
            "prompt": prompt,
            "stream": False,
            "images": [img_base64]
        }
        
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        
        description = response.json().get("response", "").strip()
        
        # Also get basic image info
        with Image.open(image_path) as img:
            basic_info = f"Image: {img.format} format, {img.size[0]}x{img.size[1]} pixels"
        
        return f"{basic_info}\n\nImage Content Analysis:\n{description}"
        
    except Exception as e:
        print(f"Error analyzing image with vision model: {e}")
        # Fallback to basic analysis
        try:
            with Image.open(image_path) as img:
                return f"Image: {img.format} format, {img.size[0]}x{img.size[1]} pixels. Note: Detailed analysis failed."
        except:
            return "Image analysis failed"

def analyze_video_content(video_path):
    """Extract key frames and analyze video"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "Cannot open video file"
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Extract key frames for analysis (first, middle, last)
        key_frames = []
        frame_positions = [0, frame_count//2, frame_count-1] if frame_count > 2 else [0]
        
        for pos in frame_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                # Save frame as image temporarily
                frame_path = f"temp_frame_{pos}.jpg"
                cv2.imwrite(frame_path, frame)
                
                # Analyze this frame
                frame_description = analyze_image_with_vision(frame_path)
                key_frames.append(f"Frame at {pos/fps:.1f}s: {frame_description}")
                
                # Clean up
                if os.path.exists(frame_path):
                    os.remove(frame_path)
        
        cap.release()
        
        video_info = f"Video: {duration:.1f} seconds, {frame_count} frames, {width}x{height} resolution, {fps:.1f} fps"
        
        if key_frames:
            frames_desc = "\n".join(key_frames[:3])  # Limit to 3 frames
            return f"{video_info}\n\nKey Frames Analysis:\n{frames_desc}"
        else:
            return video_info
        
    except Exception as e:
        print(f"Error analyzing video: {e}")
        return f"Video analysis failed: {str(e)}"

# ------------------------------------------------------------------
# AGENT DEFINITIONS
# ------------------------------------------------------------------

class BaseAgent:
    """Base class for all multimodal summarization agents"""
    
    def __init__(self, agent_id, name, description):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.agent_icon = "🤖"
        
    def build_multimodal_prompt(self, text, media_analyses):
        """Build comprehensive prompt with media analyses"""
        prompt_parts = []
        
        if text and text.strip():
            prompt_parts.append(f"TEXT CONTENT:\n{text}")
        
        if media_analyses:
            prompt_parts.append("MEDIA CONTENT ANALYSIS:")
            for i, analysis in enumerate(media_analyses, 1):
                prompt_parts.append(f"\n--- Media Item {i} ---\n{analysis}")
        
        # Add agent-specific instructions
        agent_instruction = self.get_instruction()
        if media_analyses:
            agent_instruction += "\n\nPlease analyze BOTH the text content AND the media content described above."
        
        return f"{agent_instruction}\n\n{'='*50}\n" + "\n\n".join(prompt_parts) + f"\n{'='*50}\n\n{self.get_response_format()}"
    
    def get_instruction(self):
        """Get agent-specific instruction"""
        raise NotImplementedError
        
    def get_response_format(self):
        """Get response format instruction"""
        return "Please provide your summary:"
        
    def get_system_prompt(self):
        """Get system prompt for this agent"""
        return None
        
    def get_temperature(self):
        """Get temperature parameter"""
        return 0.7
        
    def get_max_tokens(self):
        """Get max tokens for this agent"""
        return 300

    def generate_summary(self, text, media_analyses=None):
        """Generate summary using Ollama with media context"""
        if not text and not media_analyses:
            return "No content provided for summarization"
        
        # Build the complete prompt with media context
        prompt = self.build_multimodal_prompt(text, media_analyses or [])
        system_prompt = self.get_system_prompt()
        
        print(f"\n📝 Agent {self.agent_id} Prompt Preview:")
        print(f"Text length: {len(text) if text else 0} chars")
        print(f"Media analyses: {len(media_analyses) if media_analyses else 0}")
        print(f"Prompt preview: {prompt[:200]}...")
        
        try:
            payload = {
                "model": TEXT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": self.get_max_tokens(),
                    "temperature": self.get_temperature(),
                    "top_p": 0.9
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json().get("response", "").strip()
            
            # Clean up result
            result = result.replace("Summary:", "").replace("TL;DR:", "").replace("Key points:", "").strip()
            return result
            
        except requests.exceptions.Timeout:
            return "Error: Request timeout - the model took too long to respond"
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama server. Make sure Ollama is running with llama3.2:latest model."
        except Exception as e:
            return f"Error: {str(e)}"

class ExtractiveAgent(BaseAgent):
    """Extracts key information"""
    
    def __init__(self):
        super().__init__(
            agent_id=1,
            name="📄 Extractive Agent",
            description="Extracts key information from all content types"
        )
        self.agent_icon = "📄"
        
    def get_system_prompt(self):
        return """You are an expert extractive summarization agent for multimodal content.
        Your task is to extract the most important factual information from both text and media descriptions.
        Focus on concrete facts, key details, and specific information."""
        
    def get_instruction(self):
        return """Extract the 4-5 most important pieces of information from the provided content."""
        
    def get_response_format(self):
        return "Please list the extracted information as numbered points:"
        
    def get_temperature(self):
        return 0.3

class AbstractiveAgent(BaseAgent):
    """Summarizes in own words"""
    
    def __init__(self):
        super().__init__(
            agent_id=2,
            name="✨ Abstractive Agent",
            description="Summarizes content in own words concisely"
        )
        self.agent_icon = "✨"
        
    def get_system_prompt(self):
        return """You are an expert abstractive summarization agent for multimodal content.
        Your task is to synthesize information from text and media descriptions,
        then rewrite it concisely in your own words while preserving all key information."""
        
    def get_instruction(self):
        return """Synthesize and summarize the provided content in your own words."""
        
    def get_response_format(self):
        return "Please provide a concise 2-3 sentence summary:"

class BulletAgent(BaseAgent):
    """Creates bullet point summaries"""
    
    def __init__(self):
        super().__init__(
            agent_id=3,
            name="📋 Bullet Points Agent",
            description="Creates key points as bullet points"
        )
        self.agent_icon = "📋"
        
    def get_system_prompt(self):
        return """You are an expert at creating structured bullet point summaries from multimodal content.
        Your task is to extract key points from both text and media descriptions
        and present them as clear, organized bullet points."""
        
    def get_instruction(self):
        return """Extract the key points from the provided content."""
        
    def get_response_format(self):
        return "Please provide key points as bullet points (•):"
        
    def get_temperature(self):
        return 0.5

class TLDRAgent(BaseAgent):
    """Creates very concise TL;DR summaries"""
    
    def __init__(self):
        super().__init__(
            agent_id=4,
            name="⚡ TL;DR Agent",
            description="Creates extremely concise TL;DR summaries"
        )
        self.agent_icon = "⚡"
        
    def get_system_prompt(self):
        return """You are a TL;DR (Too Long; Didn't Read) summarization expert for multimodal content.
        Your task is to create extremely concise summaries that capture the absolute essence
        from both text and media descriptions."""
        
    def get_instruction(self):
        return """Create a TL;DR summary of the provided content."""
        
    def get_response_format(self):
        return "TL;DR:"
        
    def get_max_tokens(self):
        return 150

class DetailedAgent(BaseAgent):
    """Creates comprehensive detailed summaries"""
    
    def __init__(self):
        super().__init__(
            agent_id=5,
            name="📚 Detailed Agent",
            description="Creates comprehensive detailed summaries"
        )
        self.agent_icon = "📚"
        
    def get_system_prompt(self):
        return """You are a comprehensive summarization agent for multimodal content.
        Your task is to create detailed summaries that cover all important aspects
        from both text and media descriptions."""
        
    def get_instruction(self):
        return """Create a comprehensive summary of the provided content."""
        
    def get_response_format(self):
        return "Comprehensive Summary:"
        
    def get_max_tokens(self):
        return 400

# ------------------------------------------------------------------
# AGENT REGISTRY
# ------------------------------------------------------------------

class AgentRegistry:
    """Manages all available agents"""
    
    def __init__(self):
        self.agents = {}
        self.register_agents()
        
    def register_agents(self):
        """Register all available agents"""
        agents = [
            ExtractiveAgent(),
            AbstractiveAgent(),
            BulletAgent(),
            TLDRAgent(),
            DetailedAgent()
        ]
        
        for agent in agents:
            self.agents[agent.agent_id] = agent
            
    def get_agent(self, agent_id):
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def get_all_agents(self):
        """Get all registered agents"""
        return list(self.agents.values())

# ------------------------------------------------------------------
# MCTS IMPLEMENTATION
# ------------------------------------------------------------------

class MCTSNode:
    """Node in the Monte Carlo Tree Search"""
    
    def __init__(self, agent_id=None, parent=None):
        self.agent_id = agent_id
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_value = 0.0
        self.untried_agents = []
        
    @property
    def value(self):
        return self.total_value / self.visits if self.visits > 0 else 0.0
    
    def is_fully_expanded(self):
        return len(self.untried_agents) == 0
    
    def best_child(self, exploration_weight=1.414):
        """Select best child using UCT formula"""
        best_score = -float('inf')
        best_child = None
        
        for child in self.children:
            if child.visits == 0:
                uct_score = float('inf')
            else:
                exploitation = child.value
                exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
                uct_score = exploitation + exploration
            
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        
        return best_child
    
    def add_child(self, agent_id):
        """Add a new child node"""
        child = MCTSNode(agent_id=agent_id, parent=self)
        self.children.append(child)
        return child
    
    def update(self, value):
        """Update node statistics"""
        self.visits += 1
        self.total_value += value

class MCTSOptimizer:
    """MCTS optimizer for selecting best summary"""
    
    def __init__(self, agent_results, has_multimedia=False):
        self.agent_results = agent_results
        self.has_multimedia = has_multimedia
        self.root = MCTSNode()
        self.root.untried_agents = [r["agent_id"] for r in agent_results]
        self.evaluation_cache = {}
        
    def evaluate_summary(self, agent_id, summary):
        """Evaluate a summary quality"""
        cache_key = f"{agent_id}:{hash(summary[:200])}"
        
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        word_count = len(summary.split())
        
        # Base scores for different agents
        base_scores = {
            1: 0.6,  # Extractive
            2: 0.7,  # Abstractive
            3: 0.65, # Bullet
            4: 0.5,  # TL;DR
            5: 0.75  # Detailed
        }
        
        score = base_scores.get(agent_id, 0.6)
        
        # Adjust for multimedia content
        if self.has_multimedia:
            # These agents are better at handling multimedia
            if agent_id in [2, 5]:  # Abstractive and Detailed
                score += 0.2
            elif agent_id in [3]:  # Bullet points
                score += 0.1
        
        # Quality adjustments
        if word_count < 15:
            score *= 0.7  # Too short
        elif 40 <= word_count <= 200:
            score *= 1.2  # Good length
        elif word_count > 300:
            score *= 0.8  # Too long
        
        # Check if summary references media (for multimedia content)
        if self.has_multimedia:
            summary_lower = summary.lower()
            media_keywords = ['image', 'picture', 'photo', 'video', 'visual', 'graphic', 'screenshot']
            if any(keyword in summary_lower for keyword in media_keywords):
                score += 0.15  # Bonus for referencing media
        
        # Add small variation and normalize
        score += random.uniform(-0.03, 0.03)
        score = max(0.1, min(0.95, score))
        
        self.evaluation_cache[cache_key] = score
        return score
    
    def search(self, iterations=50):
        """Run MCTS search"""
        print(f"\n🔍 Starting MCTS Search ({iterations} iterations)")
        print(f"   Multimedia: {'Yes' if self.has_multimedia else 'No'}")
        print(f"   Agents to evaluate: {len(self.root.untried_agents)}")
        
        for i in range(iterations):
            # Selection
            node = self.root
            while not node.is_fully_expanded() and node.children:
                node = node.best_child()
            
            # Expansion
            if node.untried_agents:
                agent_id = node.untried_agents.pop(random.randrange(len(node.untried_agents)))
                node = node.add_child(agent_id)
            
            # Simulation
            if node.agent_id is None and node.untried_agents:
                agent_id = random.choice(node.untried_agents)
            else:
                agent_id = node.agent_id
            
            agent_result = next((r for r in self.agent_results if r["agent_id"] == agent_id), None)
            if agent_result:
                value = self.evaluate_summary(agent_id, agent_result["summary"])
            else:
                value = 0.0
            
            # Backpropagation
            current = node
            while current is not None:
                current.update(value)
                current = current.parent
        
        # Find best agent
        if not self.root.children:
            return None, 0.0, {}
        
        best_child = max(self.root.children, key=lambda c: c.value if c.visits > 0 else 0)
        confidence = best_child.value
        
        # Get scores for all agents
        agent_scores = {}
        for child in self.root.children:
            if child.visits > 0:
                agent_scores[child.agent_id] = round(child.value, 3)
        
        print(f"🏆 MCTS Results: Agent {best_child.agent_id} wins with {confidence:.3f} confidence")
        
        return best_child.agent_id, confidence, agent_scores
    
    def get_tree_structure(self):
        """Get tree structure for visualization"""
        def node_to_dict(node):
            return {
                "agent_id": node.agent_id,
                "visits": node.visits,
                "value": round(node.value, 3) if node.visits > 0 else 0.0,
                "total_value": round(node.total_value, 3),
                "children": [node_to_dict(child) for child in node.children]
            }
        
        return node_to_dict(self.root)

# ------------------------------------------------------------------
# INITIALIZE AGENT REGISTRY
# ------------------------------------------------------------------
agent_registry = AgentRegistry()

# ------------------------------------------------------------------
# FLASK ROUTES
# ------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main page"""
    return render_template("index.html")

@app.route("/health")
def health():
    """Health check endpoint"""
    try:
        response = requests.get("http://localhost:11435/api/tags", timeout=5)
        ollama_status = "connected" if response.status_code == 200 else "error"
        
        # Check for required models
        models_response = requests.get("http://localhost:11435/api/tags", timeout=5)
        models_text = models_response.text if models_response.status_code == 200 else ""
        has_llama = TEXT_MODEL in models_text
        has_llava = VISION_MODEL in models_text
        
    except:
        ollama_status = "not_connected"
        has_llama = False
        has_llava = False
    
    agents = agent_registry.get_all_agents()
    
    return jsonify({
        "status": "ok",
        "ollama": ollama_status,
        "text_model_available": has_llama,
        "vision_model_available": has_llava,
        "text_model": TEXT_MODEL,
        "vision_model": VISION_MODEL,
        "agents_count": len(agents),
        "timestamp": datetime.now().isoformat()
    })

@app.route("/upload", methods=["POST"])
def upload_files():
    """Handle multimodal file uploads"""
    text = request.form.get("text", "")
    
    # Create session directory
    session_dir, session_id = create_session_directory()
    
    video_analysis = None
    image_analyses = []
    video_path = None
    image_paths = []
    
    print(f"\n📤 Starting file upload for session: {session_id}")
    
    # Handle video upload
    if 'video' in request.files:
        video_file = request.files['video']
        if video_file and video_file.filename != '' and allowed_file(video_file.filename):
            video_filename = secure_filename(f"video_{session_id}_{video_file.filename}")
            video_path = os.path.join(session_dir, video_filename)
            video_file.save(video_path)
            print(f"✅ Video saved: {video_filename}")
            
            # Analyze video content
            print("   Analyzing video content...")
            video_analysis = analyze_video_content(video_path)
            print(f"   Video analysis complete: {len(str(video_analysis))} chars")
    
    # Handle image uploads
    if 'images' in request.files:
        image_files = request.files.getlist('images')
        print(f"   Processing {len(image_files)} image(s)...")
        
        for i, img_file in enumerate(image_files):
            if img_file and img_file.filename != '' and allowed_file(img_file.filename):
                img_filename = secure_filename(f"image_{session_id}_{i}_{img_file.filename}")
                img_path = os.path.join(session_dir, img_filename)
                img_file.save(img_path)
                image_paths.append(img_path)
                print(f"   Image saved: {img_filename}")
                
                # Analyze image content
                print(f"   Analyzing image {i+1}...")
                img_analysis = analyze_image_with_vision(img_path)
                image_analyses.append(img_analysis)
                print(f"   Image analysis complete: {len(str(img_analysis))} chars")
    
    # Combine all analyses
    media_analyses = []
    if video_analysis:
        media_analyses.append(video_analysis)
    media_analyses.extend(image_analyses)
    
    print(f"📊 Upload Summary:")
    print(f"   Text: {len(text)} chars")
    print(f"   Video: {'Yes' if video_analysis else 'No'}")
    print(f"   Images: {len(image_analyses)}")
    print(f"   Total media analyses: {len(media_analyses)}")
    
    return jsonify({
        "success": True,
        "session_id": session_id,
        "session_dir": session_dir,
        "text": text,
        "media_analyses": media_analyses,
        "has_video": video_analysis is not None,
        "has_images": len(image_analyses) > 0,
        "total_media": len(media_analyses),
        "message": f"Upload complete. Processed {len(text)} chars of text and {len(media_analyses)} media items."
    })

@app.route("/summarize_multimodal", methods=["POST"])
def summarize_multimodal():
    """Run all agents on multimodal input"""
    data = request.json
    text = data.get("text", "")
    media_analyses = data.get("media_analyses", [])
    
    print(f"\n🎬 Starting Multimodal Summarization")
    print(f"   Text length: {len(text)} chars")
    print(f"   Media analyses: {len(media_analyses)} items")
    print(f"   Has multimedia: {'Yes' if media_analyses else 'No'}")
    
    agent_results = []
    agents = agent_registry.get_all_agents()
    has_multimedia = len(media_analyses) > 0
    
    for agent in agents:
        print(f"\n🤖 Running {agent.name} (ID: {agent.agent_id})...")
        start_time = time.time()
        
        try:
            summary = agent.generate_summary(text, media_analyses)
            end_time = time.time()
            duration = round((end_time - start_time) * 1000)
            
            word_count = len(summary.split())
            char_count = len(summary)
            
            agent_results.append({
                "agent_id": agent.agent_id,
                "agent_name": agent.name,
                "agent_icon": agent.agent_icon,
                "summary": summary,
                "duration": duration,
                "has_multimedia": has_multimedia,
                "word_count": word_count,
                "char_count": char_count,
                "words_per_second": round(word_count / (duration / 1000), 1) if duration > 0 else 0
            })
            
            print(f"   ✅ Agent {agent.agent_id} complete:")
            print(f"      Summary: {word_count} words, {char_count} chars")
            print(f"      Duration: {duration}ms ({agent_results[-1]['words_per_second']} words/sec)")
            print(f"      Preview: {summary[:100]}..." if len(summary) > 100 else f"      Summary: {summary}")
            
        except Exception as e:
            print(f"   ❌ Agent {agent.agent_id} failed: {str(e)}")
            agent_results.append({
                "agent_id": agent.agent_id,
                "agent_name": agent.name,
                "summary": f"Error: {str(e)}",
                "duration": 0,
                "error": True,
                "has_multimedia": has_multimedia
            })
    
    successful = len([r for r in agent_results if not r.get("error", False)])
    print(f"\n📈 Summary Generation Complete:")
    print(f"   Successful agents: {successful}/{len(agents)}")
    print(f"   Total words generated: {sum(r.get('word_count', 0) for r in agent_results)}")
    
    return jsonify({
        "agent_results": agent_results,
        "has_multimedia": has_multimedia,
        "total_agents": len(agents),
        "successful_agents": successful,
        "total_media_items": len(media_analyses)
    })

@app.route("/mcts_optimize", methods=["POST"])
def mcts_optimize():
    """Run MCTS optimization"""
    data = request.json
    agent_results = data.get("agent_results", [])
    has_multimedia = data.get("has_multimedia", False)
    simulations = data.get("simulations", 50)
    
    print(f"\n🌳 Starting MCTS Optimization")
    print(f"   Agent results: {len(agent_results)}")
    print(f"   Has multimedia: {has_multimedia}")
    print(f"   Simulations: {simulations}")
    
    # Filter out error results
    valid_results = [r for r in agent_results if not r.get("error", False)]
    if not valid_results:
        return jsonify({
            "error": "No valid agent results to optimize",
            "winning_agent": None,
            "winning_summary": "All agents failed to generate summaries",
            "confidence": 0.0,
            "agent_scores": {},
            "simulations_run": 0
        })
    
    # Run MCTS
    mcts = MCTSOptimizer(valid_results, has_multimedia)
    winning_agent, confidence, agent_scores = mcts.search(iterations=simulations)
    
    # Get tree structure
    tree_structure = mcts.get_tree_structure()
    
    # Find winning summary
    winning_summary = None
    for result in valid_results:
        if result["agent_id"] == winning_agent:
            winning_summary = result["summary"]
            break
    
    print(f"\n🏆 MCTS Optimization Complete:")
    print(f"   Winning agent: {winning_agent}")
    print(f"   Confidence: {confidence:.3f}")
    print(f"   Agent scores: {agent_scores}")
    
    return jsonify({
        "winning_agent": winning_agent,
        "winning_summary": winning_summary,
        "confidence": confidence,
        "agent_scores": agent_scores,
        "tree_structure": tree_structure,
        "simulations_run": simulations,
        "has_multimedia": has_multimedia
    })

@app.route("/cleanup_session", methods=["POST"])
def cleanup_session():
    """Clean up session files"""
    data = request.json
    session_id = data.get("session_id")
    
    if session_id:
        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        if os.path.exists(session_dir):
            try:
                import shutil
                shutil.rmtree(session_dir)
                print(f"✅ Cleaned up session: {session_id}")
                return jsonify({"success": True, "message": f"Session {session_id} cleaned up"})
            except Exception as e:
                print(f"❌ Cleanup error: {e}")
                return jsonify({"success": False, "error": str(e)})
    
    return jsonify({"success": False, "error": "No session ID provided"})

# ------------------------------------------------------------------
# ERROR HANDLERS
# ------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found", "message": str(e)}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error", "message": str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large", "message": "Maximum file size is 50MB"}), 413

# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎬 MULTIMODAL MCTS SUMMARIZER (WITH VISION)")
    print("="*60)
    print(f"Text Model: {TEXT_MODEL}")
    print(f"Vision Model: {VISION_MODEL}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Max file size: {MAX_CONTENT_LENGTH / (1024*1024)} MB")
    
    agents = agent_registry.get_all_agents()
    print(f"\n🤖 Available Agents ({len(agents)}):")
    for agent in agents:
        print(f"  {agent.agent_id}. {agent.name} - {agent.description}")
    
    print("\n" + "="*60)
    print("🚀 Server starting on http://localhost:5000")
    print("📊 Open browser console (F12) for detailed processing logs")
    print("="*60 + "\n")
    
    # Pull required models if not available
    try:
        print("Checking for required models...")
        response = requests.get("http://localhost:11435/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            if TEXT_MODEL not in " ".join(model_names):
                print(f"⚠️  Text model '{TEXT_MODEL}' not found. Please run: ollama pull {TEXT_MODEL}")
            
            if VISION_MODEL not in " ".join(model_names):
                print(f"⚠️  Vision model '{VISION_MODEL}' not found. Please run: ollama pull {VISION_MODEL}")
        else:
            print("⚠️  Cannot connect to Ollama. Make sure Ollama is running.")
    except:
        print("⚠️  Cannot check Ollama models. Make sure Ollama is running.")
    
    app.run(host="0.0.0.0", port=5000, debug=True)