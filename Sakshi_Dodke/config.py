# Configuration file for multimodal summarization system

# Ollama Configuration
OLLAMA_URL = "http://localhost:11435/api/generate"
MODEL_NAME = "llama3.2:latest"

# Multimodal Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'mkv'}
UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max

# MCTS Configuration
MCTS_EXPLORATION_WEIGHT = 1.414  # sqrt(2)
MCTS_DEFAULT_SIMULATIONS = 50
MCTS_MAX_SIMULATIONS = 200

# Agent Configuration
AGENT_TIMEOUT = 120  # seconds for multimodal
MAX_TOKENS = {
    "default": 200,
    "extractive": 150,
    "detailed": 250,
    "tldr": 100,
    "multimodal": 300
}

# Server Configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 7860
DEBUG_MODE = True

# Media Processing
MAX_IMAGES = 10
VIDEO_FRAME_SAMPLE_RATE = 30  # Sample every Nth frame