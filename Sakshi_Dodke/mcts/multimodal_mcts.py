import random
import math
import json
from typing import List, Dict, Tuple, Optional, Any
from .mcts_node import MCTSNode
import cv2
import numpy as np
from PIL import Image
import os

class MultimodalMCTSSearch:
    """Multimodal Monte Carlo Tree Search for summary selection"""
    
    def __init__(self, agent_results: List[Dict], media_data: Dict, 
                 exploration_weight: float = 1.414):
        """
        Initialize multimodal MCTS search
        
        Args:
            agent_results: List of dictionaries with 'agent_id' and 'summary'
            media_data: Dictionary containing text, video, and image data
            exploration_weight: Weight for exploration in UCT formula
        """
        self.agent_results = agent_results
        self.media_data = media_data
        self.exploration_weight = exploration_weight
        self.root = MCTSNode()
        
        # Initialize untried agents at root
        self.root.untried_agents = [r["agent_id"] for r in agent_results]
        
        # Cache for multimodal evaluations
        self.evaluation_cache: Dict[Tuple[int, str], float] = {}
        
        # Extract media features for evaluation
        self.text_features = self.extract_text_features(media_data.get('text', ''))
        self.video_features = self.extract_video_features(media_data.get('video_path'))
        self.image_features = self.extract_image_features(media_data.get('images', []))
        self.media_type = self.determine_media_type()
        
        print(f"   Media Type: {self.media_type}")
        print(f"   Text features: {self.text_features.get('word_count', 0)} words")
        print(f"   Video features: {'Yes' if self.video_features else 'No'}")
        print(f"   Image features: {len(self.image_features)} images")
    
    def determine_media_type(self) -> str:
        """Determine the dominant media type"""
        has_text = bool(self.media_data.get('text', '').strip())
        has_video = bool(self.media_data.get('video_path'))
        has_images = len(self.media_data.get('images', [])) > 0
        
        if has_video and has_images and has_text:
            return "full_multimodal"
        elif has_video and has_text:
            return "video_text"
        elif has_images and has_text:
            return "image_text"
        elif has_video:
            return "video_only"
        elif has_images:
            return "images_only"
        else:
            return "text_only"
    
    def extract_text_features(self, text: str) -> Dict:
        """Extract features from text"""
        if not text or not text.strip():
            return {"word_count": 0, "sentence_count": 0, "avg_word_length": 0, "keywords": []}
        
        words = [w for w in text.split() if w.strip()]
        sentences = [s for s in text.split('.') if s.strip()]
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': sum(len(w) for w in words) / max(len(words), 1),
            'keywords': self.extract_keywords(text)[:10]
        }
    
    def extract_video_features(self, video_path: Optional[str]) -> Dict:
        """Extract features from video"""
        if not video_path or not os.path.exists(video_path):
            return {}
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {}
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Extract key frames (simplified)
            key_frames = []
            sample_rate = max(1, frame_count // 10)  # Get ~10 frames
            
            for i in range(0, min(frame_count, 100), sample_rate):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Convert to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Calculate basic features
                    brightness = np.mean(frame_rgb) / 255.0
                    # Calculate color variance
                    color_std = np.std(frame_rgb, axis=(0, 1)).mean() / 255.0
                    
                    key_frames.append({
                        'frame_num': i,
                        'brightness': brightness,
                        'color_variance': color_std,
                        'resolution': f"{width}x{height}"
                    })
            
            cap.release()
            
            if key_frames:
                avg_brightness = np.mean([f['brightness'] for f in key_frames])
                avg_color_var = np.mean([f['color_variance'] for f in key_frames])
            else:
                avg_brightness = 0.5
                avg_color_var = 0.1
            
            return {
                'frame_count': frame_count,
                'duration': duration,
                'fps': fps,
                'resolution': f"{width}x{height}",
                'key_frames': len(key_frames),
                'avg_brightness': avg_brightness,
                'avg_color_variance': avg_color_var,
                'has_video': True
            }
            
        except Exception as e:
            print(f"   Video feature extraction error: {e}")
            return {}
    
    def extract_image_features(self, image_paths: List[str]) -> List[Dict]:
        """Extract features from images"""
        features = []
        
        for img_path in image_paths[:5]:  # Limit to 5 images for performance
            try:
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    img_array = np.array(img)
                    
                    # Basic features
                    img_features = {
                        'size': img.size,
                        'mode': img.mode,
                        'aspect_ratio': img.size[0] / max(img.size[1], 1)
                    }
                    
                    # Calculate image statistics
                    if len(img_array.shape) == 3:  # Color image
                        avg_color = np.mean(img_array, axis=(0, 1))
                        brightness = np.mean(img_array) / 255.0
                        color_std = np.std(img_array, axis=(0, 1)).mean() / 255.0
                        img_features.update({
                            'avg_color': avg_color.tolist(),
                            'brightness': brightness,
                            'color_contrast': color_std,
                            'is_color': True
                        })
                    else:  # Grayscale
                        brightness = np.mean(img_array) / 255.0
                        contrast = np.std(img_array) / 255.0
                        img_features.update({
                            'brightness': brightness,
                            'contrast': contrast,
                            'is_color': False
                        })
                    
                    features.append(img_features)
                    img.close()
                    
            except Exception as e:
                print(f"   Image feature extraction error: {e}")
                continue
        
        return features
    
    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """Extract keywords from text"""
        if not text:
            return []
        
        # Simple keyword extraction
        words = text.lower().split()
        # Remove common stop words
        stop_words = {'the', 'and', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        filtered_words = [w for w in words if len(w) > 3 and w not in stop_words]
        
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:num_keywords]]
    
    def evaluate_multimodal_summary(self, agent_id: int, summary: str) -> float:
        """
        Evaluate a multimodal summary based on various criteria
        """
        cache_key = (agent_id, hash(summary[:200]))
        
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        # Base metrics
        word_count = len(summary.split())
        char_count = len(summary)
        
        # Initialize evaluation scores
        scores = {
            'content_coverage': 0.5,
            'conciseness': 0.5,
            'relevance': 0.5,
            'coherence': 0.5,
            'multimodal_integration': 0.3,
            'structure': 0.5
        }
        
        # Agent-specific strengths
        agent_strengths = {
            1: {'content_coverage': 0.7, 'relevance': 0.6, 'conciseness': 0.4},  # Extractive
            2: {'coherence': 0.8, 'conciseness': 0.7, 'multimodal_integration': 0.6},  # Abstractive
            3: {'structure': 0.9, 'content_coverage': 0.7, 'conciseness': 0.6},  # Bullet
            4: {'conciseness': 0.9, 'structure': 0.7, 'relevance': 0.6},  # TL;DR
            5: {'content_coverage': 0.9, 'relevance': 0.8, 'multimodal_integration': 0.7}  # Detailed
        }
        
        # Apply agent-specific strengths
        if agent_id in agent_strengths:
            for key, value in agent_strengths[agent_id].items():
                scores[key] = value
        
        # Adjust based on media type
        if self.media_type == "full_multimodal":
            # Full multimodal benefits from detailed or abstractive
            if agent_id in [2, 5]:  # Abstractive or Detailed
                scores['multimodal_integration'] += 0.2
                scores['content_coverage'] += 0.1
        
        elif self.media_type == "video_text":
            # Video+text benefits from detailed or bullet points
            if agent_id in [3, 5]:  # Bullet or Detailed
                scores['structure'] += 0.15
                scores['content_coverage'] += 0.1
        
        elif self.media_type == "image_text":
            # Image+text benefits from abstractive or bullet points
            if agent_id in [2, 3]:  # Abstractive or Bullet
                scores['multimodal_integration'] += 0.15
        
        elif self.media_type == "video_only":
            # Video only needs descriptive summaries
            if agent_id in [2, 5]:  # Abstractive or Detailed
                scores['content_coverage'] += 0.2
        
        elif self.media_type == "images_only":
            # Images only benefit from structured output
            if agent_id in [3, 5]:  # Bullet or Detailed
                scores['structure'] += 0.2
        
        # Quality checks
        if word_count < 10:
            scores['content_coverage'] *= 0.7  # Penalize very short summaries
        elif word_count > 200:
            scores['conciseness'] *= 0.7  # Penalize very long summaries
        
        # Check for multimodal references
        summary_lower = summary.lower()
        has_video_ref = any(word in summary_lower for word in ['video', 'film', 'clip', 'footage', 'recording'])
        has_image_ref = any(word in summary_lower for word in ['image', 'picture', 'photo', 'graphic', 'visual'])
        
        if self.media_data.get('video_path') and has_video_ref:
            scores['multimodal_integration'] += 0.1
        
        if self.media_data.get('images') and has_image_ref:
            scores['multimodal_integration'] += 0.1
        
        # Calculate final weighted score
        weights = {
            'content_coverage': 0.25,
            'conciseness': 0.20,
            'relevance': 0.15,
            'coherence': 0.15,
            'multimodal_integration': 0.15,
            'structure': 0.10
        }
        
        final_score = sum(scores[k] * weights[k] for k in scores)
        
        # Add small random variation
        final_score += random.uniform(-0.03, 0.03)
        
        # Normalize to [0.1, 0.95]
        final_score = max(0.1, min(0.95, final_score))
        
        # Cache the result
        self.evaluation_cache[cache_key] = final_score
        
        # Debug logging
        if random.random() < 0.1:  # Log 10% of evaluations
            print(f"   Agent {agent_id}: Score={final_score:.3f}, Words={word_count}")
        
        return final_score
    
    def selection(self, node: MCTSNode) -> MCTSNode:
        """Selection phase: traverse tree using UCT"""
        while not node.is_leaf():
            if not node.is_fully_expanded():
                return node  # Node can be expanded
            node = node.best_child(self.exploration_weight)
        return node
    
    def expansion(self, node: MCTSNode) -> MCTSNode:
        """Expansion phase: add a new child node"""
        if node.untried_agents:
            agent_id = node.untried_agents.pop(random.randrange(len(node.untried_agents)))
            return node.add_child(agent_id)
        return node
    
    def simulation(self, node: MCTSNode) -> float:
        """Simulation phase: random rollout"""
        if node.agent_id is None and node.untried_agents:
            agent_id = random.choice(node.untried_agents)
        else:
            agent_id = node.agent_id
        
        # Find the agent's summary
        agent_result = next((r for r in self.agent_results if r["agent_id"] == agent_id), None)
        if not agent_result:
            return 0.0
        
        # Evaluate the summary with multimodal consideration
        return self.evaluate_multimodal_summary(agent_id, agent_result["summary"])
    
    def backpropagation(self, node: MCTSNode, value: float):
        """Backpropagation phase: update node values"""
        while node is not None:
            node.update(value)
            node = node.parent
    
    def search(self, iterations: int = 100) -> Tuple[Optional[int], float, Dict[int, float]]:
        """Run MCTS search"""
        print(f"   Running {iterations} MCTS simulations...")
        
        for i in range(iterations):
            # 1. Selection
            node = self.selection(self.root)
            
            # 2. Expansion
            node = self.expansion(node)
            
            # 3. Simulation
            value = self.simulation(node)
            
            # 4. Backpropagation
            self.backpropagation(node, value)
            
            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"   Completed {i + 1}/{iterations} simulations")
        
        # Find best agent
        if not self.root.children:
            return None, 0.0, {}
        
        # Get scores for all agents
        agent_scores = {}
        for child in self.root.children:
            if child.visits > 0:
                agent_scores[child.agent_id] = round(child.value, 3)
        
        # Find agent with highest average value
        best_child = max(self.root.children, key=lambda c: c.value if c.visits > 0 else 0)
        confidence = best_child.value
        
        print(f"   Best agent: {best_child.agent_id} (Score: {confidence:.3f})")
        
        return best_child.agent_id, confidence, agent_scores
    
    def get_tree_structure(self) -> Dict:
        """Get tree structure for visualization"""
        def node_to_dict(node: MCTSNode) -> Dict:
            return {
                "agent_id": node.agent_id,
                "visits": node.visits,
                "value": round(node.value, 3) if node.visits > 0 else 0.0,
                "total_value": round(node.total_value, 3),
                "children": [node_to_dict(child) for child in node.children]
            }
        
        return node_to_dict(self.root)
    
    def get_media_analysis(self) -> Dict:
        """Get analysis of media content"""
        return {
            "media_type": self.media_type,
            "text_analysis": {
                "word_count": self.text_features.get('word_count', 0),
                "sentence_count": self.text_features.get('sentence_count', 0),
                "keywords": self.text_features.get('keywords', [])[:5]
            },
            "video_analysis": self.video_features if self.video_features else {"has_video": False},
            "image_analysis": {
                "image_count": len(self.image_features),
                "sample_features": self.image_features[:2] if self.image_features else []
            },
            "modality_score": self.calculate_modality_score()
        }
    
    def calculate_modality_score(self) -> float:
        """Calculate a score representing multimodal richness"""
        score = 0.0
        
        # Text contribution
        if self.text_features.get('word_count', 0) > 50:
            score += 0.3
        
        # Video contribution
        if self.video_features:
            score += 0.4
        
        # Image contribution
        if len(self.image_features) > 0:
            score += min(0.3, 0.1 * len(self.image_features))
        
        return min(1.0, score)