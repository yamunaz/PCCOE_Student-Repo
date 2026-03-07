import math
import random
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from .node import MCTSNode
from .evaluator import Evaluator

class MCTSSearch:
    def __init__(self, max_simulations: int = 3, exploration_constant: float = 1.414):
        self.max_simulations = max_simulations
        self.exploration_constant = exploration_constant
        self.evaluator = Evaluator()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            temperature=0.7 
        )

    def search(self, initial_prompt: str) -> str:
        """
        Performs MCTS to find the best coding solution for the prompt.
        """
        root = MCTSNode(state=f"# Initial request: {initial_prompt}", action="Start")
        
        for _ in range(self.max_simulations):
            node = self._select(root)
            if not node.is_terminal: # Ideally check if solution is 'done'
                if node.visits > 0: # If already visited, expand
                    node = self._expand(node, initial_prompt)
                
                score = self._simulate(node, initial_prompt)
                self._backpropagate(node, score)
        
        # Select best child (highest visits or highest avg value)
        best_child = max(root.children, key=lambda c: c.visits, default=None)
        
        if best_child:
            return best_child.state
        else:
            return "Failed to generate a solution."

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.children:
            if any(child.visits == 0 for child in node.children):
                return next(child for child in node.children if child.visits == 0)
            
            # Select child with highest UCB1
            node = max(node.children, key=lambda c: c.ucb1(self.exploration_constant))
        return node

    def _expand(self, node: MCTSNode, prompt: str) -> MCTSNode:
        # Generate 'k' possible next steps / code variations
        # For simplicity in this v1, we just generate ONE expansion per visit unless we implement multi-branching prompts
        
        # Check depth limit?
        
        generation_prompt = f"""
        You are an AI coding agent.
        Parent State:
        {node.state}
        
        User Request:
        {prompt}
        
        Generate a refined, improved, or alternative implementation of the code.
        Focus on optimizing time complexity or handling edge cases better than the parent state.
        Output ONLY the code block.
        """
        
        try:
            response = self.llm.invoke(generation_prompt)
            new_state = response.content.strip()
            # Clean up markdown code blocks if necessary
            if new_state.startswith("```"):
                new_state = new_state.split("\n", 1)[1]
            if new_state.endswith("```"):
                new_state = new_state.rsplit("\n", 1)[0]
                
            child = MCTSNode(state=new_state, parent=node, action="Refined Code")
            node.add_child(child)
            return child
        except Exception as e:
            print(f"Expansion Error: {e}")
            return node

    def _simulate(self, node: MCTSNode, prompt: str) -> float:
        return self.evaluator.evaluate(node.state, prompt)

    def _backpropagate(self, node: MCTSNode, result: float):
        while node:
            node.visits += 1
            node.value += result
            node = node.parent
