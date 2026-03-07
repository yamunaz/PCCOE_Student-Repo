from typing import List, Optional, Any

class MCTSNode:
    def __init__(self, state: str, parent: Optional['MCTSNode'] = None, action: str = None):
        """
        initializes an MCTS node.
        
        Args:
            state: The content/code/reasoning of this node.
            parent: The parent node in the tree.
            action: The action description that led to this state (e.g., "generated initial code", "fixed syntax error").
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List['MCTSNode'] = []
        self.visits: int = 0
        self.value: float = 0.0
        self.is_terminal: bool = False
    
    def add_child(self, child_node: 'MCTSNode'):
        self.children.append(child_node)

    def ucb1(self, exploration_constant: float = 1.414) -> float:
        import math
        if self.visits == 0:
            return float('inf')
        
        # Upper Confidence Bound calculation
        # Q/N + c * sqrt(ln(parent_N) / N)
        parent_visits = self.parent.visits if self.parent else 1
        return (self.value / self.visits) + exploration_constant * math.sqrt(math.log(parent_visits) / self.visits)

    def __repr__(self):
        return f"<MCTSNode state='{self.state[:20]}...' visits={self.visits} value={self.value:.2f}>"
