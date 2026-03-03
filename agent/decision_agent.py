from mllm.local_vision import LocalVisionModel
from rag.retriever import PolicyRetriever
from mcts.planner import MCTSPlanner
from agent.state import IncidentState


class EmergencyDecisionAgent:
    def __init__(self):
        self.vision = LocalVisionModel()
        self.rag = PolicyRetriever()

        # Possible emergency actions
        self.actions = [
            "dispatch_ambulance",
            "dispatch_police",
            "call_fire_brigade",
            "reroute_traffic",
            "wait"
        ]

        # Advanced MCTS planner
        self.mcts = MCTSPlanner(rollout_depth=5, simulations=200)


    def step(self, frame):
        # Step 1: Vision model analyzes the frame
        scene = self.vision.analyze(frame)

        # Step 2: RAG retrieves policy
        policy = self.rag.query(scene)

        # Step 3: Create incident state object
        state = IncidentState(scene, policy, self.actions)

        # Step 4: MCTS generates best multi-step plan
        plan = self.mcts.search(state)

        return plan, state
