import math
import random
import copy


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action

        self.children = []
        self.visits = 0
        self.total_reward = 0.0

        # actions not tried yet
        self.untried_actions = state.get_possible_actions()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, exploration_weight=1.4):
        """
        UCB1 formula:
        score = exploitation + exploration
        """
        best_score = -float("inf")
        best_node = None

        for child in self.children:
            if child.visits == 0:
                score = float("inf")
            else:
                exploitation = child.total_reward / child.visits
                exploration = exploration_weight * math.sqrt(
                    math.log(self.visits + 1) / (child.visits + 1e-9)
                )
                score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_node = child

        return best_node

    def expand(self):
        """
        Take one untried action and create a new child node.
        """
        if not self.untried_actions:
            return None

        action = self.untried_actions.pop()
        next_state = self.state.apply_action(action)

        child_node = MCTSNode(state=next_state, parent=self, action=action)
        self.children.append(child_node)

        return child_node

    def backpropagate(self, reward):
        """
        Backpropagate reward to root.
        """
        self.visits += 1
        self.total_reward += reward

        if self.parent:
            self.parent.backpropagate(reward)


class MCTSPlanner:
    """
    Advanced MCTS Planner for Emergency Decision Agent.

    Features:
    - Multi-step planning (returns sequence of actions)
    - UCB1 selection
    - Guided rollouts (not pure random)
    - Reward shaping based on how much the situation improves
    """

    def __init__(self, rollout_depth=5, simulations=300, exploration_weight=1.4):
        self.rollout_depth = rollout_depth
        self.simulations = simulations
        self.exploration_weight = exploration_weight

    def search(self, root_state):
        """
        Runs MCTS and returns best plan (sequence of actions).
        """
        root = MCTSNode(root_state)

        # If no actions possible, just wait
        if not root.untried_actions:
            return ["wait"]

        for _ in range(self.simulations):

            node = root

            # 1. Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.exploration_weight)

            # 2. Expansion
            if not node.is_fully_expanded():
                expanded = node.expand()
                if expanded is not None:
                    node = expanded

            # 3. Simulation (Rollout)
            reward = self.rollout(node.state)

            # 4. Backpropagation
            node.backpropagate(reward)

        # Choose best child based on average reward
        if not root.children:
            return ["wait"]

        best_child = max(
            root.children,
            key=lambda c: c.total_reward / (c.visits + 1e-9)
        )

        # Build plan (multi-step)
        plan = []
        current = best_child

        while current is not None and current.action is not None:
            plan.append(current.action)

            # continue down the best path greedily (no exploration)
            if current.children:
                current = max(
                    current.children,
                    key=lambda c: c.total_reward / (c.visits + 1e-9)
                )
            else:
                break

        if not plan:
            return ["wait"]

        return plan

    def rollout(self, state):
        """
        Rollout simulation (guided instead of fully random).
        """
        current_state = copy.deepcopy(state)
        start_scene = copy.deepcopy(current_state.scene)

        for _ in range(self.rollout_depth):

            actions = current_state.get_possible_actions()

            # IMPORTANT: prevent random.choice([]) crash
            if not actions:
                break

            action = self.choose_best_rollout_action(current_state.scene, actions)

            current_state = current_state.apply_action(action)

        # final reward after rollout
        return self.evaluate_state(start_scene, current_state.scene)

    def choose_best_rollout_action(self, scene, actions):
        """
        Smart rollout policy:
        - If fire exists -> prioritize fire brigade
        - If injuries -> prioritize ambulance
        - If road blocked -> prioritize police / reroute
        """

        severity = scene.get("severity", "low")
        injuries = scene.get("injuries", "no")
        fire_risk = scene.get("fire_risk", "no")
        road_blocked = scene.get("road_blocked", "no")

        # priority rules
        if fire_risk == "yes" and "call_fire_brigade" in actions:
            return "call_fire_brigade"

        if injuries == "yes" and "dispatch_ambulance" in actions:
            return "dispatch_ambulance"

        if road_blocked == "yes":
            if "reroute_traffic" in actions:
                return "reroute_traffic"
            if "dispatch_police" in actions:
                return "dispatch_police"

        # if severity high, ambulance is always good
        if severity == "high" and "dispatch_ambulance" in actions:
            return "dispatch_ambulance"

        # fallback random choice
        return random.choice(actions)

    def evaluate_state(self, start_scene, end_scene):
        """
        Reward shaping:
        Reward is high if situation improves.
        Penalize if situation worsens.
        """

        def severity_score(level):
            if level == "high":
                return 3
            if level == "medium":
                return 2
            return 1

        start_sev = severity_score(start_scene.get("severity", "low"))
        end_sev = severity_score(end_scene.get("severity", "low"))

        start_inj = 1 if start_scene.get("injuries", "no") == "yes" else 0
        end_inj = 1 if end_scene.get("injuries", "no") == "yes" else 0

        start_fire = 1 if start_scene.get("fire_risk", "no") == "yes" else 0
        end_fire = 1 if end_scene.get("fire_risk", "no") == "yes" else 0

        start_block = 1 if start_scene.get("road_blocked", "no") == "yes" else 0
        end_block = 1 if end_scene.get("road_blocked", "no") == "yes" else 0

        reward = 0

        # improvement rewards
        reward += (start_sev - end_sev) * 20
        reward += (start_inj - end_inj) * 40
        reward += (start_fire - end_fire) * 50
        reward += (start_block - end_block) * 15

        # penalty if still dangerous
        if end_fire == 1:
            reward -= 30
        if end_inj == 1:
            reward -= 20
        if end_sev == 3:
            reward -= 15
        if end_block == 1:
            reward -= 10

        # bonus if situation is safe
        if end_sev == 1 and end_fire == 0 and end_inj == 0 and end_block == 0:
            reward += 50

        return reward
