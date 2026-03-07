import copy


class IncidentState:
    def __init__(self, scene, policy, actions=None, step_count=0):
        self.scene = scene
        self.policy = policy
        self.actions = actions if actions is not None else []
        self.step_count = step_count

    def get_possible_actions(self):
        """
        Always return actions list.
        If empty, return default safe action list.
        """
        if not self.actions:
            return [
                "dispatch_ambulance",
                "dispatch_police",
                "call_fire_brigade",
                "reroute_traffic",
                "wait"
            ]
        return self.actions

    def apply_action(self, action):
        """
        Simulate state changes after applying an emergency action.
        """

        new_scene = copy.deepcopy(self.scene)

        severity = new_scene.get("severity", "low")
        injuries = new_scene.get("injuries", "no")
        fire_risk = new_scene.get("fire_risk", "no")
        road_blocked = new_scene.get("road_blocked", "no")

        # --- Action effects ---
        if action == "call_fire_brigade":
            if fire_risk == "yes":
                new_scene["fire_risk"] = "no"
                if severity == "high":
                    new_scene["severity"] = "medium"

        elif action == "dispatch_ambulance":
            if injuries == "yes":
                new_scene["injuries"] = "no"
                if severity == "high":
                    new_scene["severity"] = "medium"
                elif severity == "medium":
                    new_scene["severity"] = "low"

        elif action == "dispatch_police":
            # police helps manage road
            if road_blocked == "yes":
                new_scene["road_blocked"] = "no"

        elif action == "reroute_traffic":
            new_scene["road_blocked"] = "no"

        elif action == "wait":
            # waiting makes situation worse if fire exists
            if fire_risk == "yes":
                new_scene["severity"] = "high"
                new_scene["injuries"] = "yes"

        return IncidentState(
            scene=new_scene,
            policy=self.policy,
            actions=self.actions,
            step_count=self.step_count + 1
        )
