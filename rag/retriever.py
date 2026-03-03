class PolicyRetriever:
    def query(self, state):
        if state["severity"] == "high":
            return "High severity: dispatch ambulance immediately."
        return "Monitor situation."
