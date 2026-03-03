def evaluate(action, state):
    score = 0
    if action == "dispatch_ambulance" and state.scene["injuries"] == "yes":
        score += 10
    if action == "wait":
        score -= 5
    return score
