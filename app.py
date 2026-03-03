import time
from collections import deque, Counter

from stream.video_stream import VideoStream
from agent.decision_agent import EmergencyDecisionAgent


VIDEO_SOURCE = "data/videos/accident.mp4"

CONFIRM_FRAMES = 5          # accident must be detected in 5 frames continuously
COOLDOWN_SECONDS = 20       # after alert, wait 20 seconds before next alert
MAX_FRAMES_TO_PROCESS = 800 # stop after processing this many frames (for video file)


def is_incident(scene):
    """
    Incident trigger logic (industry-style).
    """
    if scene["severity"] in ["medium", "high"]:
        return True
    if scene["fire_risk"] == "yes":
        return True
    if scene["injuries"] == "yes":
        return True
    if scene["road_blocked"] == "yes":
        return True
    return False


def final_decision(scene, plan):
    """
    Convert MCTS plan into final emergency response actions.
    """
    actions = set(plan)

    # Force critical actions if conditions exist
    if scene["fire_risk"] == "yes":
        actions.add("call_fire_brigade")

    if scene["injuries"] == "yes":
        actions.add("dispatch_ambulance")

    if scene["severity"] in ["medium", "high"]:
        actions.add("dispatch_police")

    if scene["road_blocked"] == "yes":
        actions.add("reroute_traffic")

    # remove useless waits
    actions.discard("wait")

    # if nothing selected
    if not actions:
        return ["monitor"]

    # Sort output for clean display
    priority = [
        "dispatch_ambulance",
        "call_fire_brigade",
        "dispatch_police",
        "reroute_traffic"
    ]

    final = [a for a in priority if a in actions]
    return final


def main():
    stream = VideoStream(VIDEO_SOURCE)
    agent = EmergencyDecisionAgent()

    incident_window = deque(maxlen=CONFIRM_FRAMES)

    last_alert_time = 0
    frame_count = 0

    print(f"[INFO] Connecting to video source: {VIDEO_SOURCE}")

    for frame in stream.frames():
        frame_count += 1

        if frame_count > MAX_FRAMES_TO_PROCESS:
            print("[INFO] Video processed. Stopping.")
            break

        plan, state = agent.step(frame)
        scene = state.scene

        # Store incident flag
        incident_window.append(is_incident(scene))

        # Check if incident confirmed
        confirmed = len(incident_window) == CONFIRM_FRAMES and all(incident_window)

        # Cooldown check
        now = time.time()
        in_cooldown = (now - last_alert_time) < COOLDOWN_SECONDS

        if confirmed and not in_cooldown:
            last_alert_time = now

            final_actions = final_decision(scene, plan)

            print("\n🚨 INCIDENT CONFIRMED!")
            print("SCENE:", scene)
            print("POLICY:", state.policy)
            print("MCTS PLAN:", plan)
            print("FINAL DECISION:", final_actions)
            print("-" * 60)

        else:
            # Normal traffic flow output (optional)
            if frame_count % 80 == 0:  # print once in a while
                print("✅ Smooth Traffic Flow... Monitoring ongoing.")

    print("[INFO] Finished processing.")


if __name__ == "__main__":
    main()
