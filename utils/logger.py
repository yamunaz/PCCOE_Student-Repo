import datetime


def log_event(scene, policy, action, logfile="logs/events.log"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_text = (
        f"[{timestamp}] SCENE={scene} | POLICY={policy} | ACTION={action}\n"
    )

    with open(logfile, "a", encoding="utf-8") as f:
        f.write(log_text)
