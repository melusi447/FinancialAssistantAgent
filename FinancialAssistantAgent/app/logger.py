import datetime
import os

LOG_PATH = "logs/chat.log"
os.makedirs("logs", exist_ok=True)

def log_interaction(user_id: str, user_message: str, ai_response: str | None):
    timestamp = datetime.datetime.utcnow().isoformat()
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} | USER={user_id} | MSG={user_message} | AI={ai_response}\n")
