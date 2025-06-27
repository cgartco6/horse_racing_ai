# Configuration settings
import os
from dotenv import load_dotenv

load_dotenv()

# Telegram settings
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHANNEL_ID = os.getenv("CHANNEL_ID")

# AI Settings
TIMEZONE = "Europe/London"
SCRATCH_PROBABILITY = 0.05  # 5% chance of scratching
MODEL_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 7,
    'random_state': 42
}

# Simulation Parameters
TRACKS = ["Ascot", "Goodwood", "York", "Newmarket", "Doncaster"]
HORSE_POOL_SIZE = 200
JOCKEY_POOL_SIZE = 50
TRAINER_POOL_SIZE = 30
