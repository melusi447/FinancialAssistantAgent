"""
Frontend Runner
Starts the Gradio UI from ui/app.py.

Usage
-----
    python run_frontend.py
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/frontend.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting Financial Assistant Frontend...")

    try:
        from ui.app import main as start_ui
        start_ui()
    except ImportError as exc:
        logger.error(f"Import failed: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Frontend error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()