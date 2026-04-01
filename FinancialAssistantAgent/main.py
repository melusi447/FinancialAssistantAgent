"""
main.py
Full system launcher — starts backend and frontend as separate processes.
"""

import os
import sys
import time
import logging
import subprocess

# Force UTF-8 output on Windows to avoid emoji encoding errors
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/main.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def wait_for_backend(url: str, timeout: int = 30) -> bool:
    """Poll the backend health endpoint until it responds or times out."""
    import urllib.request
    import urllib.error
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def start_backend() -> subprocess.Popen:
    logger.info("Starting backend service...")
    proc = subprocess.Popen(
        [sys.executable, "run_backend.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    return proc


def start_frontend() -> subprocess.Popen:
    logger.info("Starting frontend service...")
    proc = subprocess.Popen(
        [sys.executable, "run_frontend.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    return proc


def stream_logs(proc: subprocess.Popen, label: str) -> None:
    """Print subprocess output with a label prefix (runs in a thread)."""
    import threading

    def _reader():
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(f"[{label}] {line}", flush=True)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()


def main():
    logger.info("Starting Financial Assistant AI...")

    from config import config
    backend_url = config.BACKEND_URL

    # ── Start backend ──────────────────────────────────────────────────────
    backend_proc = start_backend()
    stream_logs(backend_proc, "BACKEND")

    logger.info(f"Waiting for backend at {backend_url} ...")
    if wait_for_backend(backend_url, timeout=30):
        logger.info(f"Backend ready at {backend_url}")
    else:
        logger.warning(
            "Backend did not respond within 30s. "
            "It may still be loading the model. Continuing..."
        )

    # ── Start frontend ─────────────────────────────────────────────────────
    frontend_proc = start_frontend()
    stream_logs(frontend_proc, "FRONTEND")

    logger.info("=" * 50)
    logger.info("  Financial Assistant AI is running")
    logger.info(f"  API      : {backend_url}")
    logger.info(f"  API Docs : {backend_url}/docs")
    logger.info(f"  UI       : http://127.0.0.1:{config.FRONTEND_PORT}")
    logger.info("  Press Ctrl+C to stop all services")
    logger.info("=" * 50)

    # ── Keep alive — wait for both processes ───────────────────────────────
    frontend_failures = 0
    MAX_FRONTEND_FAILURES = 3

    try:
        while True:
            if backend_proc.poll() is not None:
                logger.error("Backend process exited unexpectedly.")
                break

            if frontend_proc.poll() is not None:
                frontend_failures += 1
                if frontend_failures >= MAX_FRONTEND_FAILURES:
                    logger.error(
                        f"Frontend has failed {frontend_failures} times - giving up. "
                        "Check logs/frontend.log for the error. "
                        "Most likely fix: pip install --upgrade gradio"
                    )
                    logger.info(f"Backend is still running at {backend_url}")
                    logger.info(f"API docs available at {backend_url}/docs")
                    while backend_proc.poll() is None:
                        time.sleep(5)
                    break
                else:
                    logger.warning(
                        f"Frontend exited (attempt {frontend_failures}/{MAX_FRONTEND_FAILURES}) - restarting..."
                    )
                    frontend_proc = start_frontend()
                    stream_logs(frontend_proc, "FRONTEND")

            time.sleep(2)

    except KeyboardInterrupt:
        logger.info("Shutting down...")

    finally:
        for proc, name in [(backend_proc, "backend"), (frontend_proc, "frontend")]:
            if proc.poll() is None:
                logger.info(f"Stopping {name}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        logger.info("All services stopped.")


if __name__ == "__main__":
    main()




