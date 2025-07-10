import logging
from pathlib import Path
import subprocess

from zenml import step

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@step
def push_docker_image_to_docker_hub():
    script_path = Path(__file__).parent.parent / "images_push_to_dockerhub.sh"

    if not script_path.exists():
        logging.error(f"Bash script not found: {script_path}")
        raise FileNotFoundError(f"Script {script_path} does not exist.")

    try:
        # Make script executable
        subprocess.run(["chmod", "+x", str(script_path)], check=True, capture_output=True, text=True)
        logging.info(f"Made script executable: {script_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"chmod failed: {e.stderr.strip()}")
        raise

    try:
        # Execute the script
        result = subprocess.run([str(script_path)], check=True, capture_output=True, text=True)
        logging.info(f"Script executed successfully:\n{result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Script execution failed:\n{e.stderr.strip()}")
        raise
