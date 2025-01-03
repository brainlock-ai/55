import os
#pip install GitPython
import git
import sys
import aiohttp
import asyncio
import threading
import subprocess
from packaging import version

# Function to manually load environment variables from a .env file
def load_env_file(file_path):
    if os.path.exists(file_path):
        with open(file_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

# Load environment variables from .env file
env_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cifar', '.env')
load_env_file(env_file_path)

# Get version from __init__.py
init_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "__init__.py")
with open(init_path) as f:
    for line in f:
        if line.startswith("__version__"):
            __version__ = line.split('=')[-1].replace('"', '').replace(' ', '')
            print(f"[Auto-Update] Current version: {__version__}")
            break

class AutoUpdate(threading.Thread):
    def __init__(self, container_image="cml_client_cifar_10_8_bit", check_interval=600):
        """
        :param container_image: The name of the Docker image to rebuild/restart
        :param check_interval: How often (seconds) to check for updates (default 10 minutes)
        """
        super().__init__()
        self.daemon = True
        self.container_image = container_image
        self.check_interval = check_interval

        # Initialize Git repo
        try:
            self.repo = git.Repo(search_parent_directories=True)
        except Exception as e:
            print(f"[Auto-Update] Error initializing Git repository: {e}")
            sys.exit(1)

    async def get_remote_version(self):
        """
        Asynchronously fetch the remote version string from GitHub (or any other URL).
        """
        url = "https://raw.githubusercontent.com/brainlock-ai/55/main/FHE/__init__.py"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    response.raise_for_status()
                    content = await response.text()

            # Parse __version__ line from the content
            for line in content.split("\n"):
                if line.startswith("__version__"):
                    version_info = line.split("=")[1].strip().strip(" \"'")
                    return version_info

            return None
        except Exception as e:
            print(f"[Auto-Update] Error fetching remote version: {e}")
            return None

    async def check_versions(self):
        remote_version_str = await self.get_remote_version()
        if not remote_version_str:
            return False

        local_version_obj = version.parse(__version__)
        remote_version_obj = version.parse(remote_version_str)
        print(
            f"[Auto-Update] Remote version: {remote_version_str}, "
            f"Local version: {__version__}"
        )

        return remote_version_obj > local_version_obj

    def update(self):
        """
        Perform the update:
        1. git pull
        2. Rebuild Docker image
        3. Stop and remove existing container
        4. Start new container
        5. sys.exit(0)
        """
        repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        print("[Auto-Update] Pulling latest changes from GitHub...")
        try:
            subprocess.check_call(["git", "pull"], cwd=repo_dir)
        except subprocess.CalledProcessError as e:
            subprocess.run(["git", "stash"], cwd=repo_dir)
            try:
                subprocess.check_call(["git", "pull"], cwd=repo_dir)
            except subprocess.CalledProcessError as e2:
                pass
            print(f"[Auto-Update] Error during 'git pull': {e}")
            return

        print("[Auto-Update] Rebuilding Docker image...")
        try:
            subprocess.check_call([
                "python3",
                "build_docker_client_image.py"
            ], cwd=os.path.join(repo_dir, "FHE", "cifar"))
        except subprocess.CalledProcessError as e:
            print(f"[Auto-Update] Error building Docker image: {e}")
            return
        
        print("[Auto-Update] Creating database volume if it doesn't yet exit...")
        try:
            subprocess.run(
                ["docker", "volume", "create", "postgres_data"],
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"[Auto-Update] Error creating database volume: {e}")
            return
        
        print("[Auto-Update] Creating Database container...")
        try:
            result = subprocess.check_call([
                "docker", "run", "-d",
                "--name", "postgres_container",
                "--restart", "unless-stopped",
                "-e", f"POSTGRES_USER={os.getenv('POSTGRES_USER', 'user')}",
                "-e", f"POSTGRES_PASSWORD={os.getenv('POSTGRES_PASSWORD', 'password')}",
                "-e", f"POSTGRES_DB={os.getenv('POSTGRES_DB', 'miner_data')}",
                "-v", "postgres_data:/var/lib/postgresql/data",
                "-p", "5432:5432",
                "postgres:14"
            ])
        except subprocess.CalledProcessError as e:
            if e.returncode == 125:
                print("[Auto-Update] Database container is already running, skipping")
            else:
                print("[Auto-Update] Error creating database container")

        print("[Auto-Update] Updating Docker container...")
        try:
            # Find existing container
            result = subprocess.run(
                ["docker", "ps", "-q", "--filter", f"ancestor={self.container_image}"],
                capture_output=True,
                text=True
            )
            container_id = result.stdout.strip()

            if container_id:
                # Stop and remove existing container
                subprocess.check_call(["docker", "stop", container_id])
                subprocess.check_call(["docker", "rm", container_id])

            # Start new container
            subprocess.check_call([
                "docker", "run", "-d",
                "--network=host",
                "-v", f"{os.path.expanduser('~/.bittensor/wallets')}:/root/.bittensor/wallets",
                "-e", f"POSTGRES_USER={os.getenv('POSTGRES_USER', 'user')}",
                "-e", f"POSTGRES_PASSWORD={os.getenv('POSTGRES_PASSWORD', 'password')}",
                "-e", f"POSTGRES_DB={os.getenv('POSTGRES_DB', 'miner_data')}",
                "-e", f"WALLET_NAME={os.getenv('WALLET_NAME')}",
                "-e", f"HOTKEY_NAME={os.getenv('HOTKEY_NAME')}",
                "-e", f"NETWORK={os.getenv('NETWORK')}",
                "-e", f"NETUID={os.getenv('NETUID')}",
                "--restart", "unless-stopped",
                self.container_image
            ])
        except subprocess.CalledProcessError as e:
            print(f"[Auto-Update] Error updating Docker container: {e}")
            return

        print("[Auto-Update] Update complete! New container is running.")
        sys.exit(0)

    async def main_loop(self):
        """
        Main asynchronous loop that checks for updates every `check_interval` seconds.
        """
        while True:
            # Check if remote version is newer
            try:
                should_update = await self.check_versions()
                if should_update:
                    self.update()  # Synchronous call that ends in sys.exit(0) if successful
            except Exception as e:
                print(f"[Auto-Update] Unexpected error in check loop: {e}")

            await asyncio.sleep(self.check_interval)  # Wait before next check

    def run(self):
        """
        The thread entry point. Sets up an event loop and runs main_loop().
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.main_loop())
