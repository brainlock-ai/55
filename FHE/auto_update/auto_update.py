import os
#pip install GitPython
import git
import sys
import aiohttp
import asyncio
import threading
import subprocess
import requests
from packaging import version
import json
import time

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
        elif line.startswith("CREATE_NEW_DB"):
            CREATE_NEW_DB = line.split('=')[-1].strip().lower() == 'true'
            print(f"[Auto-Update] Create new DB: {CREATE_NEW_DB}")
            break

repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

        # Check if we're already running and clean up old instances
        try:
            result = subprocess.run(["pm2", "jlist"], capture_output=True, text=True)
            processes = json.loads(result.stdout)
            auto_update_processes = [p for p in processes if p.get('name') == 'auto_update_sn_54']
            
            if auto_update_processes:
                # Find newest process
                newest_process = max(auto_update_processes, 
                                  key=lambda p: p.get('pm2_env', {}).get('pm_uptime', 0))
                newest_pm_id = newest_process.get('pm2_env', {}).get('pm_id')
                
                # Delete all other instances
                for process in auto_update_processes:
                    pm_id = process.get('pm2_env', {}).get('pm_id')
                    if pm_id != newest_pm_id:
                        subprocess.run(["pm2", "delete", str(pm_id)], check=False)
                        print(f"[Auto-Update] Cleaned up old instance with PM2 ID: {pm_id}")
                
                # If we're not the newest, exit
                current_pid = os.getpid()
                if current_pid != newest_process.get('pid'):
                    print(f"[Auto-Update] Another instance is already running (PM2 ID: {newest_pm_id}). Exiting.")
                    sys.exit(0)
                
        except Exception as e:
            print(f"[Auto-Update] Error during process check: {e}")

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
        url = "https://raw.githubusercontent.com/brainlock-ai/fhe-subnet/main/FHE/__init__.py"
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
    
    def pull_and_restart(self):
        """
        Updates the status environment variable, git pulls and restarts the current pm2 process
        """
        os.environ["AUTO_UPDATE_STATUS"] = "DOCKER_UPDATE"

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

        print("[Auto-Update] Updating Auto-Update script requirements...")
        auto_update_dir = os.path.join(repo_dir, "FHE", "auto_update")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", 
                os.path.join(auto_update_dir, "requirements.txt")
            ])
        except subprocess.CalledProcessError as e:
            print(f"[Auto-Update] Error installing requirements: {e}")
            return

        print("[Auto-Update] Restarting Auto-Update script...")
        subprocess.check_call(["pm2", "restart", "auto_update_sn_54", "--update-env"])

        sys.exit(0)

    def notify_update(self, error_message=None):
        """Get public IP and notify brainlock.ai of successful update or errors"""
        try:
            # Get public IP using ifconfig.me
            ip = requests.get('https://ifconfig.me/ip', timeout=5).text.strip()
            
            if not ip:
                print("[Auto-Update] Failed to get public IP")
                return

            # Prepare notification payload
            payload = {
                'ip': ip,
                'version': __version__,
                'status': 'error' if error_message else 'success'
            }
            
            # Add error message if present
            if error_message:
                payload['error'] = error_message
                print(f"[Auto-Update] Notifying brainlock.ai of error: {error_message}")

            # Notify brainlock.ai
            response = requests.post('https://notify.brainlock.ai/update', json=payload, timeout=10)
            print(f"[Auto-Update] Successfully notified brainlock.ai with IP: {ip}")
        except Exception as e:
            print(f"[Auto-Update] Error notifying brainlock.ai: {e}")

    def update(self):
        """
        Perform the update:
        1. Update status environment variable
        2. Rebuild Docker image
        3. Stop and remove existing container
        4. Start new container
        """
        os.environ["AUTO_UPDATE_STATUS"] = "IDLE"

        print("[Auto-Update] Rebuilding Docker image...")
        try:
            subprocess.check_call([
                "python3",
                "build_docker_client_image.py"
            ], cwd=os.path.join(repo_dir, "FHE", "cifar"))
        except subprocess.CalledProcessError as e:
            error_msg = f"Error building Docker image: {e}"
            print(f"[Auto-Update] {error_msg}")
            self.notify_update(error_msg)
            return
        
        # Database setup
        if CREATE_NEW_DB:
            print("[Auto-Update] Creating new database volume...")
            try:
                # Force remove container even if it's running
                subprocess.run(["docker", "rm", "-f", "postgres_container"], check=True)
                print("[Auto-Update] Successfully removed old postgres container")
                # Force remove volume
                subprocess.run(["docker", "volume", "rm", "-f", "postgres_data"], check=True)
                print("[Auto-Update] Successfully removed old postgres volume")
                # Create new volume
                subprocess.run(["docker", "volume", "create", "postgres_data"], check=True)
                print("[Auto-Update] Successfully created postgres volume")
            except subprocess.CalledProcessError as e:
                error_msg = f"Error cleaning up old database: {e}"
                print(f"[Auto-Update] {error_msg}")
                self.notify_update(error_msg)
                return
        else:
            print("[Auto-Update] Using existing database volume...")
            # In non-CREATE_NEW_DB mode, just ensure container is removed if it exists
            try:
                subprocess.run(["docker", "rm", "-f", "postgres_container"], check=False)
                print("[Auto-Update] Cleaned up old postgres container")
            except:
                pass

        # Create and start database container
        print("[Auto-Update] Creating Database container...")
        try:
            subprocess.check_call([
                "docker", "run", "-d",
                "--name", "postgres_container",
                "--network=host",
                "--restart", "unless-stopped",
                "-e", f"POSTGRES_USER={os.getenv('POSTGRES_USER', 'user')}",
                "-e", f"POSTGRES_PASSWORD={os.getenv('POSTGRES_PASSWORD', 'password')}",
                "-e", f"POSTGRES_DB={os.getenv('POSTGRES_DB', 'miner_data')}",
                "-v", "postgres_data:/var/lib/postgresql/data",
                "postgres:14",
                "postgres", "-c", f"port={os.getenv('POSTGRES_PORT', '5432')}"
            ])
        except subprocess.CalledProcessError as e:
            if e.returncode == 125 and not CREATE_NEW_DB:
                print("[Auto-Update] Database container is already running, skipping")
            else:
                error_msg = "Error creating database container"
                print(f"[Auto-Update] {error_msg}")
                self.notify_update(error_msg)
                return

        # Test database connection if we created a new one
        if CREATE_NEW_DB:
            print("[Auto-Update] Waiting for PostgreSQL to be ready...")
            time.sleep(10)

            print("[Auto-Update] Testing PostgreSQL connection...")
            try:
                subprocess.check_call([
                    "docker", "exec", "postgres_container",
                    "psql",
                    "-U", os.getenv('POSTGRES_USER', 'user'),
                    "-d", os.getenv('POSTGRES_DB', 'miner_data'),
                    "-p", os.getenv('POSTGRES_PORT', '5432'),
                    "-h", "localhost",
                    "-c", "\\l"
                ])
                print("[Auto-Update] PostgreSQL connection test successful")
            except subprocess.CalledProcessError as e:
                error_msg = "Failed to connect to PostgreSQL"
                print(f"[Auto-Update] {error_msg}. Container logs:")
                subprocess.run(["docker", "logs", "postgres_container"])
                self.notify_update(error_msg)
                return

        print("[Auto-Update] Updating Docker container...")
        try:
            # Find existing container
            result = subprocess.run(
                ["docker", "ps", "-q", "--filter", "name=sn54_validator"],
                capture_output=True,
                text=True
            )
            container_id = result.stdout.strip()

            if container_id:
                # Stop and remove existing container
                subprocess.check_call(["docker", "stop", container_id])
                subprocess.check_call(["docker", "rm", container_id])

            # Start new container with matching arguments from validator_setup.sh
            subprocess.check_call([
                "docker", "run", "-d",
                "--network=host",
                "--name", "sn54_validator",
                "--restart", "unless-stopped",
                "-v", f"{os.path.expanduser('~/.bittensor/wallets')}:/root/.bittensor/wallets",
                "-e", f"POSTGRES_USER={os.getenv('POSTGRES_USER', 'user')}",
                "-e", f"POSTGRES_PASSWORD={os.getenv('POSTGRES_PASSWORD', 'password')}",
                "-e", f"POSTGRES_DB={os.getenv('POSTGRES_DB', 'miner_data')}",
                "-e", f"POSTGRES_PORT={os.getenv('POSTGRES_PORT', '5432')}",
                "-e", f"WALLET_NAME={os.getenv('WALLET_NAME')}",
                "-e", f"HOTKEY_NAME={os.getenv('HOTKEY_NAME')}",
                "-e", f"NETWORK={os.getenv('NETWORK')}",
                "-e", f"NETUID={os.getenv('NETUID')}",
                self.container_image
            ])
        except subprocess.CalledProcessError as e:
            error_msg = f"Error updating Docker container: {e}"
            print(f"[Auto-Update] {error_msg}")
            self.notify_update(error_msg)
            return

        print("[Auto-Update] Update complete! New container is running.")
        self.notify_update()  # Notify success

    async def main_loop(self):
        """
        Main asynchronous loop that checks for updates every `check_interval` seconds.
        """
        while True:
            # Check if remote version is newer
            status = os.getenv("AUTO_UPDATE_STATUS", "IDLE")
            try:
                if status == "IDLE":
                    should_update = await self.check_versions()
                    if should_update:
                        self.pull_and_restart()
                elif status == "DOCKER_UPDATE":
                    self.update()
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
