"""Methods to deploy a server using Docker.

It takes as input a folder with:
    - client.zip
    - server.zip
    - processing.json

It builds a Docker image and spawns a Docker container that runs the server.

This module is untested as it would require to first build the release Docker image.
FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3347
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import time

DATE_FORMAT: str = "%Y_%m_%d_%H_%M_%S"


def delete_image(image_name: str):
    """Delete a Docker image.

    Arguments:
        image_name (str): to name of the image to delete.
    """
    to_delete = subprocess.check_output(
        f"sudo docker ps -a --filter name={image_name} -q", shell=True
    ).decode("utf-8")
    if to_delete:
        subprocess.check_output(f"sudo docker rmi {to_delete}", shell=True)


def stop_container(image_name: str):
    """Kill all containers that use a given image.

    Arguments:
        image_name (str): name of  image for which to stop  containers.
    """
    to_delete = subprocess.check_output(
        f"sudo docker ps -q --filter ancestor={image_name}", shell=True
    ).decode("utf-8")
    if to_delete:
        subprocess.check_output(f"sudo docker kill {to_delete}", shell=True)


def build_docker_image(path_to_model: Path, image_name: str, hotkey: str):
    """Build server Docker image.

    Arguments:
        path_to_model (Path): path to serialized model to serve.
        image_name (str): name to give to the image.
        hotkey (str): miner's hotkey for authentication
    """
    delete_image(image_name)

    path_of_script = Path(__file__).parent.resolve()

    cwd = os.getcwd()
    with TemporaryDirectory() as directory:
        temp_dir = Path(directory)

        files = ["server.py", "server_requirements.txt"]
        # Copy files
        for file_name in files:
            source = path_of_script / file_name
            target = temp_dir / file_name
            shutil.copyfile(src=source, dst=target)
        shutil.copytree(path_to_model, temp_dir / "dev")

        # Build image
        os.chdir(temp_dir)
        command = (
            f'sudo docker build '
            f'--tag {image_name}:latest '
            f'--build-arg MINER_HOTKEY={hotkey} '
            f'--file "{path_of_script}/Dockerfile.server" .'
        )
        subprocess.check_output(command, shell=True)
    os.chdir(cwd)


def remove_existing_container(container_name: str):
    """Remove a container if it exists.

    Arguments:
        container_name (str): name of the container to remove
    """
    try:
        subprocess.check_output(f"sudo docker rm -f {container_name}", shell=True)
    except subprocess.CalledProcessError:
        # Container doesn't exist, which is fine
        pass


def main(path_to_model: Path, image_name: str, hotkey: str):
    """Deploy function.

    Arguments:
        path_to_model (Path): path to model to server
        image_name (str): name of the Docker image
        hotkey (str): miner's hotkey for authentication
    """

    build_docker_image(path_to_model, image_name, hotkey)

    if args.only_build:
        return

    # Run newly created Docker server with network setting
    try:
        with open("./url.txt", mode="w", encoding="utf-8") as file:
            file.write("http://localhost:5000")
        
        # Remove existing container if it exists
        remove_existing_container("miner")
        
        # Launch container detached with restart policy
        subprocess.check_output(
            f"sudo docker run -d --network=host --name miner --restart always -p 5000:5000 -e MINER_HOTKEY={hotkey} {image_name}",
            shell=True
        ).strip().decode('utf-8')
        print("Docker container 'miner' launched successfully. It will persist and auto-restart even after this script ends.")
        
    except KeyboardInterrupt:
        print("\nScript interrupted. The Docker container 'miner' will continue running.")
        print("To stop it later, use: sudo docker stop miner")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-model", dest="path_to_model", type=Path, default=Path("./dev"))
    parser.add_argument("--image-name", dest="image_name", type=str, default="server")
    parser.add_argument("--only-build", dest="only_build", action="store_true")
    parser.add_argument("--hotkey", type=str, required=True, help="Miner's hotkey for authentication")
    args = parser.parse_args()
    main(path_to_model=args.path_to_model, image_name=args.image_name, hotkey=args.hotkey)
