import os
import subprocess
from pathlib import Path


def main():
    path_of_script = Path(__file__).parent.resolve()

    # Build image
    os.chdir(path_of_script)
    command = f'sudo docker build --tag compile_cifar --file "{path_of_script}/Dockerfile.compile" . && \
sudo docker run --name compile_cifar compile_cifar && \
sudo docker cp compile_cifar:/project/dev . && \
sudo docker rm "$(sudo docker ps -a --filter name=compile_cifar -q)"'
    subprocess.check_output(command, shell=True)


if __name__ == "__main__":
    main()
