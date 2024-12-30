import os
import subprocess
from pathlib import Path


def main():
    # Get the current directory (FHE/cifar)
    path_of_script = Path(__file__).parent.resolve()
    
    # Stay in the current directory where all the required files are
    os.chdir(path_of_script)
    
    # Build image using relative path to Dockerfile
    command = 'sudo docker build --tag cml_client_cifar_10_8_bit --file Dockerfile.client .'
    print(command)
    subprocess.check_output(command, shell=True)


if __name__ == "__main__":
    main()
