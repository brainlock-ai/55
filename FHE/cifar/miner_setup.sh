#!/bin/bash

# Check if .env exists, if not copy from template
if [ ! -f .env ]; then
    echo "Creating .env file from miner template..."
    cp .env.miner.template .env
    echo "Please edit the .env file with your configuration and run this script again."
    exit 1
fi

# Source the .env file
source .env

# Exit on error
set -e

# Prevent interactive prompts
export DEBIAN_FRONTEND=noninteractive

echo "Installing Docker..."
sudo -E apt-get update
sudo -E apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo -E apt-get update
sudo -E apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add user to docker group
echo "Adding user to docker group..."
sudo usermod -aG docker $USER

# Install dependencies
echo "Installing dependencies..."
sudo -E apt install -y python3-pip cargo
pip install bittensor==8.5.1

# Install npm and pm2
echo "Installing npm and pm2..."
sudo -E apt install -y npm
npm install pm2 -g
pm2 install pm2-logrotate

# Clean up docker and compile
echo "Setting up docker environment..."
docker system prune --all --force
python3 compile_with_docker.py

# Start the miner
echo "Starting miner..."
# Port 5000 is specified as external_port to broadcast the location of the FHE inference server
# This is not a traditional axon port, but rather tells the network where to find your inference container
pm2 start neurons/miner.py -- \
  --wallet.name ${WALLET_NAME} \
  --wallet.hotkey ${HOTKEY_NAME} \
  --subtensor.network ${NETWORK} \
  --netuid ${NETUID} \
  --no_force_validator_permit \
  --axon.external_port ${EXTERNAL_PORT}

echo ""
echo "Miner has been started with pm2"
echo "To check miner status, run: pm2 status"
echo "To view logs, run: pm2 logs" 