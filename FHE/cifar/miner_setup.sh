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

# Start and enable Docker service
echo "Starting and enabling Docker service..."
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
echo "Adding user to docker group..."
sudo usermod -aG docker $USER

# Install dependencies
echo "Installing dependencies..."
sudo -E apt install -y python3-pip cargo
pip install bittensor==8.5.1

# Install Node.js & npm
echo "Installing Node.js & npm via apt..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is not installed. Installing Node.js & npm via apt..."
    
    # Add NodeSource GPG key and repository
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource.gpg.key \
      | sudo gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg

    DISTRO="$(lsb_release -sc)"  # e.g., "focal", "jammy"
    NODE_VERSION="20"            # Change to "18" or "20" if desired (18.x or 20.x)

    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/nodesource.gpg] \
    https://deb.nodesource.com/node_${NODE_VERSION}.x \
    $DISTRO main" \
      | sudo tee /etc/apt/sources.list.d/nodesource.list > /dev/null

    # Update and install Node.js + npm
    sudo -E apt-get update
    sudo -E apt-get install -y nodejs
else
    echo "Node.js is already installed."
fi

# Verify Node.js and npm installation
echo "Verifying Node.js and npm installation..."
node -v
npm -v

# Check if pm2 is installed
if ! command -v pm2 &> /dev/null; then
    echo "pm2 is not installed. Installing pm2..."
    sudo npm install pm2 -g
else
    echo "pm2 is already installed."
fi

# Always install pm2-logrotate
echo "Installing pm2-logrotate..."
pm2 install pm2-logrotate

# Verify docker access using sg
echo "Verifying Docker access..."
sg docker -c '
    # Verify docker is running
    if ! docker info > /dev/null 2>&1; then
        echo "Error: Docker is not running or current user does not have docker permissions"
        echo "Please try logging out and back in for group changes to take effect"
        exit 1
    fi

    # Clean up docker and compile
    echo "Setting up docker environment..."
    docker system prune --all --force
    python3 compile_with_docker.py
'

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

# Add PM2 to startup with correct path detection
PM2_PATH=$(which pm2)
if [ -n "$PM2_PATH" ]; then
    echo "Setting up PM2 startup with path: $PM2_PATH"
    sudo env PATH=$PATH:$(dirname $PM2_PATH) $PM2_PATH startup systemd -u $USER --hp $HOME
    pm2 save
else
    echo "Error: Could not find pm2 executable"
    exit 1
fi

echo ""
echo "Miner has been started with pm2"
echo "To check miner status, run: pm2 status"
echo "To view logs, run: pm2 logs" 