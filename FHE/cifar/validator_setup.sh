#!/bin/bash

# Check if .env exists, if not copy from template
if [ ! -f .env ]; then
    echo "Creating .env file from validator template..."
    cp .env.validator.template .env
    echo "Please edit the .env file with your configuration and run this script again."
    exit 1
fi

# Source the .env file
source .env

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

sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
echo "Adding user to docker group..."
sudo usermod -aG docker $USER

# Install cargo and bittensor
echo "Installing cargo and bittensor..."
sudo -E apt install -y cargo
pip install bittensor==8.5.1

# Update PATH
echo "Updating PATH..."
export PATH=$PATH:$HOME/.local/bin

# Clean up docker and build new image
echo "Setting up docker environment..."
python3 build_docker_client_image.py

# Setup PostgreSQL container
echo "Setting up PostgreSQL container..."
USERNAME_LENGTH=8
PASSWORD_LENGTH=16

# Function to generate random strings
generate_random_string() {
    local length=$1
    # Use only alphanumeric characters
    tr -dc 'a-zA-Z0-9' < /dev/urandom | head -c "$length"
}

# Check if .env file exists
if [[ -f .env ]]; then
    echo ".env file found. Loading credentials..."
    source .env
fi

# Check if required variables are set, generate them if not
if [[ -z "$POSTGRES_USER" ]]; then
    echo "POSTGRES_USER not found. Generating a new username..."
    POSTGRES_USER=$(generate_random_string "$USERNAME_LENGTH")
    echo -e "\nPOSTGRES_USER=$POSTGRES_USER" >> .env
fi

if [[ -z "$POSTGRES_PASSWORD" ]]; then
    echo "POSTGRES_PASSWORD not found. Generating a new password..."
    POSTGRES_PASSWORD=$(generate_random_string "$PASSWORD_LENGTH")
    echo -e "\nPOSTGRES_PASSWORD=$POSTGRES_PASSWORD" >> .env
fi

if [[ -z "$POSTGRES_DB" ]]; then
    echo "POSTGRES_DB not found. Creating it..."
    POSTGRES_DB="miner_data"
    echo -e "\nPOSTGRES_DB=$POSTGRES_DB" >> .env
fi

# Before starting PostgreSQL container, add these cleanup lines
# Add this line to remove the old volume data
docker volume rm postgres_data 2>/dev/null || true
docker volume create postgres_data

# Start PostgreSQL with the generated user as the superuser
docker run -d \
    --name postgres_container \
    --restart unless-stopped \
    -e POSTGRES_USER=${POSTGRES_USER} \
    -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD} \
    -e POSTGRES_DB=${POSTGRES_DB} \
    -v postgres_data:/var/lib/postgresql/data \
    -p 5432:5432 \
    postgres:14

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
sleep 10

# Add these debug lines
echo "Testing PostgreSQL connection..."
docker exec postgres_container psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "\l" || {
    echo "Failed to connect to PostgreSQL. Container logs:"
    docker logs postgres_container
    exit 1
}

docker compose up -d

# Run the container
echo "Starting validator container..."
CONTAINER_ID=$(sudo docker run -d \
    --network=host \
    -v ~/.bittensor/wallets:/root/.bittensor/wallets \
    -e POSTGRES_USER=${POSTGRES_USER} \
    -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD} \
    -e POSTGRES_DB=${POSTGRES_DB} \
    -e WALLET_NAME=${WALLET_NAME} \
    -e HOTKEY_NAME=${HOTKEY_NAME} \
    -e NETWORK=${NETWORK} \
    -e NETUID=${NETUID} \
    ${DOCKER_IMAGE})

# Check if the container started successfully
if [ -n "$CONTAINER_ID" ]; then
    echo "Container started: $CONTAINER_ID"
else
    echo "Failed to start the container. Please check the docker run command and try again."
fi

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

# Install auto-update requirements
echo "Installing auto-update requirements..."
pip install -r ../auto_update/requirements.txt

# Use pm2 to start your application
echo "Starting auto-update script with PM2..."
sudo pm2 start python3 --name "auto_update_sn_55" -- ../auto_update/start_auto_update.py
sudo pm2 save
sudo pm2 startup

# Add PM2 to startup
sudo env PATH=$PATH:/usr/bin /usr/lib/node_modules/pm2/bin/pm2 startup systemd -u $USER --hp /home/$USER

echo "Container started with ID: $CONTAINER_ID"
echo "To view logs, run: sudo docker logs -f $CONTAINER_ID"