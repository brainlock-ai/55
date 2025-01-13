# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import os
import pathlib
import traceback
import typing
from abc import ABC, abstractmethod

from fiber.chain import chain_utils, interface, metagraph
from fiber.logging_utils import get_logger

logger = get_logger(__name__)

class BaseNeuron(ABC):
    """
    Base class for Fiber neurons. This class is abstract and should be inherited by a subclass. 
    It contains the core logic for all neurons; validators and miners.
    """

    neuron_type: str = "BaseNeuron"
    config: typing.Any

    def __init__(self):
        self.config = self.get_config()

        # Set up logging with the provided configuration.
        logger.info(self.config)

        # Build Fiber objects
        logger.info("Setting up fiber objects.")

        # Initialize substrate connection
        self.substrate = interface.get_substrate(subtensor_network=self.config.subtensor.network)
        
        # Load keypair
        self.keypair = chain_utils.load_hotkey_keypair(
            wallet_name=self.config.wallet.name,
            hotkey_name=self.config.wallet.hotkey
        )
        
        # Load coldkey
        self.coldkeypub = chain_utils.load_coldkeypub_keypair(
            wallet_name=self.config.wallet.name
        )

        # Initialize metagraph
        self.metagraph = metagraph.Metagraph(substrate=self.substrate, netuid=self.config.netuid)
        self.metagraph.sync_nodes()

        logger.info(f"Keypair: {self.keypair.ss58_address}")
        logger.info(f"Coldkey: {self.coldkeypub.ss58_address}")
        logger.info(f"Substrate: {self.substrate}")
        logger.info(f"Metagraph: {self.metagraph}")

        # Check if registered
        self.check_registered()

        # Get node ID
        self.node_id = self.substrate.query(
            "SubtensorModule", 
            "Uids", 
            [self.config.netuid, self.keypair.ss58_address]
        ).value

        self.setup_logging()
        self.step = 0

    def get_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--netuid", type=int, default=1, help="The UID for the subnet.")
        parser.add_argument("--subtensor.network", type=str, default="test", help="The subtensor network to connect to.")
        parser.add_argument("--subtensor.chain_endpoint", type=str, help="The subtensor chain endpoint.")
        parser.add_argument("--wallet.name", type=str, default="default", help="The name of the wallet to use.")
        parser.add_argument("--wallet.hotkey", type=str, default="default", help="The name of the hotkey to use.")
        parser.add_argument("--logging.debug", action="store_true", help="Enable debug logging.")
        parser.add_argument("--logging.trace", action="store_true", help="Enable trace logging.")
        parser.add_argument("--logging.logging_dir", type=str, default="~/.bittensor/miners/", help="Logging directory.")
        config = parser.parse_args()
        
        # Set up full path
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey,
                config.netuid,
                'validator' if self.neuron_type == "Validator" else 'miner',
            )
        )
        os.makedirs(config.full_path, exist_ok=True)
        return config

    def setup_logging(self):
        logger.info(f"Running neuron on subnet: {self.config.netuid} with node_id {self.node_id} on network: {self.config.subtensor.network}")

    @abstractmethod
    def run(self):
        ...

    def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """
        # Ensure miner or validator hotkey is still registered on the network.
        self.check_registered()

        if self.should_sync_metagraph():
            self.resync_metagraph()

    def check_registered(self):
        """Check if the hotkey is registered on the network."""
        try:
            node_id = self.substrate.query(
                "SubtensorModule", 
                "Uids", 
                [self.config.netuid, self.keypair.ss58_address]
            ).value
            
            if node_id is None:
                logger.error(
                    f"Hotkey {self.keypair.ss58_address} is not registered on netuid {self.config.netuid}. "
                    f"Please register before continuing."
                )
                exit()
        except Exception as e:
            logger.error(f"Failed to check registration: {str(e)}")
            exit()