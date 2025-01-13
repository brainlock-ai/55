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
import copy
import os
import pathlib
import traceback
import typing

import bittensor as bt

from abc import ABC, abstractmethod


import time
from math import floor
from typing import Callable, Any
from functools import lru_cache, update_wrapper

# TODO(developer): Change this value when updating your code base.
# Define the version of the template module.
__version__ = "0.0.0"
version_split = __version__.split(".")
spec_version = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)

def _ttl_hash_gen(seconds: int):
    """
    Internal generator function used by the `ttl_cache` decorator to generate a new hash value at regular
    time intervals specified by `seconds`.

    Args:
        seconds (int): The number of seconds after which a new hash value will be generated.

    Yields:
        int: A hash value that represents the current time interval.

    This generator is used to create time-based hash values that enable the `ttl_cache` to determine
    whether cached entries are still valid or if they have expired and should be recalculated.
    """
    start_time = time.time()
    while True:
        yield floor((time.time() - start_time) / seconds)

# LRU Cache with TTL
def ttl_cache(maxsize: int = 128, typed: bool = False, ttl: int = -1):
    """
    Decorator that creates a cache of the most recently used function calls with a time-to-live (TTL) feature.
    The cache evicts the least recently used entries if the cache exceeds the `maxsize` or if an entry has
    been in the cache longer than the `ttl` period.

    Args:
        maxsize (int): Maximum size of the cache. Once the cache grows to this size, subsequent entries
                       replace the least recently used ones. Defaults to 128.
        typed (bool): If set to True, arguments of different types will be cached separately. For example,
                      f(3) and f(3.0) will be treated as distinct calls with distinct results. Defaults to False.
        ttl (int): The time-to-live for each cache entry, measured in seconds. If set to a non-positive value,
                   the TTL is set to a very large number, effectively making the cache entries permanent. Defaults to -1.

    Returns:
        Callable: A decorator that can be applied to functions to cache their return values.

    The decorator is useful for caching results of functions that are expensive to compute and are called
    with the same arguments frequently within short periods of time. The TTL feature helps in ensuring
    that the cached values are not stale.

    Example:
        @ttl_cache(ttl=10)
        def get_data(param):
            # Expensive data retrieval operation
            return data
    """
    if ttl <= 0:
        ttl = 65536
    hash_gen = _ttl_hash_gen(ttl)

    def wrapper(func: Callable) -> Callable:
        @lru_cache(maxsize, typed)
        def ttl_func(ttl_hash, *args, **kwargs):
            return func(*args, **kwargs)

        def wrapped(*args, **kwargs) -> Any:
            th = next(hash_gen)
            return ttl_func(th, *args, **kwargs)

        return update_wrapper(wrapped, func)

    return wrapper

# 12 seconds updating block.
@ttl_cache(maxsize=1, ttl=12)
def ttl_get_block(self) -> int:
    """
    Retrieves the current block number from the blockchain. This method is cached with a time-to-live (TTL)
    of 12 seconds, meaning that it will only refresh the block number from the blockchain at most every 12 seconds,
    reducing the number of calls to the underlying blockchain interface.

    Returns:
        int: The current block number on the blockchain.

    This method is useful for applications that need to access the current block number frequently and can
    tolerate a delay of up to 12 seconds for the latest information. By using a cache with TTL, the method
    efficiently reduces the workload on the blockchain interface.

    Example:
        current_block = ttl_get_block(self)

    Note: self here is the miner or validator instance
    """
    try:
        return self.subtensor.get_current_block()
    except BrokenPipeError:
        # Attempt to reinit the subtensor connection before trying again
        try:
            self.subtensor = bt.subtensor(config=self.config, network=self.config.subtensor.network)
            return self.subtensor.get_current_block()
        except Exception as e:
            bt.logging.error(f"Error reinitializing subtensor after BrokenPipeError: {str(e)}\n{traceback.format_exc()}")
            return -1
    except Exception as e:
        bt.logging.error(f"Error getting current block: {str(e)}\n{traceback.format_exc()}")
        return -1

class BaseNeuron(ABC):
    """
    Base class for Bittensor miners. This class is abstract and should be inherited by a subclass. It contains the core logic for all neurons; validators and miners.

    In addition to creating a wallet, subtensor, and metagraph, this class also handles the synchronization of the network state via a basic checkpointing mechanism based on epoch length.
    """

    neuron_type: str = "BaseNeuron"

    axon: typing.Union[bt.axon, None]
    config: bt.config

    subtensor: "bt.subtensor"
    wallet: "bt.wallet"
    metagraph: "bt.metagraph"
    spec_version: int = spec_version

    cwd: pathlib.Path = pathlib.Path(__file__).parent.resolve()

    @property
    def block(self):
        return ttl_get_block(self)
    
    def get_config(self):
        parser = argparse.ArgumentParser()
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        bt.axon.add_args(parser)
        parser.add_argument("--netuid", type=int, default=69, help="The UID for the subnet.")
        config = bt.config(parser)
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey,
                config.netuid,
                'validator',
            )
        )
        os.makedirs(config.full_path, exist_ok=True)
        return config

    def __init__(self):
        self.config = self.get_config()

        # Set up logging with the provided configuration.
        bt.logging.set_config(config=self.config.logging)

        # If a gpu is required, set the device to cuda:N (e.g. cuda:0)
        #self.device = self.config.neuron.device

        # Log the configuration for reference.
        bt.logging.info(self.config)

        # Build Bittensor objects
        # These are core Bittensor classes to interact with the network.
        bt.logging.info("Setting up bittensor objects.")

        # The wallet holds the cryptographic key pairs for the miner.
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)

        bt.logging.info(f"Wallet: {self.wallet}")
        bt.logging.info(f"Subtensor: {self.subtensor}")
        bt.logging.info(f"Metagraph: {self.metagraph}")

        # Check if the miner is registered on the Bittensor network before proceeding further.
        self.check_registered()

        # Each miner gets a unique identity (UID) in the network for differentiation.
        self.uid = self.metagraph.hotkeys.index(
            self.wallet.hotkey.ss58_address
        )

        self.setup_logging()
        self.step = 0

    def setup_logging(self):
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(f"Running neuron on subnet: {self.config.netuid} with uid {self.uid} on network: {self.subtensor.chain_endpoint}")

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

        if self.should_set_weights():
            self.set_weights()

        # Always save state.
        self.save_state()

    def check_registered(self):
        # --- Check for registration.
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            exit()

    def should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """
        return (
            self.block - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length

    def should_set_weights(self) -> bool:
        # Don't set weights on initialization.
        if self.step == 0:
            return False

        # Check if enough epoch blocks have elapsed since the last epoch.
        if self.config.neuron.disable_set_weights:
            return False

        # Define appropriate logic for when set weights.
        return (
            (self.block - self.metagraph.last_update[self.uid])
            > self.config.neuron.epoch_length
            and self.neuron_type != "MinerNeuron"
        )  # don't set weights if you're a miner