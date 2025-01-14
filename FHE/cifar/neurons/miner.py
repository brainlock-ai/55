# neurons/miner.py

# The MIT License (MIT)
# Copyright ...
import os
import time
import subprocess
import sys
from typing import Union
import traceback
import argparse
import bittensor as bt
import pathlib
from fastapi import FastAPI
import uvicorn
from threading import Thread
from pathlib import Path

from neuron import BaseNeuron

class FHEHybridMiner(BaseNeuron):
    axon: Union[bt.axon, None] = None

    def __init__(self, config=None):
        super().__init__()
        self.config = config

        # Kill any process using the FHE server port (default 5000)
        self.kill_existing_process(self.config.fhe_server_port)
        
        # Update the base directory path to point to FHE-Subnet root
        self.base_dir = Path(__file__).parent.parent.parent.parent  # Go up four levels to reach FHE-Subnet
        
        # Update paths relative to base directory
        self.models_dir = self.base_dir / "FHE" / "cifar" / "compiled"
        self.keys_dir = self.base_dir / "FHE" / "cifar" / "neurons" / "user_keys"
        self.server_dir = self.base_dir / "FHE" / "server"
        
        self.model_name = "dev"
        self.fhe_server_port = self.config.fhe_server_port

        # Ensure directories exist
        self.keys_dir.mkdir(parents=True, exist_ok=True)

        # Start FHE server
        self.start_fhe_server()

        # Setup the axon only if not already set
        if os.getenv("AXON_SET") != "1":
            self.setup_axon()
            os.environ["AXON_SET"] = "1"  # Mark as set
        else:
            bt.logging.info("Axon setup skipped as it's already set.")

    def kill_existing_process(self, port):
        """
        Kill any process using the specified port.
        """
        try:
            # Find the process ID using the port
            result = subprocess.check_output(f"sudo fuser {port}/tcp", shell=True).decode().strip()
            if result:
                bt.logging.info(f"Killing process {result} using port {port}...")
                os.system(f"sudo kill -9 {result}")
                bt.logging.success(f"Successfully killed process {result}.")
        except subprocess.CalledProcessError:
            bt.logging.info(f"No process found on port {port}.")
        except Exception as e:
            bt.logging.error(f"Error killing process on port {port}: {e}")

    def setup_logging(self):
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(f"Running miner for subnet: {self.config.netuid} on network: {self.config.subtensor.chain_endpoint}")

    def start_fhe_server(self):
        # Use the corrected path to deploy script
        deploy_script_path = self.server_dir / "deploy_to_docker.py"
        
        if not deploy_script_path.exists():
            bt.logging.error(f"Deploy script not found at: {deploy_script_path}")
            sys.exit(1)
        
        # Get absolute path to the model directory
        model_path = self.models_dir.absolute()
        
        bt.logging.info(f"Using model path: {model_path}")
        bt.logging.info(f"Using deploy script path: {deploy_script_path}")
        
        cmd = [
            sys.executable,
            str(deploy_script_path),
            "--hotkey", str(self.wallet.hotkey.ss58_address),
            "--path-to-model", str(model_path)
        ]
        try:
            self.fhe_server_process = subprocess.Popen(cmd)
            bt.logging.info(f"Started FHE model server on port {self.fhe_server_port} with hotkey {self.wallet.hotkey.ss58_address}")
        except Exception as e:
            bt.logging.error(f"Failed to start FHE model server: {e}")
            sys.exit(1)

    def setup_axon(self):
        bt.logging.info("Setting up axon...")
        self.axon = bt.axon(wallet=self.wallet, config=self.config)

        bt.logging.info(f"Axon created: {self.axon}")

        # Serve the axon on the specified network and netuid
        try:
            bt.logging.info(f"Serving axon on network: {self.config.subtensor.network} with netuid: {self.config.netuid}")
            self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
            bt.logging.info(f"Axon served on network: {self.config.subtensor.network} with netuid: {self.config.netuid}")
        except Exception as e:
            bt.logging.error(f"Failed to serve axon: {e}")
            sys.exit(1)
                
    def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """
        self.check_registered()

    def check_registered(self):
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using btcli subnets register before trying again"
            )
            exit()

    def run(self):
        """
        Keep the miner alive.
        """
        bt.logging.info("Starting FHEHybridMiner...")

        step = 0

        while True:
            step += 1
            try:
                if step % 600 == 0:
                    self.check_register()

                if step % 24 == 0 and hasattr(self, 'miner_uid') and self.miner_uid is not None:
                    try:
                        self.metagraph = self.subtensor.metagraph(self.config.netuid)
                        bt.logging.info(
                            f"Step:{step} | "
                            f"Block:{self.block} | "
                            f"Hotkey Count:{len(self.metagraph.hotkeys)}"
                        )
                    except Exception:
                        bt.logging.warning(
                            f"Failed to sync metagraph: {traceback.format_exc()}"
                        )

                time.sleep(1)

            except KeyboardInterrupt:
                bt.logging.success("Miner killed via keyboard interrupt.")
                self.axon.stop()
                if hasattr(self, 'fhe_server_process'):
                    try:
                        self.fhe_server_process.terminate()
                        self.fhe_server_process.wait()
                        bt.logging.info("FHE model server stopped.")
                    except Exception as e:
                        bt.logging.error(f"Error stopping FHE model server: {e}")
                break
            except Exception:
                bt.logging.error(traceback.format_exc())
                continue

    def check_register(self, should_exit=False):
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"\nYour miner: {self.wallet} is not registered to the network: {self.subtensor} \n"
                "Run btcli register and try again."
            )
            if should_exit:
                sys.exit()
            self.miner_uid = None
        else:
            miner_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            bt.logging.info(f"Running miner on uid: {miner_uid}")
            self.miner_uid = miner_uid

    def __del__(self):
        if hasattr(self, 'fhe_server_process'):
            try:
                self.fhe_server_process.terminate()
                self.fhe_server_process.wait()
                bt.logging.info("FHE model server stopped.")
            except Exception as e:
                bt.logging.error(f"Error terminating FHE model server: {e}")


# Run the miner.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FHE Hybrid Miner")
    parser.add_argument("--fhe_server_port", type=int, help="FHE server port", default=5000)
    parser.add_argument(
        "--no_force_validator_permit",
        action="store_true",
        help="If set, skip minimum stake requirement check",
        default=False
    )
    parser.add_argument(
        "--netuid", type=int, default=1, help="The UID for the BrainLock subnet."
    )

    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)

    config = bt.config(parser)

    miner = FHEHybridMiner(config=config)
    miner.run()
