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
        
        # Update the base directory path to point to FHE-Subnet root
        self.base_dir = Path(__file__).parent.parent.parent.parent  # Go up four levels to reach FHE-Subnet
        
        # Update paths relative to base directory
        self.models_dir = self.base_dir / "FHE" / "cifar"
        self.keys_dir = self.base_dir / "FHE" / "cifar" / "neurons" / "user_keys"
        self.server_dir = self.base_dir / "FHE" / "server"
        
        self.model_name = "dev"
        self.fhe_server_port = self.config.fhe_server_port

        # Ensure directories exist
        self.keys_dir.mkdir(parents=True, exist_ok=True)

        # Start stake verification service
        self.start_stake_service()
        # Start FHE server after stake service is running
        self.start_fhe_server()

        # Setup the axon
        self.setup_axon()

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
        model_path = (self.models_dir / self.model_name).absolute()
        
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
        # Ensure miner or validator hotkey is still registered on the network.
        self.check_registered()

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

    def run(self):
        """
        Keep the miner alive.
        This loop maintains the miner's operations until intentionally stopped.
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

            # If someone intentionally stops the miner, it'll safely terminate operations.
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
            # In case of unforeseen errors, the miner will log the error and continue operations.
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
        """
        Destructor to ensure the FHE server is properly terminated.
        """
        if hasattr(self, 'fhe_server_process'):
            try:
                self.fhe_server_process.terminate()
                self.fhe_server_process.wait()
                bt.logging.info("FHE model server stopped.")
            except Exception as e:
                bt.logging.error(f"Error terminating FHE model server: {e}")

    def start_stake_service(self):
        """Start a local service for stake verification"""
        app = FastAPI()

        @app.get("/verify_stake/{hotkey}")
        async def verify_stake(hotkey: str):
            # Get latest metagraph
            self.metagraph = self.subtensor.metagraph(self.config.netuid)
            
            try:
                if hotkey not in self.metagraph.hotkeys:
                    return {"valid": False, "stake": 0, "error": "Hotkey not in metagraph"}
                
                uid = self.metagraph.hotkeys.index(hotkey)
                stake = float(self.metagraph.S[uid].item())
                no_force_permit = getattr(self.config, 'no_force_validator_permit', False)
                
                return {
                    "valid": no_force_permit or stake >= 10000,
                    "stake": stake,
                    "no_force_permit": no_force_permit,
                    "error": None
                }
            except Exception as e:
                bt.logging.error(f"Error verifying stake: {str(e)}")
                return {"valid": False, "stake": 0, "error": str(e)}

        # Run the service in a separate thread
        def run_service():
            uvicorn.run(app, host="127.0.0.1", port=8091)

        self.stake_service = Thread(target=run_service, daemon=True)
        self.stake_service.start()
        bt.logging.info("Started stake verification service on port 8091")


# Run the miner.
if __name__ == "__main__":
    # Setup argument parser for configurations
    parser = argparse.ArgumentParser(description="FHE Hybrid Miner")
    parser.add_argument("--fhe_server_port", type=int, help="FHE server port", default=8000)
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






























