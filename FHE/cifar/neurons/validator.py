# neurons/validator.py

# The MIT License (MIT)
# Copyright ...

from functools import partial
import random
import torch
import bittensor as bt
import asyncio
import time
from collections.abc import Generator, Iterable
import aiohttp
import traceback
import websocket
from websocket._exceptions import WebSocketConnectionClosedException
from prometheus_client import start_http_server
from retry import retry

from neuron import BaseNeuron
from client import EpistulaClient
# from metrics import metrics

# Constants
VALIDATOR_MIN_STAKE = 20_000
VALIDATOR_SHOULD_QUERY_EACH_BATCH_X_TIMES_PER_EPOCH = 2
PSEUDO_SHUFFLE_EVERY_X_BLOCK = 360
EPOCH_LENGTH = 360
SET_WEIGHTS_EVERY_X_BLOCK = 360
SCORE_EXPONENTIAL_AVG_COEFF = 0.8


class Validator(BaseNeuron):
    """Validator class that extends BaseNeuron."""

    CONNECTION_REFRESH_INTERVAL = 60 * 5  # Refresh connection every 5 minutes

    def __init__(self):
        super().__init__()
        self.cleanup_socket()  # Add this line before starting Prometheus server
        # try:
        #     # Try to start on preferred port
        #     start_http_server(8091, addr='0.0.0.0')
        # except OSError:
        #     # If preferred port fails, try alternative ports
        #     self.start_metrics_server()
        
        self.connection_timestamp = None
        self.setup_subtensor()
        self.scores = torch.zeros(256, dtype=torch.float32)
        self.weights = torch.zeros(256, dtype=torch.float32)
        
        # Initialize metagraph
        bt.logging.info("Initializing metagraph...")
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        bt.logging.info("Syncing metagraph...")
        self.metagraph.sync()  # Sync the metagraph after initialization
        bt.logging.info("Metagraph initialized and synced")
        self.weights_set_block = self.block

        # Initialize metrics for active miners
        # self.update_active_miners_metric()

    def cleanup_socket(self):
        """Cleanup any existing socket connections."""
        import socket
        import contextlib
        import psutil
        
        # Kill any process using port 8091
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for conn in proc.connections('inet'):  # Specify connection kind
                    if hasattr(conn, 'laddr') and conn.laddr.port == 8091:
                        try:
                            proc.kill()
                            bt.logging.info(f"Killed process {proc.pid} using port 8091")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        # Try to create a socket with SO_REUSEADDR
        with contextlib.suppress(Exception):
            temp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            temp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            temp_socket.bind(('0.0.0.0', 8091))
            temp_socket.close()
            bt.logging.info("Successfully cleaned up socket on port 8091")

    # def start_metrics_server(self, start_port=8091, max_attempts=10):
    #     """Start Prometheus metrics server with fallback ports."""
    #     for port in range(start_port, start_port + max_attempts):
    #         try:
    #             start_http_server(port, addr='0.0.0.0')
    #             bt.logging.info(f"Metrics server started on port {port}")
    #             return
    #         except OSError:
    #             if port == start_port + max_attempts - 1:
    #                 bt.logging.error(f"Could not find available port in range {start_port}-{port}")
    #                 raise
    #             continue

    def setup_subtensor(self):
        """Initialize subtensor connection with retry logic."""
        max_retries = 5
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                self.subtensor = bt.subtensor(
                    config=self.config,
                    network=self.config.subtensor.network,
                )
                # Test the connection
                self.subtensor.block
                self.connection_timestamp = time.time()
                bt.logging.info("Successfully connected to subtensor")
                break
            except (BrokenPipeError, ConnectionRefusedError, websocket.WebSocketConnectionClosedException) as e:
                if attempt == max_retries - 1:
                    bt.logging.error(f"Failed to connect to subtensor after {max_retries} attempts")
                    raise
                bt.logging.warning(f"Connection attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

    def should_refresh_connection(self) -> bool:
        """Check if we should refresh the connection based on time elapsed."""
        if self.connection_timestamp is None:
            return True
        
        return (time.time() - self.connection_timestamp) > self.CONNECTION_REFRESH_INTERVAL

    def should_set_weights(self):
        return self.block > self.weights_set_block + SET_WEIGHTS_EVERY_X_BLOCK

    def check_validator_permit(self) -> bool:
        """Check if this validator has permission to validate."""
        try:
            validator_index = self.get_validator_index()
            result = validator_index != -1 and self.metagraph.total_stake[validator_index] >= VALIDATOR_MIN_STAKE
            bt.logging.debug(f"Validator permit check: index={validator_index}, result={result}")
            return result
        except Exception as e:
            bt.logging.error(f"Error in check_validator_permit: {str(e)}\n{traceback.format_exc()}")
            return False

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages."""
        try:
            bt.logging.info("Starting metagraph resync...")
            self.metagraph.sync()
            bt.logging.info("Metagraph resync completed")
        except Exception as e:
            bt.logging.error(f"Error in resync_metagraph: {str(e)}\n{traceback.format_exc()}")

    def should_resync_metagraph(self) -> bool:
        """Returns true if the metagraph should be resynced."""
        try:
            current = self.block
            should_sync = current % 12 == 0  # Syncs every 12 blocks
            bt.logging.debug(f"Should resync check: block={current}, result={should_sync}")
            return should_sync
        except Exception as e:
            bt.logging.error(f"Error in should_resync_metagraph: {str(e)}\n{traceback.format_exc()}")
            return False

    def setup_logging(self):
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(f"Running validator for subnet: {self.config.netuid} on network: {self.config.subtensor.chain_endpoint}")

    def get_valid_validator_hotkeys(self):
        valid_hotkeys = []
        hotkeys = self.metagraph.hotkeys
        for index, hotkey in enumerate(hotkeys):
            if self.metagraph.total_stake[index] >= 1.024e3:
                valid_hotkeys.append(hotkey)
        return valid_hotkeys

    def get_validator_index(self):
        valid_hotkeys = self.get_valid_validator_hotkeys()
        try:
            return valid_hotkeys.index(self.wallet.hotkey.ss58_address)
        except ValueError:
            return -1

    def split_uids_in_batches(self, group_index, num_groups, queryable_uids):
        """Distributes miners into batches using a round-robin approach."""
        # Create a list to store the round-robin distribution
        batches = [[] for _ in range(num_groups)]
        
        # Distribute the miners in a round-robin fashion
        for i, uid in enumerate(queryable_uids):
            batch_index = i % num_groups
            batches[batch_index].append(uid)

        # Return the batch for the given group_index
        return batches[group_index]

    def deterministically_shuffle_and_batch_queryable_uids(self, filtered_uids): 
        """
        Pseudorandomly shuffles the list of queryable uids, and splits it in batches to reduce concurrent requests from multiple validators.
        """
        validator_uids = self.metagraph.total_stake >= VALIDATOR_MIN_STAKE
        validator_index = self.get_validator_index()            
        num_validators = (validator_uids == True).sum().item()  

        # If no validators found, assume this node is the only validator
        if num_validators == 0:
            bt.logging.warning("No validators found in network, assuming this node is the only validator")
            num_validators = 1
            validator_index = 0

        # Calculate batch duration
        batch_duration_in_blocks = max(1, int(EPOCH_LENGTH / VALIDATOR_SHOULD_QUERY_EACH_BATCH_X_TIMES_PER_EPOCH) // num_validators)
        
        batch_number_since_genesis = self.block // batch_duration_in_blocks
        # e.g., block = 2993336
        # batch_number_since_genesis = 2993336 // 10 = 299333
        batch_index_to_query = (batch_number_since_genesis + validator_index) % num_validators
        # (299333 + validator_index) % num_validators = (299333 + 4) % 17 = 1
        # as validator 4, it should then query subgroup 1

        # Add check for empty filtered_uids
        if not filtered_uids:
            bt.logging.warning("No queryable UIDs found")
            return []

        # Pseudorandomly shuffle the filtered_uids
        seed = self.block // PSEUDO_SHUFFLE_EVERY_X_BLOCK
        rng = random.Random(seed)
        shuffled_filtered_uids = list(filtered_uids[:])
        rng.shuffle(shuffled_filtered_uids)

        # Since the shuffling was seeded, all the validators will shuffle the list the exact same way
        # without having to communicate with each other, ensuring that they don't all query the same
        # miner all at once, which can happen randomly and cause deregistrations.

        batched_uids = self.split_uids_in_batches(batch_index_to_query, num_validators, shuffled_filtered_uids)

        return batched_uids

    async def run_concurrent_validations(self, batched_uids, miner_inputs):
        """Run multiple validations concurrently using aiohttp."""
        # TODO switch to httpx

        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(force_close=True, limit=100, enable_cleanup_closed=True),
            timeout=aiohttp.ClientTimeout(total=120)
        ) as session:
            tasks = []
            for uid in batched_uids:
                try:
                    axon = self.metagraph.axons[uid]
                    hotkey = str(axon.hotkey)
                    miner_client = EpistulaClient(
                        wallet=self.wallet,
                        server_ip=axon.ip,
                        server_port=str(axon.port),
                        hotkey=hotkey,
                        session=session
                    )
                    
                    start_time = time.time()
                    task = asyncio.create_task(miner_client.query())
                    tasks.append((uid, task, start_time, hotkey))
                except Exception as e:
                    # try:
                    #     # Use the correct label name 'status' and include hotkey
                    #     metrics.validation_requests.labels(status='failure', hotkey=str(axon.hotkey)).inc()
                    # except Exception as metric_error:
                    #     bt.logging.error(f"Error updating metrics: {metric_error}")
                    bt.logging.debug(f"Failed to create task for miner {uid}: {str(e)}")
                    continue
            
            if not tasks:
                return []
            
            results = []
            for uid, task, start_time, hotkey in tasks:
                try:
                    result = await asyncio.wait_for(task, timeout=120)
                    duration = time.time() - start_time
                    
                    if result is not None:
                        # try:
                        #     # Record validation attempt first
                        #     metrics.record_validation_attempt(hotkey, True, duration)
                            
                        #     # Extract score and stats from the result
                        #     score = result.get('score', 0)
                        #     stats = result.get('stats', {})
                            
                        #     # Update miner metrics with stats from the client
                        #     metrics.update_miner_metrics(hotkey, score, stats)
                            
                        # except Exception as metric_error:
                        #     bt.logging.error(f"Error updating metrics: {str(metric_error)}")
                        
                        results.append((uid, result))
                    else:
                        # try:
                        #     metrics.record_validation_attempt(hotkey, False, duration)
                        # except Exception as metric_error:
                        #     bt.logging.error(f"Error updating metrics: {str(metric_error)}")
                        bt.logging.debug(f"Null result from miner {uid}")
                except Exception as e:
                    # try:
                    #     metrics.record_validation_attempt(hotkey, False, 0.0)
                    # except Exception as metric_error:
                    #     bt.logging.error(f"Error updating metrics: {metric_error}")
                    bt.logging.debug(f"Task failed for miner {uid}: {type(e).__name__}: {str(e)}")
                    results.append((uid, None))

            return results

    def get_queryable_uids(self) -> Generator[int, None, None]:
        """
        Returns the UIDs of the miners that are queryable.
        """
        uids = self.metagraph.uids.tolist()
        # Ignore validators, they're not queryable as miners (torch.nn.Parameter)
        total_stake = (
            self.metagraph.total_stake[uids]
            if isinstance(self.metagraph.total_stake[uids], torch.Tensor)
            else torch.tensor(self.metagraph.total_stake[uids])
        )
        queryable_flags: Iterable[bool] = (
            (total_stake < VALIDATOR_MIN_STAKE)
            & torch.tensor([self.metagraph.axons[i].ip != "0.0.0.0" for i in uids])
        ).tolist()
        for uid, is_queryable in zip(uids, queryable_flags):
            if is_queryable:
                yield uid
    
    @retry(tries=3, delay=1, backoff=2)
    async def set_weights_with_timeout_and_retry(self, timeout=30):
        """
        Set weights with timeout and retry logic
        """
        try:
            sum_of_scores = self.scores[self.metagraph.uids].sum()
            self.weights = self.scores[self.metagraph.uids] / sum_of_scores
            set_weights = partial(
                self.subtensor.set_weights,
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=self.metagraph.uids,
                weights=self.weights[self.metagraph.uids],
                wait_for_inclusion=True,
                wait_for_finalization=False,
            )

            result = await asyncio.wait_for(
                asyncio.to_thread(set_weights),
                timeout=timeout
            )
            self.weights_set_block = self.block
            return result, None
        except TimeoutError:
            return None, "Timeout while setting weights"
        except Exception as e:
            return None, str(e)

    # def update_active_miners_metric(self):
    #     """Update the active miners metric."""
    #     try:
    #         # Count miners that have valid axons (non-zero IP) and are below validator stake threshold
    #         active_count = len([
    #             uid for uid in range(self.metagraph.n.item())
    #             if (
    #                 self.metagraph.axons[uid].ip != "0.0.0.0" and 
    #                 self.metagraph.total_stake[uid] < VALIDATOR_MIN_STAKE
    #             )
    #         ])
    #         metrics.active_miners.set(active_count)
    #         bt.logging.info(f"Active miners metric updated: {active_count}")
    #     except Exception as e:
    #         bt.logging.error(f"Error updating active miners metric: {str(e)}")
    #         metrics.active_miners.set(0)  # Set to 0 on error

    def run(self):
        """Main validation loop."""
        bt.logging.info("Starting validator loop.")
        
        while True:
            try:
                # Update active miners metric
                # self.update_active_miners_metric()
                
                # Add metagraph sync check
                if self.should_resync_metagraph():
                    self.resync_metagraph()
                
                # Proactively refresh connection if needed
                if self.should_refresh_connection():
                    bt.logging.info("Refreshing subtensor connection...")
                    self.setup_subtensor()

                # Create new event loop for each iteration
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    if self.should_set_weights():
                        loop.run_until_complete(self.set_weights_with_timeout_and_retry())
                    
                    # Get the UIDs of all miners in the network
                    filtered_uids = list(self.get_queryable_uids())
                    
                    if not filtered_uids:
                        bt.logging.debug("No queryable UIDs found, waiting...")
                        time.sleep(5)
                        continue
                    
                    # Get the batch of miners this validator should query
                    batched_uids = self.deterministically_shuffle_and_batch_queryable_uids(filtered_uids)
                    
                    if batched_uids:
                        results = loop.run_until_complete(
                            self.run_concurrent_validations(batched_uids, None)
                        )
                        
                        # Process results if needed
                        if results:
                            successful_validations = sum(1 for _, r in results if r is not None)
                            bt.logging.info(f"Completed batch with {successful_validations}/{len(batched_uids)} successful validations")
                            zipped_scores = zip(batched_uids, [r["score"] for _, r in results if r is not None])
                            for uid, score in zipped_scores:
                                self.scores[uid] = self.scores[uid] * SCORE_EXPONENTIAL_AVG_COEFF + (1 - SCORE_EXPONENTIAL_AVG_COEFF) * score
                
                    # Add delay between validation rounds
                    time.sleep(5)
                    
                finally:
                    # Clean up the event loop
                    loop.close()
                    
            except KeyboardInterrupt:
                bt.logging.success("Keyboard interrupt detected. Exiting validator.")
                break
            except (BrokenPipeError, ConnectionRefusedError, WebSocketConnectionClosedException) as e:
                bt.logging.error(f"Connection error in validation loop: {str(e)}")
                # Attempt to reconnect
                try:
                    self.setup_subtensor()
                    time.sleep(5)  # Wait before retrying
                    continue
                except Exception as reconnect_error:
                    bt.logging.error(f"Failed to reconnect: {str(reconnect_error)}")
                    time.sleep(10)  # Longer wait before next attempt
            except Exception as e:
                bt.logging.error(f"Error in validation loop: {str(e)} - {traceback.format_exc()}")
                time.sleep(1)  # Add small delay before retrying

    def close(self):
        """Cleanup method to properly close connections."""
        try:
            if hasattr(self, 'subtensor'):
                self.subtensor.close()
        except Exception as e:
            bt.logging.error(f"Error closing subtensor connection: {str(e)}")


if __name__ == "__main__":
    # Initialize the Validator class
    validator = Validator()

    try:
        # Run the validator loop
        validator.run()
    except Exception as e:
        bt.logging.error(f"Unhandled exception in validator: {str(e)}")
    finally:
        validator.close()
        bt.logging.info("Validator shutdown complete.")
