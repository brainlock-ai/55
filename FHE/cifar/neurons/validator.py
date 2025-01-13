# neurons/validator.py

# The MIT License (MIT)
# Copyright ...

from functools import partial
import random
import torch
import asyncio
import time
from collections.abc import Generator, Iterable
import aiohttp
import traceback
import websocket
from websocket._exceptions import WebSocketConnectionClosedException
from retry import retry
import threading
import argparse
from postgres_exporter import PostgresExporter
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
import ssl

from client import EpistulaClient
from fiber.chain import chain_utils, interface, metagraph, weights
from fiber.chain.fetch_nodes import get_nodes_for_netuid
from fiber.logging_utils import get_logger
from fiber.chain.chain_utils import query_substrate
from fiber.chain.weights import (
    blocks_since_last_update,
    min_interval_to_set_weights,
    can_set_weights,
    set_node_weights,
)

logger = get_logger(__name__)

# Constants
VALIDATOR_MIN_STAKE = 20_000
VALIDATOR_SHOULD_QUERY_EACH_BATCH_X_TIMES_PER_EPOCH = 2
PSEUDO_SHUFFLE_EVERY_X_BLOCK = 360
EPOCH_LENGTH = 360
SET_WEIGHTS_EVERY_X_BLOCK = 36
SCORE_EXPONENTIAL_AVG_COEFF = 0.8

class Validator:
    """Validator class for FHE subnet."""

    CONNECTION_REFRESH_INTERVAL = 60 * 5  # Refresh connection every 5 minutes

    def __init__(self):
        self.config = self.get_config()
        self.weights_lock = asyncio.Lock()  # Add lock for weight setting
        
        self.connection_timestamp = None
        self.scores = torch.zeros(256, dtype=torch.float32)
        self.weights = torch.zeros(256, dtype=torch.float32)
        
        # Initialize database connection
        db_url = f"postgresql://{os.getenv('POSTGRES_USER', 'user')}:{os.getenv('POSTGRES_PASSWORD', 'password')}@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'miner_data')}"
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

        # Initialize substrate connection
        self.setup_subtensor()
        
        # Load keypair
        self.keypair = chain_utils.load_hotkey_keypair(
            wallet_name=self.config.wallet_name,
            hotkey_name=self.config.wallet_hotkey
        )
        
        # Get node ID
        self.node_id = self.get_validator_index()
        
        # Start postgres exporter in a separate thread
        self.start_postgres_exporter()
        
        # Initialize weights block
        self.weights_set_block = self.block

    def get_config(self):
        parser = argparse.ArgumentParser()
        
        # Add arguments with dot notation
        parser.add_argument("--netuid", type=int, default=1, help="The UID for the subnet.")
        parser.add_argument("--subtensor.network", type=str, default="test", help="The subtensor network to connect to.")
        parser.add_argument("--chain.endpoint", type=str, help="The subtensor chain endpoint.")
        parser.add_argument("--wallet.name", type=str, default="default", help="The name of the wallet to use.")
        parser.add_argument("--wallet.hotkey", type=str, default="default", help="The name of the hotkey to use.")
        parser.add_argument("--logging.dir", type=str, default="~/.bittensor/miners/", help="Logging directory.")
        
        config = parser.parse_args()
        
        # Convert dot notation to attributes
        config.subtensor_network = getattr(config, "subtensor.network")
        config.chain_endpoint = getattr(config, "chain.endpoint")
        config.wallet_name = getattr(config, "wallet.name")
        config.wallet_hotkey = getattr(config, "wallet.hotkey")
        config.logging_dir = getattr(config, "logging.dir")
        
        # Set up full path
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/validator".format(
                config.logging_dir,
                config.wallet_name,
                config.wallet_hotkey,
                config.netuid,
            )
        )
        os.makedirs(config.full_path, exist_ok=True)
        return config

    def setup_subtensor(self):
        """Initialize substrate connection with retry logic."""
        max_retries = 5
        retry_delay = 2
        ssl_errors = (ssl.SSLError, ssl.SSLEOFError)

        for attempt in range(max_retries):
            try:
                # Initialize substrate connection
                self.substrate = interface.get_substrate(subtensor_network=self.config.subtensor_network)
                
                # Test the connection with multiple retries for the block query
                for _ in range(3):  # Try block query up to 3 times
                    try:
                        # Use proper substrate query pattern
                        self.substrate, current_block = query_substrate(self.substrate, "System", "Number", [], return_value=True)
                        break
                    except ssl_errors as e:
                        if _ < 2:  # Only retry if we haven't tried 3 times yet
                            logger.warning(f"SSL error during block query: {str(e)}. Retrying...")
                            time.sleep(1)
                            continue
                        raise  # Re-raise on final attempt
                
                self.connection_timestamp = time.time()
                logger.info("Successfully connected to substrate")
                break  # Break out of main retry loop on success
                
            except (BrokenPipeError, ConnectionRefusedError, websocket.WebSocketConnectionClosedException, *ssl_errors) as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to connect to substrate after {max_retries} attempts")
                    raise
                logger.warning(f"Connection attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

    def should_refresh_connection(self) -> bool:
        """Check if we should refresh the connection based on time elapsed or connection state."""
        if self.connection_timestamp is None:
            return True
        
        # Check if connection is stale based on time
        if (time.time() - self.connection_timestamp) > self.CONNECTION_REFRESH_INTERVAL:
            return True
            
        # Test connection health
        try:
            # Use proper substrate query pattern
            self.substrate, current_block = query_substrate(self.substrate, "System", "Number", [], return_value=True)
            return False
        except (BrokenPipeError, ConnectionRefusedError, websocket.WebSocketConnectionClosedException, ssl.SSLError, ssl.SSLEOFError) as e:
            logger.warning(f"Connection test failed: {str(e)}")
            return True
        except Exception as e:
            logger.error(f"Unexpected error testing connection: {str(e)}")
            return True

    def should_set_weights(self):
        """Check if we can set weights based on rate limits."""
        try:
            # Check if enough blocks have passed since last update
            blocks_since = blocks_since_last_update(self.substrate, self.config.netuid, self.node_id)
            min_interval = min_interval_to_set_weights(self.substrate, self.config.netuid)
            
            # First check if enough blocks have passed since last weights set
            if self.block <= self.weights_set_block + SET_WEIGHTS_EVERY_X_BLOCK:
                return False
                
            # Then check chain rate limits
            return can_set_weights(self.substrate, self.config.netuid, self.node_id)
        except Exception as e:
            logger.error(f"Error checking if should set weights: {str(e)}")
            return False

    def check_validator_permit(self) -> bool:
        """Check if this validator has permission to validate."""
        try:
            validator_index = self.get_validator_index()
            # Get nodes to check stake
            nodes = get_nodes_for_netuid(substrate=self.substrate, netuid=self.config.netuid)
            validator_stake = next((node.stake for node in nodes if node.node_id == validator_index), 0)
            result = validator_index != -1 and validator_stake >= VALIDATOR_MIN_STAKE
            logger.debug(f"Validator permit check: index={validator_index}, result={result}")
            return result
        except Exception as e:
            logger.error(f"Error in check_validator_permit: {str(e)}\n{traceback.format_exc()}")
            return False

    def get_valid_validator_hotkeys(self):
        """Get list of validator hotkeys that meet minimum stake requirement."""
        valid_hotkeys = []
        nodes = get_nodes_for_netuid(substrate=self.substrate, netuid=self.config.netuid)
        for node in nodes:
            if node.stake >= 1.024e3:
                valid_hotkeys.append(node.hotkey)
        return valid_hotkeys

    def get_validator_index(self):
        """Get the validator's node index."""
        try:
            # Use proper substrate query pattern
            self.substrate, validator_node_id = query_substrate(
                self.substrate,
                "SubtensorModule", 
                "Uids", 
                [self.config.netuid, self.keypair.ss58_address],
                return_value=True
            )
            return validator_node_id
        except Exception as e:
            logger.error(f"Failed to get validator index: {str(e)}")
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
        # Get nodes to check validator stakes
        nodes = get_nodes_for_netuid(substrate=self.substrate, netuid=self.config.netuid)
        validator_uids = [node.node_id for node in nodes if node.stake >= VALIDATOR_MIN_STAKE]
        validator_index = self.get_validator_index()            
        num_validators = len(validator_uids)

        # If no validators found, assume this node is the only validator
        if num_validators == 0:
            logger.warning("No validators found in network, assuming this node is the only validator")
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
            logger.warning("No queryable UIDs found")
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

    def get_miner_scores(self):
        """Get latest median scores for all miners directly from PostgreSQL, using up to 40 most recent entries."""
        try:
            # Get nodes to map hotkeys to UIDs
            nodes = get_nodes_for_netuid(substrate=self.substrate, netuid=self.config.netuid)
            hotkey_to_uid = {node.hotkey: node.node_id for node in nodes}
            
            session = self.Session()
            
            # Direct SQL query to get median scores from last 40 entries per miner
            query = text("""
                WITH recent_scores AS (
                    SELECT 
                        hotkey,
                        score,
                        ROW_NUMBER() OVER (PARTITION BY hotkey ORDER BY timestamp DESC) as rn
                    FROM miner_history
                    WHERE score IS NOT NULL
                ),
                last_40_scores AS (
                    SELECT 
                        hotkey,
                        score,
                        COUNT(*) OVER (PARTITION BY hotkey) as entry_count
                    FROM recent_scores
                    WHERE rn <= 40
                )
                SELECT 
                    hotkey,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY score) as median_score,
                    entry_count
                FROM last_40_scores
                GROUP BY hotkey, entry_count;
            """)
            
            # Reset scores tensor
            self.scores = torch.zeros(256, dtype=torch.float32)
            
            # Execute query and update scores
            results = session.execute(query)
            for row in results:
                try:
                    if row.hotkey in hotkey_to_uid:
                        uid = hotkey_to_uid[row.hotkey]
                        self.scores[uid] = row.median_score
                        logger.debug(f"Updated score for miner {uid} (hotkey: {row.hotkey}): {row.median_score:.6f} from {row.entry_count} entries")
                except ValueError:
                    continue
                    
            session.close()
        except Exception as e:
            logger.error(f"Error querying miner scores: {str(e)}")

    async def run_concurrent_validations(self, batched_uids, miner_inputs):
        """Run multiple validations concurrently using aiohttp."""
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(force_close=True, limit=100, enable_cleanup_closed=True),
            timeout=aiohttp.ClientTimeout(total=120)
        ) as session:
            tasks = []
            # Get all nodes first
            nodes = get_nodes_for_netuid(substrate=self.substrate, netuid=self.config.netuid)
            nodes_by_id = {node.node_id: node for node in nodes}
            
            for uid in batched_uids:
                try:
                    if uid not in nodes_by_id:
                        logger.warning(f"No node found for UID {uid}")
                        continue
                        
                    node = nodes_by_id[uid]
                    hotkey = node.hotkey
                    miner_client = EpistulaClient(
                        keypair=self.keypair,
                        server_ip=node.ip,
                        server_port=str(node.port),
                        hotkey=hotkey,
                        session=session
                    )
                    
                    start_time = time.time()
                    task = asyncio.create_task(miner_client.query())
                    tasks.append((uid, task, start_time, hotkey))
                except Exception as e:
                    logger.info(f"Failed to create task for miner {uid}: {str(e)}")
                    continue
            
            if not tasks:
                return []
            
            results = []
            for uid, task, start_time, hotkey in tasks:
                try:
                    result = await asyncio.wait_for(task, timeout=120)
                    duration = time.time() - start_time
                    
                    if result is not None and isinstance(result, dict):
                        try:
                            # Only process valid responses that have all required fields
                            is_valid = (
                                result.get('success', False) and  # Must be explicitly successful
                                (
                                    # Handle both direct score and score_stats tuple
                                    (isinstance(result.get('score'), (int, float)) and result.get('score') >= 0) or
                                    (isinstance(result.get('score_stats'), tuple) and all(isinstance(x, (int, float)) for x in result.get('score_stats')))
                                ) and
                                isinstance(result.get('stats'), dict) and  # Must have stats dictionary
                                duration > 0  # Must have non-zero duration
                            )
                            
                            if is_valid:
                                # Extract score and stats from the result
                                if 'score_stats' in result:
                                    score_stats = result.get('score_stats')
                                    mean, median, std = score_stats
                                    logger.info(f"Score stats for miner {uid} - Mean: {mean:.6f}, Median: {median:.6f}, Std: {std:.6f}")
                                
                                stats = result.get('stats', {})
                                predictions_match = stats.get('predictions_match', True)
                                
                                logger.info(f"Recorded validation for miner {uid} with duration {duration:.2f}s")
                            else:
                                if duration <= 0:
                                    logger.info(f"Skipping record for miner {uid} due to zero/negative duration: {duration}")
                                else:
                                    logger.info(f"Invalid response format from miner {uid}: {result}")
                            
                        except Exception as db_error:
                            logger.error(f"Error recording validation in database for miner {uid}: {str(db_error)}")
                        
                        results.append((uid, result))
                    else:
                        logger.info(f"Null or invalid result type from miner {uid}: {type(result)}")
                except Exception as e:
                    logger.info(f"Task failed for miner {uid}: {type(e).__name__}: {str(e)}")
                    results.append((uid, None))

            return results

    def get_queryable_uids(self) -> Generator[int, None, None]:
        """
        Returns the UIDs of the miners that are queryable.
        """
        # Get nodes from chain
        nodes = get_nodes_for_netuid(substrate=self.substrate, netuid=self.config.netuid)
        
        # Filter out validators (nodes with stake >= VALIDATOR_MIN_STAKE)
        for node in nodes:
            if node.stake < VALIDATOR_MIN_STAKE and node.ip != "0.0.0.0":
                yield node.node_id

    @retry(tries=3, delay=1, backoff=2)
    def set_weights(self):
        """
        Set weights using Fiber's implementation with retry logic
        """
        try:
            # First fetch latest scores from PostgreSQL
            self.get_miner_scores()
            
            # Get nodes from chain
            nodes = get_nodes_for_netuid(substrate=self.substrate, netuid=self.config.netuid)
            
            # Get validator node ID
            validator_node_id = self.get_validator_index()
            if validator_node_id == -1:
                logger.error("Failed to get validator node ID")
                return False
            
            # Get version key
            self.substrate, version_key = query_substrate(
                self.substrate,
                "SubtensorModule", 
                "WeightsVersionKey", 
                [self.config.netuid],
                return_value=True
            )

            # Prepare normalized weights from scores
            positive_scores = self.scores.clone()
            positive_scores[positive_scores < 0] = 0
            sum_of_scores = positive_scores.sum() or 1
            normalized_weights = positive_scores / sum_of_scores

            logger.info(f"Setting weights with scores from {len([s for s in positive_scores if s > 0])} miners")
            
            # Set weights using Fiber
            success = weights.set_node_weights(
                substrate=self.substrate,
                keypair=self.keypair,
                node_ids=[node.node_id for node in nodes],
                node_weights=[normalized_weights[node.node_id].item() for node in nodes],
                netuid=self.config.netuid,
                version_key=version_key,
                validator_node_id=validator_node_id,
                wait_for_inclusion=True,  # Wait for inclusion to ensure transaction success
                wait_for_finalization=False,
                max_attempts=3,
            )
            
            if success:
                self.weights_set_block = self.block
                logger.info("Weights set successfully.")
                return True
            else:
                logger.error("Failed to set weights")
                return False
                
        except Exception as e:
            logger.error(f"Error setting weights: {str(e)}\n{traceback.format_exc()}")
            return False

    def run(self):
        """Main validation loop."""
        logger.info("Starting validator loop.")
        
        while True:
            try:
                # Proactively refresh connection if needed
                if self.should_refresh_connection():
                    logger.info("Refreshing subtensor connection...")
                    try:
                        self.setup_subtensor()
                    except Exception as e:
                        logger.error(f"Failed to refresh connection: {str(e)}")
                        time.sleep(10)  # Longer wait on connection failure
                        continue

                # Get latest scores from PostgreSQL
                try:
                    self.get_miner_scores()
                except Exception as e:
                    logger.error(f"Failed to get miner scores: {str(e)}")
                    time.sleep(5)
                    continue

                # Set weights if needed
                if self.should_set_weights():
                    try:
                        self.set_weights()
                    except Exception as e:
                        logger.error(f"Failed to set weights: {str(e)}")
                        time.sleep(5)

                # Create new event loop for validations
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    # Get the UIDs of all miners in the network
                    try:
                        filtered_uids = list(self.get_queryable_uids())
                    except Exception as e:
                        logger.error(f"Failed to get queryable UIDs: {str(e)}")
                        time.sleep(5)
                        continue
                    
                    if not filtered_uids:
                        logger.debug("No queryable UIDs found, waiting...")
                        time.sleep(5)
                        continue
                    
                    # Get the batch of miners this validator should query
                    try:
                        batched_uids = self.deterministically_shuffle_and_batch_queryable_uids(filtered_uids)
                    except Exception as e:
                        logger.error(f"Failed to batch UIDs: {str(e)}")
                        time.sleep(5)
                        continue
                    
                    if batched_uids:
                        try:
                            results = loop.run_until_complete(
                                self.run_concurrent_validations(batched_uids, None)
                            )
                            
                            # Process results if needed
                            if results:
                                successful_validations = sum(1 for _, r in results if r is not None)
                                logger.info(f"Completed batch with {successful_validations}/{len(batched_uids)} successful validations")
                                
                                # Remove redundant score query
                                # self.get_miner_scores()
                        except Exception as e:
                            logger.error(f"Failed during validation run: {str(e)}")
                            time.sleep(5)

                    # Add delay between validation rounds
                    time.sleep(5)
                    
                finally:
                    # Clean up the event loop
                    try:
                        loop.close()
                    except Exception as e:
                        logger.error(f"Error closing event loop: {str(e)}")
                    
            except KeyboardInterrupt:
                logger.success("Keyboard interrupt detected. Exiting validator.")
                break
            except (BrokenPipeError, ConnectionRefusedError, websocket.WebSocketConnectionClosedException, ssl.SSLError, ssl.SSLEOFError) as e:
                logger.error(f"Connection error in validation loop: {str(e)}")
                # Attempt to reconnect
                try:
                    self.setup_subtensor()
                    time.sleep(5)  # Wait before retrying
                    continue
                except Exception as reconnect_error:
                    logger.error(f"Failed to reconnect: {str(reconnect_error)}")
                    time.sleep(10)  # Longer wait before next attempt
            except Exception as e:
                logger.error(f"Error in validation loop: {str(e)} - {traceback.format_exc()}")
                time.sleep(5)  # Increased delay before retrying

    def close(self):
        """Cleanup method to properly close connections."""
        try:
            if hasattr(self, 'subtensor'):
                self.subtensor.close()
        except Exception as e:
            logger.error(f"Error closing subtensor connection: {str(e)}")

    def start_postgres_exporter(self):
        """Start the Postgres exporter in a separate thread."""
        try:
            exporter = PostgresExporter()
            self.exporter_thread = threading.Thread(
                target=exporter.run_metrics_loop,
                daemon=True  # This ensures the thread stops when the main program stops
            )
            self.exporter_thread.start()
            logger.info("Started Postgres exporter thread")
        except Exception as e:
            logger.error(f"Failed to start Postgres exporter: {str(e)}")

    @property
    def block(self):
        """Get the current block number."""
        try:
            # Query the current block number from the System module
            self.substrate, current_block = query_substrate(self.substrate, "System", "Number", [], return_value=True)
            return current_block
        except Exception as e:
            logger.error(f"Error getting block number: {str(e)}")
            return -1


if __name__ == "__main__":
    # Initialize the Validator class
    validator = Validator()

    try:
        # Run the validator loop
        validator.run()
    except Exception as e:
        logger.error(f"Unhandled exception in validator: {str(e)}")
    finally:
        validator.close()
        logger.info("Validator shutdown complete.")
