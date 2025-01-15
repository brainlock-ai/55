"""Client script with Epistula authentication using Bittensor wallet.

This script does the following:
    - Uses Bittensor wallet for Epistula authentication
    - Query crypto-parameters and pre/post-processing parameters
    - Quantize the inputs using the parameters
    - Encrypt data using the crypto-parameters
    - Send the encrypted data to the server
    - Collect the data and decrypt it
    - De-quantize the decrypted results
"""

import io
import math
import os
import struct
import sys
import time
import json
import zipfile
import torch
import random
import aiohttp
import asyncio
import requests
import traceback
import numpy as np
import bittensor as bt
from pathlib import Path
from models import cnv_2w2a
from epistula import EpistulaAuth
from scoring import SimplifiedReward
from torchvision import datasets, transforms
from concrete.ml.deployment import FHEModelClient


# Add the correct models directory to the Python path
MODELS_DIR = Path(__file__).parent
sys.path.append(str(MODELS_DIR))

class EpistulaClient: 
    def __init__(self, keypair, server_ip: str, server_port: str, hotkey=None, session=None):
        self.url = f"http://{server_ip}:{server_port}"
        self.keypair = keypair
        self.hotkey = hotkey
        self.epistula = EpistulaAuth(self.keypair)
        self.session = session or aiohttp.ClientSession()
        self.setup_data()
        self.setup_model()
        self.reward_model = SimplifiedReward()

    def setup_data(self):
        """Setup CIFAR10 data for testing with augmentation."""
        # Base transformations for loading the dataset
        BASE_TRANSFORM = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        try:
            self.test_set = datasets.CIFAR10( ## https://github.com/BayesWatch/cinic-10 we can use this eventually (its larger and more diverse)
                root=".data/",
                train=False, 
                download=False,
                transform=BASE_TRANSFORM
            )
        except:
            print("Downloading CIFAR10 dataset...")
            self.test_set = datasets.CIFAR10(
                root=".data/",      
                train=False, 
                download=True,
                transform=BASE_TRANSFORM
            )

        # Store all images and labels from test set
        self.test_sub_set = torch.stack(
            [self.test_set[index][0] for index in range(len(self.test_set))]
        )
        self.test_labels = torch.tensor(
            [self.test_set[index][1] for index in range(len(self.test_set))]
        )

        # Setup augmentation transformations
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(5),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                fill=0,
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1
            ),
            transforms.Resize(32, antialias=True),
        ])

    def setup_model(self):
        """Setup and load the original model for comparison."""
        try:
            print(" model...")
            # Initialize model
            self.model = cnv_2w2a(False)
            
            # Determine device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
            
            # Find relative path to checkpoints
            dir_path = Path(__file__).parent.absolute()
            checkpoint_path = dir_path / "experiments/CNV_2W2A_2W2A_20221114_131345/checkpoints/best.tar"
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            self.model.eval()  # Set to evaluation mode
            
            # Move model to appropriate device
            self.model = self.model.to(self.device)
            
        except Exception as e:
            print(f"Error in model setup: {e}")
            print(traceback.format_exc())
            raise
    
    async def fetch_layer_outputs(self, response):
        """
        Asynchronous generator to fetch layer outputs with length-prefixed chunks.
        """
        while True:
            # Read the length header (4 bytes)
            length_bytes = await response.content.readexactly(4)
            layer_length = struct.unpack("<I", length_bytes)[0]

            if layer_length == 0:
                # End-of-stream marker received
                return  # Stop the generator

            # Read the chunk data
            chunk_data = await response.content.readexactly(layer_length)
            yield chunk_data, time.time()

    def compare_outputs(self, y_simulated, y_real, epsilon=1e-3, delta=0.01):
        """
        Compare simulated FHE outputs to real FHE outputs within defined tolerances.

        Args:
            y_simulated (np.ndarray): Simulated FHE output.
            y_real (np.ndarray): Real FHE output from the miner.
            epsilon (float): Absolute error tolerance.
            delta (float): Relative error tolerance.

        Returns:
            bool: True if outputs are within tolerance, False otherwise.
        """
        # Example usage
        # y_simulated = np.array([0.5, 0.3, 0.2])
        # y_real = np.array([0.51, 0.29, 0.21])  # Real output from miner
        absolute_error = np.abs(y_simulated - y_real)
        relative_error = np.abs(y_simulated - y_real) / np.maximum(np.abs(y_simulated), 1e-6)

        return np.all(absolute_error <= epsilon) and np.all(relative_error <= delta)

    async def get_client_zip(self):
        # Get client.zip using aiohttp with improved error handling
        bt.logging.info("Requesting client.zip...")
        try:
            async with self.session.get(
                f"{self.url}/get_client",
                headers=self.epistula.generate_headers(b"", signed_for=self.hotkey),
                ssl=False,
                timeout=aiohttp.ClientTimeout(total=20)  # Increased timeout
            ) as response:
                if response.status != 200:
                    bt.logging.info(f"Get client request failed with status {response.status}")
                    return None

                content = await response.read()
                if not content:
                    bt.logging.info("Received empty response")
                    return None

                with open("./client.zip", "wb") as f:
                    f.write(content)

        except aiohttp.ClientError as e:
            bt.logging.info(f"Network error during get_client: {e}")
            return None
    
    async def upload_evaluation_keys(self):
        # Get and upload evaluation keys
        try:
            eval_keys = self.fhe_client.get_serialized_evaluation_keys()
            bt.logging.info("Sending evaluation keys to /add_key endpoint...")
            key_response = self.epistula.request(
                'POST',
                f"{self.url}/add_key",
                signed_for=self.hotkey,
                files={"key": io.BytesIO(eval_keys)},
                timeout=20  # Add 20 second timeout for key upload
            )
            bt.logging.info("Received response from /add_key endpoint.")
            try:
                uid = key_response.json().get("uid")
                if not uid:
                    bt.logging.info("Failed to get UID from server response.")
                    return None
            except json.JSONDecodeError:
                bt.logging.info("Server response was not valid JSON.")
                return None
        except BrokenPipeError:
            bt.logging.info("Broken pipe occurred while uploading evaluation keys.")
            return None
        except requests.exceptions.RequestException as e:
            bt.logging.info(f"Request failed: {e}")
            return None
    
    async def post_with_retries(self, url, body, headers, max_retries, retry_delay):
        """
        Helper function to handle retry logic for a POST request.
        """
        for attempt in range(max_retries):
            try:
                return await self.session.post(
                    url,
                    data=body,
                    headers=headers,
                    ssl=False,
                    timeout=aiohttp.ClientTimeout(total=180),
                )
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                print(f"Network error during compute request for IP {self.url} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    # Exhausted retries
                return None
            except Exception as e:
                print(f"Error processing response: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    # Exhausted retries
                    return None
            
        return None

    async def query(self, image_index=None):
        """Main query method that can be called from validator."""
        try:
            # Cleanup any existing files
            self._cleanup_files()

            bt.logging.info(f"Connected to server at {self.url}")

            self.get_client_zip()

            # Initialize FHE client
            self.fhe_client = FHEModelClient(path_dir="./", key_dir="./keys")

            self.upload_evaluation_keys()

            # Select image (random if not specified)
            if image_index is None:
                image_index = random.randint(0, len(self.test_sub_set) - 1)
            X = self.test_sub_set[image_index:image_index+1]
            true_label = self.test_labels[image_index]

            # Generate random seed for this query
            augmentation_seed = random.randint(0, 2**32 - 1)
            bt.logging.info(f"\nUsing augmentation seed: {augmentation_seed}")

            # Apply augmentation once
            torch.manual_seed(augmentation_seed)
            augmented_X = self.augmentation(X)

            # Get local prediction using the augmented input
            with torch.no_grad():
                original_input = augmented_X.to(self.device)
                original_result = self.model(original_input)
                original_result = original_result.cpu().numpy()
                original_pred = original_result.argmax(axis=1)[0]

            # Use same augmented input for FHE inference
            clear_input = augmented_X.numpy()
            encrypted_input = self.fhe_client.quantize_encrypt_serialize(clear_input)

            # Send compute request
            url = f"{self.url}/compute"
            boundary = b'----WebKitFormBoundary' + os.urandom(16).hex().encode('ascii')
            
            # Construct payload
            payload = []
            payload.extend([
                b'--' + boundary + b'\r\n',
                b'Content-Disposition: form-data; name="model_input"; filename="model_input"\r\n',
                b'Content-Type: application/octet-stream\r\n',
                b'\r\n',
                encrypted_input,
                b'\r\n'
            ])
            payload.extend([
                b'--' + boundary + b'\r\n',
                b'Content-Disposition: form-data; name="uid"\r\n',
                b'\r\n',
                str(uid).encode(),
                b'\r\n'
            ])
            payload.extend([
                b'--' + boundary + b'--\r\n'
            ])
            
            body = b''.join(payload)
            headers = self.epistula.generate_headers(body, signed_for=self.hotkey)
            headers['Content-Type'] = f'multipart/form-data; boundary={boundary.decode()}'
            
            # Add retry logic for compute request
            max_retries = 3
            retry_delay = 2  # seconds

            # start first timer
            start_send_message_time = time.time()
            first_layer_response_time = 0.0

            response = await self.post_with_retries(
                url, body, headers, max_retries, retry_delay
            )
            if not response or response.status != 200:
                print(
                    f"Compute request failed with status "
                    f"{response.status if response else 'no response'}"
                )
                return None

            chunk_stats = []
            async for chunk_data, chunk_reception_time in self.fetch_layer_outputs(response):
                if not first_layer_response_time:
                    first_layer_response_time = chunk_reception_time

                # Deserialize and decrypt the chunk
                remote_result = self.fhe_client.deserialize_decrypt_dequantize(chunk_data)
                chunk_stats.append(
                    {
                        "result": remote_result,
                        "timestamp": chunk_reception_time,
                    }
                )

            if not chunk_stats:
                print("No data received from the server.")
                return None

            # Compare submodel outputs
        total_score = 0.0
        for i, chunk_stat in enumerate(chunk_stats):
            if i == 0:
                chunk_simulated_output = self.fhe_client.run(original_input)
            else:
                previous_chunk_result = chunk_stats[i - 1]["result"]
                chunk_simulated_output = self.fhe_client.run(previous_chunk_result)

            chunk_score = self.compare_outputs(chunk_simulated_output, chunk_stat["result"])
            total_score += chunk_score

            average_score = total_score / len(chunk_stats)

            # ---------------------
            # METRICS FOR TIMING
            # ---------------------

            # Times for inferences (one chunk == one inference, if that's your assumption)
            inference_times = [cs["timestamp"] for cs in chunk_stats]

            start_inference_time = inference_times[0]
            end_inference_time = inference_times[-1]

            # Time from request-sent to first inference
            time_to_first_inference = start_inference_time - start_send_message_time

            # Time from first inference to last inference
            time_for_all_inferences = end_inference_time - start_inference_time

            # Total time from request-sent to last inference
            total_time = end_inference_time - start_send_message_time

            # "inferences (or models) per second"
            num_inferences = len(chunk_stats)  # If each chunk is a single inference
            model_inference_per_second = (num_inferences - 1) / time_for_all_inferences if time_for_all_inferences > 0 else 0.0

            # Check if the server might have buffered results
            # e.g. by looking at the time to the 20th percentile inference
            non_streamed = False
            if num_inferences >= 5:
                twenty_percent_index = math.ceil(num_inferences * 0.20)
                time_to_twentieth_percent = inference_times[twenty_percent_index] - start_send_message_time

                # If 5% of the inferences arrived after 75% of total time,
                # it's likely non-streamed
                if time_to_twentieth_percent / total_time >= 0.75:
                    non_streamed = True
                    print("Likely non-streamed response detected.")

            print(f"Final average score: {average_score:.4f}")
            # Calculate final score using SimplifiedReward
            score, stats = self.reward_model.calculate_score(
                response_time=elapsed_time,
                predictions_match=predictions_match,
                hotkey=self.hotkey
            )
            
            # Add predictions_match to stats
            stats["predictions_match"] = predictions_match
            stats["elapsed_time"] = elapsed_time
            
            # Print results with predictions
            print("\nScoring Results:")
            print(f"Time taken: {elapsed_time:.2f}s")
            print(f"Remote prediction: {remote_pred}")
            print(f"Original prediction: {original_pred}")
            print(f"Prediction match: {'Yes' if predictions_match else 'No'}")
            print(f"Final score: {score:.2%}")
            
            # Print detailed stats
            rt_mean, rt_median, rt_std = stats["response_time_stats"]
            print(f"Response time stats - Mean: {rt_mean:.2f}s, Median: {rt_median:.2f}s, Std: {rt_std:.2f}s")
            score_mean, score_median, score_std = stats["score_stats"]
            print(f"Score stats - Mean: {score_mean:.2%}, Median: {score_median:.2%}, Std: {score_std:.2%}")
            print(f"Failure rate: {stats['failure_rate']:.2%}")

            return {
                'score': score,
                'stats': stats,
                'elapsed_time': elapsed_time,
                'predictions_match': predictions_match,
                'true_label': true_label.item(),
                'remote_pred': int(remote_pred),
                'original_pred': int(original_pred),
                'augmentation_seed': augmentation_seed
            }

        except Exception as e:
            print(f"Error during query for IP {self.url}: {str(e)}")
            return None
        finally:
            self._cleanup_files()

    def _cleanup_files(self):
        """Helper method to cleanup temporary files."""
        try:
            for file in ["./client.zip", "./compiled_model.pkl"]:
                if os.path.exists(file):
                    os.remove(file)
            if os.path.exists("./keys"):
                import shutil
                shutil.rmtree("./keys")
        except Exception as e:
            print(f"Failed to cleanup files: {e}")
