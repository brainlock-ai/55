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
from models import synthetic_cnv_2w2a
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
            self.model = synthetic_cnv_2w2a(False)
            
            # Determine device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
            
            # Find relative path to checkpoints
            dir_path = Path(__file__).parent.absolute()
            checkpoint_path = dir_path / "experiments/synthetic_model_checkpoint.pth"
            
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
    
    async def fetch_layer_outputs(self, response, iterations):
        """
        Asynchronous generator to fetch layer outputs with length-prefixed chunks.
        """
        for i in range(iterations):
            # Read the length header (4 bytes)
            length_bytes = await response.content.readexactly(4)
            layer_length = struct.unpack("<I", length_bytes)[0]

            if layer_length == 0:
                # End-of-stream marker received
                return  # Stop the generator

            # Read the chunk data
            chunk_data = await response.content.readexactly(layer_length)
            yield chunk_data, time.time()

    def compare_outputs_with_cosine_sim(self, reference_output, actual_output):
        """
        Compare two outputs using cosine similarity, returning a score in [0, 1].
        1 indicates the two outputs are perfectly aligned (cosine_sim = +1).
        0 indicates they point in opposite directions (cosine_sim = -1).

        Steps:
        1. Convert inputs to CPU numpy arrays if they're torch Tensors.
        2. Flatten both arrays into 1D vectors.
        3. Compute the dot product and norms to get the standard cosine similarity in [-1, +1].
        4. Transform it into a [0, 1] range via (1 + cos_sim) / 2.

        :param reference_output: The 'accurate' or reference result (tensor or array).
        :param actual_output:    The 'chunk' or submodel result (tensor or array).
        :return:                 A float similarity score in [0, 1].
        """

        # 1) Convert Torch tensors to numpy arrays if needed
        if isinstance(reference_output, torch.Tensor):
            reference_output = reference_output.detach().cpu().numpy()
        if isinstance(actual_output, torch.Tensor):
            actual_output = actual_output.detach().cpu().numpy()

        # 2) Flatten both arrays
        ref_vec = reference_output.ravel()
        act_vec = actual_output.ravel()

        # Avoid division by zero in case of zero vectors
        ref_norm = np.linalg.norm(ref_vec)
        act_norm = np.linalg.norm(act_vec)
        if ref_norm == 0 or act_norm == 0:
            # If either vector is all zeros, define similarity = 1 if both are zeros, else 0
            if ref_norm == 0 and act_norm == 0:
                return 1.0
            else:
                return 0.0

        # 3) Compute standard cosine similarity in [-1, 1]
        dot_product = np.dot(ref_vec, act_vec)
        cos_sim = dot_product / (ref_norm * act_norm)

        # 4) Transform cosine similarity to [0, 1]
        score = (1.0 + cos_sim) / 2.0
        return score

    async def get_client_zip(self):
        # Get client.zip using aiohttp with improved error handling
        bt.logging.info("Requesting client.zip...")
        try:
            headers = self.epistula.generate_headers(b"", signed_for=self.hotkey)
            bt.logging.debug(f"Request headers: {headers}")
            bt.logging.debug(f"Request URL: {self.url}/get_client")

            async with self.session.get(
                f"{self.url}/get_client",
                headers=headers,
                ssl=False,
                timeout=aiohttp.ClientTimeout(total=20)  # Increased timeout
            ) as response:
                if response.status != 200:
                    error_content = await response.text()
                    bt.logging.error(f"Get client request failed with status {response.status}")
                    bt.logging.error(f"Response headers: {response.headers}")
                    bt.logging.error(f"Response content: {error_content}")
                    return False

                content = await response.read()
                content_length = len(content) if content else 0
                bt.logging.debug(f"Received response with content length: {content_length} bytes")

                if not content:
                    bt.logging.error("Received empty response")
                    return False

                # Create directory if it doesn't exist
                os.makedirs(f"./{self.hotkey}", exist_ok=True)
                bt.logging.debug(f"Created/verified directory: ./{self.hotkey}")

                filepath = f"./{self.hotkey}/client.zip"
                with open(filepath, "wb") as f:
                    f.write(content)

                # Verify the file exists and has content
                file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
                bt.logging.debug(f"Wrote file {filepath} with size: {file_size} bytes")

                if not os.path.exists(filepath) or file_size == 0:
                    bt.logging.error(f"Failed to write client.zip or file is empty. Path: {filepath}, Size: {file_size}")
                    return False

                bt.logging.info(f"Successfully downloaded client.zip ({file_size} bytes)")
                return True

        except aiohttp.ClientError as e:
            bt.logging.error(f"Network error during get_client: {str(e)}")
            bt.logging.error(f"Error type: {type(e).__name__}")
            return False
        except Exception as e:
            bt.logging.error(f"Unexpected error during get_client: {str(e)}")
            bt.logging.error(f"Error type: {type(e).__name__}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            return False
    
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
                return uid
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

    def build_body_and_headers(self, uid, encrypted_input, iterations):
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
            b'--' + boundary + b'\r\n',
            b'Content-Disposition: form-data; name="iterations"\r\n',
            b'\r\n',
            str(iterations).encode(),
            b'\r\n'
        ])
        payload.extend([
            b'--' + boundary + b'--\r\n'
        ])
        
        body = b''.join(payload)
        headers = self.epistula.generate_headers(body, signed_for=self.hotkey)
        headers['Content-Type'] = f'multipart/form-data; boundary={boundary.decode()}'
        return body, headers

    async def query(self, image_index=None):
        """Main query method that can be called from validator."""
        try:
            # Cleanup any existing files
            self._cleanup_files()

            bt.logging.info(f"Connected to server at {self.url}")

            # Get client.zip and check result
            if not await self.get_client_zip():
                bt.logging.error("Failed to get client.zip, aborting query")
                return None

            # Initialize FHE client
            try:
                self.fhe_client = FHEModelClient(path_dir=f"./{self.hotkey}", key_dir="./keys")
            except Exception as e:
                bt.logging.error(f"Failed to initialize FHE client: {e}")
                return None

            # Initialize FHE client
            self.fhe_client = FHEModelClient(path_dir=f"./{self.hotkey}", key_dir="./keys")

            uid = await self.upload_evaluation_keys()

            # Select image (random if not specified)
            if image_index is None:
                image_index = random.randint(0, len(self.test_sub_set) - 1)
            X = self.test_sub_set[image_index:image_index+1]

            # Generate random seed for this query
            augmentation_seed = random.randint(0, 2**32 - 1)
            bt.logging.info(f"\nUsing augmentation seed: {augmentation_seed}")

            # Apply augmentation once
            torch.manual_seed(augmentation_seed)
            augmented_X = self.augmentation(X)
            original_input = augmented_X.to(self.device)

            # Use same augmented input for FHE inference
            clear_input = augmented_X.numpy()

            print(f"clear_input dtype: {clear_input.dtype}")
            print(f"clear_input min/max: {clear_input.min()}/{clear_input.max()}")
            print(f"unique values: {np.unique(clear_input)}")

            # 1. Multiply data by 255 to spread [0,1] float to [0,255] integer range.
            scaled_input = (clear_input * 255.0).round()  # or use .astype(np.uint8) after

            # 2. Clip values just in case of small floating rounding above 255
            scaled_input = np.clip(scaled_input, 0, 255)

            # 3. Convert to uint8
            scaled_input = scaled_input.astype(np.uint8)

            print(f"clear_input dtype: {scaled_input.dtype}")
            print(f"clear_input min/max: {scaled_input.min()}/{scaled_input.max()}")
            print(f"unique values: {np.unique(scaled_input)}")

            encrypted_input = self.fhe_client.quantize_encrypt_serialize(scaled_input)

            # How many times the miner should run the model in chain
            iterations = random.randint(5, 10)

            # Send compute request
            url = f"{self.url}/compute"
            
            body, headers = self.build_body_and_headers(uid, encrypted_input, iterations)
            
            # start first timer
            start_send_message_time = time.time()

            response = await self.post_with_retries(
                url, body, headers, max_retries=3, retry_delay=2
            )
            if not response or response.status != 200:
                print(f"Compute request failed with status {response.status if response else 'no response'}")
                return None

            previous_chunk_reception_time = start_send_message_time
            chunk_stats = []

            async for chunk_data, chunk_reception_time in self.fetch_layer_outputs(response, iterations):
                # Deserialize and decrypt the chunk
                remote_result = self.fhe_client.deserialize_decrypt_dequantize(chunk_data)
                chunk_stats.append({
                    "result": remote_result,
                    "timestamp": chunk_reception_time,
                    "inference_time": previous_chunk_reception_time - chunk_reception_time,
                })
                previous_chunk_reception_time = chunk_reception_time

            if not chunk_stats:
                print("No data received from the server.")
                return None

            # Compare submodel outputs
            total_score = 0.0
            chunk_simulated_output = None
            for i, chunk_stat in enumerate(chunk_stats):
                with torch.no_grad():
                    if i == 0:
                        chunk_simulated_output = self.fhe_client.run(scaled_input)
                    else:
                        previous_chunk_result = chunk_stats[i - 1]["result"]
                        chunk_simulated_output = self.fhe_client.run(previous_chunk_result)

                chunk_cosine_similarity_score = self.compare_outputs_with_cosine_sim(chunk_simulated_output, chunk_stat["result"])
                total_score += chunk_cosine_similarity_score

            average_cosine_similarity = total_score / len(chunk_stats)
            
            # Times for inferences (one chunk == one inference, if that's your assumption)
            inference_times = [cs["inference_time"] for cs in chunk_stats]

            end_inference_time = inference_times[-1]

            # Total time from request-sent to last inference
            total_time = end_inference_time - start_send_message_time

            # "inferences (or models) per second"
            num_inferences = len(chunk_stats)  # If each chunk is a single inference
            average_inference_per_second = num_inferences / total_time if total_time > 0 else 0.0

            # Check if the server might have buffered results
            # e.g. by looking at the time to the 20th percentile inference
            if num_inferences >= 5:
                twenty_percent_index = math.ceil(num_inferences * 0.20)
                time_to_twentieth_percent = (
                    inference_times[twenty_percent_index] - start_send_message_time
                )

                # If the first 20% of the inferences arrived after 70% of total time,
                # it's likely non-streamed
                if time_to_twentieth_percent / total_time >= 0.70:
                    print("Likely non-streamed response detected.")
                    return None

            print(f"Final average cosine similarity: {average_cosine_similarity:.4f}")
            inference_speed_and_accuracy_score = average_inference_per_second * average_cosine_similarity
            print(f"Inference score: {inference_speed_and_accuracy_score:.4f}")

            # Calculate final score using SimplifiedReward
            score, stats = self.reward_model.calculate_score(
                inference_speed_and_accuracy_score=inference_speed_and_accuracy_score,
                average_inference_per_second=average_inference_per_second,
                average_cosine_similarity=average_cosine_similarity,
                hotkey=self.hotkey
            )
            
            stats["average_cosine_similarity"] = average_cosine_similarity
            stats["elapsed_time"] = total_time
            
            # Print results with predictions
            print("\nScoring Results:")
            print(f"Time taken: {total_time:.2f}s")
            print(f"Final score: {score:.2%}")
            
            # Print detailed stats
            rt_mean, rt_median, rt_std = stats["response_time_stats"]
            print(f"Response time stats - Mean: {rt_mean:.2f}s, Median: {rt_median:.2f}s, Std: {rt_std:.2f}s")
            score_mean, score_median, score_std = stats["score_stats"]
            print(f"Score stats - Mean: {score_mean:.2%}, Median: {score_median:.2%}, Std: {score_std:.2%}")
            
            return {
                'score': score,
                'stats': stats,
                'elapsed_time': total_time,
                'average_cosine_similarity': average_cosine_similarity,
                'augmentation_seed': augmentation_seed
            }

        except Exception as e:
            print(f"Error during query for IP {self.url}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None
        finally:
            self._cleanup_files()

    def _cleanup_files(self):
        """Helper method to cleanup temporary files."""
        try:
            for file in [f"./{self.hotkey}/client.zip", "./compiled_model.pkl"]:
                if os.path.exists(file):
                    os.remove(file)
            if os.path.exists("./keys"):
                import shutil
                shutil.rmtree("./keys")
        except Exception as e:
            print(f"Failed to cleanup files: {e}")
