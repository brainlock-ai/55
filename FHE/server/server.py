"""Deployment server with Epistula authentication.

Routes:
    - Get client.zip
    - Add a key
    - Compute

All routes are protected with Epistula authentication.
"""

import asyncio
import io
import json
import os
import uuid
import time
import base58
import uvicorn
import websockets
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from hashlib import blake2b, sha256
from time import perf_counter

from fastapi import FastAPI, Form, HTTPException, UploadFile, Request, Response, Depends, File
from fastapi.responses import FileResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger
from substrateinterface import Keypair

# No relative import here because when not used in the package itself
from concrete.ml.deployment import FHEModelServer

MINER_HOTKEY = os.environ.get("MINER_HOTKEY")
if not MINER_HOTKEY:
    raise ValueError("MINER_HOTKEY environment variable must be set")

class EpistulaVerifier:
    """Handles verification of Epistula protocol authentication."""
    
    def __init__(self, allowed_delta_ms: int = 8000, cache_duration: int = 3600):
        self.ALLOWED_DELTA_MS = allowed_delta_ms
        self.MINER_HOTKEY = MINER_HOTKEY
        self.cache_duration = cache_duration  # Cache duration in seconds
        self.stake_cache: Dict[str, Tuple[float, float, bool]] = {}  # hotkey -> (timestamp, stake, valid)
        self.chain_endpoint = "wss://entrypoint-finney.opentensor.ai:443"
        logger.info(f"Initialized EpistulaVerifier with miner hotkey: {self.MINER_HOTKEY}")
    
    def ss58_decode(self, address: str, valid_ss58_format: Optional[int] = None) -> str:
        """
        Decodes given SS58 encoded address to an account ID
        Parameters
        ----------
        address: e.g. EaG2CRhJWPb7qmdcJvy3LiWdh26Jreu9Dx6R1rXxPmYXoDk
        valid_ss58_format

        Returns
        -------
        Decoded string AccountId
        """

        # Check if address is already decoded
        if address.startswith('0x'):
            return address

        if address == '':
            raise ValueError("Empty address provided")

        checksum_prefix = b'SS58PRE'

        address_decoded = base58.b58decode(address)

        if address_decoded[0] & 0b0100_0000:
            ss58_format_length = 2
            ss58_format = ((address_decoded[0] & 0b0011_1111) << 2) | (address_decoded[1] >> 6) | \
                        ((address_decoded[1] & 0b0011_1111) << 8)
        else:
            ss58_format_length = 1
            ss58_format = address_decoded[0]

        if ss58_format in [46, 47]:
            raise ValueError(f"{ss58_format} is a reserved SS58 format")

        if valid_ss58_format is not None and ss58_format != valid_ss58_format:
            raise ValueError("Invalid SS58 format")

        # Determine checksum length according to length of address string
        if len(address_decoded) in [3, 4, 6, 10]:
            checksum_length = 1
        elif len(address_decoded) in [5, 7, 11, 34 + ss58_format_length, 35 + ss58_format_length]:
            checksum_length = 2
        elif len(address_decoded) in [8, 12]:
            checksum_length = 3
        elif len(address_decoded) in [9, 13]:
            checksum_length = 4
        elif len(address_decoded) in [14]:
            checksum_length = 5
        elif len(address_decoded) in [15]:
            checksum_length = 6
        elif len(address_decoded) in [16]:
            checksum_length = 7
        elif len(address_decoded) in [17]:
            checksum_length = 8
        else:
            raise ValueError("Invalid address length")

        checksum = blake2b(checksum_prefix + address_decoded[0:-checksum_length]).digest()

        if checksum[0:checksum_length] != address_decoded[-checksum_length:]:
            raise ValueError("Invalid checksum")

        return address_decoded[ss58_format_length:len(address_decoded)-checksum_length].hex()
    
    def convert_ss58_to_hex(self, ss58_address: str) -> str:
        """Convert SS58 address to hex format."""
        address_bytes = self.ss58_decode(ss58_address)
        if isinstance(address_bytes, str):
            address_bytes = bytes.fromhex(address_bytes)
        return address_bytes.hex()
        
    async def call_rpc(self, call_params: List[str]) -> List[Dict]:
        """Call the RPC and return the results."""
        async with websockets.connect(self.chain_endpoint, ping_interval=None) as ws:
            await ws.send(json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "state_subscribeStorage",
                'params': [call_params],
            }))
            await ws.recv()  # Ignore confirmation response
            response = await ws.recv()
            changes = json.loads(response)["params"]["result"]["changes"]
            return changes
    
    async def get_stake_from_hotkey(self, hotkey: str) -> float:
        """Retrieve stakes for a hotkey asynchronously."""
        stake = 0.0
        call_params = [
            "0x" 
            + "658faa385070e074c85bf6b568cf0555"  # subtensorModule
            + "7b4e834c482cd6f103e108dacad0ab65"  # totalHotkeyStake
            + self.convert_ss58_to_hex(hotkey)
        ]
        
        try:
            call_results = await self.call_rpc(call_params)
            if call_results[0] is not None:
                stake_hex = call_results[0][1][2:]
                stake = int(stake_hex, 16)
                stake = round(stake / 1e9, 4)
                logger.debug(f"Retrieved stake for {hotkey}: {stake}")
            return stake
        except Exception as e:
            logger.error(f"Error getting stake for {hotkey}: {e}")
            raise

    async def verify_stake(self, hotkey: str) -> tuple[bool, float, str]:
        """Verify stake for a hotkey asynchronously."""
        current_time = time.time()
        
        # Check cache first
        if hotkey in self.stake_cache:
            timestamp, stake, valid = self.stake_cache[hotkey]
            if current_time - timestamp < self.cache_duration:
                logger.debug(f"Using cached stake for {hotkey}: {stake}")
                return valid, stake, ""

        # If not in cache or cache expired, verify with endpoint
        try:
            stake = await self.get_stake_from_hotkey(hotkey)
            
            # Update cache
            self.stake_cache[hotkey] = (current_time, stake, True)
            
            if stake < 10000:  # 10k Tao minimum
                return False, stake, f"Insufficient stake: {stake} < 10000"
                
            return True, stake, ""
        except Exception as e:
            logger.error(f"Stake verification error: {str(e)}")
            return False, 0, str(e)

    async def verify_signature(
        self,
        signature: str,
        body: bytes,
        timestamp: str,
        uuid_str: str,
        signed_for: Optional[str],
        signed_by: str,
        now: int,
        path: str = ""
    ) -> Optional[str]:
        # Verify stake first
        is_valid, stake, error = await self.verify_stake(signed_by)
        if not is_valid:
            logger.error(f"Stake verification failed for {signed_by}: {error}")
            return f"Stake verification failed: {error}"

        # Add miner hotkey verification
        if signed_for and signed_for != self.MINER_HOTKEY:
            logger.error(f"Request signed for {signed_for} but expected {self.MINER_HOTKEY}")
            return "Invalid signed_for address"

        # Add debug logging
        logger.debug(f"Verifying signature with params:")
        logger.debug(f"signature: {signature}")
        logger.debug(f"body hash: {sha256(body).hexdigest()}")
        logger.debug(f"timestamp: {timestamp}")
        logger.debug(f"uuid: {uuid_str}")
        logger.debug(f"signed_for: {signed_for}")
        logger.debug(f"signed_by: {signed_by}")
        logger.debug(f"current time: {now}")

        # Validate input types
        if not isinstance(signature, str):
            return "Invalid Signature"
        try:
            timestamp = int(timestamp)
        except (ValueError, TypeError):
            return "Invalid Timestamp"
        if not isinstance(signed_by, str):
            return "Invalid Sender key"
        if not isinstance(uuid_str, str):
            return "Invalid uuid"
        if not isinstance(body, bytes):
            return "Body is not of type bytes"

        # Skip staleness check for compute endpoint
        if not path.endswith("/compute"):
            if timestamp + self.ALLOWED_DELTA_MS < now:
                return "Request is too stale"

        try:
            keypair = Keypair(ss58_address=signed_by)
        except Exception as e:
            logger.error(f"Invalid Keypair for signed_by '{signed_by}': {e}")
            return "Invalid Keypair"

        # Verify signature
        message = f"{sha256(body).hexdigest()}.{uuid_str}.{timestamp}.{signed_for or ''}"
        logger.debug(f"Constructed message for verification: {message}")
        
        try:
            signature_bytes = bytes.fromhex(signature[2:])  # Remove '0x' prefix
            logger.debug(f"Parsed signature bytes (hex): {signature_bytes.hex()}")
        except ValueError as e:
            logger.error(f"Failed to parse signature: {e}")
            return "Invalid Signature Format"

        verified = keypair.verify(message, signature_bytes)
        logger.debug(f"Signature verification result: {verified}")
        
        if not verified:
            return "Signature Mismatch"
            
        return None

class RequestBodyCacheMiddleware(BaseHTTPMiddleware):
    """Middleware to cache request body for signature verification."""
    
    async def dispatch(self, request: Request, call_next):
        # Get content type
        content_type = request.headers.get('content-type', '')
        logger.debug(f"Request content type: {content_type}")
        
        # Read the raw body
        body = await request.body()
        logger.debug(f"Raw body hash: {sha256(body).hexdigest()}")
        
        # Store the raw body for signature verification
        request.state.body = body
        
        # For multipart, we need to create a new stream
        if content_type.startswith('multipart/form-data'):
            request._body = body
        
        return await call_next(request)

class TimingStats:
    """Track timing statistics for different operations"""
    def __init__(self):
        self.current_times = {}
    
    def start(self, operation: str):
        self.current_times[operation] = perf_counter()
    
    def end(self, operation: str) -> float:
        duration = perf_counter() - self.current_times[operation]
        logger.info(f"{operation}: {duration:.4f}s")
        return duration

# Create the FastAPI app instance at module level
app = FastAPI()

# Add body caching middleware
app.add_middleware(RequestBodyCacheMiddleware)

# Create the verifier instance
verifier = EpistulaVerifier()

FILE_FOLDER = Path(__file__).parent

KEY_PATH = Path(os.environ.get("KEY_PATH", FILE_FOLDER / Path("server_keys")))
CLIENT_SERVER_PATH = Path(os.environ.get("PATH_TO_MODEL", FILE_FOLDER / Path("dev")))
PORT = os.environ.get("PORT", "5000")

fhe = FHEModelServer(str(CLIENT_SERVER_PATH.resolve()))

KEYS: Dict[str, bytes] = {}

PATH_TO_CLIENT = (CLIENT_SERVER_PATH / "client.zip").resolve()
PATH_TO_SERVER = (CLIENT_SERVER_PATH / "server.zip").resolve()

assert PATH_TO_CLIENT.exists()
assert PATH_TO_SERVER.exists()

async def verify_epistula_request(request: Request, response_cls=Response):
    """Verify Epistula request with timing"""
    timer = TimingStats()
    timer.start("epistula_verification")
    
    # Get required headers directly
    signature = request.headers.get("Epistula-Request-Signature")
    timestamp = request.headers.get("Epistula-Timestamp")
    uuid_str = request.headers.get("Epistula-Uuid")
    signed_by = request.headers.get("Epistula-Signed-By")
    signed_for = request.headers.get("Epistula-Signed-For")
    
    # Validate required headers exist
    if not all([signature, timestamp, uuid_str, signed_by]):
        raise HTTPException(status_code=400, detail="Missing required Epistula headers")

    # Time signature verification
    timer.start("epistula_signature")
    try:
        error = await verifier.verify_signature(
            signature=signature,
            body=request.state.body,
            timestamp=timestamp,
            uuid_str=uuid_str,
            signed_for=signed_for,
            signed_by=signed_by,
            now=int(time.time() * 1000),
            path=request.url.path
        )
        if error:
            raise ValueError(error)
    except ValueError as e:
        timer.end("epistula_signature")
        raise HTTPException(status_code=400, detail=str(e))

    timer.end("epistula_verification")
    return True

@app.get("/get_client")
async def get_client(request: Request, _: None = Depends(verify_epistula_request)):
    """Get client.zip with Epistula authentication.

    Returns:
        FileResponse: client.zip file

    Raises:
        HTTPException: if the file can't be found locally
    """
    path_to_client = (CLIENT_SERVER_PATH / "client.zip").resolve()
    if not path_to_client.exists():
        raise HTTPException(status_code=500, detail="Could not find client.")
    return FileResponse(path_to_client, media_type="application/zip")

@app.post("/add_key")
async def add_key(
    request: Request,
    key: UploadFile,
    _: None = Depends(verify_epistula_request)
):
    """Add public key with Epistula authentication.

    Arguments:
        key (UploadFile): public key

    Returns:
        Dict[str, str]: Contains uid for the stored key
    """
    uid = str(uuid.uuid4())
    KEYS[uid] = await key.read()
    return {"uid": uid}

@app.post("/compute")
async def compute(
    request: Request,
    model_input: UploadFile = File(...),
    uid: str = Form(...),
    _: None = Depends(verify_epistula_request)
):
    """Compute the circuit over encrypted input with Epistula authentication."""
    timer = TimingStats()
    timer.start("total")
    
    try:
        # Time key lookup
        timer.start("key_lookup")
        try:
            key = KEYS[uid]
        except KeyError:
            raise HTTPException(status_code=400, detail=f"No key found for UID: {uid}")
        timer.end("key_lookup")

        # Time input reading
        timer.start("input_read")
        input_data = await model_input.read()
        timer.end("input_read")
        
        # Time FHE computation
        timer.start("fhe_computation")
        encrypted_results = fhe.run(
            serialized_encrypted_quantized_data=input_data,
            serialized_evaluation_keys=key,
        )
        timer.end("fhe_computation")

        # Time response creation
        timer.start("response_creation")
        response = Response(
            content=encrypted_results,
            media_type="application/octet-stream"
        )
        timer.end("response_creation")

        # Log total time
        total_time = timer.end("total")
        logger.info(f"Total request time: {total_time:.4f}s")
        logger.info("-" * 40)  # Separator for readability
        
        return response

    except Exception as e:
        logger.error(f"Error in compute endpoint: {str(e)}")
        raise

# Add more detailed error logging
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {str(exc)}")
    logger.error(f"Exception type: {type(exc)}")
    logger.error(f"Exception traceback:", exc_info=True)
    return {"detail": str(exc)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(PORT))